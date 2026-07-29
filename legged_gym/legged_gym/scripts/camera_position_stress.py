"""Evaluate labeled optical-center positions on one fixed single-box task."""

import faulthandler
import json
import math
from pathlib import Path
from time import monotonic

import isaacgym
import torch

from legged_gym.envs import *
from legged_gym.scripts.play_jit import (
    RANDOM_BOX_TASKS,
    configure_camera_position_stress_noise,
    configure_fixed_single_box,
    load_models,
    resolve_model_paths,
    set_per_env_camera_positions,
    validate_replay_args,
)
from legged_gym.utils import get_args, task_registry


BASELINE_POSITION_M = [0.355, 0.0, 0.065]
OUTCOME_KEYS = (
    "success_rate",
    "fall_rate",
    "timeout_rate",
    "goals_reached",
    "episode_duration_s",
    "distance_traveled_m",
)


def load_position_specs(path):
    """Load unique labeled optical-center positions from JSON."""

    if path is None:
        raise ValueError("--camera_positions_file is required.")
    position_path = Path(path).expanduser().resolve()
    with position_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, list) or not payload:
        raise ValueError("Camera position JSON must be a non-empty list.")

    specs = []
    labels = set()
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Every camera position entry must be an object.")
        label = item.get("label")
        position = item.get("position_m")
        if not isinstance(label, str) or not label or label in labels:
            raise ValueError("Camera position labels must be non-empty and unique.")
        if (
            not isinstance(position, list)
            or len(position) != 3
            or not all(isinstance(value, (int, float)) for value in position)
            or not all(math.isfinite(float(value)) for value in position)
        ):
            raise ValueError(f"Invalid position_m for {label}.")
        labels.add(label)
        specs.append(
            {
                "label": label,
                "position_m": [float(value) for value in position],
            }
        )
    return position_path, specs


def expand_trials(specs, trials_per_position):
    """Expand each position into parallel one-episode trials."""

    if trials_per_position <= 0:
        raise ValueError("--trials_per_position must be greater than zero.")
    assignments = []
    for position_index, spec in enumerate(specs):
        for trial_index in range(trials_per_position):
            assignments.append(
                {
                    "position_index": position_index,
                    "trial_index": trial_index,
                    "label": spec["label"],
                    "position_m": spec["position_m"],
                }
            )
    return assignments


def metric_values(episode, key, count):
    """Return one float per completed env while preserving env-id ordering."""

    value = episode.get(key)
    if value is None:
        return [None] * count
    if torch.is_tensor(value):
        flattened = value.detach().cpu().flatten().tolist()
    elif isinstance(value, (list, tuple)):
        flattened = list(value)
    else:
        flattened = [value]
    if len(flattened) == 1 and count > 1:
        flattened *= count
    if len(flattened) != count:
        raise RuntimeError(
            f"Episode metric {key} has {len(flattened)} values for {count} envs."
        )
    return [None if item is None else float(item) for item in flattened]


def summarize_trials(specs, records):
    """Aggregate first-episode results by camera position."""

    summaries = []
    for spec in specs:
        trials = [record for record in records if record["label"] == spec["label"]]
        episodes = len(trials)
        summary = {
            **spec,
            "offset_from_old_mean_m": [
                position - baseline
                for position, baseline in zip(
                    spec["position_m"], BASELINE_POSITION_M
                )
            ],
            "episodes": episodes,
            "waypoint_successes": sum(
                record["success_rate"] == 1.0 for record in trials
            ),
            "box_pass_successes": sum(
                record["box_pass_success"] for record in trials
            ),
            "falls": sum(record["fall_rate"] == 1.0 for record in trials),
            "timeouts": sum(record["timeout_rate"] == 1.0 for record in trials),
            "mean_goals_reached": sum(
                record["goals_reached"] for record in trials
            ) / episodes,
            "mean_duration_s": sum(
                record["episode_duration_s"] for record in trials
            ) / episodes,
        }
        summary["waypoint_success_rate"] = (
            summary["waypoint_successes"] / episodes
        )
        summary["box_pass_success_rate"] = (
            summary["box_pass_successes"] / episodes
        )
        summary["fall_rate"] = summary["falls"] / episodes
        summary["timeout_rate"] = summary["timeouts"] / episodes
        summaries.append(summary)
    return summaries


def resolve_output_path(path):
    if path is None:
        raise ValueError("--episode_output is required.")
    output_path = Path(path).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Episode output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


@torch.inference_mode()
def evaluate(args):
    validate_replay_args(args)
    if args.task not in RANDOM_BOX_TASKS:
        raise ValueError("Camera position stress requires a random-box task.")
    if args.fixed_box_height is None:
        raise ValueError("--fixed_box_height is required.")
    if args.fixed_friction is None:
        raise ValueError("--fixed_friction is required.")

    position_path, specs = load_position_specs(args.camera_positions_file)
    assignments = expand_trials(specs, args.trials_per_position)
    output_path = resolve_output_path(args.episode_output)
    args.num_envs = len(assignments)

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = len(assignments)
    configure_fixed_single_box(
        env_cfg,
        args.fixed_box_height,
        args.fixed_friction,
        args.fixed_ground_roughness,
    )
    configure_camera_position_stress_noise(env_cfg)
    set_per_env_camera_positions(
        env_cfg,
        [assignment["position_m"] for assignment in assignments],
    )

    faulthandler.enable()
    started_at = monotonic()
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.enable_viewer_sync = False
    obs = env.get_observations()
    base_model_path, vision_model_path = resolve_model_paths(args.jit_model_dir)
    base_model, depth_encoder = load_models(args.jit_model_dir, env.device)
    depth_latent = torch.zeros(env.num_envs, 32, device=env.device)
    vision_yaw = torch.zeros(env.num_envs, 2, device=env.device)
    infos = {"depth": env.depth_buffer.clone().to(env.device)[:, -1]}
    completed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    max_forward_displacement = (
        env.root_states[:, 0] - env.env_origins[:, 0]
    ).clone()
    exit_plane_displacement = (
        env.env_goals[:, 1, 0] - env.env_origins[:, 0]
    ).clone()
    records = []
    max_steps = int(env.max_episode_length) + 10

    for step in range(max_steps):
        max_forward_displacement = torch.maximum(
            max_forward_displacement,
            env.root_states[:, 0] - env.env_origins[:, 0],
        )
        depth_frame = infos.get("depth")
        if depth_frame is not None:
            depth_proprio = obs[:, : env.cfg.env.n_proprio].clone()
            depth_proprio[:, 6:8] = 0
            vision_output = depth_encoder(depth_frame, depth_proprio)
            if vision_output.shape[-1] != 34 or not torch.isfinite(
                vision_output
            ).all():
                raise RuntimeError("Vision encoder output is invalid.")
            depth_latent = vision_output[:, :32]
            vision_yaw = 1.5 * vision_output[:, -2:]

        policy_obs = obs.clone()
        policy_obs[:, 6:8] = vision_yaw
        actions = base_model(policy_obs, depth_latent)
        if actions.shape[-1] != 12 or not torch.isfinite(actions).all():
            raise RuntimeError("Base policy output is invalid.")

        obs, _, _, dones, infos = env.step(actions)
        depth_encoder.reset(dones)
        done_ids = torch.nonzero(dones, as_tuple=False).flatten().tolist()
        if done_ids:
            episode = infos.get("episode", {})
            values = {
                key: metric_values(episode, key, len(done_ids))
                for key in OUTCOME_KEYS
            }
            for metric_index, env_id in enumerate(done_ids):
                if completed[env_id]:
                    continue
                assignment = assignments[env_id]
                record = {
                    **assignment,
                    "env_id": env_id,
                    "completed_step": step + 1,
                }
                record.update(
                    {
                        key: values[key][metric_index]
                        for key in OUTCOME_KEYS
                    }
                )
                record["max_forward_displacement_m"] = float(
                    max_forward_displacement[env_id].detach().cpu()
                )
                record["exit_plane_displacement_m"] = float(
                    exit_plane_displacement[env_id].detach().cpu()
                )
                crossed_exit_plane = (
                    record["max_forward_displacement_m"]
                    >= record["exit_plane_displacement_m"]
                )
                record["crossed_exit_plane"] = crossed_exit_plane
                record["box_pass_success"] = bool(
                    record["success_rate"] == 1.0
                    or (crossed_exit_plane and record["fall_rate"] == 0.0)
                )
                records.append(record)
                completed[env_id] = True
        if bool(torch.all(completed)):
            break

    if not bool(torch.all(completed)):
        missing = torch.nonzero(~completed, as_tuple=False).flatten().tolist()
        raise RuntimeError(f"No first-episode result for envs: {missing}")

    payload = {
        "schema_version": 1,
        "policy": {
            "type": "jit_depth_student",
            "base_model": str(base_model_path),
            "vision_model": str(vision_model_path),
        },
        "task": args.task,
        "position_source": str(position_path),
        "baseline_position_m": BASELINE_POSITION_M,
        "box": {
            "height_m": args.fixed_box_height,
            "length_m": 1.2,
            "width_m": 1.2,
            "first_runup_m": 1.2,
            "ground_roughness_m": args.fixed_ground_roughness,
            "pass_plane_after_rear_m": 1.0,
        },
        "fixed_friction": args.fixed_friction,
        "noise_profile": {
            "proprioception_noise_level": 1.0,
            "contact_dropout_probability": 0.01,
            "action_delay_steps": 2,
            "camera_position_std_m": [0.0, 0.0, 0.0],
            "camera_rotation_rpy_deg": [0.0, 22.5, 0.0],
            "horizontal_fov_deg": 86,
            "depth_gaussian_std_m": 0.01,
            "depth_distance_bias_range_m": [-0.02, 0.02],
            "depth_dropout_probability": 0.01,
            "depth_occlusion_probability": 0.30,
            "depth_occlusion_fraction": 0.15,
        },
        "seed": env_cfg.seed,
        "trials_per_position": args.trials_per_position,
        "elapsed_s": monotonic() - started_at,
        "summary": summarize_trials(specs, records),
        "trials": records,
    }
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Saved camera position stress results: {output_path}")


if __name__ == "__main__":
    evaluate(get_args())
