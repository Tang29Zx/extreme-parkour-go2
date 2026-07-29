"""Evaluate exported vision policy through the production onboard safety boundary."""

import faulthandler
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep
from typing import Any, Dict, Sequence

import isaacgym  # noqa: F401 -- Isaac Gym must be imported before torch.
import numpy as np
import torch

from legged_gym.envs import *  # noqa: F401,F403 -- registers task classes.
from legged_gym.scripts.play_jit import (
    RANDOM_BOX_TASKS,
    configure_fixed_single_box,
    load_models,
    validate_replay_args,
)
from legged_gym.utils import get_args, task_registry


L1_EVENT_STEP = 5
FORWARD_COMMAND_MPS = 0.5
DEFAULT_TRIALS = 20
COUNTER_NAMES = (
    "raw_action_clip_cycles",
    "transition_guard_cycles",
    "constraint_cycles",
    "output_guard_cycles",
    "any_clip_cycles",
    "target_step_limit_cycles",
    "joint_limit_cycles",
    "pd_torque_limit_cycles",
)
JOINT_COUNTER_NAMES = (
    "raw_action_clip_by_joint",
    "transition_guard_by_joint",
    "constraint_by_joint",
    "target_step_limit_by_joint",
    "joint_limit_by_joint",
    "pd_torque_limit_by_joint",
)


@dataclass(frozen=True)
class PolicyContract:
    """Deployment values loaded from the traced package config."""

    default_q: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    action_scale: float
    clip_actions: float
    ang_vel_scale: float
    dof_pos_scale: float
    dof_vel_scale: float


@dataclass
class TrialState:
    """Per-environment onboard state and evaluation counters."""

    env_id: int
    start_q: np.ndarray
    previous_target_q: np.ndarray
    last_action: np.ndarray
    last_contacts: np.ndarray
    transition: Any
    phase: str = "dryrun"
    phase_start_step: int = 0
    prime_gate: Any = None
    policy_start_step: int = -1
    completed: bool = False
    outcome: str = ""
    detail: str = ""
    policy_cycles: int = 0
    raw_action_clip_cycles: int = 0
    transition_guard_cycles: int = 0
    constraint_cycles: int = 0
    output_guard_cycles: int = 0
    any_clip_cycles: int = 0
    target_step_limit_cycles: int = 0
    joint_limit_cycles: int = 0
    pd_torque_limit_cycles: int = 0
    raw_action_clip_by_joint: np.ndarray = field(
        default_factory=lambda: np.zeros(12, dtype=np.int64)
    )
    transition_guard_by_joint: np.ndarray = field(
        default_factory=lambda: np.zeros(12, dtype=np.int64)
    )
    constraint_by_joint: np.ndarray = field(
        default_factory=lambda: np.zeros(12, dtype=np.int64)
    )
    target_step_limit_by_joint: np.ndarray = field(
        default_factory=lambda: np.zeros(12, dtype=np.int64)
    )
    joint_limit_by_joint: np.ndarray = field(
        default_factory=lambda: np.zeros(12, dtype=np.int64)
    )
    pd_torque_limit_by_joint: np.ndarray = field(
        default_factory=lambda: np.zeros(12, dtype=np.int64)
    )
    max_abs_raw_action: float = 0.0
    max_request_command_delta: float = 0.0
    max_command_step: float = 0.0
    max_predicted_torque_ratio: float = 0.0
    max_actual_torque_ratio: float = 0.0
    max_joint_velocity_ratio: float = 0.0
    max_abs_roll_deg: float = 0.0
    max_abs_pitch_deg: float = 0.0

    def finish(self, outcome: str, detail: str = "") -> None:
        if self.completed:
            return
        self.completed = True
        self.outcome = str(outcome)
        self.detail = str(detail)
        self.phase = "completed"


def load_onboard_modules(onboard_root: Path) -> Dict[str, Any]:
    """Load production pure modules without importing the onboard legged_gym copy."""

    module_names = (
        "joint_mapping",
        "policy_context",
        "real_control_safety",
        "unitree_boundary",
    )
    missing = [name for name in module_names if not (onboard_root / f"{name}.py").is_file()]
    if missing:
        raise FileNotFoundError(
            f"Onboard safety modules are missing from {onboard_root}: {missing}"
        )
    root_text = str(onboard_root)
    if root_text not in sys.path:
        sys.path.append(root_text)
    modules = {name: importlib.import_module(name) for name in module_names}
    for name, module in modules.items():
        module_path = Path(module.__file__).resolve()
        if module_path.parent != onboard_root:
            raise RuntimeError(
                f"Loaded {name} from {module_path}, expected {onboard_root}"
            )
    return modules


def _gain_vector(values: Dict[str, float], joint_names: Sequence[str]) -> np.ndarray:
    result = []
    for name in joint_names:
        matches = [float(value) for key, value in values.items() if key in name]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one gain for joint {name}.")
        result.append(matches[0])
    return np.asarray(result, dtype=np.float64)


def load_policy_contract(config_path: Path, joint_names: Sequence[str]) -> PolicyContract:
    """Read the actor and LowCmd contract from the paired traced config."""

    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    angles = config["init_state"]["default_joint_angles"]
    scales = config["normalization"]["obs_scales"]
    return PolicyContract(
        default_q=np.asarray([float(angles[name]) for name in joint_names]),
        kp=_gain_vector(config["control"]["stiffness"], joint_names),
        kd=_gain_vector(config["control"]["damping"], joint_names),
        action_scale=float(config["control"]["action_scale"]),
        clip_actions=float(config["normalization"]["clip_actions"]),
        ang_vel_scale=float(scales["ang_vel"]),
        dof_pos_scale=float(scales["dof_pos"]),
        dof_vel_scale=float(scales["dof_vel"]),
    )


def constraint_diagnostics(
    requested_q: Sequence[float],
    previous_q: Sequence[float],
    measured_q: Sequence[float],
    measured_dq: Sequence[float],
    kp: Sequence[float],
    kd: Sequence[float],
    joint_limits_low: Sequence[float],
    joint_limits_high: Sequence[float],
    torque_limits: Sequence[float],
    max_step_rad: Sequence[float],
) -> Dict[str, np.ndarray]:
    """Reconstruct which production bounds form the active clipping interval."""

    requested = np.asarray(requested_q, dtype=np.float64)
    previous = np.asarray(previous_q, dtype=np.float64)
    measured = np.asarray(measured_q, dtype=np.float64)
    velocity = np.asarray(measured_dq, dtype=np.float64)
    p_gain = np.asarray(kp, dtype=np.float64)
    d_gain = np.asarray(kd, dtype=np.float64)
    joint_lower = np.asarray(joint_limits_low, dtype=np.float64)
    joint_upper = np.asarray(joint_limits_high, dtype=np.float64)
    torque = np.asarray(torque_limits, dtype=np.float64)
    max_step = np.asarray(max_step_rad, dtype=np.float64)
    if max_step.ndim == 0:
        max_step = np.full(12, float(max_step), dtype=np.float64)
    vectors = (
        requested,
        previous,
        measured,
        velocity,
        p_gain,
        d_gain,
        joint_lower,
        joint_upper,
        torque,
        max_step,
    )
    if any(vector.shape != (12,) for vector in vectors):
        raise ValueError("Constraint diagnostics require twelve-value vectors.")

    step_lower = previous - max_step
    step_upper = previous + max_step
    torque_lower = measured + (-torque + d_gain * velocity) / p_gain
    torque_upper = measured + (torque + d_gain * velocity) / p_gain
    lower = np.maximum.reduce((step_lower, joint_lower, torque_lower))
    upper = np.minimum.reduce((step_upper, joint_upper, torque_upper))
    infeasible = lower > upper
    clip_low = requested < lower - 1e-10
    clip_high = requested > upper + 1e-10
    clipped = np.logical_or(clip_low, clip_high)

    def binding(lower_candidate, upper_candidate):
        return np.logical_or(
            clip_low & np.isclose(lower, lower_candidate, atol=1e-10, rtol=0.0),
            clip_high & np.isclose(upper, upper_candidate, atol=1e-10, rtol=0.0),
        )

    return {
        "lower": lower,
        "upper": upper,
        "infeasible": infeasible,
        "clipped": clipped,
        "step": binding(step_lower, step_upper),
        "joint": binding(joint_lower, joint_upper),
        "torque": binding(torque_lower, torque_upper),
    }


def summarize_trials(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate trial outcomes and non-exclusive safety-limit counters."""

    if not records:
        raise ValueError("At least one trial record is required.")
    outcomes = Counter(record["outcome"] for record in records)
    total_cycles = sum(int(record["policy_cycles"]) for record in records)
    successes = outcomes.get("success", 0)
    summary = {
        "trials": len(records),
        "successes": successes,
        "success_rate": successes / len(records),
        "outcomes": dict(sorted(outcomes.items())),
        "total_policy_cycles": total_cycles,
    }
    for name in COUNTER_NAMES:
        value = sum(int(record[name]) for record in records)
        summary[name] = value
        summary[f"{name}_rate"] = value / total_cycles if total_cycles else 0.0
    for name in JOINT_COUNTER_NAMES:
        summary[name] = np.sum(
            np.asarray([record[name] for record in records], dtype=np.int64),
            axis=0,
        ).tolist()
    for name in (
        "max_abs_raw_action",
        "max_request_command_delta",
        "max_command_step",
        "max_predicted_torque_ratio",
        "max_actual_torque_ratio",
        "max_joint_velocity_ratio",
        "max_abs_roll_deg",
        "max_abs_pitch_deg",
    ):
        summary[name] = max(float(record[name]) for record in records)
    successful_cycles = [
        int(record["policy_cycles"])
        for record in records
        if record["outcome"] == "success"
    ]
    summary["successful_policy_cycles"] = successful_cycles
    summary["mean_success_time_s"] = (
        sum(successful_cycles) / len(successful_cycles) / 50.0
        if successful_cycles
        else None
    )
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _resolve_output(path: str) -> Path:
    if path is None:
        raise ValueError("--episode_output is required.")
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _synthesize_low_states(env, modules):
    mapping = modules["joint_mapping"]
    boundary = modules["unitree_boundary"]
    isaac_q = env.dof_pos.detach().cpu().numpy().astype(np.float64)
    isaac_dq = env.dof_vel.detach().cpu().numpy().astype(np.float64)
    isaac_force = torch.linalg.norm(
        env.contact_forces[:, env.feet_indices], dim=-1
    ).detach().cpu().numpy().astype(np.float64)
    gyroscope = env.base_ang_vel.detach().cpu().numpy().astype(np.float64)
    quaternion_xyzw = env.root_states[:, 3:7].detach().cpu().numpy().astype(np.float64)
    states = []
    for index in range(env.num_envs):
        quaternion = quaternion_xyzw[index]
        states.append(
            boundary.BoundaryLowState(
                motor_q=mapping.isaac_to_policy(isaac_q[index]),
                motor_dq=mapping.isaac_to_policy(isaac_dq[index]),
                foot_force=mapping.isaac_feet_to_unitree(isaac_force[index]),
                gyroscope=gyroscope[index],
                imu_quaternion_wxyz=(
                    quaternion[3],
                    quaternion[0],
                    quaternion[1],
                    quaternion[2],
                ),
            )
        )
    return states


def _build_noisy_proprio(env_cfg, rows: np.ndarray, device) -> torch.Tensor:
    proprio = torch.as_tensor(rows, device=device, dtype=torch.float32)
    enabled = bool(
        env_cfg.noise.add_noise
        and getattr(env_cfg.noise, "apply_observation_noise", False)
    )
    if not enabled:
        return proprio
    scales = env_cfg.noise.noise_scales

    def add_uniform(start, stop, amplitude):
        if float(amplitude) <= 0.0:
            return
        target = proprio[:, start:stop]
        target.add_((2.0 * torch.rand_like(target) - 1.0) * float(amplitude))

    level = float(env_cfg.noise.noise_level)
    add_uniform(0, 3, float(scales.ang_vel) * level * float(env_cfg.normalization.obs_scales.ang_vel))
    add_uniform(3, 5, float(scales.rotation) * level)
    add_uniform(13, 25, float(scales.dof_pos) * level * float(env_cfg.normalization.obs_scales.dof_pos))
    add_uniform(25, 37, float(scales.dof_vel) * level * float(env_cfg.normalization.obs_scales.dof_vel))
    dropout = float(getattr(env_cfg.noise, "contact_dropout_prob", 0.0))
    if dropout > 0.0:
        contacts = proprio[:, 49:53] > 0.0
        contacts &= torch.rand_like(proprio[:, 49:53]) >= dropout
        proprio[:, 49:53] = torch.where(contacts, 0.5, -0.5)
    return proprio


def _policy_observation(proprio, history, vision_output):
    count = proprio.shape[0]
    observation = torch.zeros(count, 753, device=proprio.device, dtype=proprio.dtype)
    actor_proprio = proprio.clone()
    actor_proprio[:, 6:8] = vision_output[:, -2:] * 1.5
    observation[:, :53] = actor_proprio
    observation[:, -530:] = history.reshape(count, -1)
    return observation


def _idle_target(decoded, contract, safety):
    target = decoded.joint_q + contract.kd * decoded.joint_dq / contract.kp
    return np.clip(
        target,
        np.maximum(safety.GO2_JOINT_LIMITS_LOW, contract.default_q - contract.clip_actions),
        np.minimum(safety.GO2_JOINT_LIMITS_HIGH, contract.default_q + contract.clip_actions),
    )


def _randomization_records(env, initial_root, initial_q, modules):
    mapping = modules["joint_mapping"]
    friction = env.friction_coeffs_tensor.detach().cpu().reshape(env.num_envs, -1)
    mass = env.mass_params_tensor.detach().cpu().reshape(env.num_envs, -1)
    motor = env.motor_strength.detach().cpu()
    delay = env.action_delay_steps.detach().cpu()
    origins = env.env_origins.detach().cpu()
    records = []
    for index in range(env.num_envs):
        records.append(
            {
                "initial_root_offset_m": (initial_root[index, :3] - origins[index]).tolist(),
                "initial_q_offset_rad": (
                    np.asarray(mapping.isaac_to_policy(initial_q[index]), dtype=np.float64)
                ).tolist(),
                "rigid_shape_friction": friction[index].tolist(),
                "mass_and_com_delta": mass[index].tolist(),
                "motor_kp_strength": motor[0, index].tolist(),
                "motor_kd_strength": motor[1, index].tolist(),
                "action_delay_steps": int(delay[index]),
            }
        )
    return records


def _trial_record(state: TrialState, randomization: Dict[str, Any]) -> Dict[str, Any]:
    record = {
        "env_id": state.env_id,
        "outcome": state.outcome,
        "detail": state.detail,
        "policy_cycles": state.policy_cycles,
        **{name: int(getattr(state, name)) for name in COUNTER_NAMES},
        **{name: getattr(state, name).tolist() for name in JOINT_COUNTER_NAMES},
        "max_abs_raw_action": state.max_abs_raw_action,
        "max_request_command_delta": state.max_request_command_delta,
        "max_command_step": state.max_command_step,
        "max_predicted_torque_ratio": state.max_predicted_torque_ratio,
        "max_actual_torque_ratio": state.max_actual_torque_ratio,
        "max_joint_velocity_ratio": state.max_joint_velocity_ratio,
        "max_abs_roll_deg": state.max_abs_roll_deg,
        "max_abs_pitch_deg": state.max_abs_pitch_deg,
        "randomization": randomization,
    }
    return record


@torch.inference_mode()
def evaluate(args) -> None:
    validate_replay_args(args)
    if args.task not in RANDOM_BOX_TASKS:
        raise ValueError("Onboard single-box evaluation requires a random-box task.")
    if args.fixed_box_height is None:
        raise ValueError("--fixed_box_height is required.")
    if args.onboard_root is None:
        raise ValueError("--onboard_root is required.")
    if not math.isfinite(args.policy_duration_s) or args.policy_duration_s <= 0.0:
        raise ValueError("--policy_duration_s must be positive and finite.")
    if not math.isfinite(args.viewer_start_delay_s) or args.viewer_start_delay_s < 0.0:
        raise ValueError("--viewer_start_delay_s must be finite and non-negative.")
    if not math.isfinite(args.viewer_hold_s) or args.viewer_hold_s < 0.0:
        raise ValueError("--viewer_hold_s must be finite and non-negative.")

    output_path = _resolve_output(args.episode_output)
    onboard_root = Path(args.onboard_root).expanduser().resolve()
    modules = load_onboard_modules(onboard_root)
    mapping = modules["joint_mapping"]
    context = modules["policy_context"]
    control = modules["real_control_safety"]
    safety = modules["unitree_boundary"]
    traced_dir = (
        Path(args.jit_model_dir).expanduser().resolve()
        if args.jit_model_dir is not None
        else onboard_root / "traced"
    )
    config_path = traced_dir / "config.json"
    for path in (traced_dir / "base_jit.pt", traced_dir / "vision_weight.pt", config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing traced asset: {path}")
    contract = load_policy_contract(config_path, mapping.POLICY_DOF_NAMES)

    num_envs = args.num_envs if args.num_envs is not None else DEFAULT_TRIALS
    if num_envs <= 0:
        raise ValueError("--num_envs must be greater than zero.")
    viewer_env_id = 0 if args.viewer_env_id is None else int(args.viewer_env_id)
    if viewer_env_id < 0 or viewer_env_id >= num_envs:
        raise ValueError("--viewer_env_id must select an existing environment.")
    if args.headless and args.viewer_env_id is not None:
        raise ValueError("--viewer_env_id requires the interactive viewer.")
    args.num_envs = num_envs
    args.rows = 1
    args.cols = 1

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = num_envs
    configure_fixed_single_box(
        env_cfg,
        args.fixed_box_height,
        args.fixed_friction,
        args.fixed_ground_roughness,
        preserve_randomization=True,
    )
    env_cfg.env.episode_length_s = 60.0

    faulthandler.enable()
    started_at = monotonic()
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.enable_viewer_sync = env.viewer is not None
    if env.viewer is not None:
        env.lookat_id = viewer_env_id
        env.lookat(viewer_env_id)
        print(f"Interactive viewer following environment {viewer_env_id}")
        if args.viewer_start_delay_s > 0.0:
            print(
                "Interactive replay starts in "
                f"{args.viewer_start_delay_s:.1f} seconds"
            )
            delay_deadline = monotonic() + args.viewer_start_delay_s
            while monotonic() < delay_deadline:
                env.render(sync_frame_time=False)
                sleep(0.02)
    device = torch.device(env.device)
    base_model, depth_encoder = load_models(str(traced_dir), device)

    expected_default = np.asarray(mapping.policy_to_isaac(contract.default_q))
    actual_default = env.default_dof_pos_all[0].detach().cpu().numpy()
    if not np.allclose(actual_default, expected_default, atol=1e-7, rtol=0.0):
        raise RuntimeError("Traced and simulation default joint positions differ.")

    initial_root = env.root_states.detach().cpu().clone()
    initial_q = env.dof_pos.detach().cpu().numpy().copy()
    low_states = _synthesize_low_states(env, modules)
    decoded = [safety.decode_low_state(state) for state in low_states]
    trials = [
        TrialState(
            env_id=index,
            start_q=decoded[index].joint_q.copy(),
            previous_target_q=decoded[index].joint_q.copy(),
            last_action=np.zeros(12, dtype=np.float64),
            last_contacts=np.zeros(4, dtype=np.bool_),
            transition=control.PolicyTransitionGuard(),
        )
        for index in range(num_envs)
    ]
    randomization = _randomization_records(
        env, initial_root, initial_q, modules
    )
    for item in randomization:
        item["initial_q_offset_rad"] = (
            np.asarray(item["initial_q_offset_rad"]) - contract.default_q
        ).tolist()

    history = torch.zeros(num_envs, 10, 53, device=device)
    context_length = torch.zeros(num_envs, device=device)
    visual_output = None
    depth_frame = env.depth_buffer[:, -2].clone().to(device)
    latest_depth_step = np.full(num_envs, -1, dtype=np.int64)
    max_policy_cycles = max(
        1,
        int(math.floor(args.policy_duration_s / float(env.dt) + 1e-9)),
    )
    max_steps = L1_EVENT_STEP + int(math.ceil(
        (control.STARTUP_RAMP_S + control.POLICY_PRIME_S) / float(env.dt)
    )) + max_policy_cycles + 100

    print(
        "Onboard evaluation: "
        f"height={args.fixed_box_height:.2f} m, trials={num_envs}, "
        f"seed={env_cfg.seed}, policy_limit={args.policy_duration_s:.2f} s"
    )
    print(
        "Randomization enabled: start pose/joints, observation/depth, camera "
        "extrinsics/FOV, friction, mass/COM, motor, pushes, action delay"
    )

    step = 0
    while step < max_steps and not all(trial.completed for trial in trials):
        sim_time = step * float(env.dt)
        env.commands.zero_()
        env.commands[:, 0] = FORWARD_COMMAND_MPS
        low_states = _synthesize_low_states(env, modules)
        decoded = [safety.decode_low_state(state) for state in low_states]

        if step == L1_EVENT_STEP:
            for index, trial in enumerate(trials):
                try:
                    control.validate_takeover_inputs(
                        decoded[index].joint_q,
                        decoded[index].joint_dq,
                        0.0,
                        0.0,
                    )
                except RuntimeError as error:
                    trial.finish("takeover_rejected", str(error))
                    continue
                trial.phase = "startup"
                trial.phase_start_step = step
                trial.start_q = decoded[index].joint_q.copy()
                trial.previous_target_q = trial.start_q.copy()

        for index, trial in enumerate(trials):
            if trial.phase != "startup" or trial.completed:
                continue
            elapsed = (step - trial.phase_start_step) * float(env.dt)
            if elapsed + 1e-12 < control.STARTUP_RAMP_S:
                continue
            trial.phase = "prime"
            trial.phase_start_step = step
            trial.last_action.fill(0.0)
            trial.last_contacts.fill(False)
            trial.prime_gate = control.PolicyPrimeGate(sim_time)
            history[index].zero_()
            context_length[index] = 0.0
        prime_reset = torch.as_tensor(
            [
                trial.phase == "prime" and trial.phase_start_step == step
                for trial in trials
            ],
            device=device,
            dtype=torch.bool,
        )
        if bool(torch.any(prime_reset)):
            depth_encoder.reset(prime_reset)

        proprio_rows = []
        for index, trial in enumerate(trials):
            row, current_contacts = safety.build_policy_proprio(
                decoded[index],
                contract.default_q,
                trial.last_action,
                trial.last_contacts,
                FORWARD_COMMAND_MPS,
                contract.ang_vel_scale,
                contract.dof_pos_scale,
                contract.dof_vel_scale,
                "parkour",
            )
            trial.last_contacts = current_contacts
            proprio_rows.append(row)
        proprio = _build_noisy_proprio(
            env_cfg, np.stack(proprio_rows), device
        )
        context_mask = torch.as_tensor(
            [trial.phase in ("prime", "policy") for trial in trials],
            device=device,
            dtype=torch.bool,
        )
        if bool(torch.any(context_mask)):
            updated_history = context.update_proprio_history(
                history, proprio, context_length
            )
            history[context_mask] = updated_history[context_mask]
            context_length[context_mask] += 1.0

        depth_updated = depth_frame is not None and bool(torch.any(context_mask))
        if depth_updated:
            visual_output = depth_encoder(depth_frame, proprio)
            if visual_output.shape != (num_envs, 34) or not torch.isfinite(
                visual_output
            ).all():
                raise RuntimeError("Depth encoder produced invalid output.")
            latest_depth_step[context_mask.detach().cpu().numpy()] = step

        for index, trial in enumerate(trials):
            if trial.phase != "prime" or trial.completed:
                continue
            if latest_depth_step[index] < 0:
                continue
            control.validate_policy_prime_inputs(
                0.0,
                (step - latest_depth_step[index]) * float(env.dt),
            )
            trial.prime_gate.record_proprio()
            if depth_updated:
                trial.prime_gate.record_depth()
            if not trial.prime_gate.ready(sim_time):
                continue
            try:
                control.validate_policy_request_input(0.0)
                control.validate_policy_entry_state(
                    decoded[index].foot_force,
                    decoded[index].roll_pitch[0],
                    decoded[index].roll_pitch[1],
                    decoded[index].joint_q,
                    trial.previous_target_q,
                    np.zeros(12),
                    np.zeros(12),
                    np.zeros(12),
                )
            except RuntimeError as error:
                trial.finish("entry_rejected", str(error))
                continue
            trial.transition.begin(trial.previous_target_q, sim_time)
            trial.phase = "policy"
            trial.phase_start_step = step
            trial.policy_start_step = step

        raw_actions = np.zeros((num_envs, 12), dtype=np.float64)
        if any(trial.phase == "policy" for trial in trials):
            if visual_output is None:
                raise RuntimeError("Policy entry has no visual context.")
            policy_observation = _policy_observation(
                proprio, history, visual_output
            )
            action_tensor = base_model(
                policy_observation, visual_output[:, :32]
            )
            if action_tensor.shape != (num_envs, 12) or not torch.isfinite(
                action_tensor
            ).all():
                raise RuntimeError("Base policy produced invalid actions.")
            raw_actions = action_tensor.detach().cpu().numpy().astype(np.float64)

        env_actions = np.zeros((num_envs, 12), dtype=np.float64)
        for index, trial in enumerate(trials):
            state = decoded[index]
            requested = trial.previous_target_q.copy()
            commanded = trial.previous_target_q.copy()
            if trial.completed:
                commanded = _idle_target(state, contract, safety)
            elif trial.phase == "dryrun":
                commanded = trial.start_q.copy()
            elif trial.phase == "startup":
                elapsed = (step - trial.phase_start_step) * float(env.dt)
                commanded = control.interpolate_pose(
                    trial.start_q,
                    contract.default_q,
                    elapsed,
                    control.STARTUP_RAMP_S,
                )
            elif trial.phase == "prime":
                commanded = contract.default_q.copy()
            elif trial.phase == "policy":
                trial.policy_cycles += 1
                raw = raw_actions[index]
                trial.max_abs_raw_action = max(
                    trial.max_abs_raw_action, float(np.max(np.abs(raw)))
                )
                try:
                    if latest_depth_step[index] < 0:
                        raise RuntimeError("runtime depth timestamp is unavailable")
                    control.validate_policy_runtime_inputs(
                        0.0,
                        (step - latest_depth_step[index]) * float(env.dt),
                        state.joint_q,
                        state.joint_dq,
                        safety.GO2_JOINT_LIMITS_LOW,
                        safety.GO2_JOINT_LIMITS_HIGH,
                        safety.GO2_JOINT_VELOCITY_LIMITS,
                    )
                    observed, clipped, requested = control.prepare_policy_action(
                        raw,
                        contract.default_q,
                        contract.clip_actions,
                        contract.action_scale,
                    )
                    trial.last_action = observed
                    raw_clip_joints = ~np.isclose(
                        observed, clipped, atol=1e-12, rtol=0.0
                    )
                    raw_clip_cycle = bool(np.any(raw_clip_joints))
                    if raw_clip_cycle:
                        trial.raw_action_clip_cycles += 1
                        trial.raw_action_clip_by_joint += raw_clip_joints

                    engagement_active = trial.transition.active
                    transition_target = (
                        trial.transition.apply(requested, sim_time)
                        if engagement_active
                        else requested.copy()
                    )
                    transition_joints = ~np.isclose(
                        transition_target, requested, atol=1e-12, rtol=0.0
                    )
                    transition_cycle = bool(np.any(transition_joints))
                    if transition_cycle:
                        trial.transition_guard_cycles += 1
                        trial.transition_guard_by_joint += transition_joints

                    max_step = (
                        control.POLICY_TRANSITION_MAX_STEP_RAD
                        if engagement_active
                        else control.POLICY_TARGET_MAX_STEP_RAD_BY_JOINT
                    )
                    diagnostics = constraint_diagnostics(
                        transition_target,
                        trial.previous_target_q,
                        state.joint_q,
                        state.joint_dq,
                        contract.kp,
                        contract.kd,
                        safety.GO2_JOINT_LIMITS_LOW,
                        safety.GO2_JOINT_LIMITS_HIGH,
                        safety.GO2_TORQUE_LIMITS,
                        max_step,
                    )
                    commanded = control.constrain_policy_target(
                        transition_target,
                        trial.previous_target_q,
                        state.joint_q,
                        state.joint_dq,
                        contract.kp,
                        contract.kd,
                        safety.GO2_JOINT_LIMITS_LOW,
                        safety.GO2_JOINT_LIMITS_HIGH,
                        safety.GO2_TORQUE_LIMITS,
                        max_step_rad=max_step,
                    )
                    expected = np.clip(
                        transition_target,
                        diagnostics["lower"],
                        diagnostics["upper"],
                    )
                    if not np.allclose(commanded, expected, atol=1e-10, rtol=0.0):
                        raise RuntimeError("Safety diagnostic and production target differ")
                    if engagement_active:
                        trial.transition.record_executed_target(commanded)

                    constraint_joints = ~np.isclose(
                        commanded, transition_target, atol=1e-12, rtol=0.0
                    )
                    constraint_cycle = bool(np.any(constraint_joints))
                    if constraint_cycle:
                        trial.constraint_cycles += 1
                        trial.constraint_by_joint += constraint_joints
                    for key, cycle_name, joint_name in (
                        ("step", "target_step_limit_cycles", "target_step_limit_by_joint"),
                        ("joint", "joint_limit_cycles", "joint_limit_by_joint"),
                        ("torque", "pd_torque_limit_cycles", "pd_torque_limit_by_joint"),
                    ):
                        hits = diagnostics[key] & constraint_joints
                        if bool(np.any(hits)):
                            setattr(trial, cycle_name, getattr(trial, cycle_name) + 1)
                            setattr(trial, joint_name, getattr(trial, joint_name) + hits)

                    output_joints = ~np.isclose(
                        commanded, requested, atol=1e-12, rtol=0.0
                    )
                    output_cycle = bool(np.any(output_joints))
                    if output_cycle:
                        trial.output_guard_cycles += 1
                    if raw_clip_cycle or output_cycle:
                        trial.any_clip_cycles += 1
                    trial.max_request_command_delta = max(
                        trial.max_request_command_delta,
                        float(np.max(np.abs(requested - commanded))),
                    )
                except RuntimeError as error:
                    trial.finish("safety_fault", str(error))
                    commanded = _idle_target(state, contract, safety)
            else:
                raise RuntimeError(f"Unsupported trial phase: {trial.phase}")

            lowcmd = safety.encode_low_cmd(commanded, contract.kp, contract.kd)
            physical_target = np.asarray(
                mapping.real_to_sim(lowcmd.motor_q), dtype=np.float64
            )
            command_step = float(
                np.max(np.abs(physical_target - trial.previous_target_q))
            )
            if trial.phase == "policy" and not trial.completed:
                trial.max_command_step = max(trial.max_command_step, command_step)
            predicted_torque = (
                lowcmd.motor_kp
                * (lowcmd.motor_q - np.asarray(low_states[index].motor_q))
                - lowcmd.motor_kd * np.asarray(low_states[index].motor_dq)
            )
            if trial.phase == "policy" and not trial.completed:
                trial.max_predicted_torque_ratio = max(
                    trial.max_predicted_torque_ratio,
                    float(
                        np.max(
                            np.abs(predicted_torque) / safety.GO2_TORQUE_LIMITS
                        )
                    ),
                )
            if not trial.completed:
                trial.previous_target_q = physical_target.copy()
            env_actions[index] = (
                physical_target - contract.default_q
            ) / contract.action_scale

        if all(trial.completed for trial in trials):
            break

        _, _, _, done, infos = env.step(
            torch.as_tensor(env_actions, device=device, dtype=torch.float32)
        )
        depth_frame = infos.get("depth")
        depth_encoder.reset(done)

        torque_ratio = torch.abs(env.torques) / env.torque_limits
        velocity_ratio = torch.abs(env.dof_vel) / torch.as_tensor(
            mapping.policy_to_isaac(safety.GO2_JOINT_VELOCITY_LIMITS),
            device=device,
            dtype=env.dof_vel.dtype,
        )
        for index, trial in enumerate(trials):
            if trial.completed:
                continue
            if trial.phase == "policy":
                trial.max_actual_torque_ratio = max(
                    trial.max_actual_torque_ratio,
                    float(torch.max(torque_ratio[index]).item()),
                )
                trial.max_joint_velocity_ratio = max(
                    trial.max_joint_velocity_ratio,
                    float(torch.max(velocity_ratio[index]).item()),
                )
                trial.max_abs_roll_deg = max(
                    trial.max_abs_roll_deg,
                    abs(math.degrees(float(env.roll[index]))),
                )
                trial.max_abs_pitch_deg = max(
                    trial.max_abs_pitch_deg,
                    abs(math.degrees(float(env.pitch[index]))),
                )
            if bool(done[index]):
                if trial.phase == "policy" and bool(env.goal_success_buf[index]):
                    trial.finish("success")
                elif bool(env.fall_buf[index]):
                    trial.finish("fall", "PhysX roll/pitch/height termination")
                elif trial.phase == "policy":
                    trial.finish("environment_reset", "non-success environment reset")
                else:
                    trial.finish("pre_policy_reset", f"reset during {trial.phase}")
                continue
            if (
                trial.phase == "policy"
                and trial.policy_cycles >= max_policy_cycles
            ):
                trial.finish("timeout", "policy duration limit reached")

        step += 1
        if step % 250 == 0:
            counts = Counter(trial.outcome or trial.phase for trial in trials)
            print(f"step={step} progress={dict(sorted(counts.items()))}")

    for trial in trials:
        if not trial.completed:
            trial.finish("evaluator_incomplete", "global step budget exhausted")

    if env.viewer is not None and args.viewer_hold_s > 0.0:
        print(
            "Interactive replay complete; holding the final frame for "
            f"{args.viewer_hold_s:.1f} seconds"
        )
        hold_deadline = monotonic() + args.viewer_hold_s
        while monotonic() < hold_deadline:
            env.render(sync_frame_time=False)
            sleep(0.02)

    records = [
        _trial_record(trial, randomization[trial.env_id]) for trial in trials
    ]
    summary = summarize_trials(records)
    payload = {
        "schema_version": 1,
        "task": args.task,
        "seed": int(env_cfg.seed),
        "training_repository_commit": _git_commit(
            Path(__file__).resolve().parents[3]
        ),
        "onboard_repository_commit": _git_commit(onboard_root),
        "traced_assets": {
            path.name: _sha256(path)
            for path in (
                traced_dir / "base_jit.pt",
                traced_dir / "vision_weight.pt",
                config_path,
            )
        },
        "box": {
            "height_m": float(args.fixed_box_height),
            "length_m": 1.2,
            "width_m": 1.2,
            "first_runup_m": 1.2,
            "ground_roughness_m": float(args.fixed_ground_roughness),
        },
        "control": {
            "frequency_hz": 1.0 / float(env.dt),
            "policy_duration_s": float(args.policy_duration_s),
            "action_scale": contract.action_scale,
            "raw_action_clip": contract.clip_actions / contract.action_scale,
            "transition_step_rad": control.POLICY_TRANSITION_MAX_STEP_RAD,
            "steady_step_rad_by_joint": list(
                control.POLICY_TARGET_MAX_STEP_RAD_BY_JOINT
            ),
            "joint_limits_low": safety.GO2_JOINT_LIMITS_LOW.tolist(),
            "joint_limits_high": safety.GO2_JOINT_LIMITS_HIGH.tolist(),
            "joint_velocity_limits": safety.GO2_JOINT_VELOCITY_LIMITS.tolist(),
            "torque_limits": safety.GO2_TORQUE_LIMITS.tolist(),
        },
        "randomization": {
            "start_position_m": list(env_cfg.env.start_pos_range),
            "start_yaw_rad": float(env_cfg.env.rand_yaw_range),
            "start_joint_offset_rad": list(env_cfg.env.dof_pos_reset_range),
            "friction": list(env_cfg.domain_rand.friction_range),
            "added_mass_kg": list(env_cfg.domain_rand.added_mass_range),
            "added_com_m": list(env_cfg.domain_rand.added_com_range),
            "motor_strength": list(env_cfg.domain_rand.motor_strength_range),
            "action_delay_steps": list(env_cfg.domain_rand.action_delay_range),
            "push_interval_s": float(env_cfg.domain_rand.push_interval_s),
            "max_push_velocity_mps": float(env_cfg.domain_rand.max_push_vel_xy),
            "camera_position": env_cfg.depth.position,
            "camera_rotation": env_cfg.depth.rotation,
            "camera_horizontal_fov_deg": env_cfg.depth.horizontal_fov,
            "depth_gaussian_std_m": float(env_cfg.depth.gaussian_noise_std),
            "depth_distance_bias_m": float(env_cfg.depth.dis_noise),
            "depth_dropout_probability": float(env_cfg.depth.depth_dropout_prob),
            "depth_occlusion_probability": float(env_cfg.depth.depth_occlusion_prob),
            "depth_occlusion_fraction": list(env_cfg.depth.depth_occlusion_size_range),
        },
        "summary": summary,
        "trials": records,
        "elapsed_wall_time_s": monotonic() - started_at,
    }
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved onboard single-box evaluation: {output_path}")


if __name__ == "__main__":
    evaluate(get_args())
