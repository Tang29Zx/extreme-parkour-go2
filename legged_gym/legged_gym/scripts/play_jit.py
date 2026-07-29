
import faulthandler
from copy import deepcopy
import math
from pathlib import Path
import subprocess
import tempfile
import time
from datetime import datetime

import isaacgym
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils import webviewer
from rsl_rl.modules import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone


RANDOM_BOX_TASKS = {
    "go2_random_box",
    "go2_random_box_clean",
    "go2_random_box_eval",
}
PRESERVED_TERRAIN_TASKS = RANDOM_BOX_TASKS | {
    "go2_five_box",
    "go2_mixed",
}


def configure_fixed_single_box(
    env_cfg,
    height,
    friction,
    ground_roughness=0.005,
    preserve_randomization=False,
):
    """Build a deterministic single-box scene while preserving start jitter."""

    env_cfg.env.episode_length_s = 15
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.max_init_terrain_level = 0
    env_cfg.terrain.num_goals = 2
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = False

    fixed_box_kwargs = deepcopy(env_cfg.terrain.random_box_kwargs)
    fixed_box_kwargs.pop("layout_presets", None)
    fixed_box_kwargs.update(
        {
            "seed": 17,
            "num_unique_layouts": 1,
            "box_count_range": (1, 1),
            "first_runup_range": (1.2, 1.2),
            "height_range": (height, height),
            "length_range": (1.2, 1.2),
            "width_range": (1.2, 1.2),
            "lateral_offset_range": (0.0, 0.0),
            "gap_distributions": (
                {"range": (0.1, 0.1), "weight": 1.0},
            ),
            "ground_roughness_distributions": (
                {
                    "range": (ground_roughness, ground_roughness),
                    "weight": 1.0,
                },
            ),
            "exit_goal_distance": 1.0,
            "end_margin": 0.5,
        }
    )
    env_cfg.terrain.random_box_kwargs = fixed_box_kwargs

    if preserve_randomization:
        if friction is not None:
            env_cfg.domain_rand.randomize_friction = True
            env_cfg.domain_rand.friction_range = [friction, friction]
        return

    env_cfg.domain_rand.randomize_friction = friction is not None
    if friction is not None:
        env_cfg.domain_rand.friction_range = [friction, friction]
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.action_delay = False

    env_cfg.noise.add_noise = False
    env_cfg.noise.apply_observation_noise = False
    env_cfg.noise.contact_dropout_prob = 0.0


def configure_camera_position_stress_noise(env_cfg):
    """Apply one conservative, reproducible sensor-noise stress profile."""

    env_cfg.noise.add_noise = True
    env_cfg.noise.apply_observation_noise = True
    env_cfg.noise.noise_level = 1.0
    env_cfg.noise.contact_dropout_prob = 0.01

    env_cfg.domain_rand.action_delay = True
    env_cfg.domain_rand.action_delay_range = [2, 2]

    stress_rotation = [0.0, math.radians(22.5), 0.0]
    env_cfg.depth.rotation = {
        "lower": stress_rotation.copy(),
        "upper": stress_rotation.copy(),
    }
    env_cfg.depth.horizontal_fov = 86
    env_cfg.depth.gaussian_noise_std = 0.01
    env_cfg.depth.dis_noise = 0.02
    env_cfg.depth.depth_dropout_prob = 0.01
    env_cfg.depth.depth_occlusion_prob = 0.30
    env_cfg.depth.depth_occlusion_size_range = [0.15, 0.15]


def set_per_env_camera_positions(env_cfg, positions):
    """Assign one exact base-frame optical center to every environment."""

    resolved = []
    for position in positions:
        values = [float(value) for value in position]
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise ValueError("Every camera position must contain three finite values.")
        resolved.append(values)
    if not resolved:
        raise ValueError("At least one camera position is required.")

    env_cfg.depth.position = {
        "mean": resolved[0].copy(),
        "std": [0.0, 0.0, 0.0],
        "per_env": resolved,
    }
    return resolved


def offset_camera_z(env_cfg, offset):
    """Move the camera distribution mean along base-frame Z."""

    if isinstance(env_cfg.depth.position, dict):
        position = deepcopy(env_cfg.depth.position)
        position["mean"][2] += offset
        env_cfg.depth.position = position
        return position["mean"], position.get("std")

    position = list(env_cfg.depth.position)
    position[2] += offset
    env_cfg.depth.position = position
    return position, None


def fix_replay_camera_pose(env_cfg, pitch_degrees, fix_position=False):
    """Fix replay orientation and optionally use the exact position mean."""

    rotation = [0.0, math.radians(pitch_degrees), 0.0]
    env_cfg.depth.rotation = {
        "lower": rotation.copy(),
        "upper": rotation.copy(),
    }

    if fix_position:
        if isinstance(env_cfg.depth.position, dict):
            position = deepcopy(env_cfg.depth.position)
            position.pop("per_env", None)
            position["std"] = [0.0, 0.0, 0.0]
            env_cfg.depth.position = position
        else:
            env_cfg.depth.position = {
                "mean": list(env_cfg.depth.position),
                "std": [0.0, 0.0, 0.0],
            }

    return rotation


def resolve_model_paths(model_dir):
    if model_dir is None:
        resolved_dir = Path(LEGGED_GYM_ROOT_DIR) / "logs" / "traced"
    else:
        resolved_dir = Path(model_dir).expanduser()
        if not resolved_dir.is_absolute():
            resolved_dir = Path(LEGGED_GYM_ROOT_DIR) / resolved_dir

    resolved_dir = resolved_dir.resolve()
    base_model_path = resolved_dir / "base_jit.pt"
    vision_model_path = resolved_dir / "vision_weight.pt"
    missing_paths = [
        path for path in (base_model_path, vision_model_path) if not path.is_file()
    ]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing exported model file(s): {missing}")
    return base_model_path, vision_model_path


def load_models(model_dir, device):
    base_model_path, vision_model_path = resolve_model_paths(model_dir)
    base_model = torch.jit.load(str(base_model_path), map_location=device).eval()

    vision_checkpoint = torch.load(
        str(vision_model_path), map_location=device, weights_only=True
    )
    if "depth_encoder_state_dict" not in vision_checkpoint:
        raise KeyError(
            f"{vision_model_path} does not contain depth_encoder_state_dict"
        )
    depth_backbone = DepthOnlyFCBackbone58x87(None, 32, 512)
    depth_encoder = RecurrentDepthBackbone(depth_backbone, None).to(device)
    depth_encoder.load_state_dict(
        vision_checkpoint["depth_encoder_state_dict"], strict=True
    )
    depth_encoder.eval()

    print(f"Loaded base model: {base_model_path}")
    print(f"Loaded vision model: {vision_model_path}")
    return base_model, depth_encoder


def configure_replay_env(env_cfg, args):
    preserve_task_terrain = args.task in PRESERVED_TERRAIN_TASKS
    mixed_defaults = args.task == "go2_mixed"
    default_num_envs = 15 if mixed_defaults else 1
    default_rows = 1
    default_cols = 15 if mixed_defaults else 1
    env_cfg.env.num_envs = (
        args.num_envs if args.num_envs is not None else default_num_envs
    )
    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = (
        args.rows if args.rows is not None else default_rows
    )
    env_cfg.terrain.num_cols = (
        args.cols if args.cols is not None else default_cols
    )

    if args.fixed_box_height is not None:
        configure_fixed_single_box(
            env_cfg,
            args.fixed_box_height,
            args.fixed_friction,
            args.fixed_ground_roughness,
        )

    camera_mean, camera_std = offset_camera_z(
        env_cfg, args.camera_z_offset
    )
    fixed_camera_pitch_deg = getattr(args, "fixed_camera_pitch_deg", None)
    fixed_camera_position = getattr(args, "fixed_camera_position", False)
    if fixed_camera_pitch_deg is not None:
        fixed_rotation = fix_replay_camera_pose(
            env_cfg,
            fixed_camera_pitch_deg,
            fix_position=fixed_camera_position,
        )
        if fixed_camera_position:
            camera_std = env_cfg.depth.position["std"]
        print(
            "Fixed replay camera pose: "
            f"rotation={fixed_rotation}, exact_position={fixed_camera_position}"
        )
    print(
        "Replay camera position: "
        f"mean={camera_mean}, std={camera_std}, "
        f"z_offset={args.camera_z_offset:.3f} m"
    )

    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
        env_cfg.domain_rand.action_delay_range = [0, 0]

    if not preserve_task_terrain:
        env_cfg.terrain.height = [0.02, 0.02]
        env_cfg.terrain.terrain_dict = {
            "smooth slope": 0.0,
            "rough slope up": 0.0,
            "rough slope down": 0.0,
            "rough stairs up": 0.0,
            "rough stairs down": 0.0,
            "discrete": 0.0,
            "stepping stones": 0.0,
            "gaps": 0.0,
            "smooth flat": 0.0,
            "pit": 0.0,
            "wall": 0.0,
            "platform": 0.0,
            "large stairs up": 0.0,
            "large stairs down": 0.0,
            "parkour": 0.0,
            "parkour_hurdle": 0.2,
            "parkour_flat": 0.0,
            "parkour_step": 0.2,
            "parkour_gap": 0.2,
            "demo": 0.2,
        }
        env_cfg.terrain.terrain_proportions = list(
            env_cfg.terrain.terrain_dict.values()
        )
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.max_difficulty = True

    env_cfg.depth.angle = [0, 1]
    if args.task not in RANDOM_BOX_TASKS:
        env_cfg.noise.add_noise = True
        env_cfg.domain_rand.randomize_friction = True
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.push_interval_s = 6
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.randomize_base_com = False


def validate_replay_args(args):
    if not args.use_camera:
        raise ValueError("play_jit.py requires --use_camera to replay the model pair.")
    if args.replay_steps is not None and args.replay_steps <= 0:
        raise ValueError("--replay_steps must be greater than zero.")
    if args.record_fps <= 0:
        raise ValueError("--record_fps must be greater than zero.")
    if args.record and args.headless:
        raise ValueError("--record requires the Isaac Gym viewer; omit --headless.")
    if not math.isfinite(args.camera_z_offset):
        raise ValueError("--camera_z_offset must be finite.")
    fixed_camera_pitch_deg = getattr(args, "fixed_camera_pitch_deg", None)
    if fixed_camera_pitch_deg is not None and not math.isfinite(
        fixed_camera_pitch_deg
    ):
        raise ValueError("--fixed_camera_pitch_deg must be finite.")
    if args.fixed_box_height is not None:
        if args.task not in RANDOM_BOX_TASKS:
            raise ValueError(
                "--fixed_box_height is only valid for random-box tasks."
            )
        if args.fixed_box_height <= 0:
            raise ValueError("--fixed_box_height must be greater than zero.")
    if args.fixed_friction is not None and args.fixed_friction < 0:
        raise ValueError("--fixed_friction must be non-negative.")
    if args.fixed_ground_roughness < 0:
        raise ValueError("--fixed_ground_roughness must be non-negative.")


def create_recorder(args, env):
    if not args.record:
        return None
    if env.viewer is None:
        raise RuntimeError("Viewer recording requires a non-headless environment.")

    if args.record_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (
            Path(LEGGED_GYM_ROOT_DIR) / "logs" / f"jit_replay_{timestamp}.mp4"
        )
    else:
        output_path = Path(args.record_path).expanduser()
        if not output_path.is_absolute():
            output_path = Path(LEGGED_GYM_ROOT_DIR) / output_path
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"Recording output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_dir = tempfile.TemporaryDirectory(prefix="extreme_jit_replay_")
    return {
        "frame_dir": frame_dir,
        "frame_count": 0,
        "fps": args.record_fps,
        "output_path": output_path,
    }


def capture_frame(recorder, env):
    if recorder is None:
        return
    frame_path = (
        Path(recorder["frame_dir"].name)
        / f"{recorder['frame_count']:06d}.png"
    )
    env.gym.write_viewer_image_to_file(env.viewer, str(frame_path))
    deadline = time.monotonic() + 2.0
    while not frame_path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Viewer frame was not written: {frame_path}")
        time.sleep(0.002)
    recorder["frame_count"] += 1


def finalize_recording(recorder):
    if recorder is None:
        return
    try:
        if recorder["frame_count"] == 0:
            raise RuntimeError("No viewer frames were captured.")
        written_frames = len(list(Path(recorder["frame_dir"].name).glob("*.png")))
        if written_frames != recorder["frame_count"]:
            raise RuntimeError(
                "Viewer recording is incomplete: "
                f"requested={recorder['frame_count']}, written={written_frames}."
            )
        frame_pattern = str(Path(recorder["frame_dir"].name) / "%06d.png")
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(recorder["fps"]),
            "-i",
            frame_pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(recorder["output_path"]),
        ]
        subprocess.run(command, check=True)
        print(f"Saved replay video: {recorder['output_path']}")
    finally:
        recorder["frame_dir"].cleanup()


@torch.inference_mode()
def play(args):
    validate_replay_args(args)
    faulthandler.enable()
    browser_viewer = webviewer.WebViewer() if args.web else None
    resolve_model_paths(args.jit_model_dir)

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    configure_replay_env(env_cfg, args)

    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    if browser_viewer is not None:
        browser_viewer.setup(env)

    base_model, depth_encoder = load_models(args.jit_model_dir, env.device)
    depth_latent = torch.zeros(env.num_envs, 32, device=env.device)
    vision_yaw = torch.zeros(env.num_envs, 2, device=env.device)
    infos = {"depth": env.depth_buffer.clone().to(env.device)[:, -1]}
    max_steps = (
        args.replay_steps
        if args.replay_steps is not None
        else 10 * int(env.max_episode_length)
    )

    print(
        "Replay configuration: "
        f"task={args.task}, device={env.device}, "
        f"envs={env.num_envs}, steps={max_steps}"
    )

    recorder = create_recorder(args, env)
    vision_output = None
    try:
        for step in range(max_steps):
            depth_frame = infos.get("depth")
            if depth_frame is not None:
                depth_proprio = obs[:, : env.cfg.env.n_proprio].clone()
                depth_proprio[:, 6:8] = 0
                vision_output = depth_encoder(depth_frame, depth_proprio)
                if vision_output.shape[-1] != 34:
                    raise RuntimeError(
                        "Vision encoder output must have 34 values, got "
                        f"{vision_output.shape[-1]}."
                    )
                if not torch.isfinite(vision_output).all():
                    raise RuntimeError("Vision encoder produced NaN or Inf.")
                depth_latent = vision_output[:, :32]
                vision_yaw = 1.5 * vision_output[:, -2:]
            elif vision_output is None:
                raise RuntimeError("The first simulated depth frame is unavailable.")

            policy_obs = obs.clone()
            policy_obs[:, 6:8] = vision_yaw
            actions = base_model(policy_obs, depth_latent)
            if actions.shape[-1] != 12:
                raise RuntimeError(
                    f"Base policy output must have 12 actions, got {actions.shape[-1]}."
                )
            if not torch.isfinite(actions).all():
                raise RuntimeError("Base policy produced NaN or Inf.")

            obs, _, _, dones, infos = env.step(actions)
            depth_encoder.reset(dones)
            capture_frame(recorder, env)

            if step == 0:
                print(
                    "First inference: "
                    f"depth={tuple(depth_frame.shape)}, "
                    f"vision={tuple(vision_output.shape)}, "
                    f"actions={tuple(actions.shape)}"
                )

            if browser_viewer is not None:
                browser_viewer.render(
                    fetch_results=True,
                    step_graphics=True,
                    render_all_camera_sensors=True,
                    wait_for_page_load=True,
                )

            if args.stop_on_done and torch.any(dones):
                episode = infos.get("episode", {})
                outcome_keys = (
                    "success_rate",
                    "fall_rate",
                    "timeout_rate",
                    "goals_reached",
                    "episode_duration_s",
                    "distance_traveled_m",
                )
                outcome = {
                    key: float(episode[key].detach().cpu().mean())
                    for key in outcome_keys
                    if key in episode
                }
                print(f"Replay episode outcome: {outcome}")
                break
    finally:
        finalize_recording(recorder)


if __name__ == "__main__":
    play(get_args())
