
import faulthandler
from pathlib import Path
import subprocess
import tempfile
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
    preserve_box_terrain = args.task in RANDOM_BOX_TASKS or args.task == "go2_five_box"
    env_cfg.env.num_envs = args.num_envs if args.num_envs is not None else 1
    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = args.rows if args.rows is not None else 1
    env_cfg.terrain.num_cols = args.cols if args.cols is not None else 1

    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
        env_cfg.domain_rand.action_delay_range = [0, 0]

    if not preserve_box_terrain:
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
    recorder["frame_count"] += 1


def finalize_recording(recorder):
    if recorder is None:
        return
    try:
        if recorder["frame_count"] == 0:
            raise RuntimeError("No viewer frames were captured.")
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
    finally:
        finalize_recording(recorder)


if __name__ == "__main__":
    play(get_args())
