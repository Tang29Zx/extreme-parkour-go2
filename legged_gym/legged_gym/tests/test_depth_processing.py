"""Focused tests for batched simulated depth-camera processing."""

from types import SimpleNamespace
import unittest
from unittest import mock

import isaacgym
import numpy as np
import torch
import torchvision

from legged_gym.envs.base import legged_robot
from legged_gym.envs.base.legged_robot import LeggedRobot
from rsl_rl.modules.depth_backbone import RecurrentDepthBackbone


def make_robot(**depth_overrides):
    depth_cfg = {
        "gaussian_noise_std": 0.0,
        "dis_noise": 0.0,
        "depth_dropout_prob": 0.0,
        "depth_occlusion_prob": 0.0,
        "depth_occlusion_size_range": [0.05, 0.15],
        "near_clip": 0.0,
        "far_clip": 2.0,
        "far_distance_noise_start": 0.0,
        "far_distance_gaussian_std": 0.0,
        "far_distance_dropout_prob": 0.0,
    }
    depth_cfg.update(depth_overrides)

    robot = object.__new__(LeggedRobot)
    robot.cfg = SimpleNamespace(depth=SimpleNamespace(**depth_cfg))
    robot.resize_transform = torchvision.transforms.Resize(
        (58, 87),
        interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
    )
    return robot


class DepthProcessingTest(unittest.TestCase):
    def test_batch_matches_independent_processing_without_noise(self):
        robot = make_robot()
        depth_images = -torch.linspace(
            0.0, 2.0, steps=4 * 60 * 106, dtype=torch.float32
        ).reshape(4, 60, 106)

        batched = robot.process_depth_images(depth_images.clone())
        independent = torch.stack(
            [
                robot.process_depth_images(image.unsqueeze(0))[0]
                for image in depth_images
            ]
        )

        self.assertEqual(tuple(batched.shape), (4, 58, 87))
        torch.testing.assert_close(batched, independent)

    def test_enabled_noise_is_batched_and_finite_for_camera_env_count(self):
        robot = make_robot(
            gaussian_noise_std=0.01,
            dis_noise=0.02,
            depth_dropout_prob=0.01,
            depth_occlusion_prob=1.0,
            depth_occlusion_size_range=[1.0, 1.0],
        )
        depth_images = -torch.ones(192, 60, 106)

        processed = robot.process_depth_images(depth_images)

        self.assertEqual(tuple(processed.shape), (192, 58, 87))
        self.assertTrue(torch.isfinite(processed).all())
        torch.testing.assert_close(processed, torch.full_like(processed, 0.5))

    def test_far_distance_dropout_does_not_change_near_pixels(self):
        robot = make_robot(
            far_clip=3.0,
            far_distance_noise_start=2.0,
            far_distance_dropout_prob=1.0,
        )
        depth_images = torch.stack(
            (
                -torch.ones(60, 106),
                -3.0 * torch.ones(60, 106),
            )
        )

        processed = robot.process_depth_images(depth_images)

        torch.testing.assert_close(
            processed[0], torch.full_like(processed[0], -1.0 / 6.0)
        )
        torch.testing.assert_close(
            processed[1], torch.full_like(processed[1], 0.5)
        )

    def test_depth_buffer_calls_batch_processor_once(self):
        class FakeGym:
            def step_graphics(self, sim):
                pass

            def render_all_camera_sensors(self, sim):
                pass

            def start_access_image_tensors(self, sim):
                pass

            def end_access_image_tensors(self, sim):
                pass

            def get_camera_image_gpu_tensor(
                self, sim, env_handle, camera_handle, image_type
            ):
                return torch.full((2, 3), -float(env_handle + 1))

        robot = object.__new__(LeggedRobot)
        robot.cfg = SimpleNamespace(
            depth=SimpleNamespace(
                use_camera=True,
                update_interval=5,
                buffer_len=2,
            )
        )
        robot.global_counter = 5
        robot.num_envs = 3
        robot.sim = object()
        robot.gym = FakeGym()
        robot.envs = [0, 1, 2]
        robot.cam_handles = [0, 1, 2]
        robot.episode_length_buf = torch.tensor([0, 2, 2])
        robot.depth_buffer = torch.tensor(
            [
                [[[1.0] * 3] * 2, [[2.0] * 3] * 2],
                [[[3.0] * 3] * 2, [[4.0] * 3] * 2],
                [[[5.0] * 3] * 2, [[6.0] * 3] * 2],
            ]
        )
        batch_calls = []

        def process_depth_images(depth_images):
            batch_calls.append(depth_images.clone())
            return depth_images

        robot.process_depth_images = process_depth_images

        with mock.patch.object(
            legged_robot.gymtorch, "wrap_tensor", side_effect=lambda tensor: tensor
        ):
            LeggedRobot.update_depth_buffer(robot)

        self.assertEqual(len(batch_calls), 1)
        self.assertEqual(tuple(batch_calls[0].shape), (3, 2, 3))
        torch.testing.assert_close(
            robot.depth_buffer[0],
            torch.full((2, 2, 3), -1.0),
        )
        torch.testing.assert_close(
            robot.depth_buffer[1],
            torch.stack(
                [torch.full((2, 3), 4.0), torch.full((2, 3), -2.0)]
            ),
        )
        torch.testing.assert_close(
            robot.depth_buffer[2],
            torch.stack(
                [torch.full((2, 3), 6.0), torch.full((2, 3), -3.0)]
            ),
        )

    def test_cpu_depth_buffer_uses_numpy_readback_without_gpu_access(self):
        class FakeGym:
            def __init__(self):
                self.start_access_calls = 0
                self.end_access_calls = 0
                self.cpu_read_calls = 0

            def step_graphics(self, sim):
                pass

            def render_all_camera_sensors(self, sim):
                pass

            def start_access_image_tensors(self, sim):
                self.start_access_calls += 1

            def end_access_image_tensors(self, sim):
                self.end_access_calls += 1

            def get_camera_image_gpu_tensor(
                self, sim, env_handle, camera_handle, image_type
            ):
                raise AssertionError("GPU camera tensor path must not be used")

            def get_camera_image(
                self, sim, env_handle, camera_handle, image_type
            ):
                self.cpu_read_calls += 1
                return np.full((2, 3), -float(env_handle + 1), dtype=np.float32)

        robot = object.__new__(LeggedRobot)
        robot.cfg = SimpleNamespace(
            depth=SimpleNamespace(
                use_camera=True,
                use_gpu_tensor=False,
                update_interval=5,
                buffer_len=2,
            )
        )
        robot.global_counter = 5
        robot.num_envs = 2
        robot.device = "cpu"
        robot.sim = object()
        robot.gym = FakeGym()
        robot.envs = [0, 1]
        robot.cam_handles = [0, 1]
        robot.episode_length_buf = torch.tensor([0, 2])
        robot.depth_buffer = torch.zeros(2, 2, 2, 3)
        batch_calls = []

        def process_depth_images(depth_images):
            batch_calls.append(depth_images.clone())
            return depth_images

        robot.process_depth_images = process_depth_images

        LeggedRobot.update_depth_buffer(robot)

        self.assertEqual(robot.gym.cpu_read_calls, 2)
        self.assertEqual(robot.gym.start_access_calls, 0)
        self.assertEqual(robot.gym.end_access_calls, 0)
        self.assertEqual(len(batch_calls), 1)
        self.assertEqual(batch_calls[0].dtype, torch.float32)
        self.assertEqual(batch_calls[0].device.type, "cpu")
        self.assertTrue(torch.isfinite(robot.depth_buffer).all())
        torch.testing.assert_close(
            robot.depth_buffer[0],
            torch.full((2, 2, 3), -1.0),
        )
        torch.testing.assert_close(
            robot.depth_buffer[1],
            torch.stack(
                [torch.zeros(2, 3), torch.full((2, 3), -2.0)]
            ),
        )

    def test_viewer_debug_lines_are_cleared_before_camera_render(self):
        calls = []

        class FakeGym:
            def clear_lines(self, viewer):
                calls.append(("clear_lines", viewer))

            def step_graphics(self, sim):
                calls.append(("step_graphics", sim))

            def render_all_camera_sensors(self, sim):
                calls.append(("render", sim))

            def get_camera_image(
                self, sim, env_handle, camera_handle, image_type
            ):
                calls.append(("read", env_handle))
                return np.full((2, 3), -1.0, dtype=np.float32)

        robot = object.__new__(LeggedRobot)
        robot.cfg = SimpleNamespace(
            depth=SimpleNamespace(
                use_camera=True,
                use_gpu_tensor=False,
                update_interval=1,
                buffer_len=1,
            )
        )
        robot.global_counter = 1
        robot.num_envs = 1
        robot.device = "cpu"
        robot.sim = "sim"
        robot.viewer = "viewer"
        robot.gym = FakeGym()
        robot.envs = [0]
        robot.cam_handles = [0]
        robot.episode_length_buf = torch.tensor([0])
        robot.depth_buffer = torch.zeros(1, 1, 2, 3)
        robot.process_depth_images = lambda depth_images: depth_images

        LeggedRobot.update_depth_buffer(robot)

        self.assertEqual(
            calls,
            [
                ("clear_lines", "viewer"),
                ("step_graphics", "sim"),
                ("render", "sim"),
                ("read", 0),
            ],
        )


class DepthRecurrentStateTest(unittest.TestCase):
    def test_reset_clears_only_completed_environment_state(self):
        env_cfg = SimpleNamespace(env=SimpleNamespace(n_proprio=53))
        encoder = RecurrentDepthBackbone(torch.nn.Identity(), env_cfg)
        encoder(torch.zeros(3, 32), torch.zeros(3, 53))
        hidden_before_reset = encoder.hidden_states.clone()

        encoder.reset(torch.tensor([False, True, False]))

        torch.testing.assert_close(
            encoder.hidden_states[:, 0], hidden_before_reset[:, 0]
        )
        torch.testing.assert_close(
            encoder.hidden_states[:, 1],
            torch.zeros_like(encoder.hidden_states[:, 1]),
        )
        torch.testing.assert_close(
            encoder.hidden_states[:, 2], hidden_before_reset[:, 2]
        )


if __name__ == "__main__":
    unittest.main()
