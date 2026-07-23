"""Focused tests for batched simulated depth-camera processing."""

from types import SimpleNamespace
import unittest
from unittest import mock

import isaacgym
import torch
import torchvision

from legged_gym.envs.base import legged_robot
from legged_gym.envs.base.legged_robot import LeggedRobot


def make_robot(**depth_overrides):
    depth_cfg = {
        "gaussian_noise_std": 0.0,
        "dis_noise": 0.0,
        "depth_dropout_prob": 0.0,
        "depth_occlusion_prob": 0.0,
        "depth_occlusion_size_range": [0.05, 0.15],
        "near_clip": 0.0,
        "far_clip": 2.0,
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


if __name__ == "__main__":
    unittest.main()
