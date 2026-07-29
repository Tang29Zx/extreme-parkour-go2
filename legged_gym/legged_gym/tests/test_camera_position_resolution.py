"""Tests for exact per-environment camera placement."""

import unittest

import isaacgym  # Must precede torch imports triggered by environment modules.

from legged_gym.envs.base.legged_robot import resolve_camera_position


class CameraPositionResolutionTest(unittest.TestCase):
    def test_selects_exact_per_env_position(self):
        config = {
            "mean": [0.355, 0.0, 0.065],
            "std": [0.015, 0.01, 0.015],
            "per_env": [
                [0.355, 0.0, 0.065],
                [0.505, 0.0, 0.065],
            ],
        }

        self.assertEqual(
            resolve_camera_position(config, 1).tolist(),
            [0.505, 0.0, 0.065],
        )

    def test_rejects_missing_per_env_position(self):
        with self.assertRaises(IndexError):
            resolve_camera_position({"per_env": [[0.0, 0.0, 0.0]]}, 1)


if __name__ == "__main__":
    unittest.main()
