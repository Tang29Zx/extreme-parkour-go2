"""Focused tests for the random-box camera mounting distribution."""

import unittest

import isaacgym  # Must precede torch imports triggered by config modules.

from legged_gym.envs.go2.go2_parkour_config import Go2ParkourCfg
from legged_gym.envs.go2.go2_random_box_config import Go2RandomBoxCfg


class RandomBoxCameraConfigTest(unittest.TestCase):
    def test_distillation_camera_mean_is_five_centimeters_higher(self):
        self.assertEqual(
            Go2RandomBoxCfg.depth.position["mean"],
            [0.355, 0.0, 0.115],
        )
        self.assertEqual(
            Go2RandomBoxCfg.depth.position["std"],
            [0.015, 0.01, 0.015],
        )

    def test_base_go2_camera_position_remains_unchanged(self):
        self.assertEqual(
            Go2ParkourCfg.depth.position,
            [0.355, 0.0, 0.065],
        )


if __name__ == "__main__":
    unittest.main()
