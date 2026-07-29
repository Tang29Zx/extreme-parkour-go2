"""Focused tests for deployment-aligned joint-limit reward shaping."""

from types import SimpleNamespace
import unittest

import isaacgym  # Must precede torch imports triggered by environment modules.
import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.go2.go2_random_box_config import Go2RandomBoxCfg


class JointLimitRewardTest(unittest.TestCase):
    def make_robot(self):
        robot = LeggedRobot.__new__(LeggedRobot)
        robot.cfg = SimpleNamespace(
            rewards=SimpleNamespace(soft_dof_vel_limit=0.8)
        )
        return robot

    def test_velocity_reward_starts_at_soft_limit_and_is_normalized(self):
        robot = self.make_robot()
        robot.dof_vel_limits = torch.tensor([20.0, 10.0, 5.0])
        robot.dof_vel = torch.tensor(
            [
                [16.0, -8.0, 0.0],
                [20.0, -10.0, 7.5],
            ]
        )

        reward = LeggedRobot._reward_dof_vel_limits(robot)

        torch.testing.assert_close(reward, torch.tensor([0.0, 0.57]))

    def test_position_reward_uses_processed_soft_limits(self):
        robot = self.make_robot()
        robot.dof_pos_limits = torch.tensor(
            [
                [-1.0, 1.0],
                [-2.0, 2.0],
            ]
        )
        robot.dof_pos = torch.tensor(
            [
                [0.0, 2.0],
                [-1.2, 2.3],
            ]
        )

        reward = LeggedRobot._reward_dof_pos_limits(robot)

        torch.testing.assert_close(reward, torch.tensor([0.0, 0.5]))

    def test_random_box_enables_deployment_limit_shaping(self):
        self.assertEqual(Go2RandomBoxCfg.rewards.soft_dof_vel_limit, 0.8)
        self.assertEqual(
            Go2RandomBoxCfg.rewards.scales.dof_vel_limits,
            -5.0,
        )
        self.assertEqual(
            Go2RandomBoxCfg.rewards.scales.dof_pos_limits,
            -2.0,
        )


if __name__ == "__main__":
    unittest.main()
