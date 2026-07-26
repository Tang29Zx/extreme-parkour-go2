"""Focused tests for viewer-requested environment resets."""

from types import SimpleNamespace
import unittest

import isaacgym  # Must precede torch.
import torch

from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.base.legged_robot import LeggedRobot


class ManualViewerResetTest(unittest.TestCase):
    def test_request_marks_only_selected_environment(self):
        task = BaseTask.__new__(BaseTask)
        task.manual_reset_buf = torch.zeros(3, dtype=torch.bool)

        task.request_manual_reset(1)

        self.assertEqual(task.manual_reset_buf.tolist(), [False, True, False])

    def test_manual_request_enters_normal_termination_buffer(self):
        robot = LeggedRobot.__new__(LeggedRobot)
        robot.num_envs = 3
        robot.device = "cpu"
        robot.reset_buf = torch.zeros(3, dtype=torch.bool)
        robot.time_out_buf = torch.zeros(3, dtype=torch.bool)
        robot.manual_reset_buf = torch.tensor([False, True, False])
        robot.goal_success_buf = torch.zeros(3, dtype=torch.bool)
        robot.goal_evaluation_episode_buf = torch.zeros(
            3, dtype=torch.bool
        )
        robot.roll = torch.zeros(3)
        robot.pitch = torch.zeros(3)
        robot.cur_goal_idx = torch.zeros(3, dtype=torch.long)
        robot.root_states = torch.zeros(3, 13)
        robot.root_states[:, 2] = 0.3
        robot.episode_length_buf = torch.zeros(3, dtype=torch.long)
        robot.max_episode_length = 100
        robot.cfg = SimpleNamespace(
            terrain=SimpleNamespace(num_goals=11)
        )

        robot.check_termination()

        self.assertEqual(robot.reset_buf.tolist(), [False, True, False])
        self.assertFalse(robot.time_out_buf.any())


if __name__ == "__main__":
    unittest.main()
