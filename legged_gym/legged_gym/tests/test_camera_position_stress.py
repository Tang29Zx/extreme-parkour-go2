"""Focused tests for camera-position stress result aggregation."""

import unittest

import isaacgym  # Must precede torch imports triggered by replay modules.

from legged_gym.scripts.camera_position_stress import (
    expand_trials,
    summarize_trials,
)


class CameraPositionStressTest(unittest.TestCase):
    def test_expands_trials_in_position_major_order(self):
        specs = [
            {"label": "a", "position_m": [0.355, 0.0, 0.065]},
            {"label": "b", "position_m": [0.255, 0.0, 0.065]},
        ]

        assignments = expand_trials(specs, 2)

        self.assertEqual([item["label"] for item in assignments], ["a", "a", "b", "b"])
        self.assertEqual([item["trial_index"] for item in assignments], [0, 1, 0, 1])

    def test_summarizes_waypoint_and_box_pass_separately(self):
        specs = [{"label": "a", "position_m": [0.355, 0.0, 0.065]}]
        records = [
            {
                "label": "a",
                "success_rate": 0.0,
                "box_pass_success": True,
                "fall_rate": 0.0,
                "timeout_rate": 1.0,
                "goals_reached": 1.0,
                "episode_duration_s": 15.0,
            },
            {
                "label": "a",
                "success_rate": 1.0,
                "box_pass_success": True,
                "fall_rate": 0.0,
                "timeout_rate": 0.0,
                "goals_reached": 2.0,
                "episode_duration_s": 5.0,
            },
        ]

        summary = summarize_trials(specs, records)[0]

        self.assertEqual(summary["waypoint_success_rate"], 0.5)
        self.assertEqual(summary["box_pass_success_rate"], 1.0)
        self.assertEqual(summary["mean_goals_reached"], 1.5)


if __name__ == "__main__":
    unittest.main()
