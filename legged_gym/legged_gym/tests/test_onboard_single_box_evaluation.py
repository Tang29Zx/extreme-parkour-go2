"""Focused tests for onboard single-box safety accounting."""

import unittest

import isaacgym  # Must precede torch imports triggered by replay modules.
import numpy as np

from legged_gym.scripts.evaluate_onboard_single_box import (
    constraint_diagnostics,
    summarize_trials,
)


class ConstraintDiagnosticsTest(unittest.TestCase):
    def make_args(self):
        return {
            "requested_q": np.zeros(12),
            "previous_q": np.zeros(12),
            "measured_q": np.zeros(12),
            "measured_dq": np.zeros(12),
            "kp": np.full(12, 40.0),
            "kd": np.ones(12),
            "joint_limits_low": np.full(12, -1.0),
            "joint_limits_high": np.full(12, 1.0),
            "torque_limits": np.full(12, 40.0),
            "max_step_rad": np.full(12, 0.2),
        }

    def test_attributes_step_joint_and_torque_bounds(self):
        arguments = self.make_args()
        arguments["requested_q"] = np.full(12, 2.0)
        arguments["joint_limits_high"][1] = 0.1
        arguments["torque_limits"][2] = 4.0

        result = constraint_diagnostics(**arguments)

        self.assertTrue(result["step"][0])
        self.assertTrue(result["joint"][1])
        self.assertTrue(result["torque"][2])
        self.assertFalse(np.any(result["infeasible"]))

    def test_reports_infeasible_intersection(self):
        arguments = self.make_args()
        arguments["previous_q"][0] = -0.5
        arguments["measured_q"][0] = 0.9
        arguments["torque_limits"][0] = 1.0

        result = constraint_diagnostics(**arguments)

        self.assertTrue(result["infeasible"][0])


class EvaluationSummaryTest(unittest.TestCase):
    def record(self, outcome, cycles, clips):
        record = {
            "outcome": outcome,
            "policy_cycles": cycles,
            **{name: 0 for name in (
                "raw_action_clip_cycles",
                "transition_guard_cycles",
                "constraint_cycles",
                "output_guard_cycles",
                "any_clip_cycles",
                "target_step_limit_cycles",
                "joint_limit_cycles",
                "pd_torque_limit_cycles",
            )},
            **{name: [0] * 12 for name in (
                "raw_action_clip_by_joint",
                "transition_guard_by_joint",
                "constraint_by_joint",
                "target_step_limit_by_joint",
                "joint_limit_by_joint",
                "pd_torque_limit_by_joint",
            )},
            "max_abs_raw_action": 0.0,
            "max_request_command_delta": 0.0,
            "max_command_step": 0.0,
            "max_predicted_torque_ratio": 0.0,
            "max_actual_torque_ratio": 0.0,
            "max_joint_velocity_ratio": 0.0,
            "max_abs_roll_deg": 0.0,
            "max_abs_pitch_deg": 0.0,
        }
        record["output_guard_cycles"] = clips
        return record

    def test_aggregates_success_and_cycle_rates(self):
        summary = summarize_trials(
            [self.record("success", 100, 25), self.record("timeout", 50, 5)]
        )

        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["total_policy_cycles"], 150)
        self.assertEqual(summary["output_guard_cycles"], 30)
        self.assertAlmostEqual(summary["output_guard_cycles_rate"], 0.2)
        self.assertEqual(summary["successful_policy_cycles"], [100])


if __name__ == "__main__":
    unittest.main()
