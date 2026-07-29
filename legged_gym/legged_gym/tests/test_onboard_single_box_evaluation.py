"""Focused tests for onboard single-box safety accounting."""

from types import SimpleNamespace
import unittest

import isaacgym  # Must precede torch imports triggered by replay modules.
import numpy as np

from legged_gym.scripts.evaluate_onboard_single_box import (
    COUNTER_NAMES,
    JOINT_COUNTER_NAMES,
    constraint_diagnostics,
    resolve_diagnostic_safety_inputs,
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

    def test_reports_recorded_rr_thigh_infeasible_intersection(self):
        arguments = self.make_args()
        joint = 7
        arguments["requested_q"][joint] = -0.2
        arguments["previous_q"][joint] = -0.013357
        arguments["measured_q"][joint] = 0.796217
        arguments["measured_dq"][joint] = 2.600477
        arguments["torque_limits"][joint] = 23.7
        arguments["max_step_rad"] = np.asarray([0.21, 0.21, 0.20] * 4)

        result = constraint_diagnostics(**arguments)

        self.assertEqual(np.flatnonzero(result["infeasible"]).tolist(), [joint])
        self.assertAlmostEqual(result["lower"][joint], 0.268728925)
        self.assertAlmostEqual(result["upper"][joint], 0.196643)


class DiagnosticSafetyInputsTest(unittest.TestCase):
    def make_modules(self):
        safety = SimpleNamespace(
            GO2_JOINT_LIMITS_LOW=np.full(12, -1.0),
            GO2_JOINT_LIMITS_HIGH=np.full(12, 1.0),
            GO2_JOINT_VELOCITY_LIMITS=np.asarray([30.0, 30.0, 20.0] * 4),
        )
        control = SimpleNamespace(
            POLICY_TORQUE_ESCAPE_MAX_STEP_RAD_BY_JOINT=(0.3, 0.3, 0.2) * 4,
        )
        return safety, control

    def make_args(self, **overrides):
        values = {
            "enable_torque_escape": True,
            "diagnostic_calf_velocity_limit": None,
            "diagnostic_position_tolerance_extra_rad": 0.0,
            "diagnostic_rr_thigh_escape_step_rad": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_defaults_preserve_production_inputs(self):
        safety, control = self.make_modules()

        result = resolve_diagnostic_safety_inputs(
            self.make_args(), safety, control
        )

        self.assertFalse(result["enabled"])
        np.testing.assert_array_equal(
            result["velocity_limits"],
            safety.GO2_JOINT_VELOCITY_LIMITS,
        )
        np.testing.assert_array_equal(
            result["runtime_joint_low"], safety.GO2_JOINT_LIMITS_LOW
        )
        np.testing.assert_array_equal(
            result["escape_steps"],
            control.POLICY_TORQUE_ESCAPE_MAX_STEP_RAD_BY_JOINT,
        )

    def test_overrides_only_requested_boundaries(self):
        safety, control = self.make_modules()

        result = resolve_diagnostic_safety_inputs(
            self.make_args(
                diagnostic_calf_velocity_limit=24.0,
                diagnostic_position_tolerance_extra_rad=0.01,
                diagnostic_rr_thigh_escape_step_rad=0.32,
            ),
            safety,
            control,
        )

        self.assertTrue(result["enabled"])
        np.testing.assert_array_equal(
            result["velocity_limits"],
            np.asarray([30.0, 30.0, 24.0] * 4),
        )
        np.testing.assert_array_equal(
            result["runtime_joint_low"], np.full(12, -1.01)
        )
        np.testing.assert_array_equal(
            result["runtime_joint_high"], np.full(12, 1.01)
        )
        expected_escape = np.asarray((0.3, 0.3, 0.2) * 4)
        expected_escape[7] = 0.32
        np.testing.assert_array_equal(result["escape_steps"], expected_escape)

    def test_rejects_tighter_or_inactive_escape_overrides(self):
        safety, control = self.make_modules()
        with self.assertRaisesRegex(ValueError, "production value"):
            resolve_diagnostic_safety_inputs(
                self.make_args(diagnostic_calf_velocity_limit=19.0),
                safety,
                control,
            )
        with self.assertRaisesRegex(ValueError, "requires"):
            resolve_diagnostic_safety_inputs(
                self.make_args(
                    enable_torque_escape=False,
                    diagnostic_rr_thigh_escape_step_rad=0.32,
                ),
                safety,
                control,
            )

class EvaluationSummaryTest(unittest.TestCase):
    def record(self, outcome, cycles, clips, escapes=0):
        record = {
            "outcome": outcome,
            "policy_cycles": cycles,
            **{name: 0 for name in COUNTER_NAMES},
            **{name: [0] * 12 for name in JOINT_COUNTER_NAMES},
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
        record["torque_escape_cycles"] = escapes
        record["torque_escape_by_joint"][7] = escapes
        return record

    def test_aggregates_success_and_cycle_rates(self):
        summary = summarize_trials(
            [
                self.record("success", 100, 25, 2),
                self.record("timeout", 50, 5, 1),
            ]
        )

        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["total_policy_cycles"], 150)
        self.assertEqual(summary["output_guard_cycles"], 30)
        self.assertAlmostEqual(summary["output_guard_cycles_rate"], 0.2)
        self.assertEqual(summary["torque_escape_cycles"], 3)
        self.assertAlmostEqual(summary["torque_escape_cycles_rate"], 0.02)
        self.assertEqual(summary["torque_escape_by_joint"][7], 3)
        self.assertEqual(summary["successful_policy_cycles"], [100])


if __name__ == "__main__":
    unittest.main()
