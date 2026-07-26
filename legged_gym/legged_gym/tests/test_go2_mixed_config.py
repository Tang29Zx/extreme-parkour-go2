"""Focused tests for the native and random-box Go2 evaluation task."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import isaacgym  # Must precede torch imports triggered by config modules.
import numpy as np
import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.go2.go2_mixed_config import (
    Go2MixedCfg,
    ORIGINAL_TRAINING_LAYOUT_INDICES,
)
from legged_gym.envs.go2.go2_random_box_config import Go2RandomBoxCfg
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.random_box_terrain import (
    build_random_box_terrain,
    resolve_random_box_layout,
)
from legged_gym.utils.terrain import Terrain


def make_box_terrain():
    cfg = Go2MixedCfg.terrain
    horizontal_scale = float(cfg.horizontal_scale)
    return SimpleNamespace(
        horizontal_scale=horizontal_scale,
        vertical_scale=float(cfg.vertical_scale),
        width=int(round(cfg.terrain_length / horizontal_scale)),
        length=int(round(cfg.terrain_width / horizontal_scale)),
        height_field_raw=np.zeros(
            (
                int(round(cfg.terrain_length / horizontal_scale)),
                int(round(cfg.terrain_width / horizontal_scale)),
            ),
            dtype=np.int16,
        ),
    )


class Go2MixedConfigTest(unittest.TestCase):
    def test_config_contains_five_native_and_ten_random_box_tracks(self):
        cfg = Go2MixedCfg
        enabled = [
            name
            for name, weight in cfg.terrain.terrain_dict.items()
            if weight > 0.0
        ]

        self.assertEqual(
            enabled,
            [
                "parkour",
                "parkour_hurdle",
                "parkour_step",
                "parkour_gap",
                "demo",
                "random_box",
            ],
        )
        self.assertEqual(cfg.env.num_envs, 15)
        self.assertTrue(cfg.env.draw_all_goals)
        self.assertEqual(
            cfg.env.goal_success_rate_groups,
            {
                "native_1_5_jump_success_rate": (0, 5),
                "custom_6_10_average_success_rate": (5, 10),
            },
        )
        self.assertEqual((cfg.terrain.num_rows, cfg.terrain.num_cols), (1, 15))
        self.assertEqual(cfg.terrain.num_goals, 8)
        self.assertEqual(cfg.terrain.fixed_terrain_difficulty, 1.0)
        self.assertEqual(
            cfg.terrain.random_box_kwargs["num_unique_layouts"], 10
        )
        self.assertEqual(
            cfg.terrain.random_box_kwargs["box_count_range"], (7, 7)
        )
        self.assertEqual(
            ORIGINAL_TRAINING_LAYOUT_INDICES,
            (51, 186, 357, 130, 221),
        )

    def test_generated_columns_follow_the_mixed_track_order(self):
        cfg = SimpleNamespace(**class_to_dict(Go2MixedCfg.terrain))
        cfg.mesh_type = "heightfield"

        np.random.seed(0)
        terrain = Terrain(cfg, num_robots=15)

        np.testing.assert_array_equal(
            terrain.terrain_type[0],
            np.array(
                [15, 16, 18, 19, 20] + [22] * 10,
                dtype=float,
            ),
        )
        self.assertEqual(terrain.goals.shape, (1, 15, 8, 3))
        self.assertTrue(np.isfinite(terrain.goals).all())

    def test_five_box_layouts_match_requested_geometry_ranges(self):
        cfg = Go2MixedCfg.terrain
        layouts = []

        for layout_index in range(5):
            terrain = make_box_terrain()
            specs = build_random_box_terrain(terrain, cfg, layout_index)
            layouts.append(
                tuple(
                    (spec["length"], spec["width"], spec["height"], spec["gap"])
                    for spec in specs
                )
            )

            self.assertEqual(len(specs), 7)
            self.assertEqual(terrain.goals.shape, (8, 2))
            self.assertEqual(terrain.layout_index, layout_index)
            for spec in specs:
                self.assertGreaterEqual(spec["length"], 0.50 - 1e-8)
                self.assertLessEqual(spec["length"], 1.00 + 1e-8)
                self.assertGreaterEqual(spec["width"], 0.50 - 1e-8)
                self.assertLessEqual(spec["width"], 1.00 + 1e-8)
                self.assertGreaterEqual(spec["height"], 0.20 - 1e-8)
                self.assertLessEqual(spec["height"], 0.50 + 1e-8)
                self.assertGreaterEqual(spec["lateral_offset"], -1.00 - 1e-8)
                self.assertLessEqual(spec["lateral_offset"], 1.00 + 1e-8)
                if spec["index"] > 0:
                    self.assertGreaterEqual(spec["gap"], 0.20 - 1e-8)
                    self.assertLessEqual(spec["gap"], 1.00 + 1e-8)

            self.assertLessEqual(
                terrain.goals[-1, 0],
                cfg.terrain_length
                - cfg.random_box_kwargs["end_margin"],
            )

        self.assertEqual(len(set(layouts)), 5)

    def test_third_box_layout_has_compact_first_four_lateral_offsets(self):
        terrain = make_box_terrain()
        specs = build_random_box_terrain(terrain, Go2MixedCfg.terrain, 2)

        np.testing.assert_allclose(
            [spec["lateral_offset"] for spec in specs[:4]],
            [-0.25, 0.25, -0.25, 0.25],
        )
        np.testing.assert_allclose(
            np.abs(
                np.diff(
                    [spec["lateral_offset"] for spec in specs[:4]]
                )
            ),
            [0.50, 0.50, 0.50],
        )

    def test_last_five_tracks_extend_original_training_layouts_to_seven_boxes(self):
        mixed_cfg = Go2MixedCfg.terrain
        training_cfg = Go2RandomBoxCfg.terrain

        for logical_index, source_index in enumerate(
            ORIGINAL_TRAINING_LAYOUT_INDICES, start=5
        ):
            mixed_terrain = make_box_terrain()
            training_terrain = make_box_terrain()
            mixed_specs = build_random_box_terrain(
                mixed_terrain, mixed_cfg, logical_index
            )
            training_specs = build_random_box_terrain(
                training_terrain, training_cfg, source_index
            )

            self.assertEqual(len(mixed_specs), 7)
            self.assertEqual(
                mixed_specs[: len(training_specs)], training_specs
            )
            self.assertEqual(mixed_terrain.layout_index, source_index)
            np.testing.assert_allclose(
                mixed_terrain.goals[: len(training_specs)],
                training_terrain.goals[: len(training_specs)],
            )
            resolved_kwargs, resolved_index = resolve_random_box_layout(
                mixed_cfg.random_box_kwargs, logical_index
            )
            expected_kwargs = dict(
                training_cfg.random_box_kwargs,
                box_count_range=(7, 7),
            )
            self.assertEqual(resolved_kwargs, expected_kwargs)
            self.assertEqual(resolved_index, source_index)

    def test_fixed_difficulty_is_used_for_every_column(self):
        terrain = Terrain.__new__(Terrain)
        terrain.cfg = SimpleNamespace(
            num_rows=1,
            num_cols=15,
            fixed_terrain_difficulty=1.0,
        )
        calls = []
        terrain.make_terrain = lambda choice, difficulty: calls.append(
            (choice, difficulty)
        )
        terrain.add_terrain_to_map = lambda generated, row, col: None

        terrain.curiculum(random=True, max_difficulty=True)

        self.assertEqual(len(calls), 15)
        self.assertTrue(all(difficulty == 1.0 for _, difficulty in calls))

    def test_goal_rendering_can_cover_all_environments(self):
        robot = LeggedRobot.__new__(LeggedRobot)
        robot.num_envs = 3
        robot.lookat_id = 1
        robot.cfg = SimpleNamespace(
            env=SimpleNamespace(
                next_goal_threshold=0.2,
                draw_all_goals=True,
            ),
            depth=SimpleNamespace(use_camera=True),
        )
        robot.terrain = SimpleNamespace(
            cfg=SimpleNamespace(
                border_size=0.0,
                horizontal_scale=1.0,
                vertical_scale=0.005,
            )
        )
        robot.terrain_goals = torch.tensor(
            [
                [
                    [[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
                    [[1.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
                    [[1.0, 3.0, 0.0], [2.0, 3.0, 0.0]],
                ]
            ]
        )
        robot.terrain_levels = torch.zeros(3, dtype=torch.long)
        robot.terrain_types = torch.arange(3, dtype=torch.long)
        robot.cur_goal_idx = torch.zeros(3, dtype=torch.long)
        robot.reached_goal_ids = torch.zeros(3, dtype=torch.bool)
        robot.height_samples = torch.zeros((10, 10))
        robot.envs = ["env0", "env1", "env2"]
        robot.gym = object()
        robot.viewer = object()

        with patch(
            "legged_gym.envs.base.legged_robot.gymutil.draw_lines"
        ) as draw_lines:
            robot._draw_goals()
            self.assertEqual(draw_lines.call_count, 6)
            self.assertEqual(
                [call.args[3] for call in draw_lines.call_args_list],
                ["env0", "env0", "env1", "env1", "env2", "env2"],
            )

            robot.cfg.env.draw_all_goals = False
            draw_lines.reset_mock()
            robot._draw_goals()
            self.assertEqual(draw_lines.call_count, 2)
            self.assertTrue(
                all(
                    call.args[3] == "env1"
                    for call in draw_lines.call_args_list
                )
            )

    def test_goal_success_marks_completed_routes_and_excludes_manual_reset(self):
        robot = LeggedRobot.__new__(LeggedRobot)
        robot.num_envs = 4
        robot.device = "cpu"
        robot.roll = torch.tensor([0.0, 0.0, 2.0, 0.0])
        robot.pitch = torch.zeros(4)
        robot.cur_goal_idx = torch.tensor([8, 0, 0, 0])
        robot.root_states = torch.tensor(
            [
                [0.0, 0.0, 0.4],
                [0.0, 0.0, 0.4],
                [0.0, 0.0, 0.4],
                [0.0, 0.0, 0.4],
            ]
        )
        robot.episode_length_buf = torch.tensor([1, 101, 1, 1])
        robot.max_episode_length = 100
        robot.manual_reset_buf = torch.tensor(
            [False, False, False, True]
        )
        robot.goal_success_buf = torch.zeros(4, dtype=torch.bool)
        robot.goal_evaluation_episode_buf = torch.zeros(
            4, dtype=torch.bool
        )
        robot.cfg = SimpleNamespace(
            terrain=SimpleNamespace(num_goals=8),
        )

        robot.check_termination()

        self.assertEqual(
            robot.goal_success_buf.tolist(),
            [True, False, False, False],
        )
        self.assertEqual(
            robot.goal_evaluation_episode_buf.tolist(),
            [True, True, True, False],
        )
        robot.cur_goal_idx.zero_()
        robot.roll.zero_()
        robot.episode_length_buf.fill_(1)
        robot.manual_reset_buf.zero_()
        robot.check_termination()
        self.assertFalse(robot.goal_evaluation_episode_buf.any().item())

    def test_training_episode_log_reports_requested_track_groups(self):
        robot = LeggedRobot.__new__(LeggedRobot)
        robot.cfg = SimpleNamespace(
            env=SimpleNamespace(
                goal_success_rate_groups={
                    "native_1_5_jump_success_rate": (0, 5),
                    "custom_6_10_average_success_rate": (5, 10),
                }
            )
        )
        robot.goal_success_buf = torch.tensor(
            [True, False, False, False, False, True, False, False, False, False, True]
        )
        robot.goal_evaluation_episode_buf = torch.tensor(
            [True, True, False, False, False, True, True, False, False, False, True]
        )
        robot.terrain_types = torch.arange(11)
        robot.extras = {"episode": {}}

        robot._log_goal_success_rates(torch.arange(11))

        self.assertEqual(
            robot.extras["episode"][
                "native_1_5_jump_success_rate"
            ].tolist(),
            [1.0, 0.0],
        )
        self.assertEqual(
            robot.extras["episode"][
                "custom_6_10_average_success_rate"
            ].tolist(),
            [1.0, 0.0],
        )

    def test_episode_outcomes_include_failures_progress_and_terrain(self):
        robot = LeggedRobot.__new__(LeggedRobot)
        robot.cfg = SimpleNamespace(
            terrain=SimpleNamespace(num_goals=8)
        )
        robot.goal_evaluation_episode_buf = torch.tensor(
            [True, True, True, False]
        )
        robot.goal_success_buf = torch.tensor(
            [True, False, False, False]
        )
        robot.fall_buf = torch.tensor([False, False, True, False])
        robot.episode_timeout_buf = torch.tensor(
            [False, True, False, False]
        )
        robot.cur_goal_idx = torch.tensor([8, 4, 2, 1])
        robot.episode_length_buf = torch.tensor([50, 100, 25, 5])
        robot.dt = 0.02
        robot.root_states = torch.zeros(4, 13)
        robot.root_states[:, :2] = torch.tensor(
            [[3.0, 4.0], [1.0, 0.0], [0.0, 2.0], [9.0, 9.0]]
        )
        robot.env_origins = torch.zeros(4, 3)
        robot.env_class = torch.tensor([15.0, 15.0, 22.0, 22.0])

        metrics = robot._collect_episode_outcomes(torch.arange(4))

        self.assertEqual(metrics["success_rate"].tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(metrics["fall_rate"].tolist(), [0.0, 0.0, 1.0])
        self.assertEqual(metrics["timeout_rate"].tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(metrics["goals_reached"].tolist(), [8.0, 4.0, 2.0])
        self.assertEqual(metrics["goal_progress"].tolist(), [1.0, 0.5, 0.25])
        self.assertEqual(metrics["episode_duration_s"].tolist(), [1.0, 2.0, 0.5])
        self.assertEqual(metrics["distance_traveled_m"].tolist(), [5.0, 1.0, 2.0])
        self.assertEqual(
            metrics["terrain_parkour_success_rate"].tolist(),
            [1.0, 0.0],
        )
        self.assertEqual(
            metrics["terrain_random_box_success_rate"].tolist(),
            [0.0],
        )


if __name__ == "__main__":
    unittest.main()
