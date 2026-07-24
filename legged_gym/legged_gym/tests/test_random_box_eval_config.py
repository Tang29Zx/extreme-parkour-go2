"""Focused tests for the fixed ten-box generalization scene."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

import isaacgym  # Must precede torch imports triggered by config modules.
import numpy as np

from legged_gym.envs.go2.go2_random_box_config import Go2RandomBoxCfg
from legged_gym.envs.go2.go2_random_box_eval_config import (
    Go2RandomBoxEvalCfg,
)
from legged_gym.utils.random_box_terrain import (
    build_random_box_terrain,
    resolve_random_box_layout,
    select_roughness_range,
)


def make_terrain(cfg=Go2RandomBoxEvalCfg.terrain):
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


class RandomBoxEvalConfigTest(unittest.TestCase):
    def test_fixed_scene_has_ten_boxes_and_expected_ranges(self):
        cfg = Go2RandomBoxEvalCfg.terrain
        first = make_terrain()
        second = make_terrain()

        first_specs = build_random_box_terrain(first, cfg, 0)
        second_specs = build_random_box_terrain(second, cfg, 0)

        self.assertEqual(first_specs, second_specs)
        np.testing.assert_array_equal(
            first.height_field_raw, second.height_field_raw
        )
        self.assertEqual(len(first_specs), 10)
        self.assertEqual(first.goals.shape, (11, 2))
        self.assertEqual(
            select_roughness_range(
                cfg.random_box_kwargs["seed"],
                cfg.random_box_kwargs["num_unique_layouts"],
                cfg.random_box_kwargs["ground_roughness_distributions"],
                0,
            )[0],
            (0.03, 0.03),
        )

        for spec in first_specs:
            self.assertGreaterEqual(spec["lateral_offset"], -0.5)
            self.assertLessEqual(spec["lateral_offset"], 0.5)
            if spec["index"] > 0:
                self.assertGreaterEqual(spec["gap"], 0.30)
                self.assertLessEqual(spec["gap"], 1.00)

        self.assertAlmostEqual(first_specs[2]["gap"], 0.35)
        self.assertAlmostEqual(first_specs[3]["height"], 0.40)
        self.assertAlmostEqual(first_specs[4]["gap"], 0.35)
        self.assertAlmostEqual(first_specs[8]["gap"], 0.35)

        self.assertLessEqual(
            first.goals[-1, 0],
            cfg.terrain_length - cfg.random_box_kwargs["end_margin"],
        )

    def test_eval_config_does_not_mutate_training_layout(self):
        training_kwargs = deepcopy(
            Go2RandomBoxCfg.terrain.random_box_kwargs
        )

        self.assertEqual(training_kwargs["num_unique_layouts"], 400)
        self.assertEqual(training_kwargs["box_count_range"], (2, 6))
        self.assertEqual(
            training_kwargs["lateral_offset_range"], (-1.0, 1.0)
        )
        self.assertEqual(
            training_kwargs["gap_distributions"],
            ({"range": (0.10, 1.50), "weight": 1.0},),
        )
        self.assertNotIn("gap_overrides", training_kwargs)
        self.assertNotIn("height_overrides", training_kwargs)
        self.assertNotIn("lateral_offset_overrides", training_kwargs)

    def test_layout_one_reproduces_original_layout_130(self):
        eval_cfg = Go2RandomBoxEvalCfg.terrain
        original_cfg = Go2RandomBoxCfg.terrain
        eval_terrain = make_terrain(eval_cfg)
        original_terrain = make_terrain(original_cfg)

        eval_specs = build_random_box_terrain(eval_terrain, eval_cfg, 1)
        original_specs = build_random_box_terrain(
            original_terrain, original_cfg, 130
        )

        self.assertEqual(eval_specs, original_specs)
        self.assertEqual(eval_terrain.box_count, original_terrain.box_count)
        np.testing.assert_allclose(
            eval_terrain.goals[: original_terrain.goals.shape[0]],
            original_terrain.goals,
        )
        np.testing.assert_allclose(
            eval_terrain.goals[original_terrain.goals.shape[0] :],
            np.repeat(
                original_terrain.goals[-1][None, :],
                eval_terrain.goals.shape[0] - original_terrain.goals.shape[0],
                axis=0,
            ),
        )

        resolved_kwargs, resolved_index = resolve_random_box_layout(
            eval_cfg.random_box_kwargs, 1
        )
        self.assertEqual(resolved_kwargs["seed"], 0)
        self.assertEqual(resolved_kwargs["num_unique_layouts"], 400)
        self.assertEqual(resolved_index, 130)

    def test_layout_two_uses_normalized_box_and_gap_ranges(self):
        cfg = Go2RandomBoxEvalCfg.terrain
        base_specs = build_random_box_terrain(make_terrain(cfg), cfg, 0)
        compact_specs = build_random_box_terrain(
            make_terrain(cfg), cfg, 2
        )

        self.assertEqual(len(base_specs), len(compact_specs))
        for base, compact in zip(base_specs, compact_specs):
            self.assertAlmostEqual(compact["length"], base["length"] - 0.30)
            self.assertGreaterEqual(compact["width"], 0.90 - 1e-8)
            self.assertLessEqual(compact["width"], 1.40 + 1e-8)
            self.assertGreaterEqual(compact["height"], 0.20 - 1e-8)
            self.assertLessEqual(compact["height"], 0.50 + 1e-8)
            if compact["index"] not in (2, 4):
                self.assertAlmostEqual(
                    compact["lateral_offset"], base["lateral_offset"]
                )
            if compact["index"] > 0:
                self.assertGreaterEqual(compact["gap"], 0.20 - 1e-8)
                if compact["index"] != 1:
                    self.assertLessEqual(compact["gap"], 0.70 + 1e-8)

        self.assertAlmostEqual(compact_specs[1]["gap"], 0.70)
        self.assertAlmostEqual(compact_specs[2]["gap"], 0.35)
        self.assertAlmostEqual(compact_specs[2]["height"], 0.45)
        self.assertAlmostEqual(compact_specs[2]["lateral_offset"], -0.25)
        self.assertAlmostEqual(
            compact_specs[2]["lateral_offset"],
            base_specs[2]["lateral_offset"] - 0.40,
        )
        self.assertAlmostEqual(compact_specs[3]["gap"], 0.25)
        self.assertAlmostEqual(compact_specs[3]["height"], 0.40)
        self.assertAlmostEqual(compact_specs[4]["gap"], 0.70)
        self.assertAlmostEqual(compact_specs[4]["height"], 0.20)
        self.assertAlmostEqual(compact_specs[4]["lateral_offset"], 0.25)
        self.assertAlmostEqual(
            compact_specs[4]["lateral_offset"],
            base_specs[4]["lateral_offset"] - 0.20,
        )
        self.assertAlmostEqual(compact_specs[5]["gap"], 0.35)
        self.assertAlmostEqual(compact_specs[5]["height"], 0.45)
        self.assertAlmostEqual(compact_specs[6]["gap"], 0.70)
        self.assertAlmostEqual(compact_specs[7]["gap"], 0.25)
        self.assertAlmostEqual(compact_specs[8]["gap"], 0.70)
        self.assertAlmostEqual(compact_specs[8]["height"], 0.20)
        self.assertAlmostEqual(compact_specs[9]["gap"], 0.35)
        self.assertAlmostEqual(compact_specs[9]["height"], 0.45)

if __name__ == "__main__":
    unittest.main()
