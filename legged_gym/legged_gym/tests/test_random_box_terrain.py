"""Focused tests for the seeded random multi-box terrain builder."""

from collections import Counter
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "utils" / "random_box_terrain.py"
SPEC = importlib.util.spec_from_file_location("random_box_terrain", MODULE_PATH)
RANDOM_BOX_TERRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RANDOM_BOX_TERRAIN)


def make_cfg():
    return SimpleNamespace(
        num_goals=7,
        random_box_kwargs={
            "seed": 0,
            "num_unique_layouts": 400,
            "box_count_range": (2, 6),
            "spawn_margin": 0.6,
            "first_runup_range": (1.0, 1.5),
            "height_range": (0.10, 0.50),
            "length_range": (0.8, 1.5),
            "width_range": (0.8, 1.5),
            "lateral_offset_range": (-1.0, 1.0),
            "gap_distributions": (
                {"range": (0.10, 1.50), "weight": 1.0},
            ),
            "ground_roughness_distributions": (
                {"range": (0.0, 0.0), "weight": 0.30},
                {"range": (0.0, 0.02), "weight": 0.50},
                {"range": (0.02, 0.06), "weight": 0.20},
            ),
            "exit_goal_distance": 1.0,
            "end_margin": 0.5,
        },
    )


def make_terrain():
    return SimpleNamespace(
        horizontal_scale=0.05,
        vertical_scale=0.005,
        width=500,
        length=80,
        height_field_raw=np.zeros((500, 80), dtype=np.int16),
    )


class RandomBoxTerrainTest(unittest.TestCase):
    def test_layout_is_reproducible_and_goal_padding_has_no_fake_boxes(self):
        first = make_terrain()
        second = make_terrain()
        first_specs = RANDOM_BOX_TERRAIN.build_random_box_terrain(
            first, make_cfg(), 17
        )
        second_specs = RANDOM_BOX_TERRAIN.build_random_box_terrain(
            second, make_cfg(), 17
        )

        np.testing.assert_array_equal(first.height_field_raw, second.height_field_raw)
        self.assertEqual(first_specs, second_specs)
        self.assertEqual(first.box_count, len(first_specs))
        padded_goals = first.goals[first.box_count:]
        np.testing.assert_allclose(
            padded_goals,
            np.repeat(first.goals[-1][None, :], len(padded_goals), axis=0),
        )
        self.assertEqual(len(first.box_specs), first.box_count)

    def test_all_400_layouts_are_balanced_and_within_bounds(self):
        cfg = make_cfg()
        box_counts = Counter()
        gap_classes = Counter()
        unique_layouts = set()

        for layout_index in range(400):
            terrain = make_terrain()
            specs = RANDOM_BOX_TERRAIN.build_random_box_terrain(
                terrain, cfg, layout_index
            )
            box_counts[len(specs)] += 1
            unique_layouts.add(
                tuple(
                    (
                        spec["front_x"],
                        spec["length"],
                        spec["width"],
                        spec["height"],
                        spec["lateral_offset"],
                    )
                    for spec in specs
                )
            )

            self.assertGreaterEqual(specs[0]["gap"], 1.0)
            self.assertLessEqual(specs[0]["gap"], 1.5)
            for spec in specs:
                self.assertGreaterEqual(spec["height"], 0.10)
                self.assertLessEqual(spec["height"], 0.50)
                self.assertGreaterEqual(spec["length"], 0.8)
                self.assertLessEqual(spec["length"], 1.5)
                self.assertGreaterEqual(spec["width"], 0.8)
                self.assertLessEqual(spec["width"], 1.5)
                self.assertGreaterEqual(spec["lateral_offset"], -1.0)
                self.assertLessEqual(spec["lateral_offset"], 1.0)
                self.assertGreaterEqual(spec["y_min"], 0.0)
                self.assertLessEqual(spec["y_max"], 4.0)
                self.assertLessEqual(spec["rear_x"], 25.0)
                if spec["gap_class"] >= 0:
                    gap_classes[spec["gap_class"]] += 1

            self.assertLessEqual(terrain.goals[-1, 0], 24.5)

        self.assertEqual(box_counts, Counter({2: 80, 3: 80, 4: 80, 5: 80, 6: 80}))
        self.assertEqual(set(gap_classes), {0})
        sampled_gaps = [
            spec["gap"]
            for layout_index in range(400)
            for spec in RANDOM_BOX_TERRAIN.build_random_box_terrain(
                make_terrain(), cfg, layout_index
            )[1:]
        ]
        self.assertGreaterEqual(min(sampled_gaps), 0.10)
        self.assertLessEqual(max(sampled_gaps), 1.50)
        self.assertLess(abs(float(np.mean(sampled_gaps)) - 0.80), 0.05)
        self.assertEqual(len(unique_layouts), 400)

    def test_roughness_mix_is_balanced_and_reproducible(self):
        cfg = make_cfg()
        selected = [
            RANDOM_BOX_TERRAIN.select_roughness_range(
                cfg.random_box_kwargs["seed"],
                cfg.random_box_kwargs["num_unique_layouts"],
                cfg.random_box_kwargs["ground_roughness_distributions"],
                layout_index,
            )
            for layout_index in range(400)
        ]
        self.assertEqual(Counter(item[1] for item in selected), {0: 120, 1: 200, 2: 80})
        self.assertEqual(selected, [
            RANDOM_BOX_TERRAIN.select_roughness_range(
                cfg.random_box_kwargs["seed"],
                cfg.random_box_kwargs["num_unique_layouts"],
                cfg.random_box_kwargs["ground_roughness_distributions"],
                layout_index,
            )
            for layout_index in range(400)
        ])


if __name__ == "__main__":
    unittest.main()
