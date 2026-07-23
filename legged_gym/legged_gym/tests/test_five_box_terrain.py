"""Focused tests for the deterministic five-box heightfield generator."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "utils" / "five_box_terrain.py"
SPEC = importlib.util.spec_from_file_location("five_box_terrain", MODULE_PATH)
FIVE_BOX_TERRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIVE_BOX_TERRAIN)


def make_cfg():
    return SimpleNamespace(
        num_goals=6,
        five_box_kwargs={
            "seed": 0,
            "num_unique_layouts": 4,
            "spawn_margin": 0.6,
            "first_gap_range": (0.8, 1.2),
            "gap_distributions": (
                {"range": (0.45, 0.75), "weight": 0.2},
                {"range": (0.75, 1.20), "weight": 0.6},
                {"range": (1.20, 1.60), "weight": 0.2},
            ),
            "high_box_threshold": 0.4,
            "post_high_min_gap": 0.8,
            "exit_goal_distance": 1.0,
            "boxes": (
                {"length": 1.2, "width": 1.2, "height": 0.20},
                {"length": 1.2, "width": 1.2, "height": 0.30},
                {"length": 1.2, "width": 1.2, "height": 0.40},
                {"length": 1.2, "width": 1.2, "height": 0.40},
                {
                    "length": 1.2,
                    "width": 1.2,
                    "height": 0.40,
                    "gap_range": (1.60, 2.00),
                },
            ),
        },
    )


def make_terrain():
    return SimpleNamespace(
        horizontal_scale=0.05,
        vertical_scale=0.005,
        width=360,
        length=40,
        height_field_raw=np.zeros((360, 40), dtype=np.int16),
    )


class FiveBoxTerrainTest(unittest.TestCase):
    def test_layout_is_reproducible_and_matches_geometry(self):
        first = make_terrain()
        second = make_terrain()
        first_specs = FIVE_BOX_TERRAIN.build_five_box_terrain(first, make_cfg(), 2)
        second_specs = FIVE_BOX_TERRAIN.build_five_box_terrain(second, make_cfg(), 2)

        np.testing.assert_array_equal(first.height_field_raw, second.height_field_raw)
        self.assertEqual(first_specs, second_specs)
        self.assertEqual([spec["height"] for spec in first_specs], [0.2, 0.3, 0.4, 0.4, 0.4])
        self.assertTrue(all(np.isclose(spec["rear_x"] - spec["front_x"], 1.2) for spec in first_specs))
        self.assertTrue(all(np.isclose(spec["y_max"] - spec["y_min"], 1.2) for spec in first_specs))
        self.assertGreaterEqual(first_specs[-1]["gap"], 1.60)
        self.assertLessEqual(first_specs[-1]["gap"], 2.00)
        self.assertTrue(np.all(np.diff(first.goals[:, 0]) > 0.0))
        self.assertTrue(np.allclose(first.env_origin, [0.6, 1.0, 0.0]))

    def test_four_seeded_layouts_repeat_by_index(self):
        layouts = []
        for layout_index in range(4):
            terrain = make_terrain()
            specs = FIVE_BOX_TERRAIN.build_five_box_terrain(
                terrain, make_cfg(), layout_index
            )
            layouts.append(tuple(spec["gap"] for spec in specs))

        self.assertGreater(len(set(layouts)), 1)


if __name__ == "__main__":
    unittest.main()
