"""Evaluation task containing native and fixed random-box Go2 terrains."""

from copy import deepcopy

from legged_gym.envs.go2.go2_parkour_config import (
    Go2ParkourCfg,
    Go2ParkourCfgPPO,
)
from legged_gym.envs.go2.go2_random_box_config import Go2RandomBoxCfg


ORIGINAL_TRAINING_LAYOUT_INDICES = (51, 186, 357, 130, 221)


def _build_layout_presets():
    presets = {
        2: {
            "source_layout_index": 2,
            "lateral_offset_overrides": {
                0: -0.25,
                1: 0.25,
                2: -0.25,
                3: 0.25,
            },
        },
    }
    for logical_index, source_index in enumerate(
        ORIGINAL_TRAINING_LAYOUT_INDICES, start=5
    ):
        presets[logical_index] = dict(
            deepcopy(Go2RandomBoxCfg.terrain.random_box_kwargs),
            source_layout_index=source_index,
            box_count_range=(7, 7),
        )
    return presets


class Go2MixedCfg(Go2ParkourCfg):
    class env(Go2ParkourCfg.env):
        num_envs = 15
        draw_all_goals = True
        goal_success_rate_groups = {
            "native_1_5_jump_success_rate": (0, 5),
            "custom_6_10_average_success_rate": (5, 10),
        }

    class terrain(Go2ParkourCfg.terrain):
        num_rows = 1
        num_cols = 15
        max_init_terrain_level = 0
        curriculum = False
        fixed_terrain_difficulty = 1.0
        height = [0.02, 0.02]

        terrain_dict = {
            "smooth slope": 0.0,
            "rough slope up": 0.0,
            "rough slope down": 0.0,
            "rough stairs up": 0.0,
            "rough stairs down": 0.0,
            "discrete": 0.0,
            "stepping stones": 0.0,
            "gaps": 0.0,
            "smooth flat": 0.0,
            "pit": 0.0,
            "wall": 0.0,
            "platform": 0.0,
            "large stairs up": 0.0,
            "large stairs down": 0.0,
            "parkour": 1.0,
            "parkour_hurdle": 1.0,
            "parkour_flat": 0.0,
            "parkour_step": 1.0,
            "parkour_gap": 1.0,
            "demo": 1.0,
            "five_box": 0.0,
            "random_box": 10.0,
        }
        terrain_proportions = list(terrain_dict.values())

        random_box_kwargs = deepcopy(
            Go2RandomBoxCfg.terrain.random_box_kwargs
        )
        random_box_kwargs.update(
            {
                "seed": 0,
                "num_unique_layouts": 10,
                "box_count_range": (7, 7),
                "height_range": (0.20, 0.50),
                "length_range": (0.50, 1.00),
                "width_range": (0.50, 1.00),
                "lateral_offset_range": (-1.00, 1.00),
                "gap_distributions": (
                    {"range": (0.20, 1.00), "weight": 1.0},
                ),
                "ground_roughness_distributions": (
                    {"range": (0.02, 0.02), "weight": 1.0},
                ),
                "layout_presets": _build_layout_presets(),
            }
        )

    class depth(Go2ParkourCfg.depth):
        camera_num_envs = 15
        camera_terrain_num_rows = 1
        camera_terrain_num_cols = 15
        preserve_custom_terrain_with_camera = True


class Go2MixedCfgPPO(Go2ParkourCfgPPO):
    pass
