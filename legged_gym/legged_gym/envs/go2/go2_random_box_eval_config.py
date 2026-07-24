"""Fixed ten-box generalization evaluation task."""

from copy import deepcopy

from legged_gym.envs.go2.go2_random_box_config import (
    Go2RandomBoxCfg,
    Go2RandomBoxCfgPPO,
)


class Go2RandomBoxEvalCfg(Go2RandomBoxCfg):
    class env(Go2RandomBoxCfg.env):
        num_envs = 1

    class terrain(Go2RandomBoxCfg.terrain):
        terrain_length = 30.0
        terrain_width = 4.0
        num_rows = 1
        num_cols = 1
        max_init_terrain_level = 0
        num_goals = 11

        random_box_kwargs = deepcopy(
            Go2RandomBoxCfg.terrain.random_box_kwargs
        )
        random_box_kwargs.update(
            {
                "seed": 20260724,
                "num_unique_layouts": 3,
                "box_count_range": (10, 10),
                "lateral_offset_range": (-0.5, 0.5),
                "gap_distributions": (
                    {"range": (0.30, 1.00), "weight": 1.0},
                ),
                # Keys are zero-based box indices. These three transitions are
                # intentionally close to probe short-gap behavior.
                "gap_overrides": {
                    2: 0.35,
                    4: 0.35,
                    8: 0.35,
                },
                "height_overrides": {
                    3: 0.40,
                },
                "layout_presets": {
                    1: dict(
                        deepcopy(
                            Go2RandomBoxCfg.terrain.random_box_kwargs
                        ),
                        source_layout_index=130,
                        gap_overrides={},
                        height_overrides={},
                    ),
                    2: {
                        "source_layout_index": 0,
                        "length_range": (0.50, 1.20),
                        "width_range": (0.90, 1.40),
                        "height_range": (0.20, 0.50),
                        "gap_distributions": (
                            {"range": (0.20, 0.70), "weight": 1.0},
                        ),
                        "gap_overrides": {
                            1: 0.70,
                            2: 0.35,
                            3: 0.25,
                            4: 0.70,
                            5: 0.35,
                            6: 0.70,
                            7: 0.25,
                            8: 0.70,
                            9: 0.35,
                        },
                        # Alternate low-to-high challenges while giving every
                        # high-to-low transition the full 0.70 m approach gap.
                        "height_overrides": {
                            2: 0.45,
                            3: 0.40,
                            4: 0.20,
                            5: 0.45,
                            8: 0.20,
                            9: 0.45,
                        },
                        # Positive y is left while facing world +x.
                        "lateral_offset_overrides": {
                            2: -0.25,
                            4: 0.25,
                        },
                    },
                },
                "ground_roughness_distributions": (
                    {"range": (0.03, 0.03), "weight": 1.0},
                ),
            }
        )

    class depth(Go2RandomBoxCfg.depth):
        camera_terrain_num_rows = 1
        camera_terrain_num_cols = 1


class Go2RandomBoxEvalCfgPPO(Go2RandomBoxCfgPPO):
    pass
