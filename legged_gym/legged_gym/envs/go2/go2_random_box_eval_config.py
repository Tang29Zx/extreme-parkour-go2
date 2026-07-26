"""Fixed ten-box generalization evaluation task."""

from copy import deepcopy

from legged_gym.envs.go2.go2_random_box_config import (
    Go2RandomBoxCfg,
    Go2RandomBoxCfgPPO,
)


class Go2RandomBoxEvalCfg(Go2RandomBoxCfg):
    class env(Go2RandomBoxCfg.env):
        num_envs = 1

    class domain_rand(Go2RandomBoxCfg.domain_rand):
        push_robots = False

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
                "num_unique_layouts": 4,
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
                            1: 0.40,
                            2: 0.35,
                            3: 0.25,
                            4: 0.90,
                            5: 0.35,
                            6: 0.90,
                            7: 0.25,
                            8: 0.40,
                            9: 0.25,
                        },
                        # Alternate low-to-high challenges while giving every
                        # high-to-low transition an explicit approach gap.
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
                    3: {
                        "source_layout_index": 0,
                        "length_range": (0.50, 0.95),
                        "width_range": (0.80, 1.10),
                        "height_range": (0.15, 0.50),
                        "lateral_offset_range": (-0.65, 0.65),
                        "gap_distributions": (
                            {"range": (0.20, 0.95), "weight": 1.0},
                        ),
                        "gap_overrides": {
                            1: 0.20,
                            2: 0.85,
                            3: 0.20,
                            4: 0.90,
                            5: 0.25,
                            6: 0.95,
                            7: 0.20,
                            8: 0.85,
                            9: 0.20,
                        },
                        "height_overrides": {
                            0: 0.20,
                            1: 0.50,
                            2: 0.15,
                            3: 0.48,
                            4: 0.20,
                            5: 0.50,
                            6: 0.15,
                            7: 0.45,
                            8: 0.15,
                            9: 0.50,
                        },
                        # Positive y is left while facing world +x.
                        "lateral_offset_overrides": {
                            0: 0.00,
                            1: 0.55,
                            2: -0.55,
                            3: 0.65,
                            4: -0.60,
                            5: 0.60,
                            6: -0.65,
                            7: 0.55,
                            8: -0.55,
                            9: 0.60,
                        },
                        "ground_roughness_distributions": (
                            {"range": (0.06, 0.06), "weight": 1.0},
                        ),
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
