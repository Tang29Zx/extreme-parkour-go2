"""Go2 configuration for replaying the native policy on five boxes."""

from legged_gym.envs.go2.go2_parkour_config import (
    Go2ParkourCfg,
    Go2ParkourCfgPPO,
)


class Go2FiveBoxCfg(Go2ParkourCfg):
    """Keep the native policy interface while replacing only the terrain."""

    class terrain(Go2ParkourCfg.terrain):
        terrain_length = 18.0
        terrain_width = 2.0
        num_rows = 4
        num_cols = 4
        max_init_terrain_level = 3
        num_goals = 6
        curriculum = False
        max_difficulty = False
        origin_zero_z = True
        height = [0.0, 0.0]

        terrain_dict = {
            name: 0.0 for name in Go2ParkourCfg.terrain.terrain_dict
        }
        terrain_dict["five_box"] = 1.0
        terrain_proportions = list(terrain_dict.values())

        five_box_kwargs = {
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
        }

    class commands(Go2ParkourCfg.commands):
        curriculum = False
        resampling_time = 60.0

        class ranges(Go2ParkourCfg.commands.ranges):
            lin_vel_x = [0.5, 0.5]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

        class max_ranges(Go2ParkourCfg.commands.max_ranges):
            lin_vel_x = [0.5, 0.5]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class domain_rand(Go2ParkourCfg.domain_rand):
        # The native observation path always expects these latent tensors.
        # Degenerate ranges keep them deterministic without changing its schema.
        randomize_friction = True
        friction_range = [1.0, 1.0]
        randomize_base_mass = False
        randomize_base_com = False
        randomize_motor = False
        motor_strength_range = [1.0, 1.0]
        push_robots = False
        action_delay = False

    class noise(Go2ParkourCfg.noise):
        add_noise = False

    class viewer(Go2ParkourCfg.viewer):
        pos = [4.0, -5.0, 3.0]
        lookat = [7.0, 0.0, 0.3]


class Go2FiveBoxCfgPPO(Go2ParkourCfgPPO):
    """Use the unchanged Extreme Parkour network and optimizer schema."""

    class runner(Go2ParkourCfgPPO.runner):
        experiment_name = "parkour_A100"
        run_name = "five_box_capability"
