"""Independent random multi-box task with moderate sim-to-real randomization."""

from legged_gym.envs.go2.go2_parkour_config import (
    Go2ParkourCfg,
    Go2ParkourCfgPPO,
)


class Go2RandomBoxCfg(Go2ParkourCfg):
    class env(Go2ParkourCfg.env):
        episode_length_s = 60
        randomize_start_pos = True
        start_pos_range = [0.05, 0.05]
        randomize_start_yaw = True
        rand_yaw_range = 0.05
        dof_pos_reset_range = [-0.05, 0.05]

    class terrain(Go2ParkourCfg.terrain):
        terrain_length = 25.0
        terrain_width = 4.0
        num_rows = 10
        num_cols = 40
        max_init_terrain_level = 9
        num_goals = 7
        curriculum = False
        max_difficulty = False
        origin_zero_z = True
        height = [0.0, 0.0]
        preserve_custom_terrain_with_camera = True
        horizontal_scale_camera = 0.05
        measure_horizontal_offset = 0.02
        measure_point_jitter = 0.01

        terrain_dict = {
            name: 0.0 for name in Go2ParkourCfg.terrain.terrain_dict
        }
        terrain_dict["five_box"] = 0.0
        terrain_dict["random_box"] = 1.0
        terrain_proportions = list(terrain_dict.values())

        random_box_kwargs = {
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
        randomize_friction = True
        friction_range = [0.7, 1.3]
        randomize_base_mass = True
        added_mass_range = [-0.5, 1.5]
        randomize_base_com = True
        added_com_range = [-0.03, 0.03]
        randomize_motor = True
        motor_strength_range = [0.90, 1.10]
        push_robots = True
        push_interval_s = 8.0
        max_push_vel_xy = 0.20
        action_delay = True
        action_delay_range = [0, 2]

    class noise(Go2ParkourCfg.noise):
        add_noise = True
        apply_observation_noise = True
        noise_level = 1.0
        contact_dropout_prob = 0.01

        class noise_scales(Go2ParkourCfg.noise.noise_scales):
            rotation = 0.015
            dof_pos = 0.01
            dof_vel = 0.45
            lin_vel = 0.05
            ang_vel = 0.06
            goal_yaw = 0.02
            height_measurements = 0.03

    class depth(Go2ParkourCfg.depth):
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 40
        position = {
            "mean": [0.355, 0.0, 0.065],
            "std": [0.015, 0.01, 0.015],
        }
        rotation = {
            "lower": [-0.0349066, 0.349066, -0.0349066],
            "upper": [0.0349066, 0.436332, 0.0349066],
        }
        horizontal_fov = [86, 90]
        gaussian_noise_std = 0.01
        dis_noise = 0.02
        depth_dropout_prob = 0.01
        depth_occlusion_prob = 0.30
        depth_occlusion_size_range = [0.05, 0.15]

    class viewer(Go2ParkourCfg.viewer):
        pos = [5.0, -7.0, 4.0]
        lookat = [9.0, 0.0, 0.4]


class Go2RandomBoxCfgPPO(Go2ParkourCfgPPO):
    class runner(Go2ParkourCfgPPO.runner):
        experiment_name = "parkour_A100"
        run_name = "random_box_sim2real"
