"""Geometry-only adaptation task for the random multi-box terrain."""

from legged_gym.envs.go2.go2_random_box_config import (
    Go2RandomBoxCfg,
    Go2RandomBoxCfgPPO,
)


class Go2RandomBoxCleanCfg(Go2RandomBoxCfg):
    class terrain(Go2RandomBoxCfg.terrain):
        measure_horizontal_offset = 0.0
        measure_point_jitter = 0.0

    class domain_rand(Go2RandomBoxCfg.domain_rand):
        action_delay = False
        action_delay_range = [0, 0]

    class noise(Go2RandomBoxCfg.noise):
        add_noise = False
        apply_observation_noise = False
        contact_dropout_prob = 0.0


class Go2RandomBoxCleanCfgPPO(Go2RandomBoxCfgPPO):
    class runner(Go2RandomBoxCfgPPO.runner):
        run_name = "random_box_geometry_from_29500"
