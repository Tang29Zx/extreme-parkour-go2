"""Focused tests for configurable JIT replay scenes."""

from types import SimpleNamespace
import unittest

import isaacgym  # Must precede torch imports triggered by replay modules.

from legged_gym.scripts.play_jit import (
    configure_camera_position_stress_noise,
    configure_fixed_single_box,
    offset_camera_z,
    set_per_env_camera_positions,
)


def make_env_cfg():
    return SimpleNamespace(
        env=SimpleNamespace(episode_length_s=60),
        terrain=SimpleNamespace(
            num_rows=10,
            num_cols=40,
            max_init_terrain_level=9,
            num_goals=7,
            curriculum=False,
            max_difficulty=False,
            random_box_kwargs={
                "layout_presets": {0: {}},
                "spawn_margin": 0.6,
            },
        ),
        domain_rand=SimpleNamespace(
            randomize_friction=True,
            friction_range=[0.7, 1.3],
            randomize_base_mass=True,
            randomize_base_com=True,
            randomize_motor=True,
            push_robots=True,
            action_delay=True,
        ),
        noise=SimpleNamespace(
            add_noise=True,
            apply_observation_noise=True,
            contact_dropout_prob=0.01,
        ),
        depth=SimpleNamespace(
            position={
                "mean": [0.355, 0.0, 0.065],
                "std": [0.015, 0.01, 0.015],
            }
        ),
    )


class PlayJitConfigTest(unittest.TestCase):
    def test_fixed_single_box_replaces_geometry_and_randomization(self):
        cfg = make_env_cfg()

        configure_fixed_single_box(
            cfg,
            height=0.45,
            friction=0.5,
            ground_roughness=0.01,
        )

        self.assertEqual(cfg.env.episode_length_s, 15)
        self.assertEqual(cfg.terrain.num_goals, 2)
        self.assertEqual(cfg.terrain.random_box_kwargs["box_count_range"], (1, 1))
        self.assertEqual(cfg.terrain.random_box_kwargs["height_range"], (0.45, 0.45))
        self.assertEqual(cfg.terrain.random_box_kwargs["length_range"], (1.2, 1.2))
        self.assertEqual(
            cfg.terrain.random_box_kwargs["ground_roughness_distributions"],
            ({"range": (0.01, 0.01), "weight": 1.0},),
        )
        self.assertNotIn("layout_presets", cfg.terrain.random_box_kwargs)
        self.assertEqual(cfg.domain_rand.friction_range, [0.5, 0.5])
        self.assertFalse(cfg.domain_rand.randomize_base_mass)
        self.assertFalse(cfg.domain_rand.push_robots)
        self.assertFalse(cfg.noise.add_noise)

    def test_camera_z_offset_preserves_other_distribution_values(self):
        cfg = make_env_cfg()

        mean, std = offset_camera_z(cfg, 0.05)

        self.assertEqual(mean, [0.355, 0.0, 0.115])
        self.assertEqual(std, [0.015, 0.01, 0.015])
        self.assertEqual(cfg.depth.position["std"], [0.015, 0.01, 0.015])

    def test_fixed_single_box_can_preserve_randomization(self):
        cfg = make_env_cfg()

        configure_fixed_single_box(
            cfg,
            height=0.25,
            friction=None,
            ground_roughness=0.005,
            preserve_randomization=True,
        )

        self.assertEqual(cfg.terrain.num_goals, 2)
        self.assertEqual(cfg.terrain.random_box_kwargs["height_range"], (0.25, 0.25))
        self.assertTrue(cfg.domain_rand.randomize_friction)
        self.assertEqual(cfg.domain_rand.friction_range, [0.7, 1.3])
        self.assertTrue(cfg.domain_rand.randomize_base_mass)
        self.assertTrue(cfg.domain_rand.randomize_base_com)
        self.assertTrue(cfg.domain_rand.randomize_motor)
        self.assertTrue(cfg.domain_rand.push_robots)
        self.assertTrue(cfg.domain_rand.action_delay)
        self.assertTrue(cfg.noise.add_noise)
        self.assertTrue(cfg.noise.apply_observation_noise)

    def test_stress_noise_profile_is_fixed_and_conservative(self):
        cfg = make_env_cfg()
        cfg.depth.rotation = {}
        cfg.depth.horizontal_fov = [86, 90]
        cfg.depth.gaussian_noise_std = 0.0
        cfg.depth.dis_noise = 0.0
        cfg.depth.depth_dropout_prob = 0.0
        cfg.depth.depth_occlusion_prob = 0.0
        cfg.depth.depth_occlusion_size_range = [0.05, 0.15]
        cfg.domain_rand.action_delay_range = [0, 2]

        configure_camera_position_stress_noise(cfg)

        self.assertTrue(cfg.noise.add_noise)
        self.assertEqual(cfg.noise.contact_dropout_prob, 0.01)
        self.assertEqual(cfg.domain_rand.action_delay_range, [2, 2])
        self.assertEqual(cfg.depth.horizontal_fov, 86)
        self.assertEqual(cfg.depth.rotation["lower"], cfg.depth.rotation["upper"])
        self.assertAlmostEqual(cfg.depth.rotation["lower"][1], 0.3926990817)
        self.assertEqual(cfg.depth.depth_occlusion_size_range, [0.15, 0.15])

    def test_per_env_positions_are_exact(self):
        cfg = make_env_cfg()

        positions = set_per_env_camera_positions(
            cfg,
            [[0.355, 0.0, 0.065], [0.205, 0.0, 0.065]],
        )

        self.assertEqual(positions[1], [0.205, 0.0, 0.065])
        self.assertEqual(cfg.depth.position["std"], [0.0, 0.0, 0.0])
        self.assertEqual(cfg.depth.position["per_env"], positions)


if __name__ == "__main__":
    unittest.main()
