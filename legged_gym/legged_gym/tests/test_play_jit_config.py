"""Focused tests for configurable JIT replay scenes."""

from types import SimpleNamespace
import unittest

import isaacgym  # Must precede torch imports triggered by replay modules.

from legged_gym.scripts.play_jit import (
    configure_fixed_single_box,
    offset_camera_z,
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

        configure_fixed_single_box(cfg, height=0.45, friction=0.5)

        self.assertEqual(cfg.env.episode_length_s, 15)
        self.assertEqual(cfg.terrain.num_goals, 2)
        self.assertEqual(cfg.terrain.random_box_kwargs["box_count_range"], (1, 1))
        self.assertEqual(cfg.terrain.random_box_kwargs["height_range"], (0.45, 0.45))
        self.assertEqual(cfg.terrain.random_box_kwargs["length_range"], (1.2, 1.2))
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


if __name__ == "__main__":
    unittest.main()
