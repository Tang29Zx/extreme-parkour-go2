"""Focused tests for durable local training logs."""

import csv
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils.training_logger import (
    LocalMetricsWriter,
    append_run_manifest,
)


class LocalMetricsWriterTest(unittest.TestCase):
    def test_writes_metrics_text_checkpoint_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = LocalMetricsWriter(
                temp_dir,
                enable_tensorboard=False,
            )
            writer.write(
                iteration=12,
                total_timesteps=3456,
                wall_time_s=7.5,
                metrics={
                    "Train/mean_reward": 2.5,
                    "Episode/success_rate": 0.75,
                },
            )
            writer.append_text("\x1b[1miteration 12\x1b[0m")
            checkpoint_path = os.path.join(temp_dir, "model_12.pt")
            writer.record_checkpoint(
                checkpoint_path,
                iteration=12,
                total_timesteps=3456,
                training_mode="rl",
            )
            append_run_manifest(
                temp_dir,
                {"task": "go2", "git": {"dirty": True}},
            )
            writer.close()

            with open(
                Path(temp_dir) / "metrics.csv",
                newline="",
                encoding="utf-8",
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["iteration"] for row in rows}, {"12"})
            self.assertEqual(
                {row["metric"] for row in rows},
                {"Train/mean_reward", "Episode/success_rate"},
            )

            train_log = (Path(temp_dir) / "train.log").read_text(
                encoding="utf-8"
            )
            self.assertEqual(train_log, "iteration 12\n")

            with open(
                Path(temp_dir) / "checkpoints.csv",
                newline="",
                encoding="utf-8",
            ) as stream:
                checkpoint_rows = list(csv.DictReader(stream))
            self.assertEqual(checkpoint_rows[0]["iteration"], "12")
            self.assertEqual(checkpoint_rows[0]["training_mode"], "rl")
            self.assertEqual(
                checkpoint_rows[0]["path"],
                os.path.abspath(checkpoint_path),
            )

            manifests = (Path(temp_dir) / "run_manifest.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(manifests), 1)
            manifest = json.loads(manifests[0])
            self.assertEqual(manifest["task"], "go2")
            self.assertTrue(manifest["git"]["dirty"])
            self.assertIn("timestamp", manifest)

    def test_appends_iterations_without_replacing_existing_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = LocalMetricsWriter(
                temp_dir,
                enable_tensorboard=False,
            )
            writer.write(1, 10, 1.0, {"Loss/value_function": 3.0})
            writer.write(2, 20, 2.0, {"Loss/value_function": 2.0})
            writer.close()

            with open(
                Path(temp_dir) / "metrics.csv",
                newline="",
                encoding="utf-8",
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["iteration"] for row in rows], ["1", "2"])
            self.assertEqual([row["value"] for row in rows], ["3.0", "2.0"])

    def test_episode_aggregation_accepts_sparse_dynamic_keys(self):
        runner = OnPolicyRunner.__new__(OnPolicyRunner)
        runner.device = "cpu"

        metrics = runner._collect_episode_metrics(
            [
                {
                    "success_rate": torch.tensor([1.0, 0.0]),
                    "terrain_parkour_success_rate": torch.tensor([1.0]),
                },
                {
                    "success_rate": torch.tensor([1.0]),
                    "terrain_random_box_success_rate": torch.tensor([0.0]),
                },
            ]
        )

        self.assertAlmostEqual(metrics["Episode/success_rate"], 2.0 / 3.0)
        self.assertEqual(metrics["Episode/terrain_parkour_success_rate"], 1.0)
        self.assertEqual(
            metrics["Episode/terrain_random_box_success_rate"], 0.0
        )

    def test_checkpoint_contains_recoverable_training_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = OnPolicyRunner.__new__(OnPolicyRunner)
            actor_critic = torch.nn.Linear(2, 1)
            estimator = torch.nn.Linear(2, 1)
            runner.alg = SimpleNamespace(
                actor_critic=actor_critic,
                estimator=estimator,
                optimizer=torch.optim.Adam(actor_critic.parameters()),
            )
            runner.if_depth = False
            runner.current_learning_iteration = 12
            runner.tot_timesteps = 3456
            runner.tot_time = 7.5
            runner.training_mode = "rl"
            runner.metrics_writer = LocalMetricsWriter(
                temp_dir,
                enable_tensorboard=False,
            )
            path = os.path.join(temp_dir, "model_12.pt")

            runner.save(path, iteration=12)
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )

            self.assertEqual(checkpoint["checkpoint_format_version"], 2)
            self.assertEqual(checkpoint["iter"], 12)
            self.assertEqual(checkpoint["total_timesteps"], 3456)
            self.assertEqual(checkpoint["total_time"], 7.5)
            self.assertEqual(checkpoint["training_mode"], "rl")
            runner.metrics_writer.close()


if __name__ == "__main__":
    unittest.main()
