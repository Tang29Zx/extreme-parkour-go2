"""Durable local metrics for training and checkpoint provenance."""

import csv
from datetime import datetime, timezone
import json
import os
import re
import warnings


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _utc_timestamp():
    return datetime.now(timezone.utc).isoformat()


def append_run_manifest(log_dir, manifest):
    """Append one JSON launch record without overwriting previous runs."""
    if log_dir is None:
        return
    os.makedirs(log_dir, exist_ok=True)
    record = dict(manifest)
    record.setdefault("timestamp", _utc_timestamp())
    path = os.path.join(log_dir, "run_manifest.jsonl")
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")


class LocalMetricsWriter:
    """Write scalar metrics to CSV, TensorBoard, and a plain text log."""

    def __init__(self, log_dir, enable_tensorboard=True):
        self.log_dir = log_dir
        self.summary_writer = None
        if log_dir is None:
            return

        os.makedirs(log_dir, exist_ok=True)
        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.summary_writer = SummaryWriter(
                    log_dir=log_dir,
                    flush_secs=10,
                )
            except (ImportError, ModuleNotFoundError) as error:
                warnings.warn(
                    "TensorBoard is unavailable; metrics.csv remains enabled: "
                    + str(error)
                )

    def write(self, iteration, total_timesteps, wall_time_s, metrics):
        """Append one iteration of finite scalar metrics."""
        if self.log_dir is None:
            return

        rows = []
        timestamp = _utc_timestamp()
        for metric, value in sorted(metrics.items()):
            scalar = float(value)
            rows.append(
                (
                    timestamp,
                    int(iteration),
                    int(total_timesteps),
                    float(wall_time_s),
                    metric,
                    scalar,
                )
            )
            if self.summary_writer is not None:
                self.summary_writer.add_scalar(metric, scalar, int(iteration))

        if rows:
            path = os.path.join(self.log_dir, "metrics.csv")
            new_file = not os.path.exists(path) or os.path.getsize(path) == 0
            with open(path, "a", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                if new_file:
                    writer.writerow(
                        [
                            "timestamp",
                            "iteration",
                            "total_timesteps",
                            "wall_time_s",
                            "metric",
                            "value",
                        ]
                    )
                writer.writerows(rows)

        if self.summary_writer is not None:
            self.summary_writer.flush()

    def append_text(self, text):
        """Append an ANSI-free copy of the terminal iteration summary."""
        if self.log_dir is None:
            return
        path = os.path.join(self.log_dir, "train.log")
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(_ANSI_ESCAPE.sub("", text).rstrip() + "\n")

    def record_checkpoint(
        self,
        path,
        iteration,
        total_timesteps,
        training_mode,
    ):
        """Append checkpoint metadata that can be read without PyTorch."""
        if self.log_dir is None:
            return
        manifest_path = os.path.join(self.log_dir, "checkpoints.csv")
        new_file = (
            not os.path.exists(manifest_path)
            or os.path.getsize(manifest_path) == 0
        )
        with open(
            manifest_path, "a", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            if new_file:
                writer.writerow(
                    [
                        "timestamp",
                        "iteration",
                        "total_timesteps",
                        "training_mode",
                        "path",
                    ]
                )
            writer.writerow(
                [
                    _utc_timestamp(),
                    int(iteration),
                    int(total_timesteps),
                    training_mode,
                    os.path.abspath(path),
                ]
            )

    def close(self):
        if self.summary_writer is not None:
            self.summary_writer.close()
