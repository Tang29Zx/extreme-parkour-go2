# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin


import numpy as np
import os
import subprocess
import sys
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import class_to_dict, get_args, task_registry
from shutil import copyfile
import torch
import wandb
from rsl_rl.utils.training_logger import append_run_manifest


def get_git_metadata():
    """Return reproducibility metadata without reading repository contents."""
    metadata = {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=LEGGED_GYM_ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=LEGGED_GYM_ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        metadata["commit"] = commit.stdout.strip()
        metadata["dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return metadata

def train(args):
    args.headless = True
    # args.proj_name = "extreme_parkour_a1_test"
    # args.exptid = "327-22-53"
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid
    os.makedirs(log_pth, exist_ok=True)
    if args.debug:
        mode = "disabled"
        args.rows = 10
        args.cols = 8
        args.num_envs = 64
    else:
        mode = "online"
    
    if args.no_wandb:
        mode = "disabled"
    wandb.init(project=args.proj_name, name=args.exptid, entity="michael_zhang", group=args.exptid[:3], mode=mode, dir="../../logs")
    wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot_config.py", policy="now")
    wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot.py", policy="now")

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args)
    append_run_manifest(
        log_pth,
        {
            "format_version": 1,
            "command": sys.argv,
            "task": args.task,
            "project": args.proj_name,
            "experiment_id": args.exptid,
            "start_iteration": ppo_runner.current_learning_iteration,
            "args": vars(args),
            "git": get_git_metadata(),
            "env_cfg": class_to_dict(env_cfg),
            "train_cfg": class_to_dict(train_cfg),
        },
    )
    try:
        ppo_runner.learn(
            num_learning_iterations=train_cfg.runner.max_iterations,
            init_at_random_ep_len=True,
        )
    finally:
        ppo_runner.metrics_writer.close()
        wandb.finish()

if __name__ == '__main__':
    # Log configs immediately
    args = get_args()
    train(args)
