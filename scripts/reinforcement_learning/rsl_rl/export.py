"""
Script to export a trained policy to ONNX or JIT.

This script loads a trained policy and exports it to a specific format.
The exported model is saved in the same directory as the checkpoint.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Export a trained policy to ONNX or JIT.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--resume", type=str, default=None, help="Path to the checkpoint.")
parser.add_argument("--onnx", action="store_true", default=False, help="Export to ONNX.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""
Rest of the code.
"""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, export_policy_as_jit, export_policy_as_onnx
from rsl_rl.runners import OnPolicyRunner


def main():
    """Export a trained policy to ONNX or JIT."""
    # parse configuration
    env_cfg = isaaclab_tasks.utils.parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
    agent_cfg: RslRlOnPolicyRunnerCfg = isaaclab_tasks.utils.load_cfg_from_registry(
        args_cli.task, "rsl_rl_cfg_entry_point"
    )

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # create agent
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    
    # load checkpoint
    resume_path = args_cli.resume
    if not os.path.exists(resume_path):
        print(f"[Error] Checkpoint not found: {resume_path}")
        sys.exit(1)
        
    print(f"[INFO] Loading checkpoint from: {resume_path}")
    runner.load(resume_path)

    # create directory for exported models
    export_dir = os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(export_dir, exist_ok=True)

    # export policy to JIT
    print(f"[INFO] Exporting policy to JIT in: {export_dir}")
    export_policy_as_jit(
        runner.alg.actor_critic, 
        runner.obs_normalizer, 
        path=os.path.join(export_dir, "policy.pt")
    )

    # export policy to ONNX
    if args_cli.onnx:
        print(f"[INFO] Exporting policy to ONNX in: {export_dir}")
        export_policy_as_onnx(
            runner.alg.actor_critic,
            runner.obs_normalizer,
            path=os.path.join(export_dir, "policy.onnx")
        )

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()