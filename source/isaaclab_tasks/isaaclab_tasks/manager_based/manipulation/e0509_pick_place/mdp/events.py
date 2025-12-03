# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for the e0509 pick and place task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_gripper_to_open(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset gripper joints to open position (0.0 rad)."""
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Get gripper joint indices (last 4 joints)
    gripper_joint_ids = slice(-4, None)
    
    # Set gripper joints to open position (0.0 rad)
    robot.data.joint_pos[env_ids, gripper_joint_ids] = 0.0
    robot.data.joint_vel[env_ids, gripper_joint_ids] = 0.0
    
    # Write to simulation
    robot.write_joint_state_to_sim(
        robot.data.joint_pos[env_ids],
        robot.data.joint_vel[env_ids],
        env_ids=env_ids,
    )


def randomize_target_object(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """Randomly position medicine_cabinet and select it as the target."""
    # Store target object index in env (only medicine_cabinet for now)
    if not hasattr(env, "episode_target_object_idx"):
        env.episode_target_object_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    # Always select medicine_cabinet (index 0 since it's the only object)
    env.episode_target_object_idx[env_ids] = 0
    
    # Get medicine_cabinet object
    obj: RigidObject = env.scene["medicine_cabinet"]
    
    # Randomize position for reset environments
    num_resets = len(env_ids)
    
    # Get current positions to preserve the environment offset
    current_pos = obj.data.root_pos_w[env_ids].clone()
    
    # Random X offset: 0.29 ~ 0.6 (relative to each environment's origin)
    random_x_offset = torch.rand(num_resets, device=env.device) * (0.6 - 0.29) + 0.29
    
    # Random Y offset: -0.07 ~ 0.37 (relative to each environment's origin)
    random_y_offset = torch.rand(num_resets, device=env.device) * (0.37 - (-0.07)) + (-0.07)
    
    # Z: fixed at table height
    z_position = -0.95
    
    # Calculate environment origins
    # Each environment has its origin at (0, 0) in its local space
    # We need to get the world position offset for each environment
    # Use robot position as reference since it's at the environment origin
    robot = env.scene["robot"]
    robot_base_pos = robot.data.root_pos_w[env_ids, :2].clone()
    
    # Robot is at (0.96, 0.095) relative to each env origin
    # So env origin = robot_pos - (0.96, 0.095)
    env_origins = robot_base_pos.clone()
    env_origins[:, 0] -= 0.96  # Subtract robot's local x offset
    env_origins[:, 1] -= 0.095  # Subtract robot's local y offset
    
    # Create new pose (position + quaternion)
    new_positions = torch.zeros(num_resets, 7, device=env.device)
    new_positions[:, 0] = env_origins[:, 0] + random_x_offset  # x = env_origin_x + offset
    new_positions[:, 1] = env_origins[:, 1] + random_y_offset  # y = env_origin_y + offset
    new_positions[:, 2] = z_position  # z (same for all)
    new_positions[:, 3:] = obj.data.root_quat_w[env_ids]  # keep original orientation
    
    # Apply positions
    obj.write_root_pose_to_sim(new_positions, env_ids=env_ids)
    
    # Store target object position for observations
    if not hasattr(env, "target_object_pos"):
        env.target_object_pos = torch.zeros(env.num_envs, 3, device=env.device)
    
    env.target_object_pos[env_ids, 0] = new_positions[:, 0]
    env.target_object_pos[env_ids, 1] = new_positions[:, 1]
    env.target_object_pos[env_ids, 2] = z_position
