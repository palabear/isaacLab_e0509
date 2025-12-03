# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for the e0509 pick and place task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("medicine_cabinet"),
) -> torch.Tensor:
    """The position of the TARGET object in the robot's root frame (dynamically selected)."""
    robot: RigidObject = env.scene[robot_cfg.name]
    
    # Use stored target object position from randomize_target_object event
    if hasattr(env, "target_object_pos"):
        object_pos_w = env.target_object_pos
    else:
        # Fallback to medicine_cabinet if not initialized
        object: RigidObject = env.scene["medicine_cabinet"]
        object_pos_w = object.data.root_pos_w[:, :3]
    
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b


def goal_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_marker"),
) -> torch.Tensor:
    """The position of the goal marker in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    goal: RigidObject = env.scene[goal_cfg.name]
    goal_pos_w = goal.data.root_pos_w[:, :3]
    goal_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, goal_pos_w)
    return goal_pos_b
