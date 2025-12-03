# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for the e0509 pick and place task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def end_effector_collision_with_ground(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.02,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Terminate if end-effector (gripper) touches or goes below ground level.
    
    Args:
        env: The environment.
        minimum_height: Minimum allowed height above ground (in meters). Default is 2cm.
        ee_frame_cfg: The scene entity configuration for the end-effector frame.
    
    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Get end-effector z position (height)
    ee_height = ee_frame.data.target_pos_w[..., 0, 2]
    
    # Ground is at z = -1.05 (from scene config)
    ground_level = -1.05
    
    # Terminate if gripper is too close to or below ground
    return ee_height < (ground_level + minimum_height)


def robot_body_collision_with_ground(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate if any robot link goes too close to ground level.
    
    This checks if any part of the robot body (links) is touching the ground,
    which indicates an unsafe configuration.
    
    Args:
        env: The environment.
        minimum_height: Minimum allowed height for robot links above ground (in meters).
        robot_cfg: The scene entity configuration for the robot.
    
    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Get positions of all robot links including gripper parts
    body_positions = robot.data.body_pos_w
    
    # Ground is at z = -1.05
    ground_level = -1.05
    
    # Get minimum z-position across ALL links (including gripper links) for each environment
    min_link_height = torch.min(body_positions[:, :, 2], dim=1)[0]
    
    # Terminate if any link is too close to ground
    # Use stricter threshold (0.10m = 10cm) to catch gripper collisions early
    collision_threshold = max(minimum_height, 0.10)
    return min_link_height < (ground_level + collision_threshold)


def unsafe_robot_configuration(
    env: ManagerBasedRLEnv,
    ee_min_height: float = 0.02,
    body_min_height: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Combined termination for any unsafe robot configuration.
    
    Terminates if:
    - End-effector touches ground
    - Robot body touches ground
    
    Args:
        env: The environment.
        ee_min_height: Minimum height for end-effector above ground.
        body_min_height: Minimum height for robot body above ground.
        robot_cfg: The scene entity configuration for the robot.
        ee_frame_cfg: The scene entity configuration for the end-effector frame.
    
    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    # Check end-effector collision
    ee_collision = end_effector_collision_with_ground(env, ee_min_height, ee_frame_cfg)
    
    # Check robot body collision
    body_collision = robot_body_collision_with_ground(env, body_min_height, robot_cfg)
    
    # Terminate if either condition is met
    return torch.logical_or(ee_collision, body_collision)


def gripper_table_collision(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.03,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Terminate if gripper goes below table surface (simple height check).
    
    Args:
        env: The environment.
        minimum_height: Minimum allowed height above table surface (in meters). Default is 3cm.
        ee_frame_cfg: The scene entity configuration for the end-effector frame.
    
    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    # Get end-effector z position
    ee_height = ee_frame.data.target_pos_w[..., 0, 2]
    
    # Table surface is at approximately z = -0.95
    table_surface_z = -0.95
    
    # Terminate if gripper goes below table surface + small margin
    return ee_height < (table_surface_z + minimum_height)


def object_successfully_grasped(
    env: ManagerBasedRLEnv,
    height_threshold: float = 0.10,  # Lowered from 0.15 - easier success
    hold_duration: int = 10,
    object_cfg: SceneEntityCfg = SceneEntityCfg("medicine_cabinet"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate successfully when object is grasped and lifted slightly.
    
    Relaxed criteria for easier success detection.
    """
    object: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Check if object is lifted
    object_height = object.data.root_pos_w[:, 2]
    table_height = 0.0
    is_lifted = object_height > (table_height + height_threshold)
    
    # Check if gripper is closed (grasping)
    gripper_pos = robot.data.joint_pos[:, -4]
    is_grasping = gripper_pos > 0.5  # Gripper more than half closed
    
    # Basic success: lifted + grasping
    success_condition = is_lifted & is_grasping
    
    # For stability tracking, we could add a counter in env extras
    # For now, return basic success condition
    # (holding duration would require state tracking in environment)
    
    return success_condition


def object_collision_termination(
    env: ManagerBasedRLEnv,
    min_height: float = -0.05,
) -> torch.Tensor:
    """Terminate if any object collides with ground or falls."""
    object_names = ["medicine_cabinet"]
    
    termination = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    for obj_name in object_names:
        obj: RigidObject = env.scene[obj_name]
        obj_height = obj.data.root_pos_w[:, 2]
        
        # Terminate if object fell below minimum height (collision with ground/table)
        fell = obj_height < min_height
        termination = termination | fell
    
    return termination


def any_object_collision_termination(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.08,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate if robot (any part) touches ANY object."""
    robot: Articulation = env.scene[robot_cfg.name]
    object_names = ["medicine_cabinet"]
    
    termination = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Get all robot body positions (including all links and gripper)
    robot_body_pos = robot.data.body_pos_w  # (num_envs, num_bodies, 3)
    
    for obj_name in object_names:
        obj: RigidObject = env.scene[obj_name]
        obj_pos = obj.data.root_pos_w  # (num_envs, 3)
        
        # Calculate distance from each robot body part to object
        # Expand obj_pos to match robot_body_pos dimensions
        obj_pos_expanded = obj_pos.unsqueeze(1)  # (num_envs, 1, 3)
        
        # Distance from each body part to object
        distances = torch.norm(robot_body_pos - obj_pos_expanded, dim=2)  # (num_envs, num_bodies)
        
        # Check if ANY body part is too close
        min_distance = torch.min(distances, dim=1)[0]  # (num_envs,)
        is_collision = min_distance < distance_threshold
        
        termination = termination | is_collision
    
    return termination



def object_disturbed_termination(
    env: ManagerBasedRLEnv,
    position_threshold: float = 0.15,
    height_threshold: float = 0.03,
) -> torch.Tensor:
    """Terminate if ANY object is knocked over or moved significantly from initial position."""
    object_names = ["medicine_cabinet"]
    
    # Initial positions (set in scene config)
    initial_positions = {
        "medicine_cabinet": torch.tensor([0.5, -0.2, -0.95], device=env.device),
    }
    
    termination = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    for obj_name in object_names:
        obj: RigidObject = env.scene[obj_name]
        current_pos = obj.data.root_pos_w
        initial_pos = initial_positions[obj_name]
        
        # Check horizontal displacement (XY plane)
        horizontal_displacement = torch.norm(current_pos[:, :2] - initial_pos[:2], dim=1)
        moved_too_much = horizontal_displacement > position_threshold
        
        # Check if object fell (Z position too low)
        fell_down = current_pos[:, 2] < (initial_pos[2] - height_threshold)
        
        # Terminate if object moved too much or fell
        termination = termination | moved_too_much | fell_down
    
    return termination


def object_out_of_bounds(
    env: ManagerBasedRLEnv,
    x_limit: float = 1.0,
    y_limit: float = 1.0,
    z_min: float = -1.2,
    z_max: float = 0.5,
    object_cfg: SceneEntityCfg = SceneEntityCfg("medicine_cabinet"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate if object goes out of bounds relative to robot base."""
    object: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Get positions
    object_pos_w = object.data.root_pos_w[:, :3]
    robot_base_pos_w = robot.data.root_pos_w[:, :3]
    
    # Calculate relative position (object - robot base)
    relative_pos = object_pos_w - robot_base_pos_w
    
    # Check if object is outside allowed region (relative to robot)
    out_x = torch.abs(relative_pos[:, 0]) > x_limit
    out_y = torch.abs(relative_pos[:, 1]) > y_limit
    out_z_low = object_pos_w[:, 2] < z_min  # Absolute Z for floor check
    out_z_high = object_pos_w[:, 2] > z_max  # Absolute Z for ceiling check
    
    out_of_bounds = torch.logical_or(
        torch.logical_or(out_x, out_y),
        torch.logical_or(out_z_low, out_z_high)
    )
    
    return out_of_bounds


def arm_object_collision(
    env: ManagerBasedRLEnv,
    force_threshold: float = 1.0,
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """로봇팔-물체 충돌 감지 (그리퍼 제외)
    
    ContactSensor를 사용하여 로봇 링크와 물체 사이의 접촉력을 감지합니다.
    일정 임계값 이상의 힘이 감지되면 충돌로 판정합니다.
    
    Args:
        env: The environment.
        force_threshold: 충돌 판정 임계값 (Newton). 기본 1.0N
        contact_sensor_cfg: ContactSensor configuration
        
    Returns:
        충돌 발생한 환경들의 boolean tensor
    """
    contact_sensor: ContactSensor = env.scene[contact_sensor_cfg.name]
    
    # Get contact forces - net_forces_w는 (num_envs, num_bodies, 3) shape
    net_forces = contact_sensor.data.net_forces_w
    
    # 각 body의 접촉력 크기 계산
    force_magnitudes = torch.norm(net_forces, dim=-1)  # (num_envs, num_bodies)
    
    # 환경별 최대 접촉력
    max_force_per_env = torch.max(force_magnitudes, dim=1)[0]  # (num_envs,)
    
    # 임계값 초과 시 충돌로 판정
    collision = max_force_per_env > force_threshold
    
    return collision


