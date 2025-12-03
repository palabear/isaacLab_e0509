# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the e0509 pick and place task.

[개선된 철학: Soft Guidance]
1. Approach: 곱하기(*) 대신 더하기(+)를 사용하여, 자세가 나빠도 일단 접근하면 보상을 줌.
2. Target: 물체 중심(0.0)이 아니라 공중(Pre-grasp)을 목표로 설정.
3. Grasp: 가까우면 닫고, 멀면 여는 전략 유도.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def approach_pregrasp_pose(
    env: ManagerBasedRLEnv,
    pregrasp_height: float = 0.15,
    object_cfg: SceneEntityCfg = SceneEntityCfg("medicine_cabinet"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    거리 보상: 물체 위 'Pre-grasp' 지점까지의 거리
    반환: exp(-distance) [0~1]
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    # 1. 목표 지점 계산 (물체 위치 + Z축 오프셋)
    object_pos = object.data.root_pos_w[:, :3].clone()
    object_pos[:, 2] += pregrasp_height  # 물체 머리 위로 목표 설정
    
    # 2. EE 위치
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    
    # 3. 거리 계산
    distance = torch.norm(ee_pos - object_pos, dim=1)
    
    # 4. 보상 변환 (Log-scale과 유사한 깔때기)
    # 가까울수록 점수가 급격히 오름
    reward = torch.exp(-distance * 3.0) 
    
    return torch.clamp(reward, 0.0, 1.0)


def downward_orientation_score(
    env: ManagerBasedRLEnv,
    strictness: float = 2.0,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    자세 보상: 그리퍼가 수직 아래를 향하는지 평가
    반환: [0~1] (1.0 = 완벽한 수직)
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]
    
    # Z축 벡터 추출
    x, y = ee_quat[:, 1], ee_quat[:, 2]
    z_axis_z = 1.0 - 2.0 * (x * x + y * y)
    
    # 점수 변환 (-1:수직 -> 1.0점, 0:수평 -> 0.5점)
    alignment = torch.clamp((-z_axis_z + 1.0) / 2.0, 0.0, 1.0)
    
    # Strictness 적용 (높을수록 엄격함)
    return torch.pow(alignment, strictness)


def approach_and_orient_reward(
    env: ManagerBasedRLEnv,
    pregrasp_height: float = 0.10,
    orientation_strictness: float = 2.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("medicine_cabinet"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    [핵심 수정] 거리 보상과 자세 보상의 '가중 합'
    곱하기(*)를 쓰면 초기에 보상이 0이 되어 학습이 안 됨.
    더하기(+)를 써서, 자세가 나빠도 접근만 하면 부분 점수를 줌.
    """
    dist_rew = approach_pregrasp_pose(
        env, pregrasp_height, object_cfg, ee_frame_cfg
    )
    
    orient_rew = downward_orientation_score(
        env, orientation_strictness, ee_frame_cfg
    )
    
    # 거리 60% + 자세 40% 비중
    # 로봇이 일단 가까이 가는 법을 먼저 배우게 됨
    return 0.6 * dist_rew + 0.4 * orient_rew


def gripper_close_encourage(
    env: ManagerBasedRLEnv,
    switch_dist: float = 0.04,
    object_cfg: SceneEntityCfg = SceneEntityCfg("medicine_cabinet"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    그리퍼 전략: 4cm 밖에서는 열고(Open), 4cm 안에서는 닫아라(Close)
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    
    # 거리 계산
    object_pos = object.data.root_pos_w[:, :3]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(ee_pos - object_pos, dim=1)
    
    # 그리퍼 값 정규화 (0.0 ~ 1.0)
    gripper_pos = robot.data.joint_pos[:, -4:]
    gripper_val = torch.mean(gripper_pos, dim=1) / 0.8 # 대략적인 max값으로 나눔
    gripper_val = torch.clamp(gripper_val, 0.0, 1.0)
    
    is_close = distance < switch_dist
    
    # 가까우면 닫을수록 이득, 멀면 열수록 이득
    close_reward = gripper_val
    open_reward = 1.0 - gripper_val
    
    reward = torch.where(is_close, close_reward, open_reward)
    return torch.nan_to_num(reward, 0.0)


def lift_success_bonus(
    env: ManagerBasedRLEnv,
    lift_height: float = 0.05,
    gripper_threshold: float = 0.5,
    object_cfg: SceneEntityCfg = SceneEntityCfg("medicine_cabinet"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    최종 성공 보상: 들어올리면 큰 점수
    """
    object: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    
    current_height = object.data.root_pos_w[:, 2]
    
    # 초기 높이 저장 및 리셋 처리
    if not hasattr(env, '_object_initial_height'):
        env._object_initial_height = current_height.clone()
        env._last_episode_buf = env.episode_length_buf.clone()
    else:
        reset_mask = env.episode_length_buf < env._last_episode_buf
        if reset_mask.any():
            env._object_initial_height[reset_mask] = current_height[reset_mask]
        env._last_episode_buf = env.episode_length_buf.clone()
    
    # 성공 조건: 높이 상승 + 그리퍼 닫힘
    height_diff = current_height - env._object_initial_height
    is_lifted = height_diff > lift_height
    
    gripper_pos = robot.data.joint_pos[:, -4:]
    is_closed = torch.all(gripper_pos > gripper_threshold, dim=1)
    
    success = torch.logical_and(is_lifted, is_closed)
    return success.float()


def object_stability_penalty(
    env: ManagerBasedRLEnv,
    velocity_threshold: float = 0.15,
    safe_distance: float = 0.10,
    object_cfg: SceneEntityCfg = SceneEntityCfg("medicine_cabinet"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    페널티: 멀리 있는데 물체가 움직이면(쳐서 날리면) 감점
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    object_vel = torch.norm(object.data.root_lin_vel_w[:, :3], dim=1)
    is_moving_fast = object_vel > velocity_threshold
    
    object_pos = object.data.root_pos_w[:, :3]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(ee_pos - object_pos, dim=1)
    is_far = distance > safe_distance
    
    invalid = torch.logical_and(is_far, is_moving_fast)
    return invalid.float()