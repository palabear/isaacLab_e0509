# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, matrix_from_quat
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
def object_ee_distance_separate(
    env: ManagerBasedRLEnv,
    std_xy: float,
    std_z: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward agent: first reward z-lift, then reward xy positioning."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    delta = cube_pos_w - ee_pos_w
    dx, dy, dz = delta[:, 0], delta[:, 1], delta[:, 2]
    # XY 평면 Gaussian 보상
    reward_xy = torch.exp(-(dx**2 + dy**2) / (2 * std_xy**2))
    # Z Gaussian 보상 (XY 가까워야 의미 있음)
    reward_z = torch.exp(-(dz**2) / (2 * std_z**2))
    # 최종 보상
    reward = reward_xy * reward_z
    return reward
def spoon_gripper_perpendicular(  # 숟가락과 그리퍼 수직 정도 계산 함수
    env: ManagerBasedRLEnv,  # 환경
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),  # 물체(숟가락) 설정
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")  # 엔드 이펙터(그리퍼) 설정
) -> torch.Tensor:  # 반환 타입: 토치 텐서
    """숟가락의 x축과 그리퍼 close 축(gripper x축)이 수직일 때 보상을 준다.
    두 벡터의 외적을 이용하여 수직 정도를 계산한다.
    - 외적의 크기가 크면 두 축이 수직 (보상 증가)
    - 외적의 크기가 작으면 두 축이 평행 (보상 감소)
    """
    # 환경에서 물체 객체 가져오기
    object: RigidObject = env.scene[object_cfg.name]  # 씬에서 물체 가져오기
    # 환경에서 엔드 이펙터 객체 가져오기
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]  # 씬에서 엔드 이펙터 가져오기
    # 숟가락의 쿼터니언을 회전 행렬로 변환하여 x축 벡터 추출
    spoon_quat = object.data.root_quat_w  # 숟가락 쿼터니언 (환경 수, 4)
    spoon_rot_matrix = matrix_from_quat(spoon_quat)  # 회전 행렬로 변환 (환경 수, 3, 3)
    spoon_x_world = spoon_rot_matrix[:, 0, :]  # 회전 행렬의 첫 번째 열 (x축)
    # 그리퍼의 쿼터니언을 회전 행렬로 변환하여 x축 벡터 추출
    gripper_quat = ee_frame.data.target_quat_w  # 그리퍼 쿼터니언 (환경 수, 타겟 수, 4)
    gripper_quat = gripper_quat[:, 0, :]  # 첫 번째 타겟만 선택 (환경 수, 4)
    gripper_rot_matrix = matrix_from_quat(gripper_quat)  # 회전 행렬로 변환 (환경 수, 3, 3)
    gripper_x_world = gripper_rot_matrix[:, 0, :]  # 회전 행렬의 첫 번째 열 (x축)
    # 숟가락 x축을 정규화
    spoon_x_world = torch.nn.functional.normalize(spoon_x_world, dim=-1)  # 숟가락 벡터 정규화
    # 그리퍼 x축을 정규화
    gripper_x_world = torch.nn.functional.normalize(gripper_x_world, dim=-1)  # 그리퍼 벡터 정규화
    # 두 벡터의 내적 계산 (코사인 유사도)
    dot = (spoon_x_world * gripper_x_world).sum(dim=-1)  # 내적 계산
    # 직교 정도를 보상으로 변환 (수직일수록 보상 증가)
    reward = 1.0 - torch.abs(dot)  # 보상 = 1 - |내적| (0~1, 1이면 직교)
    # 계산된 보상 반환
    return reward  # 보상 반환
def gripper_horizontal(  # 그리퍼 수평성 보상 함수
    env: ManagerBasedRLEnv,  # 환경
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),  # 엔드 이펙터 설정
) -> torch.Tensor:  # 반환 타입: 토치 텐서
    """그리퍼가 수평(z축이 아래를 향함)일 때 보상을 준다.
    그리퍼의 z축이 -z 방향(아래)을 향할수록 보상이 증가한다.
    - z축이 (0, 0, -1) 방향이면: 보상 = 1.0 (완벽한 수평)
    - z축이 (0, 0, 1) 방향이면: 보상 = 0.0 (역방향)
    """
    # 환경에서 엔드 이펙터 객체 가져오기
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]  # 씬에서 엔드 이펙터 가져오기
    # 그리퍼의 z축 방향 벡터 (아래쪽을 향해야 함)
    # 쿼터니언을 회전 행렬로 변환하여 z축 추출
    # 쿼터니언을 회전 행렬로 변환
    gripper_quat = ee_frame.data.target_quat_w  # 그리퍼 쿼터니언 (환경 수, 타겟 수, 4)
    gripper_quat = gripper_quat[:, 0, :]  # 첫 번째 타겟만 선택 (환경 수, 4)
    ee_rot_matrix = matrix_from_quat(gripper_quat)  # 회전 행렬로 변환 (환경 수, 3, 3)
    # 회전 행렬의 두 번째 열을 y축으로 추출 (90도 회전)
    gripper_y_axis = ee_rot_matrix[:, 1, :]  # 회전 행렬의 y축 벡터 추출
    # 목표 방향 설정: (0, 0, -1) - 아래쪽을 향함
    target_axis = torch.tensor([0.0, 0.0, -1.0], device=gripper_y_axis.device)  # 목표 벡터 정의
    # 그리퍼 y축을 정규화 (이미 단위 벡터이지만 안전하게)
    gripper_y_axis = torch.nn.functional.normalize(gripper_y_axis, dim=-1)  # 벡터 정규화
    # 두 벡터의 내적 계산 (-1 ~ 1 범위)
    # 내적이 1에 가까우면 같은 방향 (옆으로 누운 자세)
    # 내적이 -1에 가까우면 반대 방향
    dot_product = (gripper_y_axis * target_axis).sum(dim=-1)  # 내적 계산
    # 내적을 보상으로 변환: -1 ~ 1을 0 ~ 1 범위로
    # dot = 1 (완벽한 수평) → reward = 1.0
    # dot = 0 (수직) → reward = 0.0
    # dot = -1 (역방향) → reward = 0.0
    reward = torch.clamp(dot_product, min=0.0)  # 내적을 0 이상으로 클램핑하여 보상 계산
    # 계산된 보상 반환
    return reward  # 보상 반환


def approach_velocity_penalty(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.15,  # 15cm 이내에서 속도 체크
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """물체에 가까워질수록 빠르게 움직이면 패널티를 준다.
    충돌을 방지하기 위해 접근 시 속도를 줄이도록 유도한다.
    
    Args:
        env: 환경
        distance_threshold: 패널티를 시작할 거리 (미터)
        object_cfg: 물체 설정
        ee_frame_cfg: 엔드 이펙터 설정
        robot_cfg: 로봇 설정
    
    Returns:
        패널티 값 (가까울수록, 빠를수록 큰 음수)
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    
    # 물체와 엔드 이펙터 사이의 거리
    object_pos = object.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(object_pos - ee_pos, dim=-1)
    
    # 엔드 이펙터의 속도 (joint velocity 기반 근사)
    # 실제로는 ee frame의 linear velocity를 사용해야 하지만, 
    # joint velocity의 L2 norm으로 근사
    joint_vel = robot.data.joint_vel[:, :6]  # 첫 6개 조인트 (arm)
    velocity_magnitude = torch.norm(joint_vel, dim=-1)
    
    # 거리가 threshold 이내일 때만 패널티 적용
    # 거리가 가까울수록 패널티 증가 (거리 역수 개념)
    distance_factor = torch.clamp(1.0 - distance / distance_threshold, min=0.0)
    
    # 패널티 = 거리 factor * 속도 크기
    # 가까이 있을 때 빠르게 움직이면 큰 패널티
    penalty = distance_factor * velocity_magnitude
    
    return -penalty  # 음수 패널티로 반환