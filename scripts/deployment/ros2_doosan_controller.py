#!/usr/bin/env python3
"""
ROS 2 Node for Doosan Robot Control using Isaac Lab trained JIT policy

Observation structure (29 dims) matching Isaac Lab training:
- joint_pos_rel (10): 6 (robot arm) + 4 (gripper joints) relative to default
- joint_vel_rel (10): 6 (robot arm) + 4 (gripper joints) velocities
- object_position (3): Object position in robot root frame
- last_action (6): Previous robot arm action (gripper action removed)
Total: 10 + 10 + 3 + 6 = 29 dimensions
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped
import torch
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation


class DoosanIsaacLabController(Node):
    def __init__(self):
        super().__init__('doosan_isaaclab_controller')
        
        # =====================================================================
        # 1. JIT 모델 로드
        # =====================================================================
        model_path = '/home/jiwoo/IsaacLab/logs/rsl_rl/e0509_pick_place/2025-12-05_09-11-46/exported/policy.pt'
        self.model = torch.jit.load(model_path)
        self.model.eval()
        
        self.get_logger().info(f'✅ JIT model loaded from: {model_path}')
        
        # =====================================================================
        # 2. 로봇 설정 (E0509 6-DOF 로봇 + 그리퍼 4개 관절)
        # =====================================================================
        # Isaac Lab에서 학습한 관절 순서: 로봇 팔(6) + 그리퍼(4) = 10개
        self.robot_joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        self.gripper_joint_names = ['rh_l1_joint', 'rh_l2_joint', 'rh_r1_joint', 'rh_r2_joint']
        self.all_joint_names = self.robot_joint_names + self.gripper_joint_names
        self.num_robot_joints = len(self.robot_joint_names)  # 6
        self.num_gripper_joints = len(self.gripper_joint_names)  # 4
        self.num_total_joints = len(self.all_joint_names)  # 10
        
        # Default joint positions (Isaac Lab의 use_default_offset=True와 동일)
        # 로봇 팔: E0509의 기본 자세
        # 그리퍼: 0.0 (닫힌 상태)
        self.default_joint_pos = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 로봇 팔 6개
            0.0, 0.0, 0.0, 0.0              # 그리퍼 4개
        ], dtype=np.float32)
        
        # 두산 로봇 관절 한계 (라디안, 안전 범위) - 로봇 팔만
        # 실제 로봇 사양에 맞게 수정 필요!
        self.joint_lower_limits = np.array([-6.28, -6.28, -2.61, -6.28, -6.28, -6.28])
        self.joint_upper_limits = np.array([ 6.28,  6.28,  2.61,  6.28,  6.28,  6.28])
        
        # Action scaling factor (Isaac Lab training과 동일: 0.1)
        self.action_scale = 0.1
        
        # 속도 제한 (rad/s) - 안전을 위해 낮게 설정
        self.max_joint_velocity = 0.5  # rad/s
        
        # Robot base frame (물체 위치 변환용)
        # Isaac Lab 학습 환경과 동일한 로봇 위치
        self.robot_base_pos = np.array([0.96, 0.095, -0.95])
        self.robot_base_quat = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        
        # =====================================================================
        # 3. 관찰(Observation) 버퍼
        # =====================================================================
        # 전체 관절: 로봇 팔(6) + 그리퍼(4) = 10개
        self.current_joint_pos = np.zeros(self.num_total_joints, dtype=np.float32)
        self.current_joint_vel = np.zeros(self.num_total_joints, dtype=np.float32)
        # 액션은 로봇 팔만 (그리퍼 액션 삭제됨)
        self.previous_action = np.zeros(self.num_robot_joints, dtype=np.float32)
        
        # 물체 위치 (world frame, 실제로는 vision system에서 업데이트)
        # Default: medicine_cabinet initial position from training
        self.object_position_world = np.array([0.5, -0.2, -0.95], dtype=np.float32)
        
        # 관찰 이력 (필요시 사용)
        self.obs_history = deque(maxlen=10)
        
        # =====================================================================
        # 4. ROS 2 Publishers & Subscribers
        # =====================================================================
        
        # Subscriber: 로봇의 현재 관절 상태 수신
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',  # 두산 로봇의 joint_states 토픽
            self.joint_state_callback,
            10
        )
        
        # Subscriber: 물체 위치 수신 (vision system)
        self.object_detection_sub = self.create_subscription(
            PoseStamped,
            '/object_detection/pose',
            self.object_detection_callback,
            10
        )
        
        # Publisher: ros2_control Forward Command Controller
        # 위치 제어 명령 전송
        self.cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/forward_position_controller/commands',
            10
        )
        
        # =====================================================================
        # 5. 제어 타이머 (Isaac Lab 기준: 120Hz 물리, decimation=4 → 30Hz)
        # =====================================================================
        control_freq = 30  # Hz (Isaac Lab action frequency)
        self.control_timer = self.create_timer(
            1.0 / 30.0,
            self.control_loop
        )
        
        self.get_logger().info(f'🤖 Doosan Isaac Lab Controller started at {control_freq} Hz')
        self.get_logger().warn('⚠️  Make sure robot is in REMOTE/ROS MODE before running!')
        
        self.initialized = False
    
    
    def joint_state_callback(self, msg: JointState):
        """로봇의 현재 관절 상태 수신 (로봇 팔 6개 + 그리퍼 4개)"""
        try:
            # 첫 수신 시 joint 이름 확인
            if not self.initialized:
                self.get_logger().info(f'Received joint names: {msg.name}')
                self.get_logger().info(f'Expected joint names: {self.all_joint_names}')
            
            # 로봇 팔 관절 (6개)
            found_robot_joints = 0
            for i, joint_name in enumerate(self.robot_joint_names):
                if joint_name in msg.name:
                    idx = msg.name.index(joint_name)
                    self.current_joint_pos[i] = msg.position[idx]
                    if len(msg.velocity) > idx:
                        self.current_joint_vel[i] = msg.velocity[idx]
                    found_robot_joints += 1
                else:
                    self.get_logger().warn(f'Joint {joint_name} not found in message!')
            
            # 그리퍼 관절 (4개) - 없으면 0으로 유지
            found_gripper_joints = 0
            for i, joint_name in enumerate(self.gripper_joint_names):
                if joint_name in msg.name:
                    idx = msg.name.index(joint_name)
                    self.current_joint_pos[self.num_robot_joints + i] = msg.position[idx]
                    if len(msg.velocity) > idx:
                        self.current_joint_vel[self.num_robot_joints + i] = msg.velocity[idx]
                    found_gripper_joints += 1
                # 그리퍼 관절이 없는 경우 0으로 유지 (닫힌 상태로 가정)
            
            # 관절 개수 체크 (로봇 팔은 필수, 그리퍼는 선택)
            if found_robot_joints != self.num_robot_joints and not self.initialized:
                self.get_logger().error(f'Only found {found_robot_joints}/{self.num_robot_joints} robot joints!')
            
            if not self.initialized:
                self.initialized = True
                self.get_logger().info(f'✅ Initial joint state received:')
                self.get_logger().info(f'   Robot joints: {found_robot_joints}/{self.num_robot_joints}')
                self.get_logger().info(f'   Gripper joints: {found_gripper_joints}/{self.num_gripper_joints}')
                self.get_logger().info(f'   Current joint pos: {self.current_joint_pos}')
                self.get_logger().info(f'   Default joint pos: {self.default_joint_pos}')
                self.get_logger().info(f'   Object position (world): {self.object_position_world}')
                self.get_logger().info(f'   Robot base pos: {self.robot_base_pos}')
        
        except Exception as e:
            self.get_logger().error(f'Error in joint_state_callback: {e}')
    
    
    def object_detection_callback(self, msg: PoseStamped):
        """
        물체 감지 결과 수신
        World frame에서의 물체 위치를 저장
        
        Args:
            msg: geometry_msgs/PoseStamped (world frame의 물체 위치)
        """
        # World frame에서의 물체 위치 저장
        self.object_position_world = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
    
    
    def world_to_robot_frame(self, pos_world):
        """
        World frame 좌표를 robot root frame으로 변환
        Isaac Lab의 subtract_frame_transforms와 동일한 변환
        
        Args:
            pos_world: World frame에서의 위치 [x, y, z]
            
        Returns:
            pos_robot: Robot root frame에서의 위치 [x, y, z]
        """
        # Robot base의 역변환 적용
        # pos_robot = R^T * (pos_world - robot_base_pos)
        
        # Quaternion을 rotation matrix로 변환
        rot = Rotation.from_quat([
            self.robot_base_quat[1],  # x
            self.robot_base_quat[2],  # y  
            self.robot_base_quat[3],  # z
            self.robot_base_quat[0],  # w
        ])
        rot_matrix = rot.as_matrix()
        
        # 상대 위치 계산
        pos_relative = pos_world - self.robot_base_pos
        
        # Robot frame으로 회전
        pos_robot = rot_matrix.T @ pos_relative
        
        return pos_robot
    
    
    def build_observation(self):
        """
        Isaac Lab 학습 시와 동일한 observation 구성
        
        실제 모델 입력: 29차원
        ObservationsCfg.PolicyCfg:
        - joint_pos_rel (10): 6 (robot arm) + 4 (gripper) relative to default
        - joint_vel_rel (10): 6 (robot arm) + 4 (gripper) velocities
        - object_position (3): Object position in robot root frame
        - last_action (6): Previous robot arm action (gripper action removed)
        Total: 10 + 10 + 3 + 6 = 29
        
        Returns:
            obs: numpy array of shape (29,)
        """
        # 1. Joint positions relative to default (mdp.joint_pos_rel)
        # 전체 10개: 로봇 팔 6개 + 그리퍼 4개
        joint_pos_rel = self.current_joint_pos - self.default_joint_pos  # (10,)
        
        # 2. Joint velocities (mdp.joint_vel_rel)
        # 전체 10개: 로봇 팔 6개 + 그리퍼 4개
        joint_vel_rel = self.current_joint_vel  # (10,)
        
        # 3. Object position in robot root frame
        # World frame → Robot frame 변환 (e0509_mdp.object_position_in_robot_root_frame)
        object_pos_robot = self.world_to_robot_frame(self.object_position_world)  # (3,)
        
        # 4. Previous action (mdp.last_action)
        # 로봇 팔 액션만 (6개) - 그리퍼 액션은 삭제됨
        last_action = self.previous_action.copy()  # (6,)
        
        # Isaac Lab과 동일한 순서로 concatenate
        obs = np.concatenate([
            joint_pos_rel,      # 10
            joint_vel_rel,      # 10
            object_pos_robot,   # 3
            last_action,        # 6
        ]).astype(np.float32)  # Total: 29 dimensions
        
        return obs
    
    
    def control_loop(self):
        """메인 제어 루프 - JIT 모델 추론 및 명령 발행"""
        
        if not self.initialized:
            return
        
        try:
            # ================================================================
            # 1. Observation 구성 (Isaac Lab과 동일: 29 dimensions)
            # ================================================================
            obs = self.build_observation()
            
            # NaN 체크
            if np.isnan(obs).any():
                self.get_logger().error(f'❌ NaN detected in observation! {obs}')
                return
            
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # [1, 29]
            
            # ================================================================
            # 2. JIT 모델 추론
            # ================================================================
            with torch.no_grad():
                raw_action = self.model(obs_tensor).squeeze().cpu().numpy()
            
            # 출력값 체크 및 클리핑
            if np.any(np.abs(raw_action) > 1000):
                self.get_logger().warn(f'⚠️  Large action detected: {raw_action}, clipping...')
                raw_action = np.clip(raw_action, -10.0, 10.0)
            
            # NaN 체크
            if np.isnan(raw_action).any():
                self.get_logger().error(f'❌ NaN detected in action!')
                self.get_logger().error(f'   Observation shape: {obs.shape}')
                self.get_logger().error(f'   Observation values: {obs}')
                self.get_logger().error(f'   Observation min/max: {obs.min():.3f} / {obs.max():.3f}')
                self.get_logger().error(f'   Has NaN in obs: {np.isnan(obs).any()}')
                self.get_logger().error(f'   Has Inf in obs: {np.isinf(obs).any()}')
                # 긴급: previous_action 초기화
                self.previous_action = np.zeros(self.num_robot_joints, dtype=np.float32)
                return
            
            # 출력 차원 확인 (로봇 팔 액션만 6개)
            if len(raw_action) != self.num_robot_joints:
                self.get_logger().error(
                    f'❌ Action dimension mismatch! Expected {self.num_robot_joints}, got {len(raw_action)}'
                )
                return
            
            # ================================================================
            # 3. Action Scaling & Delta Control
            # ================================================================
            # Isaac Lab의 JointPositionActionCfg와 동일하게 적용
            # action_scale을 곱한 후 현재 위치에 더함 (Delta control)
            
            scaled_action = raw_action * self.action_scale
            
            # 목표 관절 위치 = 현재 위치 + 스케일된 액션 (로봇 팔만)
            target_joint_pos = self.current_joint_pos[:self.num_robot_joints] + scaled_action
            
            # ================================================================
            # 4. Safety Clipping (관절 한계 및 속도 제한)
            # ================================================================
            # 관절 위치 한계
            target_joint_pos = np.clip(
                target_joint_pos,
                self.joint_lower_limits,
                self.joint_upper_limits
            )
            
            # 속도 제한 (갑작스러운 움직임 방지) - 로봇 팔만
            dt = 1.0 / 30.0  # control frequency
            max_pos_change = self.max_joint_velocity * dt
            position_delta = target_joint_pos - self.current_joint_pos[:self.num_robot_joints]
            position_delta = np.clip(position_delta, -max_pos_change, max_pos_change)
            target_joint_pos = self.current_joint_pos[:self.num_robot_joints] + position_delta
            
            # ================================================================
            # 5. ROS 2 명령 발행
            # ================================================================
            cmd_msg = Float64MultiArray()
            cmd_msg.data = target_joint_pos.tolist()
            self.cmd_pub.publish(cmd_msg)
            
            # ================================================================
            # 6. 이전 액션 저장 (다음 observation용)
            # ================================================================
            # 안전하게 클리핑 (-10 ~ 10 범위)
            self.previous_action = np.clip(raw_action, -10.0, 10.0).astype(np.float32)
            
            # 주기적 로그 (디버깅용)
            if self.get_clock().now().nanoseconds % 1_000_000_000 < 33_000_000:  # ~1초마다
                obj_pos_robot = self.world_to_robot_frame(self.object_position_world)
                self.get_logger().info(
                    f'Obs: [{obs[0]:.2f}, {obs[1]:.2f}, ...] | '
                    f'Object(robot): [{obj_pos_robot[0]:.2f}, {obj_pos_robot[1]:.2f}, {obj_pos_robot[2]:.2f}] | '
                    f'Action: [{raw_action[0]:.2f}, {raw_action[1]:.2f}, ...]'
                )
        
        except Exception as e:
            self.get_logger().error(f'❌ Error in control loop: {e}')


def main(args=None):
    rclpy.init(args=args)
    
    node = DoosanIsaacLabController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Shutting down Doosan Isaac Lab Controller')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
