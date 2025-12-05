#!/usr/bin/env python3
"""
Mock Joint State Publisher for testing without real robot
Subscribes to commands and publishes updated joint states
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np


class MockJointStatePublisher(Node):
    def __init__(self):
        super().__init__('mock_joint_state_publisher')
        
        # Publisher: Joint states
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        
        # Subscriber: Commands from controller
        self.cmd_sub = self.create_subscription(
            Float64MultiArray,
            '/forward_position_controller/commands',
            self.command_callback,
            10
        )
        
        # 30Hz로 발행 (컨트롤러와 동일)
        self.timer = self.create_timer(1.0 / 30.0, self.publish_joint_states)
        
        # Joint names: 로봇 팔(6) + 그리퍼(4)
        self.robot_joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        self.gripper_joint_names = ['rh_l1_joint', 'rh_l2_joint', 'rh_r1_joint', 'rh_r2_joint']
        self.all_joint_names = self.robot_joint_names + self.gripper_joint_names
        
        # Current joint positions: 로봇 팔(6) + 그리퍼(4) = 10개
        # 학습 데이터 평균값 근처로 초기화 (normalization 범위 내)
        # 학습 시 평균: [-0.09, -0.79, -1.22, -0.15, -1.17, 0.45]
        self.joint_positions = np.array([0.0, -0.8, -1.2, 0.0, -1.2, 0.5,  # robot arm
                                         0.0, 0.0, 0.0, 0.0],                # gripper (closed)
                                        dtype=np.float32)
        self.joint_velocities = np.zeros(10, dtype=np.float32)
        
        # Target positions from controller (로봇 팔만 6개 받음)
        self.target_positions = self.joint_positions[:6].copy()
        
        # Simple position control gain
        self.position_gain = 0.3  # 30% of error per timestep
        
        self.get_logger().info('🤖 Mock Joint State Publisher started (10 joints: 6 robot + 4 gripper)')
        self.get_logger().info('   - Subscribing to: /forward_position_controller/commands')
        self.get_logger().info('   - Publishing to: /joint_states')
        self.counter = 0
    
    def command_callback(self, msg):
        """Receive target positions from controller (로봇 팔 6개만)"""
        if len(msg.data) == 6:
            new_targets = np.array(msg.data, dtype=np.float32)
            # NaN 체크
            if not np.isnan(new_targets).any():
                self.target_positions = new_targets
    
    def publish_joint_states(self):
        # Simple position control: 로봇 팔 6개만 움직임 (그리퍼는 0으로 고정)
        if not np.isnan(self.target_positions).any() and not np.isnan(self.joint_positions[:6]).any():
            position_error = self.target_positions - self.joint_positions[:6]
            self.joint_velocities[:6] = position_error * self.position_gain * 30.0  # scale by frequency
            self.joint_positions[:6] += position_error * self.position_gain
        
        # 그리퍼는 0으로 고정 (닫힌 상태)
        self.joint_positions[6:] = 0.0
        self.joint_velocities[6:] = 0.0
        
        # Publish joint states (전체 10개)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ''
        
        msg.name = self.all_joint_names
        msg.position = self.joint_positions.tolist()
        msg.velocity = self.joint_velocities.tolist()
        msg.effort = [0.0] * 10
        
        self.joint_state_pub.publish(msg)
        
        # 주기적 로그
        if self.counter % 30 == 0:  # 1초마다
            self.get_logger().info(
                f'Joint Pos: [{self.joint_positions[0]:.3f}, {self.joint_positions[1]:.3f}, '
                f'{self.joint_positions[2]:.3f}, ...] | '
                f'Target: [{self.target_positions[0]:.3f}, {self.target_positions[1]:.3f}, ...]'
            )
        
        self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = MockJointStatePublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Mock Joint State Publisher stopped')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
