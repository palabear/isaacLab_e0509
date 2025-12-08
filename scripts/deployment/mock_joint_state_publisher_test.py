#!/usr/bin/env python3
"""
Mock Joint State Publisher for Testing Sim2Real Controller

This script simulates a robot that follows commands from the controller.
It subscribes to ServojRtStream commands and publishes resulting joint states.

Usage:
    python3 mock_joint_state_publisher_test.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np


class MockJointStatePublisher(Node):
    def __init__(self):
        super().__init__('mock_joint_state_publisher')
        
        # Publisher: sends current joint states (mimic Doosan driver)
        self.publisher = self.create_publisher(JointState, '/dsr01/joint_states', 10)
        
        # Subscriber: receives ServojRtStream commands from controller
        # Using Float64MultiArray as a simple alternative to ServojRtStream
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/dsr01/servoj_rt_stream_mock',
            self.servoj_callback,
            10
        )
        
        # Timer (100 Hz to match real robot)
        self.timer = self.create_timer(0.01, self.publish_joint_states)
        
        # Joint configuration (6 arm + 4 gripper = 10 total)
        # ⚠️ Must match controller JOINT_NAMES exactly!
        self.joint_names = [
            'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6',
            'rh_l1', 'rh_r1_joint', 'rh_l2', 'rh_r2'
        ]
        
        # Current state (home position from e0509.py)
        self.current_pos = np.array([0.0, 0.0, 1.5708, 0.0, 1.5708, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.current_vel = np.zeros(10)
        
        # Target position (initially same as current)
        self.target_pos = self.current_pos.copy()
        
        # Simulation parameters
        self.dt = 0.01
        self.position_gain = 5.0  # How fast robot follows commands
        
        # Logging control
        self.step_count = 0
        self.log_interval_target = 50  # Log received targets every 50 msgs (1 msg per 20ms = 1 sec)
        self.log_interval_state = 100  # Log current state every 100 steps (1 sec)
        
        self.get_logger().info('🤖 Mock Robot Simulator Started')
        self.get_logger().info(f'   Publishing to: /dsr01/joint_states')
        self.get_logger().info(f'   Subscribing to: /dsr01/servoj_rt_stream_mock')
        self.get_logger().info(f'   Rate: 100 Hz')
        self.get_logger().info(f'   Joints: {self.joint_names}')
    
    def servoj_callback(self, msg: Float64MultiArray):
        """Receive servoj commands from controller (in DEGREES)."""
        if len(msg.data) >= 6:
            # Convert degrees to radians (controller sends in degrees)
            target_deg = np.array(msg.data[:6])
            target_rad = target_deg / 57.2958  # deg to rad
            
            # Update target position for arm joints
            self.target_pos[:6] = target_rad
            
            # Log received command (only periodically to reduce spam)
            if self.step_count % self.log_interval_target == 0:
                self.get_logger().info(f'📥 Received target: {[f"{x:.3f}" for x in target_rad]}')
            
            self.step_count += 1
    
    def publish_joint_states(self):
        """Simulate robot dynamics and publish joint states."""
        # Simple PD controller: move towards target position
        position_error = self.target_pos - self.current_pos
        
        # Update velocity (proportional control)
        self.current_vel = self.position_gain * position_error
        
        # Limit velocity for realistic motion
        max_vel = 1.0  # rad/s
        self.current_vel = np.clip(self.current_vel, -max_vel, max_vel)
        
        # Update position (integrate velocity)
        self.current_pos += self.current_vel * self.dt
        
        # Log current state periodically (every 1 second)
        if self.step_count % self.log_interval_state == 0:
            self.get_logger().info(f'🤖 Current pos: {[f"{x:.3f}" for x in self.current_pos[:6]]}')
        
        # Publish joint state
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ''
        msg.name = self.joint_names
        msg.position = self.current_pos.tolist()
        msg.velocity = self.current_vel.tolist()
        msg.effort = [0.0] * 10
        
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    publisher = MockJointStatePublisher()
    
    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        publisher.get_logger().info('Shutting down...')
    finally:
        publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
