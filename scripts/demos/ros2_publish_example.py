#!/usr/bin/env python3
"""
ROS2 토픽 publish 예제
Isaac Sim 내장 rclpy 사용 - 시스템 ROS2와 통신 가능
"""

import argparse
import sys
import os

# ROS2 환경 변수 설정 (Isaac Sim 내장 라이브러리 사용)
os.environ["ROS_DISTRO"] = "jazzy"
os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"

# Isaac Sim ROS2 라이브러리 경로 추가
ros2_lib_path = os.path.expanduser("~/env_isaacsim/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib")
if "LD_LIBRARY_PATH" in os.environ:
    os.environ["LD_LIBRARY_PATH"] = f"{ros2_lib_path}:{os.environ['LD_LIBRARY_PATH']}"
else:
    os.environ["LD_LIBRARY_PATH"] = ros2_lib_path

# Isaac Sim 먼저 초기화
from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="ROS2 토픽 publish 예제")
parser.add_argument("--headless", action="store_true", help="Headless 모드")
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

# Isaac Sim ROS2 브리지 활성화
import omni
import carb

# ROS2 브리지 확장 활성화
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)

# 확장이 로드될 때까지 대기
import time
for _ in range(10):
    if ext_manager.is_extension_enabled("isaacsim.ros2.bridge"):
        break
    time.sleep(0.5)

# ROS2 import (Isaac Sim 내장)
try:
    import rclpy
    carb.log_info("rclpy successfully imported")
except ImportError as e:
    carb.log_error(f"Failed to import rclpy: {e}")
    carb.log_error("Make sure isaacsim.ros2.bridge extension is enabled")
    simulation_app.close()
    sys.exit(1)
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import numpy as np
from datetime import datetime


class SimpleROS2Publisher(Node):
    """간단한 ROS2 Publisher 노드"""
    
    def __init__(self):
        super().__init__('isaac_sim_publisher')
        
        # Publishers 생성
        self.string_pub = self.create_publisher(String, 'hello_topic', 10)
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.pose_pub = self.create_publisher(PoseStamped, 'robot_pose', 10)
        
        # 타이머 생성 (10Hz)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.counter = 0
        
        self.get_logger().info('🚀 ROS2 Publisher 시작!')
        self.get_logger().info('📡 토픽:')
        self.get_logger().info('  - /hello_topic (std_msgs/String)')
        self.get_logger().info('  - /joint_states (sensor_msgs/JointState)')
        self.get_logger().info('  - /robot_pose (geometry_msgs/PoseStamped)')
    
    def timer_callback(self):
        """주기적으로 메시지 publish"""
        self.counter += 1
        
        # 1. String 메시지
        msg = String()
        msg.data = f'Hello from Isaac Sim! Count: {self.counter}'
        self.string_pub.publish(msg)
        
        # 2. JointState 메시지 (가상의 6축 로봇)
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.header.frame_id = "base_link"
        joint_msg.name = [f'joint_{i+1}' for i in range(6)]
        
        # 사인파로 움직이는 조인트 각도
        t = self.counter * 0.1
        joint_msg.position = [
            np.sin(t) * 0.5,
            np.cos(t) * 0.5,
            np.sin(t * 2) * 0.3,
            np.cos(t * 2) * 0.3,
            np.sin(t * 3) * 0.2,
            np.cos(t * 3) * 0.2,
        ]
        joint_msg.velocity = [0.0] * 6
        joint_msg.effort = [0.0] * 6
        self.joint_pub.publish(joint_msg)
        
        # 3. PoseStamped 메시지 (원형 경로)
        pose_msg = PoseStamped()
        pose_msg.header.stamp = joint_msg.header.stamp
        pose_msg.header.frame_id = "world"
        
        radius = 0.5
        pose_msg.pose.position.x = radius * np.cos(t)
        pose_msg.pose.position.y = radius * np.sin(t)
        pose_msg.pose.position.z = 0.5 + 0.1 * np.sin(t * 4)
        
        # 간단한 quaternion (z축 회전)
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = np.sin(t / 2)
        pose_msg.pose.orientation.w = np.cos(t / 2)
        
        self.pose_pub.publish(pose_msg)
        
        # 로그 (5초마다)
        if self.counter % 50 == 0:
            self.get_logger().info(f'📤 {self.counter}개 메시지 전송됨')


def main():
    """메인 함수"""
    
    # ROS2 초기화
    rclpy.init()
    
    # Publisher 노드 생성
    publisher = SimpleROS2Publisher()
    
    print("\n" + "="*60)
    print("ROS2 토픽 Publisher 실행 중...")
    print("="*60)
    print("\n다른 터미널에서 확인하려면:")
    print("  source /opt/ros/jazzy/setup.bash")
    print("  ros2 topic list")
    print("  ros2 topic echo /hello_topic")
    print("  ros2 topic echo /joint_states")
    print("  ros2 topic echo /robot_pose")
    print("\nRViz2로 시각화:")
    print("  rviz2")
    print("  - Add -> By topic -> /joint_states -> JointState")
    print("  - Add -> By topic -> /robot_pose -> Pose")
    print("\n종료: Ctrl+C")
    print("="*60 + "\n")
    
    try:
        # ROS2 스핀 (메시지 처리 루프)
        while simulation_app.is_running() and rclpy.ok():
            rclpy.spin_once(publisher, timeout_sec=0.01)
            simulation_app.update()
            
    except KeyboardInterrupt:
        print("\n종료 중...")
    finally:
        # 정리
        publisher.destroy_node()
        rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
