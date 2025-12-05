#!/usr/bin/env python3
"""
물체 위치 발행 테스트용 Mock 노드
실제 vision system 구현 전 테스트용
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import math


class MockObjectDetector(Node):
    def __init__(self):
        super().__init__('mock_object_detector')
        
        # 물체 위치 발행 (로봇 베이스 프레임 기준)
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/object_detection/pose',
            10
        )
        
        # 1Hz로 발행
        self.timer = self.create_timer(1.0, self.publish_object_pose)
        
        self.get_logger().info('🎯 Mock Object Detector started')
        self.counter = 0
    
    def publish_object_pose(self):
        """고정된 위치의 물체 발행 (테스트용)"""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'  # 또는 'world'
        
        # 물체 위치 (로봇 앞쪽 0.5m, 왼쪽 0.2m)
        # 실제로는 카메라에서 받아온 값
        msg.pose.position.x = 0.5
        msg.pose.position.y = -0.2
        msg.pose.position.z = 0.0
        
        # 작은 움직임 시뮬레이션 (선택사항)
        t = self.counter * 0.1
        msg.pose.position.x += 0.02 * math.sin(t)
        
        msg.pose.orientation.w = 1.0
        
        self.pose_pub.publish(msg)
        self.counter += 1
        
        if self.counter % 10 == 0:
            self.get_logger().info(
                f'Publishing object at: ({msg.pose.position.x:.3f}, '
                f'{msg.pose.position.y:.3f}, {msg.pose.position.z:.3f})'
            )


def main(args=None):
    rclpy.init(args=args)
    node = MockObjectDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
