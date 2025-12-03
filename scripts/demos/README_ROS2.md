# ROS2 토픽 예제

Isaac Sim 내장 ROS2로 토픽을 publish하고 시스템 ROS2로 받는 예제입니다.

## 실행 방법

### 터미널 1: Isaac Sim에서 토픽 publish

```bash
cd /home/jiwoo/IsaacLab
./isaaclab.sh -p scripts/demos/ros2_publish_example.py --headless
```

### 터미널 2: 시스템 ROS2로 토픽 확인

```bash
# ROS2 환경 소싱
source /opt/ros/jazzy/setup.bash

# 토픽 리스트 확인
ros2 topic list

# 토픽 내용 확인
ros2 topic echo /hello_topic
ros2 topic echo /joint_states
ros2 topic echo /robot_pose

# 토픽 정보 확인
ros2 topic info /hello_topic
ros2 topic hz /hello_topic
```

### 터미널 3: RViz2로 시각화 (선택사항)

```bash
source /opt/ros/jazzy/setup.bash
rviz2
```

RViz2에서:
1. Fixed Frame을 `world`로 설정
2. Add → By topic → `/joint_states` → JointState
3. Add → By topic → `/robot_pose` → Pose
4. Add → TF (변환 프레임 표시)

## 발행되는 토픽

| 토픽 | 메시지 타입 | 주기 | 설명 |
|------|------------|------|------|
| `/hello_topic` | `std_msgs/String` | 10Hz | 간단한 문자열 메시지 |
| `/joint_states` | `sensor_msgs/JointState` | 10Hz | 6축 로봇의 조인트 상태 (사인파) |
| `/robot_pose` | `geometry_msgs/PoseStamped` | 10Hz | 원형 경로를 따라 움직이는 포즈 |

## 핵심 개념

1. **Isaac Sim (Python 3.11)**: 내장 rclpy로 토픽 publish
2. **시스템 ROS2 (Python 3.12)**: DDS로 메시지 수신
3. **DDS**: Python 버전 무관하게 프로세스 간 통신 처리

## 문제 해결

### 토픽이 보이지 않을 때
```bash
# ROS_DOMAIN_ID 확인 (같아야 함)
echo $ROS_DOMAIN_ID

# 네트워크 확인
ros2 doctor --report
```

### RViz2가 메시지를 받지 못할 때
- Fixed Frame이 메시지의 frame_id와 일치하는지 확인
- `/joint_states`는 `base_link`, `/robot_pose`는 `world` 프레임 사용
