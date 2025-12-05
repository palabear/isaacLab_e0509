# Doosan Robot Isaac Lab Deployment Guide

## 개요
Isaac Lab에서 학습된 JIT 모델(`policy.pt`)을 두산 로봇에 적용하는 ROS 2 컨트롤러입니다.

## 파일 구조
```
scripts/deployment/
├── ros2_doosan_controller.py      # 메인 ROS 2 노드
└── launch_doosan_controller.sh    # 실행 스크립트
```

## 필요 사항

### 1. ROS 2 환경
```bash
# ROS 2 설치 확인
ros2 --version

# 두산 로보틱스 ROS 2 패키지
# https://github.com/doosan-robotics/doosan-robot2
```

### 2. Python 패키지
```bash
pip install torch rclpy scipy
```

### 3. 로봇 설정
- 두산 로봇 티칭 펜던트에서 **Remote Mode** 또는 **ROS Mode** 활성화
- `ros2_control` Forward Position Controller 설정
- **중요**: 로봇의 default/home position을 `[0, 0, 0, 0, 0, 0]`으로 설정

## 사용 방법

### 1단계: 로봇 설정 확인
```bash
# 로봇 joint_states 토픽 확인
ros2 topic echo /joint_states

# Forward position controller 확인
ros2 topic list | grep forward_position_controller
```

### 2단계: 관절 한계 및 파라미터 조정
`ros2_doosan_controller.py` 파일에서 다음 파라미터를 로봇 사양에 맞게 수정:

```python
# 관절 한계 (라디안)
self.joint_lower_limits = np.array([...])  # 실제 값으로 수정
self.joint_upper_limits = np.array([...])  # 실제 값으로 수정

# Action scale (Isaac Lab 학습 시와 동일)
self.action_scale = 0.1  # 필요시 조정

# 최대 속도 (안전)
self.max_joint_velocity = 0.5  # rad/s
```

### 3단계: Observation 구조 매칭 ✅ VERIFIED

Isaac Lab 학습 환경에서 확인한 정확한 observation 구조:

**소스**: `e0509_pick_place_env_cfg.py` → `ObservationsCfg.PolicyCfg`

```python
# 총 21차원 (정확히 매칭됨)
obs = [
    joint_pos_rel (6),       # mdp.joint_pos_rel - 기본 자세 대비 상대 위치
    joint_vel_rel (6),       # mdp.joint_vel_rel - 관절 속도
    object_position (3),     # e0509_mdp.object_position_in_robot_root_frame
    last_action (6)          # mdp.last_action - 이전 액션
]
```

**✅ 현재 구현 상태**: `ros2_doosan_controller.py`의 `build_observation()`이 위 구조와 **정확히 일치**합니다.

**핵심 구현 사항**:
1. **joint_pos_rel**: `current_joint_pos - default_joint_pos` (상대 위치)
2. **object_position**: World frame → Robot root frame 좌표 변환 (`world_to_robot_frame()`)
3. **Robot base position**: `[0.96, 0.095, -0.95]` (학습 환경과 동일)

#### 물체 위치 획득 방법

**Option 1: Mock 데이터로 테스트 (권장)**
```bash
# 별도 터미널에서 mock object detector 실행
python3 scripts/deployment/mock_object_detector.py
```

**Option 2: 실제 Vision System 연동**
- ArUco 마커 사용
- YOLO/Detectron2 Object Detection
- MoCap 시스템 (OptiTrack, Vicon)
- RealSense/ZED Camera depth + detection

`ros2_doosan_controller.py`의 `object_detection_callback()` 함수를 실제 토픽에 맞게 활성화하세요.

### 4단계: 실행

#### 기본 실행 (Mock 물체 위치 사용)
```bash
# 터미널 1: Mock object detector (테스트용)
python3 scripts/deployment/mock_object_detector.py

# 터미널 2: 로봇 컨트롤러
./scripts/deployment/launch_doosan_controller.sh
```

#### 실제 Vision System 사용
```bash
# 터미널 1: Vision system (예: ArUco detector)
ros2 run your_vision_pkg object_detector

# 터미널 2: 로봇 컨트롤러
./scripts/deployment/launch_doosan_controller.sh
```

## 주요 토픽

### Subscribe
- `/joint_states` (sensor_msgs/JointState)
  - 로봇의 현재 관절 위치, 속도 수신

- `/object_detection/pose` (geometry_msgs/PoseStamped) - **선택사항**
  - 물체 위치 수신 (vision system에서)
  - 현재는 mock_object_detector.py에서 발행

### Publish
- `/forward_position_controller/commands` (std_msgs/Float64MultiArray)
  - 목표 관절 위치 명령 송신

## 안전 기능

### 1. 관절 한계 체크
```python
target_joint_pos = np.clip(
    target_joint_pos,
    self.joint_lower_limits,
    self.joint_upper_limits
)
```

### 2. 속도 제한
```python
max_pos_change = self.max_joint_velocity * dt
position_delta = np.clip(position_delta, -max_pos_change, max_pos_change)
```

### 3. 초기화 확인
- `joint_states` 수신 전까지 명령 발행 안 함

## 문제 해결

### 1. 로봇이 움직이지 않음
- [ ] 로봇이 Remote/ROS Mode인지 확인
- [ ] `/forward_position_controller/commands` 토픽이 존재하는지 확인
- [ ] `joint_states` 토픽이 수신되는지 확인

### 2. 관절 순서가 맞지 않음
`joint_state_callback()`에서 관절 매핑 확인:
```python
self.joint_names = ['joint_1', 'joint_2', ...]  # 로봇 실제 이름으로 수정
```

### 3. Observation 차원 불일치
Isaac Lab 학습 시: **21 dimensions**
```python
[joint_pos(6), joint_vel(6), object_position(3), actions(6)]
```

확인 방법:
```python
obs = self.build_observation(self.object_position_robot_frame)
print(f"Observation shape: {obs.shape}")  # Should be (21,)
```

### 4. 로봇이 너무 빠르게/느리게 움직임
- `self.action_scale` 조정 (현재 0.1)
- `self.max_joint_velocity` 조정 (현재 0.5 rad/s)

### 5. Protective Stop 발생
- 두산 로봇 충돌 감지 민감도 조정
- Action scale을 더 작게 설정
- 제어 주파수 낮추기

## Isaac Lab 학습 환경과의 차이점

| 항목 | Isaac Lab | 실제 로봇 |
|------|-----------|----------|
| 제어 주파수 | 30 Hz (decimation=4) | 30 Hz (동일하게 설정) |
| Action range | -1.0 ~ 1.0 | -1.0 ~ 1.0 |
| Action scale | 0.1 | 0.1 (동일하게 설정) |
| 물리 엔진 | PhysX (완벽) | 실제 물리 (마찰, 지연 등) |
| Observation | 시뮬레이션 값 | 센서 값 (노이즈 포함) + **물체 위치는 vision system 필요** |

## 다음 단계

### 1. 물체 인식 추가
Pick-and-place 작업을 위해서는 물체 위치를 observation에 추가해야 합니다:
- 카메라 + Object Detection (YOLO, etc.)
- ArUco 마커
- MoCap 시스템

### 2. Gripper 제어 추가
현재는 팔 관절만 제어합니다. Gripper 제어 추가 필요:
```python
# Gripper action topic
self.gripper_pub = self.create_publisher(
    Float64MultiArray,
    '/gripper_controller/commands',
    10
)
```

### 3. Sim-to-Real Transfer 개선
- Domain Randomization 결과 확인
- Action smoothing/filtering 추가
- Model fine-tuning with real data

## 참고 자료
- [Doosan Robotics ROS 2 Package](https://github.com/doosan-robotics/doosan-robot2)
- [ros2_control Documentation](https://control.ros.org/)
- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
