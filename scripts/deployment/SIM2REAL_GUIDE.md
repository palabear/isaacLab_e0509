# Sim2Real E0509 Controller - 사용 가이드

## 개요

Isaac Lab에서 학습된 PPO 정책을 **실제 Doosan E0509 로봇**에서 실행하는 완전한 Sim2Real 스크립트입니다.

## 시스템 구성

```
Isaac Lab Training → Policy Model (.pt) → Real Robot Deployment
                      ↓
            Observation (25-dim)
            Action (6-dim)
            Safety Checks
                      ↓
            Real E0509 Robot
```

## 실행 방법

### 1. ROS2 환경 설정

```bash
# ROS2 Jazzy 소스
source /opt/ros/jazzy/setup.bash

# 작업 디렉토리 이동
cd /home/jiwoo/IsaacLab
```

### 2. 실제 로봇 준비

- 로봇을 **REMOTE/ROS 모드**로 설정
- Emergency stop 버튼 확인
- `/joint_states` topic이 publish되는지 확인:
  ```bash
  ros2 topic hz /joint_states
  # Expected: ~100 Hz
  ```

### 3. Sim2Real Controller 실행

```bash
python3 scripts/deployment/sim2real_e0509_controller.py
```

**출력 예시:**
```
======================================================================
🚀 Sim2Real E0509 Policy Controller Initialized
======================================================================
   Model: /home/jiwoo/IsaacLab/logs/rsl_rl/.../exported/policy.pt
   Device: cuda
   Control Rate: 50.0 Hz
   Observation Dim: 25
   Action Dim: 6
======================================================================
⚠️  Make sure robot is in REMOTE/ROS MODE!
⚠️  Emergency stop button should be accessible!
Waiting for sensor data...
📊 Step    50 | Freq:  50.0 Hz | Action: [ 0.050, -0.100, ...] | Target: [...]
```

## 주요 기능

### 1. Observation Processing (Sim2Real 핵심)

**25차원 Observation 구성:**
```python
observation = [
    joint_pos_rel (6),     # current - default (⚠️ 상대 위치!)
    joint_vel (6),         # 현재 속도
    target_pose (7),       # target pos(3) + quat(4)
    previous_actions (6)   # 이전 action
]
```

**정규화 (Normalization):**
- RSL-RL 학습 시 적용된 mean/std를 ONNX에서 로드
- `obs_normalized = (obs - mean) / std`

### 2. Action Processing

**Action 스케일:**
```python
# Isaac Lab 학습 환경과 동일
action_scale = 0.5

# Target position 계산
target_pos = current_pos + action * 0.5
```

### 3. Safety Checks

#### a) Action Change Rate Limiting
```python
MAX_ACTION_CHANGE = 0.2  # 스텝 간 최대 action 변화량

if action_change > MAX_ACTION_CHANGE:
    # 스케일 다운
    action = prev_action + (action - prev_action) * scale
```

#### b) Joint Limits
```python
JOINT_LIMITS_LOWER = [-6.2832, -6.2832, -2.7053, ...]
JOINT_LIMITS_UPPER = [ 6.2832,  6.2832,  2.7053, ...]

target_pos = np.clip(target_pos, LOWER, UPPER)
```

#### c) Velocity Limits
```python
MAX_JOINT_VELOCITY = 1.0  # rad/s (보수적 설정)

velocity = (target - current) / dt
if velocity > MAX_VELOCITY:
    # 스케일 다운
```

#### d) Emergency Stop
```python
# 5회 연속 경고 시 자동 정지
if warning_count >= 5:
    emergency_stop = True
    # 로봇 정지 명령 전송
```

## Default Joint Positions (중요!)

**⚠️ CRITICAL: Isaac Lab 학습 환경과 동일해야 함!**

```python
default_joint_pos = [
    0.0,      # joint_1
    0.0,      # joint_2
    1.5708,   # joint_3 (90°)
    0.0,      # joint_4
    1.5708,   # joint_5 (90°)
    0.0,      # joint_6
]
```

이 값은 `source/isaaclab_tasks/.../e0509/e0509.py`와 일치해야 합니다!

## 수정이 필요한 부분

### 1. Model Path (필수)
```python
# Line 92-93
self.model_path = '/home/jiwoo/IsaacLab/logs/rsl_rl/.../exported/policy.pt'
self.onnx_path = '/home/jiwoo/IsaacLab/logs/rsl_rl/.../exported/policy.onnx'
```

### 2. Target Pose (선택)
```python
# Line 147-148
# TODO: /object_detection/pose에서 실시간 업데이트
self.target_position = np.array([0.55, 0.0, 0.15])
self.target_orientation = np.array([1.0, 0.0, 0.0, 0.0])
```

### 3. Safety Parameters (상황에 따라 조정)
```python
# Line 62-66
MAX_JOINT_VELOCITY = 1.0        # 속도 제한 (rad/s)
MAX_ACTION_CHANGE = 0.2         # Action 변화 제한
CONTROL_FREQUENCY = 50.0        # 제어 주파수 (Hz)
max_consecutive_warnings = 5    # 긴급 정지 threshold
```

## ROS2 Topics

### Subscribed Topics

| Topic | Type | Rate | Description |
|-------|------|------|-------------|
| `/joint_states` | `sensor_msgs/JointState` | 100 Hz | 실제 로봇 센서 데이터 |
| `/object_detection/pose` | `geometry_msgs/PoseStamped` | Variable | Medicine cabinet 위치 (선택) |

### Published Topics

| Topic | Type | Rate | Description |
|-------|------|------|-------------|
| `/joint_trajectory_controller/joint_trajectory` | `trajectory_msgs/JointTrajectory` | 50 Hz | 로봇 제어 명령 |

## 디버깅

### Topic 확인
```bash
# 모든 topic 확인
ros2 topic list

# Joint states 확인
ros2 topic echo /joint_states

# 로봇 명령 확인
ros2 topic echo /joint_trajectory_controller/joint_trajectory

# 주파수 확인
ros2 topic hz /joint_states
ros2 topic hz /joint_trajectory_controller/joint_trajectory
```

### 로그 확인
```bash
# ROS2 로그 레벨 설정
ros2 run sim2real_e0509_controller --ros-args --log-level DEBUG
```

### 문제 해결

#### 1. "Waiting for sensor data..." 계속 대기
**원인**: `/joint_states` topic이 publish되지 않음

**해결**:
```bash
# Topic 확인
ros2 topic list | grep joint_states

# 로봇 컨트롤러 재시작
# (로봇별 방법 다름)
```

#### 2. "Action change too large!" 경고 반복
**원인**: 정책이 급격한 동작 요구

**해결**:
```python
# MAX_ACTION_CHANGE 값 증가
MAX_ACTION_CHANGE = 0.3  # 0.2 → 0.3
```

#### 3. "EMERGENCY STOP" 발생
**원인**: 5회 연속 안전 경고

**해결**:
1. 노드 재시작
2. Safety parameter 조정
3. 로봇 상태 확인 (충돌, 한계값 도달 등)

#### 4. Observation normalization 오류
**원인**: ONNX에서 mean/std 로드 실패

**확인**:
```bash
# ONNX 파일 존재 확인
ls /home/jiwoo/IsaacLab/logs/rsl_rl/.../exported/policy.onnx

# Python에서 수동 확인
python3
>>> import onnx
>>> model = onnx.load('policy.onnx')
>>> for init in model.graph.initializer:
...     print(init.name, init.dims)
```

## 테스트 시나리오

### Test 1: Mock Joint States로 테스트
```bash
# Terminal 1: Mock publisher
python3 scripts/deployment/mock_joint_state_publisher.py

# Terminal 2: Sim2Real controller
python3 scripts/deployment/sim2real_e0509_controller.py
```

### Test 2: 안전 모드로 실행 (매우 보수적)
```python
# sim2real_e0509_controller.py 수정
MAX_JOINT_VELOCITY = 0.5       # 1.0 → 0.5
MAX_ACTION_CHANGE = 0.1        # 0.2 → 0.1
action_scale = 0.25            # 0.5 → 0.25
```

### Test 3: 실제 로봇 연동
```bash
# 1. 로봇 홈 위치로 이동
# 2. REMOTE/ROS 모드 설정
# 3. Emergency stop 준비
# 4. Controller 실행
python3 scripts/deployment/sim2real_e0509_controller.py

# 5. 관찰하면서 천천히 테스트
```

## 성능 모니터링

### 실시간 통계
```
📊 Step   250 | Freq:  50.1 Hz | Action: [...] | Target: [...]
```

- **Step**: 제어 스텝 카운트
- **Freq**: 실제 제어 주파수 (50 Hz 목표)
- **Action**: 현재 policy action
- **Target**: 목표 joint position

### 주의사항

⚠️ **반드시 확인**:
1. 로봇이 REMOTE/ROS 모드인지
2. Emergency stop 버튼 준비
3. 로봇 작업 공간에 장애물 없는지
4. Joint limits 올바른지
5. Default position이 학습 환경과 일치하는지

⚠️ **첫 실행 시**:
1. `MAX_ACTION_CHANGE = 0.1`로 매우 보수적으로 시작
2. 로봇 동작 관찰
3. 안전하다고 판단되면 점진적으로 증가

## 참고 파일

- **환경 설정**: `source/isaaclab_tasks/.../e0509/lift_env_cfg.py`
- **로봇 구성**: `source/isaaclab_tasks/.../e0509/e0509.py`
- **Agent 설정**: `source/isaaclab_tasks/.../e0509/agents/rsl_rl_ppo_cfg.py`
- **학습 모델**: `logs/rsl_rl/e0509_pick_place/.../exported/policy.pt`

## 라이센스

BSD-3-Clause License
