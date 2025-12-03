# E0509 Pick and Place Task

Isaac Lab 기반 E0509 로봇 픽앤플레이스 강화학습 환경

---

## 개요

E0509 6-DOF 로봇팔과 Robotiq 2F-85 그리퍼를 사용하여 테이블 위의 물체를 잡고 들어올리는 태스크입니다.

### 환경 구성
- **로봇**: E0509 6-DOF 매니퓰레이터 + Robotiq 2F-85 그리퍼
- **물체**: Medicine Cabinet (2.5kg)
- **목표**: 물체를 8cm 이상 들어올리기

---

## 설치 및 실행

### 학습 시작
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-E0509-PickPlace-v0 \
    --num_envs 1024 \
    --max_iterations 20000
```

### 연속 학습
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-E0509-PickPlace-v0 \
    --num_envs 1024 \
    --max_iterations 20000 \
    --resume \
    --load_run [디렉토리명]  # 예: 2025-12-02_20-56-40
```

### 학습 결과 검증
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-E0509-PickPlace-v0 \
    --num_envs 16 \
    --checkpoint /path/to/model.pt
```

---

## 관찰 (Observations)

| 항목 | 설명 | 차원 |
|------|------|------|
| `joint_pos` | 관절 위치 (상대값) | 10 |
| `joint_vel` | 관절 속도 (상대값) | 10 |
| `object_position` | 물체 위치 (로봇 base 기준) | 3 |
| `actions` | 이전 action | 7 |

**총 관찰 차원**: 30

---

## 행동 (Actions)

| 항목 | 제어 방식 | 범위 | 차원 |
|------|-----------|------|------|
| `arm_action` | Joint position (6-DOF) | [-1, 1] | 6 |
| `gripper_action` | Mimic gripper (4 joints) | [-1, 1] → [0, 1.1] | 1 |

**총 행동 차원**: 7

- 그리퍼는 1개의 action으로 4개 joint 동시 제어 (mimic)
- `-1`: 완전 열림, `+1`: 완전 닫힘

---

## 보상 함수 (Rewards)

### Stage 1: 접근 및 방향 정렬 (10.0)
```python
approach_and_orient = exp(-distance) × orientation_gate
```
- 물체 15cm 위 공중 목표로 접근
- 수직 자세(아래 방향) 유지 필수
- `orientation_strictness=4.0`: 기울어지면 급격히 감소

### Stage 2: 그리퍼 제어 (5.0)
```python
grasp_encourage = distance_factor × gripper_closure
```
- 12cm 이내에서 그리퍼 닫기 유도
- 거리와 그리퍼 닫힘 정도 곱셈 보상

### Stage 3: 성공 (15.0)
```python
lift_success = is_lifted(8cm) AND is_grasping(50%)
```
- 물체를 8cm 이상 들어올리면 압도적 보상
- 그리퍼가 절반 이상 닫혀있어야 함

### 페널티
- `object_stability`: 물체 날리기 방지 (-1.0)
- `action_rate`: 부드러운 동작 유도 (-0.01)

---

## 종료 조건 (Terminations)

| 조건 | 설명 |
|------|------|
| `time_out` | 15초 경과 |
| `object_out_of_bounds` | 물체가 작업 영역 이탈 |
| `arm_collision` | 로봇팔-물체 충돌 (5N 이상) |

- 그리퍼-물체 접촉은 허용 (< 5N)
- 로봇팔이 물체를 치면 즉시 종료

---

## 주요 설정

### 시뮬레이션
- Physics: 120Hz
- Control: 30Hz (decimation=4)
- Episode: 15초 (450 steps)

### 초기화
- Joint 랜덤화: ±0.3 rad (±17도)
- Gripper: 항상 열린 상태로 시작

### End-Effector Frame
```python
offset = (0.0, -0.15, 0.0)  # gripper base → end-effector mesh
debug_vis = False            # 시각화 끄기
```

---

## 파일 구조

```
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/e0509_pick_place/
├── __init__.py
├── e0509_pick_place_env_cfg.py  # 환경 설정
└── mdp/
    ├── __init__.py
    ├── observations.py           # 관찰 함수
    ├── rewards.py                # 보상 함수
    ├── terminations.py           # 종료 조건
    ├── events.py                 # 초기화 이벤트
    └── gripper_action.py         # Mimic 그리퍼 액션
```

---

## 학습 팁

### Hyperparameters (RSL RL)
- `num_envs`: 1024 (GPU 메모리에 따라 조정)
- `max_iterations`: 20000+ (약 2-3시간)
- Learning rate: 기본값 사용
- Episode length: 15초 (450 steps)

### 주요 조정 포인트

**방향 엄격도 조정**
```python
orientation_strictness: 4.0  # 높을수록 수직 자세 강요
```

**충돌 감지 임계값**
```python
force_threshold: 5.0  # 낮추면 민감, 높이면 둔감
```

**그리퍼 offset 보정**
```python
offset: (0.0, -0.15, 0.0)  # Y축 조정하여 정확한 위치 맞추기
```

---

## 문제 해결

### 로봇이 쓰러짐
- `orientation_strictness` 증가 (4.0 → 6.0)
- `approach_and_orient` weight 증가 (10.0 → 15.0)

### 그리퍼가 물체를 못 잡음
- End-effector offset 조정
- `debug_vis=True`로 화살표 위치 확인
- `grasp_encourage` weight 증가

### 충돌이 너무 자주 발생
- `force_threshold` 증가 (5.0 → 10.0)
- 또는 `arm_collision` termination 비활성화

---

## License

BSD-3 License (Isaac Lab)

## 참고

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab)
- [Reinforcement Learning Guide](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
