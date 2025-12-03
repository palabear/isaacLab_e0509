#!/bin/bash

# E0509 Pick-Place 연속 학습 스크립트
# 사용법: ./continue_training.sh [checkpoint_dir] [iterations]

# 기본값 설정
CHECKPOINT_DIR="${1:-/home/jiwoo/IsaacLab/logs/rsl_rl/e0509_pick_place/2025-12-02_20-56-40}"
ITERATIONS="${2:-10000}"

echo "=========================================="
echo "E0509 Pick-Place 연속 학습"
echo "=========================================="
echo "체크포인트: $CHECKPOINT_DIR"
echo "추가 학습 반복: $ITERATIONS"
echo "=========================================="

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-E0509-PickPlace-v0 \
    --num_envs 1024 \
    --max_iterations $ITERATIONS \
    --resume \
    --load_run $CHECKPOINT_DIR

echo "=========================================="
echo "학습 완료!"
echo "=========================================="
