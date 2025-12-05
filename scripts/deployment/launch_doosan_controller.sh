#!/bin/bash
# Launch script for Doosan Robot Controller with Isaac Lab JIT policy

echo "=================================================="
echo "🤖 Doosan Robot Isaac Lab Controller"
echo "=================================================="
echo ""

# ROS 2 환경 소싱 (설치된 ROS 2 버전에 맞게 수정)
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash
    echo "✅ ROS 2 Jazzy sourced"
elif [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    echo "✅ ROS 2 Humble sourced"
elif [ -f "/opt/ros/foxy/setup.bash" ]; then
    source /opt/ros/foxy/setup.bash
    echo "✅ ROS 2 Foxy sourced"
else
    echo "❌ ROS 2 installation not found!"
    exit 1
fi

# Doosan ROS 2 workspace 소싱 (실제 경로로 수정 필요)
# if [ -f "~/doosan_ws/install/setup.bash" ]; then
#     source ~/doosan_ws/install/setup.bash
#     echo "✅ Doosan workspace sourced"
# fi

echo ""
echo "⚠️  SAFETY CHECKLIST:"
echo "  1. Robot is in REMOTE/ROS MODE"
echo "  2. Emergency stop is accessible"
echo "  3. Workspace is clear of obstacles"
echo "  4. Joint limits are properly configured"
echo ""
read -p "Press ENTER to continue or Ctrl+C to cancel..."

# Check if torch is installed in system python
if ! python3 -c "import torch" 2>/dev/null; then
    echo "Installing torch and scipy for system Python..."
    pip3 install torch scipy --break-system-packages
fi

# Python 노드 실행
python3 /home/jiwoo/IsaacLab/scripts/deployment/ros2_doosan_controller.py

echo ""
echo "🛑 Controller stopped"
