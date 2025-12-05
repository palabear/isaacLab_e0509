#!/bin/bash
# Launch Mock Joint State Publisher

if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash
elif [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
else
    echo "❌ ROS 2 not found!"
    exit 1
fi

echo "🤖 Starting Mock Joint State Publisher..."
python3 /home/jiwoo/IsaacLab/scripts/deployment/mock_joint_state_publisher.py
