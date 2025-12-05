#!/bin/bash
# Launch script for Mock Object Detector

# ROS 2 환경 소싱
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash
elif [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
elif [ -f "/opt/ros/foxy/setup.bash" ]; then
    source /opt/ros/foxy/setup.bash
else
    echo "❌ ROS 2 not found!"
    exit 1
fi

echo "🎯 Starting Mock Object Detector..."
python3 /home/jiwoo/IsaacLab/scripts/deployment/mock_object_detector.py
