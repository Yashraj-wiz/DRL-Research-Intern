#!/bin/bash
# Eval entrypoint — runs inside gotgtrl:noetic container (see run_test_all.sh),
# with the host repo mounted at /repo
set -e

echo "[eval] Installing xvfb..."
apt-get update -qq > /dev/null 2>&1
apt-get install -yqq xvfb > /dev/null 2>&1
Xvfb :99 -screen 0 1280x1024x24 &
export DISPLAY=:99
sleep 2

source /opt/ros/noetic/setup.bash
source /workspace/catkin_ws/devel/setup.bash

echo "[eval] Copying host files into workspace..."
cp /repo/catkin_ws/src/gtrl/scripts/SAC/test_all.py     /workspace/catkin_ws/src/gtrl/scripts/SAC/test_all.py
cp /repo/catkin_ws/src/gtrl/scripts/SAC/config.yaml     /workspace/catkin_ws/src/gtrl/scripts/SAC/config.yaml
cp /repo/catkin_ws/src/gtrl/scripts/Environments/env_lab.py /workspace/catkin_ws/src/gtrl/scripts/Environments/env_lab.py
mkdir -p /workspace/catkin_ws/src/gtrl/scripts/SAC/final_models
cp /repo/catkin_ws/src/gtrl/scripts/SAC/final_models/*.pth /workspace/catkin_ws/src/gtrl/scripts/SAC/final_models/
mkdir -p /workspace/catkin_ws/src/gtrl/scripts/SAC/results

echo "[eval] Starting bulk model evaluation..."
export ROS_MASTER_URI=http://localhost:11311
cd /workspace/catkin_ws/src/gtrl/scripts
python3 -u SAC/test_all.py --episodes "${EPISODES:-5}" 2>&1 | tee /tmp/test_all.log

echo "[eval] Copying results back to host..."
cp /tmp/test_all.log /repo/test_eval_output.log 2>/dev/null || true
cp /workspace/catkin_ws/src/gtrl/scripts/SAC/results/evaluation_summary.json \
   /repo/catkin_ws/src/gtrl/scripts/SAC/results/evaluation_summary.json 2>/dev/null || true

echo "EVAL_COMPLETE"
