#!/bin/bash

# Configuration
IMAGE_NAME="gotgtrl:noetic"
CONTAINER_NAME="gotgtrl_evaluation"
WORKDIR="/workspace"
HOST_WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Display usage
if [ -z "$1" ]; then
  echo "Usage: ./run_test.sh <model_prefix> [num_episodes]"
  echo "Example: ./run_test.sh gtrl_run2_ep1100_reward-50_seed420 10"
  exit 1
fi

MODEL_PREFIX=$1
EPISODES=${2:-10}

echo "=============================================="
echo "  GoT-GTRL Evaluation Launcher"
echo "  Model: $MODEL_PREFIX"
echo "  Episodes: $EPISODES"
echo "=============================================="

# Ensure no existing container
docker rm -f $CONTAINER_NAME > /dev/null 2>&1

echo "[1/2] Making sure script is executable..."
chmod +x "$HOST_WORKSPACE/catkin_ws/src/gtrl/scripts/SAC/test.py"

echo "[2/2] Starting Docker container for evaluation..."
docker run --name $CONTAINER_NAME \
  --rm \
  --gpus all \
  --net=host \
  --ipc=host \
  --privileged \
  --env="DISPLAY=$DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --env="NVIDIA_DRIVER_CAPABILITIES=all" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="$HOST_WORKSPACE/catkin_ws/src/gtrl/scripts/SAC/final_curves:/workspace/catkin_ws/src/gtrl/scripts/SAC/final_curves" \
  --volume="$HOST_WORKSPACE/catkin_ws/src/gtrl/scripts/SAC/final_models:/workspace/catkin_ws/src/gtrl/scripts/SAC/final_models" \
  --volume="$HOST_WORKSPACE/catkin_ws/src/gtrl/scripts/SAC/test.py:/workspace/catkin_ws/src/gtrl/scripts/SAC/test.py" \
  --volume="$HOST_WORKSPACE/catkin_ws/src/gtrl/scripts/Environments/env_lab.py:/workspace/catkin_ws/src/gtrl/scripts/Environments/env_lab.py" \
  --workdir="$WORKDIR/catkin_ws/src/gtrl/scripts" \
  $IMAGE_NAME \
  bash -c "source /opt/ros/noetic/setup.bash && source /workspace/catkin_ws/devel/setup.bash && apt-get update -qq && apt-get install -yqq xvfb && (rosrun rviz rviz -d /workspace/catkin_ws/src/gtrl/launch/scout.rviz &) && cd /workspace/catkin_ws/src/gtrl/scripts && python3 SAC/test.py --model $MODEL_PREFIX --episodes $EPISODES"

echo "Evaluation Complete!"
