#!/bin/bash
# ============================================================
# run_test_all.sh — GoT-GTRL Bulk Evaluation Launcher
# Starts a fresh container that runs eval_entrypoint.sh: every
# *_actor.pth in final_models is evaluated with the deterministic
# policy and results go to results/evaluation_summary.json
# Usage: bash run_test_all.sh [episodes_per_model]
# ============================================================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="gotgtrl:noetic"
CONTAINER_NAME="gotgtrl_evaluation_bulk"
EPISODES=${1:-5}

echo "=============================================="
echo "  GoT-GTRL Bulk Evaluation Launcher"
echo "  Models dir : final_models/"
echo "  Episodes/model : $EPISODES"
echo "=============================================="

# Kill any stale eval container (training container is separate — leave it)
docker rm -f "$CONTAINER_NAME" > /dev/null 2>&1

# Ensure results dir exists on host
mkdir -p "$REPO_DIR/catkin_ws/src/gtrl/scripts/SAC/results"

echo "Launching evaluation container (Gazebo renders on a virtual display)..."
docker run \
  --name "$CONTAINER_NAME" \
  --rm \
  --gpus all \
  --net=host \
  --ipc=host \
  --env="EPISODES=$EPISODES" \
  --env="QT_X11_NO_MITSHM=1" \
  --env="NVIDIA_DRIVER_CAPABILITIES=all" \
  --volume="$REPO_DIR:/repo" \
  "$IMAGE_NAME" \
  bash /repo/eval_entrypoint.sh

echo ""
echo "=============================================="
echo "  Bulk Evaluation Complete!"
echo "  Results: catkin_ws/src/gtrl/scripts/SAC/results/evaluation_summary.json"
echo "=============================================="
