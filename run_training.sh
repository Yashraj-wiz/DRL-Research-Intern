#!/bin/bash
# ============================================================
# run_training.sh — GoT-GTRL One-Command Training Launcher
# ============================================================
# Runs the complete training stack inside Docker:
#   roscore → Gazebo (headless) → main.py (GoT-SAC) → TensorBoard
#
# Usage:
#   bash run_training.sh
#   (seed, episodes, run name and PRE_TRAIN are set in
#    catkin_ws/src/gtrl/scripts/SAC/config.yaml)
#
# Outputs (persisted on host via volume mounts):
#   ./catkin_ws/src/gtrl/scripts/SAC/runs/        ← TensorBoard logs
#   ./catkin_ws/src/gtrl/scripts/SAC/final_models/ ← Checkpoints
#   ./catkin_ws/src/gtrl/scripts/SAC/final_curves/ ← Reward curves (.npy)
#   ./catkin_ws/src/gtrl/scripts/SAC/results/      ← CSV exports + PNG graphs
#
# Memory note:
#   The replay buffer is pre-allocated: BUFFER_SIZE=50000 →
#   50k × (128×160×4 float32 × 2) ≈ 31 GB of RAM
#   (100k caused OOM/SIGKILL on a 62 GB machine)
# ============================================================

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="gotgtrl:noetic"
CONTAINER_NAME="gotgtrl_training"
SCRIPTS_DIR="${REPO_DIR}/catkin_ws/src/gtrl/scripts"

echo "=============================================="
echo "  GoT-GTRL Training Launcher"
echo "  Config: catkin_ws/src/gtrl/scripts/SAC/config.yaml"
echo "  Image: ${IMAGE_NAME}"
echo "  Repo:  ${REPO_DIR}"
echo "=============================================="

# ── Pre-flight checks ─────────────────────────────────────────────────────────
echo "[1/4] Checking Docker image exists..."
if ! docker image inspect "${IMAGE_NAME}" > /dev/null 2>&1; then
    echo "  ERROR: Docker image '${IMAGE_NAME}' not found."
    echo "  Run:   cd ${REPO_DIR} && docker build -t ${IMAGE_NAME} ."
    exit 1
fi

# ── Enable X11 forwarding ─────────────────────────────────────────────────────
echo "[2/4] Enabling X11 forwarding..."
xhost +local:docker 2>/dev/null || echo "  WARNING: xhost not available (headless OK)"

# ── Create output directories on host ────────────────────────────────────────
mkdir -p "${SCRIPTS_DIR}/SAC/final_models"
mkdir -p "${SCRIPTS_DIR}/SAC/final_curves"
mkdir -p "${SCRIPTS_DIR}/SAC/results"
mkdir -p "${SCRIPTS_DIR}/SAC/runs"

# ── Stop any existing container ───────────────────────────────────────────────
echo "[3/4] Stopping existing container (if any)..."
docker stop "${CONTAINER_NAME}" 2>/dev/null || true
docker rm   "${CONTAINER_NAME}" 2>/dev/null || true
sleep 2

# ── Launch training container (detached) ─────────────────────────────────────
echo "[4/4] Launching training container..."

docker run -d \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --net=host \
    --ipc=host \
    --shm-size=4g \
    -e DISPLAY="${DISPLAY}" \
    -e IGN_IP=127.0.0.1 \
    -e QT_X11_NO_MITSHM=1 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "${SCRIPTS_DIR}":/workspace/catkin_ws/src/gtrl/scripts \
    "${IMAGE_NAME}" \
    /bin/bash -c '
        set -e
        source /opt/ros/noetic/setup.bash
        source /workspace/catkin_ws/devel/setup.bash
        export GAZEBO_MODEL_PATH=/root/.gazebo/models
        export GAZEBO_RESOURCE_PATH=/workspace/catkin_ws/src/gtrl/launch
        export IGN_IP=127.0.0.1

        echo ">>> [1/4] Installing Xvfb for headless rendering..."
        apt-get update -qq && apt-get install -yqq xvfb > /dev/null 2>&1
        echo "    Xvfb ready."

        echo ">>> [2/4] Starting TensorBoard on port 6006..."
        tensorboard \
            --logdir /workspace/catkin_ws/src/gtrl/scripts/SAC/runs \
            --port 6006 \
            --bind_all \
            --reload_interval 10 &
        sleep 2
        echo "    TensorBoard running at http://localhost:6006"

        echo ">>> [3/4] Starting GoT-SAC training..."
        echo "    Settings: scripts/SAC/config.yaml (PRE_TRAIN checkpoint: see main.py)"
        echo "    NOTE: env_lab.py will start roscore + Gazebo automatically."
        echo "--------------------------------------------------------------"
        cd /workspace/catkin_ws/src/gtrl/scripts/SAC

        echo ">>> Starting RViz (background)..."
        rosrun rviz rviz -d /workspace/catkin_ws/src/gtrl/launch/scout.rviz &

        python3 -u main.py
        TRAIN_EXIT=$?
        echo "--------------------------------------------------------------"
        echo ">>> Training finished (exit=${TRAIN_EXIT})."

        echo ">>> [4/4] Post-training: exporting TensorBoard graphs..."
        python3 post_training_analysis.py && \
            echo "    All graphs saved to /workspace/catkin_ws/src/gtrl/scripts/SAC/results/" || \
            echo "    WARNING: post_training_analysis.py failed — check results dir manually."

        echo "=============================================="
        echo "  TRAINING COMPLETE"
        echo "  Checkpoints: ./catkin_ws/src/gtrl/scripts/SAC/final_models/"
        echo "  Graphs/CSV:  ./catkin_ws/src/gtrl/scripts/SAC/results/"
        echo "  TensorBoard: http://localhost:6006  (still running)"
        echo "=============================================="
    '

echo ""
echo "=============================================="
echo "  Container '${CONTAINER_NAME}' launched."
echo "  Streaming live logs below..."
echo "  (Press Ctrl+C to detach — training continues)"
echo ""
echo "  TensorBoard:    http://localhost:6006"
echo "  Stop training:  bash kill_training.sh"
echo "=============================================="
echo ""

# ── Stream logs live in this terminal ────────────────────────────────────────
docker logs -f "${CONTAINER_NAME}"

# ── After container exits: run post-training analysis on host ─────────────────
echo ""
echo "=============================================="
echo "  Container exited. Running post-training analysis on host..."
echo "=============================================="

python3 "${SCRIPTS_DIR}/SAC/post_training_analysis.py" \
    2>&1 | sed 's|/workspace/catkin_ws/src/gtrl/scripts|'"${SCRIPTS_DIR}"'|g' \
    || echo "  WARNING: host-side analysis failed — check container results dir."

echo ""
echo "=============================================="
echo "  Done. Output files:"
echo "  Graphs + CSV: ${SCRIPTS_DIR}/SAC/results/"
echo "  Checkpoints:  ${SCRIPTS_DIR}/SAC/final_models/"
echo "  TensorBoard:  docker logs ${CONTAINER_NAME} | grep -i tensorboard"
echo "=============================================="
