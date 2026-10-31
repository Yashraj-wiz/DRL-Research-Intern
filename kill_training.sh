#!/bin/bash
# ============================================================
# kill_training.sh — GoT-GTRL Clean Shutdown
# ============================================================
# Gracefully stops the training container and cleans up
# any lingering ROS / Gazebo processes.
#
# Usage:
#   bash kill_training.sh
# ============================================================

CONTAINER_NAME="gotgtrl_training"

echo "=============================================="
echo "  GoT-GTRL Training Shutdown"
echo "=============================================="

echo "[1/3] Stopping Docker container '${CONTAINER_NAME}'..."
docker stop "${CONTAINER_NAME}" 2>/dev/null && echo "  Container stopped." || echo "  Container not running."

echo "[2/3] Removing container..."
docker rm "${CONTAINER_NAME}" 2>/dev/null && echo "  Container removed." || echo "  Nothing to remove."

echo "[3/3] Killing any lingering host-side processes..."
pkill -f "tensorboard" 2>/dev/null || true

echo ""
echo "  Done. Training stack is fully stopped."
echo "=============================================="

echo ""
echo "=============================================="
echo "  Displaying Training Plot..."
echo "=============================================="
python3 plot_results.py || true
