#!/bin/bash
# ============================================================
# smoke_test.sh — GoT-GTRL Full Stack Smoke Test
# ============================================================
# Tests: XACRO parse, Python imports, Gazebo launch, ROS topics
# Usage: bash smoke_test.sh
# ============================================================

set +e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="gotgtrl:noetic"
PASS=0; FAIL=0

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  [PASS]${NC} $1"; ((PASS++)); }
fail() { echo -e "${RED}  [FAIL]${NC} $1"; ((FAIL++)); }
hdr()  { echo -e "\n${YELLOW}══ $1 ══${NC}"; }

# ── 0. Pre-flight ─────────────────────────────────────────────────────────────
hdr "0. Pre-flight"
if sg docker -c "docker image inspect $IMAGE" > /dev/null 2>&1; then
    ok "Docker image '$IMAGE' exists"
else
    fail "Docker image '$IMAGE' not found — run: docker build -t $IMAGE ."
    echo "Cannot continue without image. Exiting."
    exit 1
fi

# ── 1. XACRO Parse ────────────────────────────────────────────────────────────
hdr "1. XACRO Parse Test"

for xacro_file in \
    "/workspace/catkin_ws/src/gtrl/xacro/bringup/bringup_limo.xacro" \
    "/workspace/catkin_ws/src/gtrl/xacro/bringup/bringup_scout.xacro"; do

    name=$(basename $xacro_file)
    result=$(sg docker -c "docker run --rm $IMAGE /bin/bash -c \
        'source /opt/ros/noetic/setup.bash && \
         source /workspace/catkin_ws/devel/setup.bash && \
         rosrun xacro xacro $xacro_file > /dev/null 2>&1 && echo OK || echo FAIL'" 2>&1)
    if echo "$result" | grep -q "^OK"; then
        ok "$name parses cleanly"
    else
        fail "$name — parse error: $result"
    fi
done

# ── 2. Python / ML Script Tests ───────────────────────────────────────────────
hdr "2. Python / ML Script Tests"

py_result=$(sg docker -c "docker run --rm --gpus all \
    -v $REPO/catkin_ws/src/gtrl/scripts/SAC/runs:/workspace/catkin_ws/src/gtrl/scripts/SAC/runs \
    -v $REPO/catkin_ws/src/gtrl/scripts/SAC/verify_day3.py:/workspace/catkin_ws/src/gtrl/scripts/SAC/verify_day3.py \
    $IMAGE \
    /bin/bash -c 'source /opt/ros/noetic/setup.bash && \
    source /workspace/catkin_ws/devel/setup.bash && \
    python3 /workspace/catkin_ws/src/gtrl/scripts/SAC/verify_day3.py 2>&1'" 2>&1)

echo "$py_result"

if echo "$py_result" | grep -q "STEP 1: PASS"; then
    ok "GoT actor forward pass (CUDA, shape 4×2)"
else
    fail "GoT actor forward pass"
fi
if echo "$py_result" | grep -q "STEP 2: PASS"; then
    ok "TensorBoard write + volume mount"
else
    fail "TensorBoard write"
fi

# ── 3. main.py Import Test ────────────────────────────────────────────────────
hdr "3. main.py Import Test"

import_result=$(sg docker -c "docker run --rm --gpus all $IMAGE \
    /bin/bash -c 'source /opt/ros/noetic/setup.bash && \
    source /workspace/catkin_ws/devel/setup.bash && \
    cd /workspace/catkin_ws/src/gtrl/scripts/SAC && \
    python3 -c \"
import sys
sys.path.insert(0, \\\"/workspace/catkin_ws/src/gtrl/scripts\\\")
import yaml, torch, numpy, cv2, squaternion, einops, cpprb
from SAC.GoalFormer import GoT
from SAC.got_sac_network import GoTPolicy, QNetwork
from SAC.DRL import SAC
print(\\\"ALL IMPORTS OK\\\")
print(\\\"PyTorch:\\\", torch.__version__)
print(\\\"CUDA:\\\", torch.cuda.is_available())
\" 2>&1'" 2>&1)

echo "$import_result"
if echo "$import_result" | grep -q "ALL IMPORTS OK"; then
    ok "All Python imports (GoT, SAC, DRL, cv2, cpprb)"
else
    fail "Python import error — check above"
fi

# ── 4. Gazebo + ROS Topics Test (headless) ────────────────────────────────────
hdr "4. Gazebo Launch + ROS Topics Test (headless, ~25s)"

echo "  Starting container..."
sg docker -c "docker stop gotgtrl_smoke 2>/dev/null; docker rm gotgtrl_smoke 2>/dev/null" > /dev/null 2>&1 || true

sg docker -c "docker run -d \
    --name gotgtrl_smoke \
    --gpus all \
    --net=host --ipc=host \
    -e IGN_IP=127.0.0.1 \
    -e QT_X11_NO_MITSHM=1 \
    $IMAGE \
    /bin/bash -c '
        source /opt/ros/noetic/setup.bash
        source /workspace/catkin_ws/devel/setup.bash
        export GAZEBO_MODEL_PATH=/root/.gazebo/models
        export IGN_IP=127.0.0.1
        roscore -p 11311 &
        sleep 5
        roslaunch gtrl main_limo.launch gui:=false paused:=false &
        sleep 30
        echo TOPICS_START
        rostopic list
        echo TOPICS_END
        sleep 5
    '" > /dev/null

echo "  Waiting 40s for Gazebo to load..."
sleep 40

topic_output=$(sg docker -c "docker logs gotgtrl_smoke 2>&1")
echo ""
echo "  Topic list from container:"
echo "$topic_output" | sed -n '/TOPICS_START/,/TOPICS_END/p' | grep -v "TOPICS_"

required_topics="/limo/cmd_vel /limo/odom /camera/fisheye/image_raw /front_laser/scan"
for topic in $required_topics; do
    if echo "$topic_output" | grep -q "$topic"; then
        ok "Topic active: $topic"
    else
        fail "Topic MISSING: $topic"
    fi
done

echo "  Cleaning up container..."
sg docker -c "docker stop gotgtrl_smoke 2>/dev/null; docker rm gotgtrl_smoke 2>/dev/null" > /dev/null 2>&1 || true

# ── 5. config.yaml sanity check ───────────────────────────────────────────────
hdr "5. config.yaml Sanity Check"

config="$REPO/catkin_ws/src/gtrl/scripts/SAC/config.yaml"
if [ -f "$config" ]; then
    ok "config.yaml exists"
    pre_train=$(grep "PRE_TRAIN" $config | awk '{print $NF}')
    robot=$(grep "ROBOT" $config | awk '{print $NF}')
    if [ "$pre_train" = "False" ]; then ok "PRE_TRAIN: False (training from scratch)"; else fail "PRE_TRAIN unexpected: $pre_train"; fi
else
    fail "config.yaml not found"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}══════════════════════════════════════${NC}"
echo -e "  SMOKE TEST RESULTS"
echo -e "${GREEN}  PASS: $PASS${NC}  |  ${RED}FAIL: $FAIL${NC}"
echo -e "${YELLOW}══════════════════════════════════════${NC}"

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}  All checks passed — ready for training!${NC}"
    echo "  Run: bash run_training.sh"
else
    echo -e "${RED}  $FAIL check(s) failed — fix before training.${NC}"
    exit 1
fi
