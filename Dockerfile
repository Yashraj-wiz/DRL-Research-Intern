# GoT-GTRL Training Environment
# Base: ROS Noetic + Gazebo 11 on Ubuntu 20.04
FROM osrf/ros:noetic-desktop-full

LABEL maintainer="GoT-GTRL Team"
LABEL description="ROS Noetic + Gazebo 11 + PyTorch for GoT-GTRL navigation training"

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=noetic

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # ROS tools
    python3-catkin-tools \
    python3-osrf-pycommon \
    python3-pip \
    python3-dev \
    # ROS packages needed by GoT-GTRL
    ros-noetic-cv-bridge \
    ros-noetic-ros-numpy \
    ros-noetic-gazebo-ros \
    ros-noetic-gazebo-plugins \
    ros-noetic-robot-state-publisher \
    ros-noetic-joint-state-publisher \
    ros-noetic-xacro \
    ros-noetic-rviz \
    ros-noetic-sensor-msgs \
    ros-noetic-nav-msgs \
    ros-noetic-geometry-msgs \
    ros-noetic-visualization-msgs \
    ros-noetic-gazebo-msgs \
    # Utilities
    git \
    wget \
    curl \
    nano \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# ── Python ML dependencies (Miniconda for clean Python 3.10 on Ubuntu 20.04) ─
# RTX 5080 (sm_120) needs CUDA 13.0+ and PyTorch 2.4+.
# Use Miniconda to avoid Ubuntu 20.04's Python 3.8 limitation.
RUN curl -sS https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh \
      -o miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH="/opt/conda/bin:${PATH}"

# Install PyTorch (latest stable — resolves at build time)
RUN pip install --no-cache-dir torch torchvision

# Install all Python dependencies for GoT-GTRL
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    tqdm \
    natsort \
    cpprb \
    einops \
    squaternion \
    opencv-python-headless \
    rospkg \
    defusedxml \
    catkin_pkg \
    empy==3.3.4 \
    tensorboard \
    matplotlib \
    pyyaml

# ── Fix cv_bridge / libffi conflict caused by conda shadowing system libs ────
# conda ships libp11-kit.so.0 which is missing the ffi_type_pointer symbol,
# causing 'ImportError: undefined symbol: ffi_type_pointer' when cv_bridge
# tries to import its boost module. Solution: remove conda's copy so the
# system /usr/lib version is used instead.
RUN rm -f /opt/conda/lib/libp11-kit.so.0* /opt/conda/lib/libp11-kit.so \
 && rm -f /opt/conda/lib/libffi.so.7* /opt/conda/lib/libffi.so \
 || true

# ── Workspace setup ──────────────────────────────────────────────────────────
WORKDIR /workspace

# Copy the catkin workspace into the container
COPY catkin_ws /workspace/catkin_ws

# ── Build the catkin workspace ────────────────────────────────────────────────
SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/noetic/setup.bash && \
    cd /workspace/catkin_ws && \
    catkin_make -DPYTHON_EXECUTABLE=/opt/conda/bin/python 2>&1 | tail -30

# ── Gazebo models ─────────────────────────────────────────────────────────────
# RRC1.world defines its geometry inline (plus a few Gazebo Fuel meshes fetched
# at runtime), so no local model library is needed; keep GAZEBO_MODEL_PATH valid.
RUN mkdir -p /root/.gazebo/models

# ── Create output directories ─────────────────────────────────────────────────
RUN mkdir -p \
    /workspace/catkin_ws/src/gtrl/scripts/SAC/final_models \
    /workspace/catkin_ws/src/gtrl/scripts/SAC/final_curves \
    /workspace/catkin_ws/src/gtrl/scripts/SAC/results \
    /workspace/catkin_ws/src/gtrl/scripts/SAC/runs

# ── Environment variables ─────────────────────────────────────────────────────
ENV GAZEBO_RESOURCE_PATH=/workspace/catkin_ws/src/gtrl/launch
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ros/noetic/lib
ENV GAZEBO_MODEL_PATH=/root/.gazebo/models
ENV QT_X11_NO_MITSHM=1
# Required to prevent Ignition Transport multicast crashes in headless mode:
ENV IGN_IP=127.0.0.1

# ── Entrypoint: source ROS + workspace ───────────────────────────────────────
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source /workspace/catkin_ws/devel/setup.bash" >> /root/.bashrc && \
    echo "export GAZEBO_RESOURCE_PATH=/workspace/catkin_ws/src/gtrl/launch" >> /root/.bashrc && \
    echo "export GAZEBO_MODEL_PATH=/root/.gazebo/models" >> /root/.bashrc && \
    echo "export QT_X11_NO_MITSHM=1" >> /root/.bashrc && \
    echo "export IGN_IP=127.0.0.1" >> /root/.bashrc

WORKDIR /workspace/catkin_ws/src/gtrl/scripts/SAC

# Default: drop into bash with everything sourced
CMD ["/bin/bash", "--rcfile", "/root/.bashrc"]
