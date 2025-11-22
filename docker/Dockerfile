# 1) ROS Melodic base (Ubuntu 18.04, arm64)
FROM ros:melodic-ros-base-bionic

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_PYTHON_VERSION=3

# 2) Python3 + 필수 라이브러리 + 빌드 도구 + TF
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip python3-dev \
      python3-numpy python3-yaml python3-opencv \
      build-essential cmake git \
      libboost-all-dev \
      ros-melodic-tf \
      && rm -rf /var/lib/apt/lists/*

# 3) ROS Python3 유틸 (catkin_tools 없이)
RUN pip3 install --no-cache-dir \
      rospkg catkin_pkg

# 4) Python3용 cv_bridge 빌드 (catkin_make 사용)
RUN mkdir -p /opt/cvbridge_ws/src
WORKDIR /opt/cvbridge_ws/src

# vision_opencv (cv_bridge 포함) clone
RUN git clone -b melodic https://github.com/ros-perception/vision_opencv.git

WORKDIR /opt/cvbridge_ws

# ROS 환경 소싱 + catkin_make로 cv_bridge만 빌드 + install
RUN /bin/bash -c "source /opt/ros/melodic/setup.bash && \
    catkin_make \
      -DPYTHON_EXECUTABLE=/usr/bin/python3 \
      -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
      -DPYTHON_LIBRARY=/usr/lib/aarch64-linux-gnu/libpython3.6m.so \
      --pkg cv_bridge && \
    catkin_make install"

# 5) 유저 워크스페이스 디렉토리 (호스트 마운트 위치에 맞춤)
RUN mkdir -p /root/LEGEND_WS/drive_ws/src
WORKDIR /root/LEGEND_WS/drive_ws

# 6) 엔트리포인트 스크립트
COPY entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]






