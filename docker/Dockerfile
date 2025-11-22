# Jetson Nano에서 자동으로 arm64v8 버전이 받아지는 공식 ROS 이미지
FROM ros:melodic-ros-base-bionic

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    ROS_PYTHON_VERSION=3

# 기본 패키지 + Python3 + OpenCV + catkin-tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      locales tzdata \
      python3-pip python3-all-dev python3-rospkg \
      python3-numpy python3-yaml python3-opencv \
      python3-catkin-tools \
      build-essential cmake git \
      libboost-all-dev \
      && rm -rf /var/lib/apt/lists/* && \
    locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8

# pip 쪽 ROS helper (rosdep 쪽에서 쓰는 rospkg, catkin_pkg)
RUN pip3 install --no-cache-dir rospkg catkin_pkg

# ==========================
#  Python3용 cv_bridge 빌드
# ==========================
# 참고: vision_opencv를 따로 워크스페이스에 가져와서 python3로 cv_bridge만 빌드하는 방식 :contentReference[oaicite:5]{index=5}
RUN mkdir -p /opt/cvbridge_ws/src
WORKDIR /opt/cvbridge_ws

# catkin workspace 설정 (python3 경로는 Jetson 기본값 기준: 3.6)
RUN catkin init && \
    catkin config \
      -DPYTHON_EXECUTABLE=/usr/bin/python3 \
      -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
      -DPYTHON_LIBRARY=/usr/lib/aarch64-linux-gnu/libpython3.6m.so \
      --install

# vision_opencv (cv_bridge) 소스 가져오기
WORKDIR /opt/cvbridge_ws/src
RUN git clone -b melodic https://github.com/ros-perception/vision_opencv.git

WORKDIR /opt/cvbridge_ws
# 필요하면 apt에 깔린 cv_bridge 버전 확인 후 맞춰서 checkout 할 수도 있음
#   apt-cache show ros-melodic-cv-bridge | grep Version
RUN catkin build cv_bridge

# ROS + cv_bridge 환경 자동 로드
RUN echo "source /opt/ros/melodic/setup.bash" >> /root/.bashrc && \
    echo "source /opt/cvbridge_ws/install/setup.bash --extend" >> /root/.bashrc && \
    echo "export ROS_PYTHON_VERSION=3" >> /root/.bashrc

# 유저 프로젝트 워크스페이스 디렉토리 (컨테이너 내부 기준)
RUN mkdir -p /root/LEGEND_WS/drive_ws/src
WORKDIR /root/LEGEND_WS/drive_ws

# entrypoint 스크립트
COPY entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]

