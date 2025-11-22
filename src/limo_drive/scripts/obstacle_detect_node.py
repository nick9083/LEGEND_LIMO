#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from math import pi
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool


class Detect:
    """
    라이다로 전방 FOV 안에 일정 거리 이내 장애물이 있는지 판단하는 클래스.
    원래 A.Drive.py 안에 있던 Detect 로직을 거의 그대로 분리한 것.
    """

    def __init__(self):
        # 파라미터: 필요하면 launch에서 ~네임스페이스 기준으로 override
        self.FOV_degree = rospy.get_param("~fov_degree", 55.0)       # 전방 시야각(±deg)
        self.FOV_range = rospy.get_param("~fov_range", 0.45)          # 장애물 판단 거리(m)
        self.filter_degree = rospy.get_param("~filter_degree", 1.0)  # 잡음 필터용 최소 각도 차
        self.standard_degree = rospy.get_param("~standard_degree", 10.0)

        # 내부 상태
        self.msg = LaserScan()
        self.lidar_flag = False
        self.last_degree = 0.0
        self.nuber_of_object = 0  # 원 코드 그대로 이름 유지

        # 라이다 토픽 이름도 파라미터로 뺌 (기본은 /scan)
        scan_topic = rospy.get_param("~scan_topic", "/scan")
        rospy.Subscriber(scan_topic, LaserScan, self.laser_callback)

    def laser_callback(self, msg: LaserScan):
        # 처음 한 번만 각도 배열 계산
        if not self.lidar_flag:
            self.degrees = [
                (msg.angle_min + index * msg.angle_increment) * 180.0 / pi
                for index, _ in enumerate(msg.ranges)
            ]
            self.lidar_flag = True
        self.msg = msg

    def lidar_filter(self, msg: LaserScan):
        detect_degree = []
        for index, value in enumerate(msg.ranges):
            if (
                -self.FOV_degree < self.degrees[index] < self.FOV_degree
                and 0.0 < value < self.FOV_range
            ):
                current_range = value
                current_degree = self.degrees[index]

                # 너무 촘촘한 포인트는 하나로 취급
                if abs(current_degree - self.last_degree) > self.filter_degree:
                    self.last_degree = current_degree
                    detect_degree.append(current_degree)
        return detect_degree

    def count_object_function(self, detect_degree):
        count_objects = 0
        if len(detect_degree) != 0:
            count_objects = 1
            for index in range(len(detect_degree) - 1):
                between_degree = abs(
                    detect_degree[index] - detect_degree[index + 1]
                )
                if self.standard_degree < between_degree:
                    count_objects += 1
        return count_objects

    def status(self, count_objects):
        """
        원래 코드 그대로:
        - nuber_of_object 는 0에서 고정
        - 그래서 count_objects 가 0이 아니면 항상 True, 0이면 False
          => "장애물이 1개 이상 있냐 없냐" 플래그 역할을 함.
        """
        if self.nuber_of_object != count_objects:
            state = True
        else:
            state = False
        return state

    def think(self):
        """
        메인 판단 함수:
        - 라이다 필터링 → 물체 개수 계산 → 상태(True/False) 리턴
        """
        detect_degree = self.lidar_filter(self.msg)
        count_objects = self.count_object_function(detect_degree)
        rospy.logdebug("Obstacle count: %d", count_objects)

        state = self.status(count_objects)
        return state


class ObstacleDetectNode:
    """
    FSM 에서 쓸 sta 플래그를 퍼블리시하는 ROS 노드 래퍼.
    - 입력 : LaserScan (/scan)
    - 출력 : Bool (/obstacle_state)
    """

    def __init__(self):
        rospy.init_node("obstacle_detect_node")

        # Detect 로직 인스턴스
        self.detector = Detect()

        # 퍼블리셔: FSM 노드에서 이걸 구독해서 sta 로 사용
        out_topic = rospy.get_param("~out_topic", "/obstacle_state")
        self.pub_state = rospy.Publisher(out_topic, Bool, queue_size=1)

        self.rate = rospy.Rate(rospy.get_param("~rate", 10.0))

    def spin(self):
        while not rospy.is_shutdown():
            # 아직 라이다 데이터 안 들어왔으면 스킵
            if self.detector.lidar_flag:
                state = self.detector.think()
                msg = Bool()
                msg.data = state
                self.pub_state.publish(msg)
            self.rate.sleep()


if __name__ == "__main__":
    try:
        node = ObstacleDetectNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
