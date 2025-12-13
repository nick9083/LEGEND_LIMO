#!/usr/bin/env python3 

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from math import *


class Limo_obstacle_avoidence:
    def __init__(self):
        # 노드 이름만 변경
        rospy.init_node("base_obstacle_avoid")

        # === 파라미터 ===
        scan_topic   = rospy.get_param("~scan_topic", "/scan")
        cmd_topic    = rospy.get_param("~cmd_topic", "/cmd_vel_obstacle")
        enable_topic = rospy.get_param("~enable_topic", "/obstacle_enable")

        # 구독 / 퍼블리시
        rospy.Subscriber(scan_topic, LaserScan, self.laser_callback, queue_size=1)
        rospy.Subscriber(enable_topic, Bool, self.enable_cb, queue_size=1)

        self.pub = rospy.Publisher(cmd_topic, Twist, queue_size=3)
        self.rate = rospy.Rate(30)

        self.cmd_vel_msg = Twist()

        self.msg = None
        self.lidar_flag = False
        self.dist_data = 0
        self.direction = None
        self.is_scan = False

        self.obstacle_ranges = []
        self.center_list_left = []
        self.center_list_right = []

        self.scan_dgree = 50
        self.min_dist = 0.2

        self.speed = 0
        self.angle = 0
        self.default_speed = 0.15
        self.default_angle = 0.0
        self.turning_speed = 0.08
        self.backward_speed = -0.08

        self.OBSTACLE_PERCEPTION_BOUNDARY = 10

        self.ranges_length = None

        # FSM에서 오는 enable 플래그
        self.enabled = True  # FSM 없을 때 단독 테스트용으로 기본 True

    # --------------------------------
    # enable 콜백 (FSM -> base avoid on/off)
    # --------------------------------
    def enable_cb(self, msg: Bool):
        self.enabled = msg.data

        # 꺼질 때는 한 번 0 명령을 보내서 FSM의 last_obstacle_cmd 도 0으로 덮어줌
        if not self.enabled:
            self.cmd_vel_msg.linear.x = 0.0
            self.cmd_vel_msg.angular.z = 0.0
            self.pub.publish(self.cmd_vel_msg)

    # --------------------------------
    # LiDAR 콜백
    # --------------------------------
    def laser_callback(self, msg):
        self.msg = msg

        if len(self.obstacle_ranges) > self.OBSTACLE_PERCEPTION_BOUNDARY:
            self.obstacle_exit = True
        else:
            self.obstacle_exit = False

        self.is_scan = True
        self.obstacle_ranges = []

    # --------------------------------
    # LiDAR 스캔 처리
    # --------------------------------
    def LiDAR_scan(self):
        obstacle = []

        if not self.lidar_flag:
            self.degrees = [
                (self.msg.angle_min + (i * self.msg.angle_increment)) * 180.0 / pi
                for i, data in enumerate(self.msg.ranges)
            ]
            self.ranges_length = len(self.msg.ranges)
            self.lidar_flag = True

        for i, data in enumerate(self.msg.ranges):
            if 0 < data < 0.3 and -self.scan_dgree < self.degrees[i] < self.scan_dgree:
                obstacle.append(i)
                self.dist_data = data

        if obstacle:
            first = obstacle[0]
            first_dst = first
            last = obstacle[-1]
            last_dst = self.ranges_length - last
            self.obstacle_ranges = self.msg.ranges[first : last + 1]
        else:
            first, first_dst, last, last_dst = 0, 0, 0, 0

        return first, first_dst, last, last_dst

    # --------------------------------
    # 장애물 방향에 따른 속도/조향 결정
    # --------------------------------
    def move_direction(self, last, first):
        # 매 루프에서 center 리스트는 비워주자 (안 비우면 계속 쌓임)
        self.center_list_left = []
        self.center_list_right = []

        if self.direction == "right":
            # 오른쪽 장애물 → 왼쪽 gap 중심으로
            for i in range(first):
                self.center_list_left.append(i)
            if self.center_list_left:
                Lcenter = self.center_list_left[floor(first / 2)]
                center_angle_left = -self.msg.angle_increment * Lcenter
                self.angle = center_angle_left
            else:
                self.angle = self.default_angle
            self.speed = self.default_speed

        elif self.direction == "left":
            # 왼쪽 장애물 → 오른쪽 gap 중심으로
            for i in range(len(self.msg.ranges) - last):
                self.center_list_right.append(last + i)
            if self.center_list_right:
                Rcenter = self.center_list_right[
                    floor((len(self.center_list_right) - 1) / 2.0)
                ]
                center_angle_right = self.msg.angle_increment * Rcenter
                self.angle = center_angle_right / 2.5
            else:
                self.angle = self.default_angle
            self.speed = self.default_speed

        elif self.direction == "right_back":
            # If there is no gap ahead, move backward
            self.angle = -7
            self.speed = -0.15
        elif self.direction == "left_back":
            self.angle = 7
            self.speed = -0.15

        elif self.direction == "back":
            self.angle = self.default_angle
            self.speed = self.backward_speed
        else:
            # 장애물 없으면 직진
            self.angle = self.default_angle
            self.speed = self.default_speed

    # --------------------------------
    # 공간 비교 (어느 쪽으로 피할지)
    # --------------------------------
    def compare_space(self, first_dst, last_dst):
        if self.obstacle_exit:
            if first_dst > last_dst and self.dist_data > self.min_dist:
                self.direction = "right"
            elif first_dst < last_dst and self.dist_data > self.min_dist:
                self.direction = "left"
            elif first_dst > last_dst and self.dist_data < self.min_dist:
                self.direction = "right_back"
            elif first_dst < last_dst and self.dist_data < self.min_dist:
                self.direction = "left_back"
        else:
            self.direction = "front"

    # --------------------------------
    # 메인 루프
    # --------------------------------
    def main(self):
        # FSM에서 disable이면 연산 스킵 + 0속도 유지
        if not self.enabled:
            self.cmd_vel_msg.linear.x = 0.0
            self.cmd_vel_msg.angular.z = 0.0
            self.pub.publish(self.cmd_vel_msg)
            self.rate.sleep()
            return

        # LiDAR 스캔 아직 없음
        if not self.is_scan or self.msg is None:
            self.cmd_vel_msg.linear.x = 0.0
            self.cmd_vel_msg.angular.z = 0.0
            self.pub.publish(self.cmd_vel_msg)
            self.rate.sleep()
            return

        # 1) LiDAR 스캔 처리
        first, first_dst, last, last_dst = self.LiDAR_scan()

        # 2) 어느 쪽 공간이 더 넓은지
        self.compare_space(first_dst, last_dst)

        # 3) 방향에 따라 속도/조향 결정
        self.move_direction(last, first)

        # 4) cmd_vel 메시지 채우고 publish
        self.cmd_vel_msg.linear.x = self.speed
        self.cmd_vel_msg.angular.z = self.angle
        self.pub.publish(self.cmd_vel_msg)

        self.rate.sleep()


if __name__ == "__main__":
    node = Limo_obstacle_avoidence()
    try:
        while not rospy.is_shutdown():
            node.main()
    except rospy.ROSInterruptException:
        pass
