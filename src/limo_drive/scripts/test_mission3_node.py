#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Bool


# ---------------- 공통 유틸 ----------------

def sanitize_range(r, range_max):
    """NaN / inf / 0 / 음수 → range_max 로 보정"""
    if r is None:
        return range_max
    if r != r:                # NaN 체크
        return range_max
    if r <= 0.01 or math.isinf(r):
        return range_max
    return r


def find_longest_free_gap_angle(msg, min_dist=1.0, fov_deg=120.0):
    """
    LaserScan에서 전방 fov_deg 범위 안에서
    min_dist(m) 이상 떨어진 구간들 중
    가장 길게 연속된 구간(gap)의 중심 각도(rad, ROS 기준: 왼쪽 +) 리턴

    여기서 min_dist = 1.0 → "100cm 앞까지 빈공간" 기준
    """
    half = math.radians(fov_deg / 2.0)
    n = len(msg.ranges)

    best_start = None
    best_len = 0
    cur_start = None

    for i in range(n):
        angle = msg.angle_min + i * msg.angle_increment

        # FOV 밖이면 gap 끊기
        if angle < -half or angle > half:
            if cur_start is not None:
                cur_len = i - cur_start
                if cur_len > best_len:
                    best_len = cur_len
                    best_start = cur_start
                cur_start = None
            continue

        r = sanitize_range(msg.ranges[i], msg.range_max)
        safe = r > min_dist  # 1m 이상이면 "멀리까지 빈공간"

        if safe:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                cur_len = i - cur_start
                if cur_len > best_len:
                    best_len = cur_len
                    best_start = cur_start
                cur_start = None

    # 마지막까지 이어진 gap 처리
    if cur_start is not None:
        cur_len = n - cur_start
        if cur_len > best_len:
            best_len = cur_len
            best_start = cur_start

    # 안전 구간 없음 → 정면 0
    if best_start is None or best_len == 0:
        return 0.0

    center_idx = best_start + best_len // 2
    center_angle = msg.angle_min + center_idx * msg.angle_increment
    return center_angle  # ROS 기준: 왼쪽 +, 오른쪽 -


# ---------------- 미션3 노드 ----------------

class Mission3GapAndEmergencyAvoidNode(object):
    def __init__(self):
        # 노드 이름만 미션3용으로
        rospy.init_node("mission3_node")

        # ===== 파라미터 =====
        # 토픽 이름들 (★ 기본값만 미션3용으로 변경)
        self.scan_topic      = rospy.get_param("~scan_topic", "/scan")
        self.cmd_topic       = rospy.get_param("~cmd_topic", "/cmd_vel_obstacle_m3")
        self.enable_topic    = rospy.get_param("~enable_topic", "/mission3_enable")
        self.debug_deg_topic = rospy.get_param("~debug_deg_topic",
                                               "/mission3_free_gap_angle_deg")

        # 멀리 보는 gap 기준 (알고리즘/값 그대로)
        self.free_dist      = rospy.get_param("~free_dist", 0.9)       # 90cm
        self.fov_deg        = rospy.get_param("~fov_deg", 120.0)       # 전방 ±60도
        self.linear_speed   = rospy.get_param("~linear_speed", 0.40)   # 평상시 전진 속도
        self.k_ang          = rospy.get_param("~k_ang", 1.0)           # gap 조향 gain
        self.max_yaw        = rospy.get_param("~max_yaw", 1.0)         # 최대 조향 속도

        # 30cm 안 emergency 회피 기준 (그대로)
        self.emergency_dist = rospy.get_param("~emergency_dist", 0.30)  # 30cm
        self.min_dist_back  = rospy.get_param("~min_dist_back", 0.20)   # 이 이하면 뒤로
        self.scan_degree    = rospy.get_param("~scan_degree", 55.0)     # ±scan_degree 내만 근접장애물 판단

        self.default_speed   = 0.25
        self.backward_speed  = rospy.get_param("~backward_speed", 0.15)

        # LiDAR 관련 상태
        self.lidar_flag      = False
        self.degrees         = []
        self.ranges_length   = 0
        self.dist_data       = 999.0
        self.obstacle_ranges = []
        self.direction       = "front"  # front / right / left / right_back / left_back / back

        # enable (FSM에서 /mission3_enable 들어옴) ★
        self.enabled         = True

        # 퍼블리셔
        self.cmd_pub         = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.debug_deg_pub   = rospy.Publisher(self.debug_deg_topic, Float32, queue_size=1)

        # 서브스크라이버
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_cb, queue_size=1)
        rospy.Subscriber(self.enable_topic, Bool, self.enable_cb, queue_size=1)

        rospy.loginfo("Mission3GapAndEmergencyAvoidNode started.")
        rospy.loginfo("  scan_topic=%s, cmd_topic=%s, enable_topic=%s",
                      self.scan_topic, self.cmd_topic, self.enable_topic)

    def enable_cb(self, msg: Bool):
        """
        FSM MUX 에서 오는 /mission3_enable 신호
        True  → 연산/회피 활성
        False → 이 노드 계산 스킵 (CPU 아낌)
        """
        self.enabled = msg.data

    # ---------- LiDAR_scan 그대로 ----------
    def LiDAR_scan(self, l_msg):
        obstacle_idx = []

        # 최초 1회 각도 테이블 생성 (deg)
        if not self.lidar_flag:
            self.degrees = [
                (l_msg.angle_min + i * l_msg.angle_increment) * 180.0 / math.pi
                for i in range(len(l_msg.ranges))
            ]
            self.ranges_length = len(l_msg.ranges)
            self.lidar_flag = True

        # 근접 장애물 찾기 (0 < data < emergency_dist, ±scan_degree 안)
        min_obstacle_dist = l_msg.range_max

        for i, data in enumerate(l_msg.ranges):
            if 0.0 < data < self.emergency_dist and \
               -self.scan_degree < self.degrees[i] < self.scan_degree:
                obstacle_idx.append(i)
                # 가장 가까운 장애물 거리 기록
                if data < min_obstacle_dist:
                    min_obstacle_dist = data

        has_obstacle = False
        first = first_dst = last = last_dst = 0

        if obstacle_idx:
            has_obstacle = True
            first = obstacle_idx[0]
            first_dst = first
            last = obstacle_idx[-1]
            last_dst = self.ranges_length - last
            self.obstacle_ranges = l_msg.ranges[first:last + 1]
            self.dist_data = min_obstacle_dist
        else:
            self.obstacle_ranges = []
            self.dist_data = l_msg.range_max

        return first, first_dst, last, last_dst, has_obstacle

    # ---------- 방향 결정 그대로 ----------
    def decide_direction(self, first_dst, last_dst, has_obstacle):
        if has_obstacle:
            # first_dst > last_dst → 오른쪽으로 공간 많다 → 오른쪽으로 회피
            if first_dst > last_dst and self.dist_data > self.min_dist_back:
                self.direction = "right"
            elif first_dst < last_dst and self.dist_data > self.min_dist_back:
                self.direction = "left"
            # 너무 가까우면 뒤로 + 회피
            elif first_dst > last_dst and self.dist_data <= self.min_dist_back:
                self.direction = "right_back"
            elif first_dst < last_dst and self.dist_data <= self.min_dist_back:
                self.direction = "left_back"
        else:
            self.direction = "front"

    # ---------- direction → Twist 그대로 ----------
    def make_cmd_from_direction(self, free_gap_angle_ros):
        """
        free_gap_angle_ros : 멀리(1m) 기준으로 찾은 gap 각도(rad, 왼쪽+)
        self.direction     : LiDAR_scan + decide_direction 에서 결정된 값
        """
        cmd = Twist()

        if self.direction == "front":
            # ☆ 정상 모드: 1m 기준 가장 긴 gap 방향으로 전진
            ang_z = self.k_ang * free_gap_angle_ros
            if ang_z > self.max_yaw:
                ang_z = self.max_yaw
            elif ang_z < -self.max_yaw:
                ang_z = -self.max_yaw

            cmd.linear.x = self.default_speed
            cmd.angular.z = ang_z

        elif self.direction == "right":
            # 오른쪽으로 회피 (ROS 기준 오른쪽은 -)
            cmd.linear.x = self.default_speed * 0.5
            cmd.angular.z = -self.max_yaw * 0.7

        elif self.direction == "left":
            # 왼쪽으로 회피
            cmd.linear.x = self.default_speed * 0.5
            cmd.angular.z = self.max_yaw * 0.7

        elif self.direction == "right_back":
            # 오른쪽으로 돌면서 후진
            cmd.linear.x = -self.backward_speed
            cmd.angular.z = -self.max_yaw * 0.7

        elif self.direction == "left_back":
            # 왼쪽으로 돌면서 후진
            cmd.linear.x = -self.backward_speed
            cmd.angular.z = self.max_yaw * 0.7

        elif self.direction == "back":
            # 그냥 직진 후진 (필요하면 사용)
            cmd.linear.x = -self.backward_speed
            cmd.angular.z = 0.0

        else:
            # 안전빵: 멈춤
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        return cmd

    # ---------- 메인 콜백 그대로 (enable만 mission3 기준) ----------
    def scan_cb(self, msg: LaserScan):
        # FSM에서 mission3_disable 이면 계산 스킵
        if not self.enabled:
            return

        # 1) 100cm 기준 가장 긴 빈공간 각도 찾기 (멀리용)
        free_angle_ros = find_longest_free_gap_angle(
            msg,
            min_dist=self.free_dist,
            fov_deg=self.fov_deg
        )

        # 2) 30cm 안 근접 장애물 스캔
        first, first_dst, last, last_dst, has_obstacle = self.LiDAR_scan(msg)

        # 3) 근접 장애물이 있으면 direction 결정 / 없으면 front
        self.decide_direction(first_dst, last_dst, has_obstacle)

        # 4) 디버그: 네가 원하는 기준(정면0, 오른쪽+, 왼쪽-) 각도 publish
        deg_ros = free_angle_ros * 180.0 / math.pi    # ROS: 왼쪽+
        deg_user = -deg_ros                           # USER: 오른쪽+
        if deg_user > 60.0:
            deg_user = 60.0
        elif deg_user < -60.0:
            deg_user = -60.0

        dbg = Float32()
        dbg.data = deg_user
        self.debug_deg_pub.publish(dbg)

        rospy.loginfo(
            "[Mission3] dir=%s, free_gap=ROS %.1fdeg (left+), USER %.1fdeg (right+), dist_data=%.3f",
            self.direction, deg_ros, deg_user, self.dist_data
        )

        # 5) direction + free_gap_angle 로 최종 cmd_vel 생성
        cmd = self.make_cmd_from_direction(free_angle_ros)
        self.cmd_pub.publish(cmd)


if __name__ == "__main__":
    try:
        node = Mission3GapAndEmergencyAvoidNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
