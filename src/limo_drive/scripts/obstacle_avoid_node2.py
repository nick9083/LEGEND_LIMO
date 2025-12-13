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


def find_longest_free_gap_angle(msg,
                                min_dist=1.0,
                                fov_deg=120.0,
                                min_gap_width_m=0.0,
                                max_gap_width_m=999.0):
    """
    LaserScan에서 전방 fov_deg 범위 안에서
    min_dist(m) 이상 떨어진 구간들 중 gap들을 찾는다.

    - chosen_* : gap 폭이 [min_gap_width_m, max_gap_width_m] 안에 있는
                 gap들 중에서 가장 넓은 것
                 (없으면 chosen_center=0.0, chosen_width=0.0)
    - longest_*: 폭 조건 상관 없이, 전방에서 가장 '길이(인덱스 개수)'가 긴 gap

    리턴:
      (chosen_center_angle, chosen_width_m,
       longest_center_angle, longest_width_m)

    각도 단위는 rad(ROS 기준: 왼쪽 +)
    폭 단위는 m (대충 min_dist * 각도폭)
    """
    half = math.radians(fov_deg / 2.0)
    n = len(msg.ranges)

    # 조건 만족 gap 중 가장 넓은 것
    best_start = None
    best_len = 0
    best_width_m = 0.0

    # 디버그용: 조건 무시하고 가장 긴 gap
    fb_start = None
    fb_len = 0
    fb_width_m = 0.0

    cur_start = None

    def process_gap(start_idx, end_idx_exclusive):
        nonlocal best_start, best_len, best_width_m
        nonlocal fb_start, fb_len, fb_width_m

        length = end_idx_exclusive - start_idx
        if length <= 0:
            return

        angle_start = msg.angle_min + start_idx * msg.angle_increment
        angle_end   = msg.angle_min + (end_idx_exclusive - 1) * msg.angle_increment
        angle_width = abs(angle_end - angle_start)  # rad

        width_m = min_dist * angle_width

        # --- longest gap (조건 무시) 기록 ---
        if length > fb_len:
            fb_len = length
            fb_start = start_idx
            fb_width_m = width_m

        # --- 폭 조건 체크 ---
        if width_m < min_gap_width_m or width_m > max_gap_width_m:
            return

        # 조건 만족 gap 중 가장 넓은 것
        if width_m > best_width_m:
            best_width_m = width_m
            best_start = start_idx
            best_len = length

    # 메인 루프: safe 구간(gap) 찾기
    for i in range(n):
        angle = msg.angle_min + i * msg.angle_increment

        # FOV 밖이면 gap 끊기
        if angle < -half or angle > half:
            if cur_start is not None:
                process_gap(cur_start, i)
                cur_start = None
            continue

        r = sanitize_range(msg.ranges[i], msg.range_max)
        safe = r > min_dist  # min_dist 이상이면 "멀리까지 빈공간"

        if safe:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                process_gap(cur_start, i)
                cur_start = None

    # 마지막까지 이어진 gap 처리
    if cur_start is not None:
        process_gap(cur_start, n)

    # gap 자체가 하나도 없는 경우
    if fb_start is None or fb_len == 0:
        return 0.0, 0.0, 0.0, 0.0

    # longest gap 중심각
    fb_center_idx = fb_start + fb_len // 2
    fb_center_angle = msg.angle_min + fb_center_idx * msg.angle_increment

    # 조건 만족 gap이 없으면 chosen은 0
    if best_start is None or best_len == 0:
        chosen_center_angle = 0.0
        chosen_width_m = 0.0
    else:
        chosen_center_idx = best_start + best_len // 2
        chosen_center_angle = msg.angle_min + chosen_center_idx * msg.angle_increment
        chosen_width_m = best_width_m

    longest_center_angle = fb_center_angle
    longest_width_m = fb_width_m

    return chosen_center_angle, chosen_width_m, longest_center_angle, longest_width_m


# ---------------- 메인 클래스 ----------------

class Limo_obstacle_avoidence:
    def __init__(self):
        rospy.init_node("base_obstacle_avoid")

        # ===== 토픽/게이팅 =====
        self.scan_topic      = rospy.get_param("~scan_topic", "/scan")
        self.cmd_topic       = rospy.get_param("~cmd_topic", "/cmd_vel_obstacle")
        self.enable_topic    = rospy.get_param("~enable_topic", "/obstacle_enable")
        self.debug_deg_topic = rospy.get_param("~debug_deg_topic", "/free_gap_angle_deg")

        # ===== gap 기반 주행 파라미터 =====
        self.free_dist      = rospy.get_param("~free_dist", 0.7)
        self.fov_deg        = rospy.get_param("~fov_deg", 100.0)
        self.linear_speed   = rospy.get_param("~linear_speed", 0.20)
        self.k_ang          = rospy.get_param("~k_ang", 1.0)
        self.max_yaw        = rospy.get_param("~max_yaw", 1.0)
        self.min_gap_width_m = rospy.get_param("~min_gap_width_m", 0.20)
        self.max_gap_width_m = rospy.get_param("~max_gap_width_m", 0.60)

        # ===== 이머전시(근접) 파라미터 =====
        # (아래 로직은 1번 코드의 LiDAR_scan/decide_direction에서 그대로 사용)
        self.emergency_dist = rospy.get_param("~emergency_dist", 0.30)
        self.min_dist_back  = rospy.get_param("~min_dist_back", 0.17)
        self.scan_degree    = rospy.get_param("~scan_degree", 60.0)

        self.default_speed  = self.linear_speed
        self.backward_speed = rospy.get_param("~backward_speed", 0.15)

        # ===== LiDAR 상태 =====
        self.lidar_flag      = False
        self.degrees         = []
        self.ranges_length   = 0
        self.dist_data       = 999.0
        self.obstacle_ranges = []
        self.direction       = "front"  # front / right / left / right_back / left_back / back

        # 마지막으로 선택된 gap 폭
        self.last_chosen_width = 0.0

        # enable (FSM)
        self.enabled = True

        # pub/sub
        self.cmd_pub       = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.debug_deg_pub = rospy.Publisher(self.debug_deg_topic, Float32, queue_size=1)

        rospy.Subscriber(self.scan_topic,   LaserScan, self.scan_cb,   queue_size=1)
        rospy.Subscriber(self.enable_topic, Bool,       self.enable_cb, queue_size=1)

        rospy.loginfo("base_obstacle_avoid started.")
        rospy.loginfo("  scan_topic=%s, cmd_topic=%s, enable_topic=%s",
                      self.scan_topic, self.cmd_topic, self.enable_topic)

    # ---------- enable 콜백 ----------
    def enable_cb(self, msg: Bool):
        """
        FSM MUX 에서 오는 /obstacle_enable 신호
        True  → 연산/회피 활성
        False → 이 노드 계산 스킵 (CPU 아낌)
        """
        self.enabled = msg.data

        # 꺼질 때는 한 번 0 명령을 보내서 FSM의 last_obstacle_cmd 도 0으로 덮어줌
        if not self.enabled:
            z = Twist()
            z.linear.x = 0.0
            z.angular.z = 0.0
            self.cmd_pub.publish(z)

    # ---------- LiDAR_scan: 근접 장애물 ----------
    # ★★★★★ 여기부터 이머전시 로직: 1번 코드 그대로 (변경 금지) ★★★★★
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

    # ---------- 방향 결정 ----------
    def decide_direction(self, first_dst, last_dst, has_obstacle):
        if has_obstacle:
            # first_dst > last_dst → 오른쪽에 더 공간 → 오른쪽 회피
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
    # ★★★★★ 여기까지 이머전시 로직: 1번 코드 그대로 (변경 금지) ★★★★★

    # ---------- direction 에 따라 Twist 생성 ----------
    def make_cmd_from_direction(self, free_gap_angle_ros):
        """
        free_gap_angle_ros : 선택된 gap의 중심각(rad, 왼쪽+)
        self.last_chosen_width : 그 gap 폭 (0이면 조건 만족 gap 없음)
        """
        cmd = Twist()

        if self.direction == "front":
            # 조건 만족 gap 이 없으면 => 안전하게 직진/정지 (원하는 대로 조절)
            if self.last_chosen_width <= 0.0:
                cmd.linear.x = 0.15   # 살짝 직진만 (원하면 0.0으로 꺼도 됨)
                cmd.angular.z = 0.0
                return cmd

            # 정상 모드: 선택된 gap 방향으로 전진
            ang_z = self.k_ang * free_gap_angle_ros
            if ang_z > self.max_yaw:
                ang_z = self.max_yaw
            elif ang_z < -self.max_yaw:
                ang_z = -self.max_yaw

            cmd.linear.x = self.default_speed
            cmd.angular.z = ang_z

        elif self.direction == "right":
            cmd.linear.x = self.default_speed * 0.5
            cmd.angular.z = -self.max_yaw * 0.7

        elif self.direction == "left":
            cmd.linear.x = self.default_speed * 0.5
            cmd.angular.z = self.max_yaw * 0.7

        elif self.direction == "right_back":
            cmd.linear.x = -self.backward_speed
            cmd.angular.z = -self.max_yaw * 0.7

        elif self.direction == "left_back":
            cmd.linear.x = -self.backward_speed
            cmd.angular.z = self.max_yaw * 0.7

        elif self.direction == "back":
            cmd.linear.x = -self.backward_speed
            cmd.angular.z = 0.0

        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        return cmd

    # ---------- 메인 콜백 ----------
    def scan_cb(self, msg: LaserScan):
        # FSM에서 obstacle_disable 이면 계산 스킵
        if not self.enabled:
            return

        # 1) gap 정보 (조건 만족 gap + longest gap)
        (free_angle_ros,
         chosen_width_m,
         longest_center_ros,
         longest_width_m) = find_longest_free_gap_angle(
            msg,
            min_dist=self.free_dist,
            fov_deg=self.fov_deg,
            min_gap_width_m=self.min_gap_width_m,
            max_gap_width_m=self.max_gap_width_m
        )

        self.last_chosen_width = chosen_width_m

        # 2) 근접 장애물 스캔 (이머전시)
        first, first_dst, last, last_dst, has_obstacle = self.LiDAR_scan(msg)

        # 3) direction 결정 (이머전시)
        self.decide_direction(first_dst, last_dst, has_obstacle)

        # 4) 디버그: 각도(사용자 기준 오른쪽+)
        deg_ros_chosen  = free_angle_ros * 180.0 / math.pi
        deg_user = -deg_ros_chosen
        if deg_user > 60.0:
            deg_user_clamped = 60.0
        elif deg_user < -60.0:
            deg_user_clamped = -60.0
        else:
            deg_user_clamped = deg_user

        dbg = Float32()
        dbg.data = deg_user_clamped
        self.debug_deg_pub.publish(dbg)

        # 5) 최종 cmd publish
        cmd = self.make_cmd_from_direction(free_angle_ros)
        self.cmd_pub.publish(cmd)


if __name__ == "__main__":
    try:
        node = Limo_obstacle_avoidence()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
