#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from math import pi, floor

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


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


# ---------------- 베이스 장애물 회피 노드 ----------------

class Limo_obstacle_avoidence:
    def __init__(self):
        # 노드 이름: base obstacle avoid
        rospy.init_node("base_obstacle_avoid")

        # === 파라미터 ===
        scan_topic     = rospy.get_param("~scan_topic", "/scan")
        cmd_topic      = rospy.get_param("~cmd_topic", "/cmd_vel_obstacle")
        enable_topic   = rospy.get_param("~enable_topic", "/obstacle_enable")
        debug_deg_topic = rospy.get_param("~debug_deg_topic",
                                          "/base_free_gap_angle_deg")

        # 가까운 영역(emergency) 설정
        self.scan_degree = rospy.get_param("~scan_degree", 50.0)       # ±50도
        self.emergency_dist = rospy.get_param("~emergency_dist", 0.40) # 30cm 안
        self.min_dist = rospy.get_param("~min_dist", 0.20)             # 이 이하면 back
        self.OBSTACLE_PERCEPTION_BOUNDARY = rospy.get_param(
            "~obstacle_perception_boundary", 10
        )

        # 속도 관련
        self.default_speed = rospy.get_param("~default_speed", 0.15)
        self.default_angle = 0.0
        self.backward_speed = rospy.get_param("~backward_speed", -0.08)

        # 먼 영역(gap 기반) 설정
        self.free_dist      = rospy.get_param("~free_dist", 0.7)
        self.fov_deg        = rospy.get_param("~fov_deg", 150.0)
        self.k_ang          = rospy.get_param("~k_ang", 1.0)
        self.max_yaw        = rospy.get_param("~max_yaw", 1.0)
        self.min_gap_width_m = rospy.get_param("~min_gap_width_m", 0.25)
        self.max_gap_width_m = rospy.get_param("~max_gap_width_m", 0.7)

        # ROS IO
        rospy.Subscriber(scan_topic, LaserScan, self.laser_callback, queue_size=1)
        rospy.Subscriber(enable_topic, Bool, self.enable_cb, queue_size=1)

        self.pub       = rospy.Publisher(cmd_topic, Twist, queue_size=3)
        self.debug_pub = rospy.Publisher(debug_deg_topic, Float32, queue_size=1)

        self.rate = rospy.Rate(30)
        self.cmd_vel_msg = Twist()

        # 상태 변수들
        self.msg = None
        self.is_scan = False

        self.lidar_flag = False
        self.degrees = []
        self.ranges_length = None

        self.dist_data = float("inf")
        self.direction = "front"

        self.obstacle_ranges = []
        self.center_list_left = []
        self.center_list_right = []

        self.speed = 0.0
        self.angle = 0.0

        self.last_chosen_width = 0.0

        # FSM 에서 오는 enable 플래그
        self.enabled = True  # FSM 없을 때 단독 테스트용으로 기본 True

        rospy.loginfo("[BASE_OBS] node started. scan=%s, cmd=%s, enable=%s",
                      scan_topic, cmd_topic, enable_topic)

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
    def laser_callback(self, msg: LaserScan):
        self.msg = msg
        self.is_scan = True

    # --------------------------------
    # LiDAR 스캔 처리 (가까운 emergency window)
    # --------------------------------
    def LiDAR_scan(self):
        """
        emergency window 안(±scan_degree, dist < emergency_dist) 의 점들을 모아서
        군집 크기가 일정 이상이면 'emergency 존재'로 판단.
        """
        if self.msg is None:
            return 0, 0, 0, 0, False

        obstacle = []

        # 각도 테이블 1회 생성
        if not self.lidar_flag:
            self.degrees = [
                (self.msg.angle_min + (i * self.msg.angle_increment)) * 180.0 / pi
                for i, _ in enumerate(self.msg.ranges)
            ]
            self.ranges_length = len(self.msg.ranges)
            self.lidar_flag = True

        min_dist_val = float("inf")

        for i, data in enumerate(self.msg.ranges):
            if 0.0 < data < self.emergency_dist and \
               -self.scan_degree < self.degrees[i] < self.scan_degree:
                obstacle.append(i)
                if data < min_dist_val:
                    min_dist_val = data

        if obstacle:
            self.dist_data = min_dist_val
            first = obstacle[0]
            last = obstacle[-1]
            first_dst = first
            last_dst = self.ranges_length - last
            self.obstacle_ranges = self.msg.ranges[first:last + 1]
        else:
            self.dist_data = float("inf")
            self.obstacle_ranges = []
            first = first_dst = last = last_dst = 0

        # 군집 크기가 기준 이상이면 emergency 로 취급
        has_emergency = (len(obstacle) >= self.OBSTACLE_PERCEPTION_BOUNDARY)

        return first, first_dst, last, last_dst, has_emergency

    # --------------------------------
    # 공간 비교 (어느 쪽으로 피할지) — emergency 영역에서만 사용
    # --------------------------------
    def compare_space(self, first_dst, last_dst, has_emergency: bool):
        if has_emergency:
            if first_dst > last_dst and self.dist_data > self.min_dist:
                self.direction = "right"
            elif first_dst < last_dst and self.dist_data > self.min_dist:
                self.direction = "left"
            elif first_dst > last_dst and self.dist_data <= self.min_dist:
                self.direction = "right_back"
            elif first_dst < last_dst and self.dist_data <= self.min_dist:
                self.direction = "left_back"
        else:
            self.direction = "front"

    # --------------------------------
    # 장애물 방향에 따른 속도/조향 결정 (emergency 로직 그대로)
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

        elif self.direction in ("right_back", "left_back", "back"):
            # 너무 가까우면 뒤로 (간단히 직선 후진)
            self.angle = self.default_angle
            self.speed = self.backward_speed

        else:
            # 장애물 없으면 직진
            self.angle = self.default_angle
            self.speed = self.default_speed

    # --------------------------------
    # 먼 영역 gap 기반 방향 결정
    # --------------------------------
    def compute_far_gap_cmd(self):
        """
        긴 거리 window 에서 gap 기반으로 조향각 잡기.
        emergency 가 없을 때만 사용.
        """
        chosen_center, chosen_width, longest_center, longest_width = \
            find_longest_free_gap_angle(
                self.msg,
                min_dist=self.free_dist,
                fov_deg=self.fov_deg,
                min_gap_width_m=self.min_gap_width_m,
                max_gap_width_m=self.max_gap_width_m,
            )

        if chosen_width > 0.0:
            theta = chosen_center
            self.last_chosen_width = chosen_width
        else:
            theta = longest_center
            self.last_chosen_width = longest_width

        # deg 디버그 (USER 기준: 오른쪽 +)
        deg_ros = theta * 180.0 / math.pi     # ROS: 왼쪽 +, 오른쪽 -
        deg_user = -deg_ros
        dbg = Float32()
        dbg.data = deg_user
        self.debug_pub.publish(dbg)

        # 속도/조향
        self.speed = self.default_speed
        self.angle = clamp(self.k_ang * theta, -self.max_yaw, self.max_yaw)

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

        # 1) 가까운 emergency window 검사
        first, first_dst, last, last_dst, has_emergency = self.LiDAR_scan()

        if has_emergency:
            # ====== 응급 영역: 기존 로직 그대로 ======
            self.compare_space(first_dst, last_dst, has_emergency)
            self.move_direction(last, first)
        else:
            # ====== 긴 거리 gap 기반 steering ======
            self.compute_far_gap_cmd()

        # 2) cmd_vel 메시지 채우고 publish
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
