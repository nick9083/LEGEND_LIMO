#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math, numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool   # ★ 추가: enable 게이팅용


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ============================================================
# tf 없이 쓰는 euler_from_quaternion
# 입력: [x, y, z, w]
# 출력: (roll, pitch, yaw)
# ============================================================
def euler_from_quaternion(quat):
    x, y, z, w = quat

    # roll (x축)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    # pitch (y축)
    t2 = +2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    pitch = math.asin(t2)

    # yaw (z축)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw


class DWAObstacleAvoid:
    def __init__(self):

        # ===== ROS Param =====
        self.frame_rate      = rospy.get_param("~rate", 15)
        self.dt              = rospy.get_param("~dt", 0.1)
        self.predict_time    = rospy.get_param("~predict_time", 1.4)

        # 속도 및 제한
        self.max_speed       = rospy.get_param("~max_speed", 0.8)
        self.min_speed       = rospy.get_param("~min_speed", -0.08)
        self.max_yawrate     = rospy.get_param("~max_yawrate", 1.5)
        self.max_accel       = rospy.get_param("~max_accel", 1.2)
        self.max_dyawrate    = rospy.get_param("~max_dyawrate", 3.0)

        # 샘플 resolution
        self.v_reso          = rospy.get_param("~v_reso", 0.03)
        self.yawrate_reso    = rospy.get_param("~yawrate_reso", 0.15)

        # 로봇 크기
        self.robot_radius    = rospy.get_param("~robot_radius", 0.16)
        self.obst_margin     = rospy.get_param("~obstacle_margin", 0.05)

        # 안전 Dist
        self.stop_clearance  = rospy.get_param("~stop_clearance", 0.14)
        self.slow_clearance  = rospy.get_param("~slow_clearance", 0.22)
        self.gap_clearance   = rospy.get_param("~gap_clearance", 0.50)   # 50cm 기준

        # FTG ROI
        self.front_roi_deg   = rospy.get_param("~front_roi_deg", 120)

        # Cost
        self.w_heading       = rospy.get_param("~w_heading",    1.9)
        self.w_clearance     = rospy.get_param("~w_clearance",  0.9)
        self.w_velocity      = rospy.get_param("~w_velocity",   1.0)
        self.w_smooth        = rospy.get_param("~w_smooth",     0.25)

        # 후진 패널티
        self.backward_penalty = rospy.get_param("~backward_penalty", 3.0)

        # Internal states
        self.has_scan = False
        self.has_odom = False
        self.scan_angles = None
        self.scan_pts = None
        self.ranges = None

        self.x = self.y = self.yaw = 0.0
        self.vx = self.wz = 0.0

        # ★ enable 플래그
        self.enabled = True  # FSM 없으면 기본 True로 단독 테스트 가능

        # ROS IO (토픽 이름은 파라미터로도 바꿀 수 있게)
        scan_topic   = rospy.get_param("~scan_topic", "/scan")
        odom_topic   = rospy.get_param("~odom_topic", "/odom")
        cmd_topic    = rospy.get_param("~cmd_topic", "/cmd_vel_obstacle")  # ★ FSM용 출력
        enable_topic = rospy.get_param("~enable_topic", "/obstacle_enable")  # ★ 게이팅

        rospy.Subscriber(scan_topic, LaserScan, self.cb_scan, queue_size=1)
        rospy.Subscriber(odom_topic, Odometry, self.cb_odom, queue_size=1)
        rospy.Subscriber(enable_topic, Bool, self.cb_enable, queue_size=1)  # ★ enable 구독
        self.pub_cmd = rospy.Publisher(cmd_topic, Twist, queue_size=1)

    # =====================
    #    Callbacks
    # =====================

    def cb_scan(self, msg: LaserScan):
        n = len(msg.ranges)
        angles = msg.angle_min + np.arange(n) * msg.angle_increment
        ranges = np.array(msg.ranges, dtype=np.float32)

        ranges = np.where(
            np.isfinite(ranges) & (ranges > 0.01),
            ranges,
            msg.range_max
        )

        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)

        self.scan_pts = np.stack([xs, ys], axis=1)
        self.scan_angles = angles
        self.ranges = ranges
        self.has_scan = True

    def cb_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        self.x, self.y = p.x, p.y
        # ★ 여기서 우리가 만든 euler_from_quaternion 사용
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.yaw = yaw

        self.vx = msg.twist.twist.linear.x
        self.wz = msg.twist.twist.angular.z
        self.has_odom = True

    def cb_enable(self, msg: Bool):
        # ★ FSM에서 /obstacle_enable 들어옴
        self.enabled = msg.data

    # ========================================
    # FTG 기반 GAP 헤딩 (조향 방향)
    # ========================================
    def compute_gap_heading(self):
        if self.ranges is None:
            return 0.0

        half = math.radians(self.front_roi_deg / 2.0)
        mask = (self.scan_angles > -half) & (self.scan_angles < half)

        ang = self.scan_angles[mask]
        dist = self.ranges[mask]

        thresh = self.gap_clearance
        safe = dist > thresh

        if not np.any(safe):
            return 0.0

        idx = np.where(safe)[0]
        gaps = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        best_gap = max(gaps, key=len)
        a_center = ang[best_gap[len(best_gap)//2]]

        return float(a_center)

    # ========================================
    #    Pre-Steering: 미리 조향하게 하는 핵심
    # ========================================
    def front_min_clearance(self, width_deg=60):
        if self.ranges is None:
            return 10.0

        half = math.radians(width_deg / 2)
        mask = (self.scan_angles > -half) & (self.scan_angles < half)

        if not np.any(mask):
            return 10.0

        return float(np.min(self.ranges[mask]))

    # ========================================
    # DWA Core
    # ========================================
    def dynamic_window(self):
        vs_min = max(self.min_speed, self.vx - self.max_accel * self.dt)
        vs_max = min(self.max_speed, self.vx + self.max_accel * self.dt)
        ws_min = self.wz - self.max_dyawrate * self.dt
        ws_max = self.wz + self.max_dyawrate * self.dt
        return vs_min, vs_max, ws_min, ws_max

    def simulate(self, v, w):
        x = y = yaw = 0.0
        path = []
        t = 0.0
        while t < self.predict_time:
            x += v * math.cos(yaw) * self.dt
            y += v * math.sin(yaw) * self.dt
            yaw += w * self.dt
            path.append((x, y, yaw))
            t += self.dt
        return path

    def clearance_along_path(self, path):
        if self.scan_pts is None:
            return 10.0

        pts = self.scan_pts
        rad = self.robot_radius + self.obst_margin
        min_clear = 10.0

        for x, y, yaw in path:
            c = math.cos(-yaw)
            s = math.sin(-yaw)

            px = pts[:, 0] - x
            py = pts[:, 1] - y

            rx = c * px - s * py
            ry = s * px + c * py

            d2 = rx*rx + ry*ry
            if len(d2) == 0:
                continue

            m = float(np.sqrt(d2.min()))
            if m < min_clear:
                min_clear = m

            if min_clear <= rad:
                return min_clear

        return min_clear

    def eval_traj(self, v, w, th_goal):
        path = self.simulate(v, w)
        clear = self.clearance_along_path(path)

        xf, yf, yawf = path[-1]
        head_err = abs(self.norm_angle(yawf - th_goal))

        cost = (
            self.w_heading    * head_err
            - self.w_clearance * clear
            - self.w_velocity  * v
            + self.w_smooth    * abs(w - self.wz)
        )

        if v < 0:
            cost += self.backward_penalty

        return cost, clear

    @staticmethod
    def norm_angle(a):
        while a > math.pi:
            a -= 2*math.pi
        while a < -math.pi:
            a += 2*math.pi
        return a

    # ========================================
    # pick_control()
    # ========================================
    def pick_control(self):

        th_goal = self.compute_gap_heading()
        vs_min, vs_max, ws_min, ws_max = self.dynamic_window()

        best_cost = float("inf")
        best_v = 0.0
        best_w = 0.0
        best_clr = 0.0

        v = vs_min
        while v <= vs_max + 1e-9:
            w = ws_min
            while w <= ws_max + 1e-9:
                cost, clr = self.eval_traj(v, w, th_goal)
                if cost < best_cost:
                    best_cost = cost
                    best_v = v
                    best_w = w
                    best_clr = clr
                w += self.yawrate_reso
            v += self.v_reso

        v_cmd = best_v
        w_cmd = best_w
        clr   = best_clr

        # ======================================================
        #   🔥 Pre-Steering Layer (조향을 더 빨리 시작)
        # ======================================================
        front_clr = self.front_min_clearance(width_deg=60)
        PRE_STEER_DIST = 0.45      # 이 거리 안에서 서서히 조향
        if front_clr < PRE_STEER_DIST:
            alpha = (PRE_STEER_DIST - front_clr) / PRE_STEER_DIST
            alpha = clamp(alpha, 0.0, 1.0)

            desired_w = clamp(th_goal * 1.6,
                               -self.max_yawrate, self.max_yawrate)

            w_cmd = (1.0 - alpha) * w_cmd + alpha * desired_w
            v_cmd = min(v_cmd, self.max_speed * (1.0 - 0.4 * alpha))

        # ======================================================
        #   기존 안전 정책 (정지/감속)
        # ======================================================
        if clr < self.stop_clearance:
            if v_cmd > 0:
                v_cmd = 0.0
            w_cmd = clamp(th_goal * 1.8,
                          -self.max_yawrate, self.max_yawrate)

        elif clr < self.slow_clearance:
            v_cmd = min(v_cmd, 0.25)

        # 최종 클램프 — 후진 포함
        v_cmd = clamp(v_cmd, self.min_speed, self.max_speed)
        w_cmd = clamp(w_cmd, -self.max_yawrate, self.max_yawrate)

        return v_cmd, w_cmd

    def publish(self, v, w):
        msg = Twist()
        msg.linear.x  = float(v)
        msg.angular.z = float(w)
        self.pub_cmd.publish(msg)

    def spin(self):
        rate = rospy.Rate(self.frame_rate)
        rospy.loginfo("[Mission3][DWA] started.")
        idle_cnt = 0

        while not rospy.is_shutdown():

            # ★ 게이팅: obstacle_enable == False 이면 연산 싹 스킵 + 정지 명령
            if not self.enabled:
                self.publish(0.0, 0.0)
                rate.sleep()
                continue

            if not (self.has_scan and self.has_odom):
                self.publish(0.0, 0.0)
                idle_cnt += 1
                if idle_cnt % (self.frame_rate*3) == 0:
                    rospy.logwarn("Waiting scan/odom...")
                rate.sleep()
                continue

            try:
                v, w = self.pick_control()
                self.publish(v, w)
            except Exception as e:
                rospy.logerr("DWA error: %s", str(e))
                self.publish(0.0, 0.0)

            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("mission3_dwa")
    node = DWAObstacleAvoid()
    node.spin()
