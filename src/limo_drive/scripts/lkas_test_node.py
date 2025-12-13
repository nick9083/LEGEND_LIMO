#!/usr/bin/env python3
import rospy
import numpy as np
import cv2

from math import atan2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Int32


class LKAS:
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node("LKAS_node")

        # ---- Publishers ----
        self.ctrl_pub = rospy.Publisher("/cmd_vel_lkas", Twist, queue_size=1)
        self.debug_pub = rospy.Publisher("/sliding_windows/compressed", CompressedImage, queue_size=1)
        self.lane_departure_pub = rospy.Publisher("/lane_departure", Bool, queue_size=1)

        # ---- Subscribers ----
        rospy.Subscriber("/camera/rgb/image_raw/compressed", CompressedImage, self.img_CB, queue_size=1)
        rospy.Subscriber("/scan", LaserScan, self.scan_CB, queue_size=1)
        rospy.Subscriber("/lkas_enable", Bool, self.enable_cb, queue_size=1)
        rospy.Subscriber("/mission_mode", Int32, self.mission_cb, queue_size=1)

        # ---- State ----
        self.enabled = True
        self.mission_mode = 2  # 기본: 급곡선 주행(카메라)

        self.cmd_vel_msg = Twist()
        self.last_pub_time = rospy.get_time()

        # ---- Camera / Warp ----
        self.img_x = 0
        self.img_y = 0
        self.offset_x = 80

        # ---- Control params (미션별로 자동 세팅) ----
        self.speed = 0.12
        self.turn_gain = 0.0038    # 픽셀 오차 -> 각속도 변환 게인
        self.lookahead_y_ratio = 0.62  # centerline에서 조향 기준 y(화면 높이 비율)

        # ---- Sliding window ----
        self.nwindows = 10
        self.margin = 80
        self.minpix = 35

        # ---- Lane model ----
        self.lane_width_px = None  # 한쪽만 잡힐 때 사용(학습형으로 갱신)
        self.prev_center_x = None  # 튐 완화용

        # ---- Lidar ----
        self.scan_ranges = None
        self.scan_angle_min = None
        self.scan_angle_inc = None
        self.scan_valid = False

        # 안전 반경/전방 감지
        self.front_obs_dist = 0.55     # 전방 위험거리(카메라 미션에서도 안전 개입)
        self.front_fov_deg = 35.0

    # =========================
    # Callbacks
    # =========================
    def enable_cb(self, msg: Bool):
        self.enabled = msg.data

    def mission_cb(self, msg: Int32):
        self.mission_mode = int(msg.data)
        self.apply_mission_params(self.mission_mode)

    def scan_CB(self, msg: LaserScan):
        self.scan_ranges = np.array(msg.ranges, dtype=np.float32)
        self.scan_ranges[np.isinf(self.scan_ranges)] = 10.0
        self.scan_ranges[np.isnan(self.scan_ranges)] = 10.0
        self.scan_ranges = np.clip(self.scan_ranges, 0.02, 10.0)

        self.scan_angle_min = msg.angle_min
        self.scan_angle_inc = msg.angle_increment
        self.scan_valid = True

    def img_CB(self, data: CompressedImage):
        if not self.enabled:
            return

        now = rospy.get_time()
        if now - self.last_pub_time < 0.1:
            return

        img = self.bridge.compressed_imgmsg_to_cv2(data)
        self.img_y, self.img_x = img.shape[0], img.shape[1]

        # 1) 미션별 조향 계산
        if self.mission_mode in [3, 4]:
            # 라이다 미션: 카메라 무시하고 라이다 조향
            steer = self.lidar_steer_follow_gap()
            self.publish_cmd(self.speed, steer)
            self.last_pub_time = now
            return

        # 카메라 미션: 1,2,5,6
        warp = self.img_warp(img)
        binary = self.make_lane_binary(warp, mission=self.mission_mode)

        # 2) 슬라이딩 윈도우로 좌/우 차선 추정
        debug_img, lane_ok, center_x, departure = self.lane_center_from_windows(binary)

        # 디버그 이미지 publish
        self.publish_debug(debug_img)

        # 3) 차선이 약하면(혹은 center 불안정) -> 튐 완화(이전 center 유지)
        if (not lane_ok) and (self.prev_center_x is not None):
            center_x = self.prev_center_x

        if center_x is None:
            # 완전 실패: 직진 + 속도 낮추기
            self.publish_cmd(0.08, 0.0)
            self.lane_departure_pub.publish(Bool(True))
            self.last_pub_time = now
            return

        # 4) 조향 계산(lookahead 오차 기반)
        img_center = binary.shape[1] * 0.5
        err_px = (center_x - img_center)

        # 튐 방지: center low-pass
        if self.prev_center_x is None:
            self.prev_center_x = center_x
        else:
            self.prev_center_x = 0.7 * self.prev_center_x + 0.3 * center_x

        steer_cam = -err_px * self.turn_gain

        # 5) 안전: 전방 장애물 너무 가까우면 라이다로 일부 개입
        steer = steer_cam
        if self.front_obstacle_close():
            steer_lidar = self.lidar_steer_follow_gap()
            steer = 0.55 * steer_cam + 0.45 * steer_lidar
            # 위험하면 속도도 살짝 낮춤
            v = min(self.speed, 0.10)
        else:
            v = self.speed

        self.publish_cmd(v, steer)
        self.lane_departure_pub.publish(Bool(departure))
        self.last_pub_time = now

    # =========================
    # Mission params
    # =========================
    def apply_mission_params(self, mode: int):
        # 기본값
        self.speed = 0.12
        self.turn_gain = 0.0038
        self.lookahead_y_ratio = 0.62

        # 미션별 튜닝(필요하면 여기만 조절)
        if mode == 1:  # 지그재그
            self.speed = 0.11
            self.turn_gain = 0.0042
            self.lookahead_y_ratio = 0.58
        elif mode == 2:  # 급곡선
            self.speed = 0.10
            self.turn_gain = 0.0048
            self.lookahead_y_ratio = 0.52
        elif mode == 5:  # 색깔차선
            self.speed = 0.10
            self.turn_gain = 0.0040
            self.lookahead_y_ratio = 0.58
        elif mode == 6:  # 경로선택
            self.speed = 0.11
            self.turn_gain = 0.0040
            self.lookahead_y_ratio = 0.58
        elif mode == 3:  # 장애물회피(라이다)
            self.speed = 0.09
        elif mode == 4:  # 라바콘(라이다)
            self.speed = 0.08

    # =========================
    # Warp
    # =========================
    def img_warp(self, img):
        h, w = img.shape[0], img.shape[1]

        # src: 사다리꼴 (환경에 맞게 조정 가능)
        src_center_offset = [200, 315]
        src = np.array(
            [
                [0, h - 1],
                [src_center_offset[0], src_center_offset[1]],
                [w - src_center_offset[0], src_center_offset[1]],
                [w - 1, h - 1],
            ],
            dtype=np.float32,
        )

        dst = np.array(
            [
                [self.offset_x, h],
                [self.offset_x, 0],
                [w - self.offset_x, 0],
                [w - self.offset_x, h],
            ],
            dtype=np.float32,
        )

        M = cv2.getPerspectiveTransform(src, dst)
        warp_img = cv2.warpPerspective(img, M, (w, h))
        return warp_img

    # =========================
    # Lane Binary (숫자/중앙 오인 억제 포함)
    # =========================
    def make_lane_binary(self, warp_bgr, mission=2):
        """
        목표:
        - 숫자/문구(중앙) 성분은 최대한 제거
        - 좌/우 차선 성분은 살림
        """
        hsv = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2HSV)

        # 공통: 흰색/노란색 차선
        white = cv2.inRange(hsv, (0, 0, 205), (179, 60, 255))
        yellow = cv2.inRange(hsv, (15, 80, 60), (45, 255, 255))
        mask = cv2.bitwise_or(white, yellow)

        # 미션5(색깔차선): 색 차선도 추가로 잡아줌(파랑/빨강 계열을 넓게)
        if mission == 5:
            # 파랑(대략)
            blue = cv2.inRange(hsv, (90, 80, 40), (140, 255, 255))
            # 빨강(HSV에서 양 끝)
            red1 = cv2.inRange(hsv, (0, 90, 40), (10, 255, 255))
            red2 = cv2.inRange(hsv, (170, 90, 40), (179, 255, 255))
            red = cv2.bitwise_or(red1, red2)
            mask = cv2.bitwise_or(mask, blue)
            mask = cv2.bitwise_or(mask, red)

        # 노이즈 정리(작게)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

        # ---- 숫자 억제 핵심: 연결성분에서 "좌/우 차선 후보"만 남김 ----
        h, w = mask.shape
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        keep = np.zeros((h, w), dtype=np.uint8)

        # 좌/우 영역 기준(중앙 글자 제거)
        left_band_end = int(w * 0.46)
        right_band_start = int(w * 0.54)

        # 바닥에 일정 이상 닿는 성분만(차선은 대체로 하단에 존재)
        bottom_band = int(h * 0.35)
        bottom_touch = h - bottom_band

        # 급곡선에서 차선이 중앙으로 살짝 들어오더라도 완전히 죽지 않게:
        # "바닥에 충분히 닿는 성분"이면 중앙이어도 살릴 수 있게 옵션 제공
        strict_center_exclude = int(w * 0.10)
        cx0 = (w // 2) - strict_center_exclude
        cx1 = (w // 2) + strict_center_exclude

        min_area = int(h * w * 0.00018)

        for i in range(1, num):
            x, y, ww, hh, area = stats[i]
            if area < min_area:
                continue

            cy = centroids[i][1]
            cx = centroids[i][0]
            bottom = y + hh

            # 바닥에 어느 정도 닿아야 함
            if bottom < bottom_touch:
                continue

            # 중앙 억제(너무 중앙에 붙은 성분은 숫자일 확률 높음)
            # 단, 좌/우 밴드에 속하면 통과
            in_left = (cx < left_band_end)
            in_right = (cx > right_band_start)

            if (not in_left) and (not in_right):
                # 정말 중앙이면 제거
                if cx0 <= cx <= cx1:
                    continue
                # 중앙이지만 너무 약하게만 중앙이면(급곡선 대비) 통과 가능
                # (그래도 숫자 방지를 위해 cy가 너무 위면 제거)
                if cy < h * 0.55:
                    continue

            keep[labels == i] = 255

        # 슬라이딩 윈도우가 잘 잡게 약간 두껍게
        keep = cv2.dilate(keep, k, iterations=1)

        binary = (keep > 0).astype(np.uint8)  # 0/1
        return binary

    # =========================
    # Sliding window -> center x
    # =========================
    def lane_center_from_windows(self, binary):
        h, w = binary.shape

        # 히스토그램은 하단부만
        hist = np.sum(binary[h // 2:, :], axis=0)

        midpoint = w // 2
        leftx_base = int(np.argmax(hist[:midpoint]))
        rightx_base = int(np.argmax(hist[midpoint:]) + midpoint)

        # 초기값이 너무 중앙/너무 가장자리면 신뢰 낮음
        lane_ok = True
        if leftx_base < 5 or rightx_base > (w - 5):
            lane_ok = False

        # 슬라이딩 윈도우
        nwindows = self.nwindows
        window_height = h // nwindows
        margin = self.margin

        nonzero = binary.nonzero()
        nonzeroy = np.array(nonzero[0], dtype=np.int32)
        nonzerox = np.array(nonzero[1], dtype=np.int32)

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        out_img = (np.dstack((binary, binary, binary)) * 255).astype(np.uint8)

        # minpix 동적(이미지 크기 따라)
        minpix = max(self.minpix, int((2 * margin * window_height) * 0.0025))

        for window in range(nwindows):
            win_y_low = h - (window + 1) * window_height
            win_y_high = h - window * window_height

            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 0, 255), 2)

            good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)

            if len(good_left) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left]))
            if len(good_right) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right]))

        left_lane_inds = np.concatenate(left_lane_inds) if len(left_lane_inds) else np.array([], dtype=int)
        right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) else np.array([], dtype=int)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # ---- 좌/우 둘 다 없으면 실패 ----
        if (len(leftx) < 30) and (len(rightx) < 30):
            return out_img, False, None, True

        # ---- 한쪽만 있을 때 lane_width_px 추정/활용 ----
        if len(leftx) >= 30 and len(rightx) >= 30:
            # 둘 다 있으면 lane width 갱신
            y_eval = int(h * 0.85)
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            lx = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
            rx = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
            width = float(rx - lx)
            if width > w * 0.25 and width < w * 0.85:
                self.lane_width_px = width
        else:
            left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) >= 30 else None
            right_fit = np.polyfit(righty, rightx, 2) if len(rightx) >= 30 else None

        # ---- centerline 생성 ----
        y_look = int(h * self.lookahead_y_ratio)

        if len(leftx) >= 30 and len(rightx) >= 30:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

            lx = left_fit[0] * y_look**2 + left_fit[1] * y_look + left_fit[2]
            rx = right_fit[0] * y_look**2 + right_fit[1] * y_look + right_fit[2]
            center_x = 0.5 * (lx + rx)

            # 디버그 polyline
            ploty = np.linspace(0, h - 1, 20)
            lxs = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
            rxs = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
            cxs = 0.5 * (lxs + rxs)

            cv2.polylines(out_img, [np.int32(np.column_stack((lxs, ploty)))], False, (0, 255, 0), 3)
            cv2.polylines(out_img, [np.int32(np.column_stack((rxs, ploty)))], False, (0, 0, 255), 3)
            cv2.polylines(out_img, [np.int32(np.column_stack((cxs, ploty)))], False, (255, 0, 0), 3)

        else:
            # 한쪽만 있을 때
            if self.lane_width_px is None:
                # 폭을 아직 모르겠으면 실패 처리 쪽으로
                return out_img, False, None, True

            if len(leftx) >= 30:
                left_fit = np.polyfit(lefty, leftx, 2)
                lx = left_fit[0] * y_look**2 + left_fit[1] * y_look + left_fit[2]
                center_x = lx + 0.5 * self.lane_width_px
            else:
                right_fit = np.polyfit(righty, rightx, 2)
                rx = right_fit[0] * y_look**2 + right_fit[1] * y_look + right_fit[2]
                center_x = rx - 0.5 * self.lane_width_px

            lane_ok = False  # 신뢰 낮음

        # ---- 차선 이탈 판단(대략) ----
        # center_x가 화면 중앙에서 너무 벗어나면 이탈로 판단
        img_center = w * 0.5
        departure = (abs(center_x - img_center) > (w * 0.42))

        # lookahead 점 표시
        cv2.circle(out_img, (int(center_x), y_look), 6, (255, 255, 0), -1)

        return out_img, lane_ok, float(center_x), departure

    # =========================
    # Lidar: Follow-the-Gap
    # =========================
    def lidar_steer_follow_gap(self):
        if not self.scan_valid or self.scan_ranges is None:
            return 0.0

        ranges = self.scan_ranges.copy()

        # 전방 위주로만 사용(너무 뒤는 무시)
        angles = self.scan_angle_min + np.arange(len(ranges)) * self.scan_angle_inc
        fov = np.deg2rad(100.0)  # 라이다 활용 폭
        use = np.abs(angles) < (fov * 0.5)

        r = ranges[use]
        a = angles[use]

        # 가장 가까운 장애물 기준으로 bubble
        closest_idx = int(np.argmin(r))
        closest_dist = float(r[closest_idx])

        bubble_radius = 0.32  # m
        # 거리->각도 bubble 크기 근사
        if closest_dist > 0.05:
            bubble_ang = np.arctan2(bubble_radius, closest_dist)
        else:
            bubble_ang = np.deg2rad(25.0)

        bubble_mask = np.abs(a - a[closest_idx]) < bubble_ang
        r[bubble_mask] = 0.0

        # gap 찾기: r>threshold인 연속 구간 중 최대
        free = r > 0.85  # 이 값이 낮으면 과감히 지나가고, 높으면 안전 위주
        if not np.any(free):
            # 전방이 막혀있으면 느리게 직진(또는 약간 회전)
            return 0.35

        # 최대 연속 구간
        idx = np.where(free)[0]
        splits = np.where(np.diff(idx) > 1)[0]
        segments = np.split(idx, splits + 1)

        best_seg = max(segments, key=len)
        # best point: 가장 먼 점(or 중앙점)
        seg_r = r[best_seg]
        best_local = int(best_seg[np.argmax(seg_r)])

        best_angle = float(a[best_local])

        # 각도 -> angular.z (부호는 ROS 기준에 맞게)
        steer = best_angle * 1.2
        steer = float(np.clip(steer, -1.2, 1.2))
        return steer

    def front_obstacle_close(self):
        if not self.scan_valid or self.scan_ranges is None:
            return False

        angles = self.scan_angle_min + np.arange(len(self.scan_ranges)) * self.scan_angle_inc
        fov = np.deg2rad(self.front_fov_deg)
        front = np.abs(angles) < (fov * 0.5)

        dmin = float(np.min(self.scan_ranges[front])) if np.any(front) else 10.0
        return dmin < self.front_obs_dist

    # =========================
    # Publish helpers
    # =========================
    def publish_cmd(self, v, steer):
        msg = Twist()
        msg.linear.x = float(np.clip(v, 0.0, 0.25))
        msg.angular.z = float(np.clip(steer, -1.6, 1.6))
        self.ctrl_pub.publish(msg)

    def publish_debug(self, bgr_img):
        try:
            out = CompressedImage()
            out.header.stamp = rospy.Time.now()
            out.format = "jpeg"
            out.data = np.array(cv2.imencode(".jpg", bgr_img)[1]).tobytes()
            self.debug_pub.publish(out)
        except Exception:
            pass


if __name__ == "__main__":
    lkas = LKAS()
    rospy.spin()