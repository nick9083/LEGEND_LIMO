#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from std_msgs.msg import Bool

import numpy as np
import cv2


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class LKAS:
    """
    LKAS node - Approach #1 (Control-side robustness)

    추가된 것
      1) curvature(곡률 proxy) + confidence 기반 속도 스케줄링
      2) lookahead y 기반 오프셋 계산 (하단 80px이 비어도 덜 흔들림)
      3) confidence 낮을수록 측정치를 덜 믿고(hold/low-pass) 조향 안정화
      4) steering rate limit / deadband / clamp
      5) 디버그 이미지에 conf, radius, v, wz 표시
    """

    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node("LKAS_node", anonymous=False)

        # --- Topics ---
        self.pub_dbg = rospy.Publisher(
            "/sliding_windows/compressed", CompressedImage, queue_size=1
        )
        rospy.Subscriber(
            "/camera/rgb/image_raw/compressed", CompressedImage, self.img_CB, queue_size=1
        )

        self.ctrl_pub = rospy.Publisher("/cmd_vel_lkas", Twist, queue_size=1)
        rospy.Subscriber("/lkas_enable", Bool, self.enable_cb, queue_size=1)

        # --- Params (tunable) ---
        self.base_speed = rospy.get_param("~base_speed", 0.16)
        self.curve_speed = rospy.get_param("~curve_speed", 0.10)         # 급커브 속도
        self.low_conf_speed = rospy.get_param("~low_conf_speed", 0.08)   # 신뢰도 낮을 때 속도

        self.turn_mult = rospy.get_param("~turn_mult", 0.12)             # 기존 trun_mutip 역할
        self.max_ang = rospy.get_param("~max_ang", 1.2)                  # |angular.z| 제한
        self.max_ang_rate = rospy.get_param("~max_ang_rate", 2.0)        # rad/s, rate limit
        self.steer_deadband = rospy.get_param("~steer_deadband", 0.01)   # 작은 떨림 제거

        # lookahead: 하단이 안 보일 때를 대비해, 바닥에서 일정 px 위에서 오프셋 평가
        self.lookahead_px = rospy.get_param("~lookahead_px", 140)        # 80px 문제면 120~180 추천

        # curvature proxy threshold (pixel radius)
        self.radius_thresh_px = rospy.get_param("~radius_thresh_px", 900.0)

        # confidence thresholds
        self.conf_slow_thresh = rospy.get_param("~conf_slow_thresh", 0.55)
        self.conf_hold_thresh = rospy.get_param("~conf_hold_thresh", 0.35)

        # confidence 계산용 파라미터
        self.min_total_pixels = rospy.get_param("~min_total_pixels", 450)     # 좌+우 총 픽셀
        self.min_side_pixels = rospy.get_param("~min_side_pixels", 150)       # 한쪽 픽셀
        self.width_min_ratio = rospy.get_param("~lane_width_min_ratio", 0.30) # w 대비 최소 폭
        self.width_max_ratio = rospy.get_param("~lane_width_max_ratio", 0.80) # w 대비 최대 폭
        self.width_std_max = rospy.get_param("~lane_width_std_max", 25.0)     # px (BEV 기준)
        self.offset_jump_max = rospy.get_param("~offset_jump_max", 0.40)      # m (차량 오프셋 급변)

        # warp 관련 기본값
        self.img_x = 0
        self.img_y = 0
        self.offset_x = rospy.get_param("~bev_offset_x", 80)

        # 상태
        self.enabled = True
        self.last_pub_time = rospy.get_time()
        self.nothing_flag = False

        # 윈도우 파라미터 (img_CB에서 세팅)
        self.nwindows = 10
        self.window_height = 0

        # 필터/히스토리
        self.offset_filt = 0.0
        self.last_ang = 0.0
        self.last_good_offset = 0.0
        self.last_offset_meas = 0.0

    def enable_cb(self, msg: Bool):
        self.enabled = bool(msg.data)
        if not self.enabled:
            stop = Twist()
            self.ctrl_pub.publish(stop)
            self.last_ang = 0.0

    # ---------------------------------------------------------------------
    # 색 기반 차선 검출
    # ---------------------------------------------------------------------
    def detect_color(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        yellow_lower = np.array([15, 80, 0], dtype=np.uint8)
        yellow_upper = np.array([45, 255, 255], dtype=np.uint8)

        white_lower = np.array([0, 0, 200], dtype=np.uint8)
        white_upper = np.array([179, 64, 255], dtype=np.uint8)

        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        white_mask = cv2.inRange(hsv, white_lower, white_upper)

        blend_mask = yellow_mask | white_mask
        return cv2.bitwise_and(img, img, mask=blend_mask)

    # ---------------------------------------------------------------------
    # Bird-Eye-View warp
    # ---------------------------------------------------------------------
    def img_warp(self, img):
        self.img_x, self.img_y = img.shape[1], img.shape[0]

        src_center_offset = [200, 315]
        src = np.array(
            [
                [0, self.img_y - 1],
                [src_center_offset[0], src_center_offset[1]],
                [self.img_x - src_center_offset[0], src_center_offset[1]],
                [self.img_x - 1, self.img_y - 1],
            ],
            dtype=np.float32,
        )

        dst = np.array(
            [
                [self.offset_x, self.img_y],
                [self.offset_x, 0],
                [self.img_x - self.offset_x, 0],
                [self.img_x - self.offset_x, self.img_y],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(src, dst)
        warp_img = cv2.warpPerspective(img, matrix, (self.img_x, self.img_y))
        return warp_img

    # ---------------------------------------------------------------------
    # 이진화 + 중앙 영역 마스크
    # ---------------------------------------------------------------------
    def img_binary(self, blend_line):
        center_y_ratio = 0.55
        up_ratio = 0.50
        down_ratio = 0.50
        half_width_ratio = 0.30

        gray = cv2.cvtColor(blend_line, cv2.COLOR_BGR2GRAY)
        _, binary_line = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY)

        h, w = binary_line.shape
        cx = w // 2
        cy = int(h * center_y_ratio)
        up = int(h * up_ratio)
        dn = int(h * down_ratio)
        hw = int(w * half_width_ratio)

        x0, x1 = max(0, cx - hw), min(w, cx + hw)
        y0, y1 = max(0, cy - up), min(h, cy + dn)
        binary_line[y0:y1, x0:x1] = 0

        return binary_line

    # ---------------------------------------------------------------------
    # nothing일 때 기본 픽셀 위치
    # ---------------------------------------------------------------------
    def detect_nothing(self):
        offset = int(self.img_x * 0.140625)
        self.nothing_left_x_base = offset
        self.nothing_right_x_base = self.img_x - offset

        self.nothing_pixel_left_x = np.full(self.nwindows, self.nothing_left_x_base, dtype=np.int32)
        self.nothing_pixel_right_x = np.full(self.nwindows, self.nothing_right_x_base, dtype=np.int32)

        base_y = int(self.window_height / 2)
        self.nothing_pixel_y = np.arange(0, self.nwindows * base_y, base_y, dtype=np.int32)

    # ---------------------------------------------------------------------
    # 슬라이딩 윈도우 탐색 + 통계(stats) 반환
    # ---------------------------------------------------------------------
    def window_search(self, binary_line):
        h, w = binary_line.shape

        bottom_half = binary_line[h // 2 :, :]
        histogram = np.sum(bottom_half, axis=0)

        midpoint = w // 2
        left_x_base = int(np.argmax(histogram[:midpoint]))
        right_x_base = int(np.argmax(histogram[midpoint:]) + midpoint)

        left_x_current = left_x_base if left_x_base != 0 else self.nothing_left_x_base
        right_x_current = right_x_base if right_x_base != midpoint else self.nothing_right_x_base

        out_img = (np.dstack((binary_line, binary_line, binary_line)).astype(np.uint8) * 255)

        nwindows = self.nwindows
        window_height = self.window_height
        margin = 80
        min_pix = int((margin * 2 * window_height) * 0.0031)

        lane_y, lane_x = binary_line.nonzero()
        lane_y = lane_y.astype(np.int32)
        lane_x = lane_x.astype(np.int32)

        left_lane_idx_list = []
        right_lane_idx_list = []

        left_hit = 0
        right_hit = 0

        for window in range(nwindows):
            win_y_low = h - (window + 1) * window_height
            win_y_high = h - window * window_height

            left_low = left_x_current - margin
            left_high = left_x_current + margin
            right_low = right_x_current - margin
            right_high = right_x_current + margin

            if left_x_current != 0:
                cv2.rectangle(out_img, (left_low, win_y_low), (left_high, win_y_high), (0, 255, 0), 2)
            if right_x_current != midpoint:
                cv2.rectangle(out_img, (right_low, win_y_low), (right_high, win_y_high), (0, 0, 255), 2)

            in_window = (lane_y >= win_y_low) & (lane_y < win_y_high)

            good_left_idx = np.where(in_window & (lane_x >= left_low) & (lane_x < left_high))[0]
            good_right_idx = np.where(in_window & (lane_x >= right_low) & (lane_x < right_high))[0]

            left_lane_idx_list.append(good_left_idx)
            right_lane_idx_list.append(good_right_idx)

            if len(good_left_idx) > min_pix:
                left_x_current = int(np.mean(lane_x[good_left_idx]))
                left_hit += 1
            if len(good_right_idx) > min_pix:
                right_x_current = int(np.mean(lane_x[good_right_idx]))
                right_hit += 1

        left_lane_idx = np.concatenate(left_lane_idx_list) if left_lane_idx_list else np.array([], dtype=int)
        right_lane_idx = np.concatenate(right_lane_idx_list) if right_lane_idx_list else np.array([], dtype=int)

        left_x_raw = lane_x[left_lane_idx]
        left_y_raw = lane_y[left_lane_idx]
        right_x_raw = lane_x[right_lane_idx]
        right_y_raw = lane_y[right_lane_idx]

        left_cnt_raw = int(len(left_x_raw))
        right_cnt_raw = int(len(right_x_raw))

        # fallback 유지
        if left_cnt_raw == 0 and right_cnt_raw == 0:
            left_x = self.nothing_pixel_left_x
            left_y = self.nothing_pixel_y
            right_x = self.nothing_pixel_right_x
            right_y = self.nothing_pixel_y
        else:
            if left_cnt_raw == 0:
                left_x = right_x_raw - self.img_x // 2
                left_y = right_y_raw
                right_x = right_x_raw
                right_y = right_y_raw
            elif right_cnt_raw == 0:
                right_x = left_x_raw + self.img_x // 2
                right_y = left_y_raw
                left_x = left_x_raw
                left_y = left_y_raw
            else:
                left_x, left_y, right_x, right_y = left_x_raw, left_y_raw, right_x_raw, right_y_raw

        def safe_polyfit(y, x):
            if len(x) < 3 or len(y) < 3:
                return None
            try:
                return np.polyfit(y, x, 2)
            except Exception:
                return None

        left_fit = safe_polyfit(left_y, left_x)
        right_fit = safe_polyfit(right_y, right_x)

        if left_fit is None and right_fit is None:
            left_fit = np.array([0.0, 0.0, float(self.nothing_left_x_base)], dtype=np.float32)
            right_fit = np.array([0.0, 0.0, float(self.nothing_right_x_base)], dtype=np.float32)
        elif left_fit is None:
            left_fit = right_fit.copy()
            left_fit[2] -= (self.img_x / 2.0)
        elif right_fit is None:
            right_fit = left_fit.copy()
            right_fit[2] += (self.img_x / 2.0)

        plot_y = np.linspace(0, h - 1, 20)
        left_fit_x = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
        right_fit_x = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]
        center_fit_x = (right_fit_x + left_fit_x) / 2.0

        width = right_fit_x - left_fit_x
        width_mean = float(np.mean(width))
        width_std = float(np.std(width))

        left_pts = np.int32(np.column_stack((left_fit_x, plot_y)))
        right_pts = np.int32(np.column_stack((right_fit_x, plot_y)))
        center_pts = np.int32(np.column_stack((center_fit_x, plot_y)))

        cv2.polylines(out_img, [left_pts], False, (0, 0, 255), 5)
        cv2.polylines(out_img, [right_pts], False, (0, 255, 0), 5)
        cv2.polylines(out_img, [center_pts], False, (255, 0, 0), 3)

        center_fit = (left_fit + right_fit) / 2.0

        stats = {
            "left_cnt_raw": left_cnt_raw,
            "right_cnt_raw": right_cnt_raw,
            "left_hit": left_hit,
            "right_hit": right_hit,
            "width_mean": width_mean,
            "width_std": width_std,
            "img_w": w,
            "img_h": h,
        }

        return out_img, left_fit, right_fit, center_fit, stats

    # ---------------------------------------------------------------------
    # 픽셀 → 미터 변환 비율 (기존 유지)
    # ---------------------------------------------------------------------
    def meter_per_pixel(self):
        world_warp = np.array(
            [[97, 1610], [109, 1610], [109, 1606], [97, 1606]], dtype=np.float32
        )

        dx_x = world_warp[0, 0] - world_warp[3, 0]
        dy_x = world_warp[0, 1] - world_warp[3, 1]
        meter_x = dx_x * dx_x + dy_x * dy_x

        dx_y = world_warp[0, 0] - world_warp[1, 0]
        dy_y = world_warp[0, 1] - world_warp[1, 1]
        meter_y = dx_y * dx_y + dy_y * dy_y

        meter_per_pix_x = meter_x / float(self.img_x if self.img_x else 1.0)
        meter_per_pix_y = meter_y / float(self.img_y if self.img_y else 1.0)
        return meter_per_pix_x, meter_per_pix_y

    # ---------------------------------------------------------------------
    # lookahead y에서의 차량 오프셋 계산
    # ---------------------------------------------------------------------
    def calc_vehicle_offset_y(self, img_w, y_query, left_fit, right_fit):
        y = float(clamp(y_query, 0, self.img_y - 1))
        y2 = y * y

        a_l, b_l, c_l = left_fit
        a_r, b_r, c_r = right_fit

        x_left = a_l * y2 + b_l * y + c_l
        x_right = a_r * y2 + b_r * y + c_r

        img_center_x = img_w / 2.0
        lane_center_x = (x_left + x_right) / 2.0

        pixel_offset = img_center_x - lane_center_x
        meter_per_pix_x, _ = self.meter_per_pixel()
        vehicle_offset_m = pixel_offset * (2.0 * meter_per_pix_x)
        return float(vehicle_offset_m)

    # ---------------------------------------------------------------------
    # curvature proxy: center poly에서 pixel radius 계산
    # ---------------------------------------------------------------------
    def curvature_radius_px(self, center_fit, y_eval):
        a, b, _ = center_fit
        y = float(y_eval)
        denom = abs(2.0 * a)
        if denom < 1e-6:
            return 1e9
        num = (1.0 + (2.0 * a * y + b) ** 2) ** 1.5
        return float(num / denom)

    # ---------------------------------------------------------------------
    # confidence 계산 (0~1)
    # ---------------------------------------------------------------------
    def compute_confidence(self, stats, offset_meas, offset_prev):
        w = float(stats["img_w"])
        left_cnt = stats["left_cnt_raw"]
        right_cnt = stats["right_cnt_raw"]
        total = left_cnt + right_cnt

        width_mean = stats["width_mean"]
        width_std = stats["width_std"]

        conf = 1.0

        if total < self.min_total_pixels:
            conf *= 0.6
        if left_cnt < self.min_side_pixels:
            conf *= 0.7
        if right_cnt < self.min_side_pixels:
            conf *= 0.7
        if left_cnt < 30 and right_cnt < 30:
            conf *= 0.2

        width_min = w * self.width_min_ratio
        width_max = w * self.width_max_ratio
        if not (width_min <= width_mean <= width_max):
            conf *= 0.3
        if width_std > self.width_std_max:
            conf *= 0.5

        if abs(offset_meas - offset_prev) > self.offset_jump_max:
            conf *= 0.5

        return float(clamp(conf, 0.0, 1.0))

    # ---------------------------------------------------------------------
    # cmd 생성
    # ---------------------------------------------------------------------
    def build_cmd(self, speed, ang_z):
        msg = Twist()
        msg.linear.x = float(speed)
        msg.angular.z = float(ang_z)
        return msg

    # ---------------------------------------------------------------------
    # 콜백
    # ---------------------------------------------------------------------
    def img_CB(self, data):
        if not self.enabled:
            return

        now = rospy.get_time()

        # 1) 이미지 변환
        img = self.bridge.compressed_imgmsg_to_cv2(data)

        # 2) 윈도우 파라미터
        self.nwindows = 10
        self.window_height = img.shape[0] // self.nwindows

        # 3) warp → 색 검출 → 이진화
        warp_img = self.img_warp(img)
        blend_img = self.detect_color(warp_img)
        binary_img = self.img_binary(blend_img)

        # 4) nothing 초기값
        if not self.nothing_flag:
            self.detect_nothing()
            self.nothing_flag = True

        # 5) sliding window + fits
        sliding_window_img, left_fit, right_fit, center_fit, stats = self.window_search(binary_img)

        h, w = binary_img.shape
        y_look = int(clamp(h - 1 - self.lookahead_px, 0, h - 1))

        # 6) offset(measure) at lookahead y
        offset_meas = self.calc_vehicle_offset_y(w, y_look, left_fit, right_fit)

        # 7) curvature proxy
        radius_px = self.curvature_radius_px(center_fit, y_look)

        # 8) confidence
        conf = self.compute_confidence(stats, offset_meas, self.last_offset_meas)
        self.last_offset_meas = offset_meas

        # 9) offset filtering / hold
        if conf < self.conf_hold_thresh:
            target = self.last_good_offset
        else:
            self.last_good_offset = offset_meas
            target = offset_meas

        alpha = 0.15 + (1.0 - conf) * 0.70
        self.offset_filt = alpha * self.offset_filt + (1.0 - alpha) * target

        # 10) steering
        ang_raw = -self.offset_filt * self.turn_mult

        if abs(ang_raw) < self.steer_deadband:
            ang_raw = 0.0
        ang_raw = float(clamp(ang_raw, -self.max_ang, self.max_ang))

        dt = max(1e-3, now - self.last_pub_time)
        max_step = self.max_ang_rate * dt
        ang = self.last_ang + clamp(ang_raw - self.last_ang, -max_step, max_step)
        self.last_ang = float(ang)

        # 11) speed scheduling
        speed = self.base_speed
        if radius_px < self.radius_thresh_px:
            speed = min(speed, self.curve_speed)
        if conf < self.conf_slow_thresh:
            speed = min(speed, self.low_conf_speed)

        # 12) publish at 0.1s
        if now - self.last_pub_time >= 0.1:
            self.ctrl_pub.publish(self.build_cmd(speed, ang))
            self.last_pub_time = now

        # Debug image
        try:
            dbg = sliding_window_img.copy()
            cv2.putText(dbg, f"conf={conf:.2f}  radius_px={radius_px:.0f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(dbg, f"v={speed:.2f}  wz={ang:.2f}  y_look={y_look}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.line(dbg, (0, y_look), (w - 1, y_look), (255, 255, 0), 2)

            msg = CompressedImage()
            msg.header = data.header
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode(".jpg", dbg)[1]).tobytes()
            self.pub_dbg.publish(msg)
        except Exception:
            pass


if __name__ == "__main__":
    node = LKAS()
    rospy.spin()