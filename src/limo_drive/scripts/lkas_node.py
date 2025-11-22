#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from std_msgs.msg import Bool


import numpy as np
import cv2
from math import atan


class LKAS:
    def __init__(self):
        # CvBridge
        self.bridge = CvBridge()

        # ROS node
        rospy.init_node("LKAS_node")

        # 디버그 이미지 publish (buffer 최소화)
        self.pub = rospy.Publisher(
            "/sliding_windows/compressed",
            CompressedImage,
            queue_size=1
        )

        # 카메라 subscribe (최신 프레임만 유지)
        rospy.Subscriber(
            "/camera/rgb/image_raw/compressed",
            CompressedImage,
            self.img_CB,
            queue_size=1
        )

        # cmd_vel publisher
        self.ctrl_pub = rospy.Publisher("/cmd_vel_lkas", Twist, queue_size=1)
        self.speed = 0.24
        self.trun_mutip = 0.14

        # 상태 변수들
        self.start_time = rospy.get_time()
        self.nothing_flag = False
        self.cmd_vel_msg = Twist()

        self.frame_skip = 3     # 3프레임 마다 계산
        self._frame_count = 0

        # warp 관련 기본값
        self.img_x = 0
        self.img_y = 0
        self.offset_x = 40  # BEV에서 좌우 여유. 필요하면 조절

        self.enabled = True   # 기본값: FSM 없이 단독 돌릴 때도 동작하도록
        rospy.Subscriber("/lkas_enable", Bool, self.enable_cb, queue_size=1)

    def enable_cb(self, msg: Bool):
        self.enabled = msg.data

    # ---------------------------------------------------------------------
    # 색 기반 차선 검출
    # ---------------------------------------------------------------------
    def detect_color(self, img):
        # BGR → HSV 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 노란색 범위
        yellow_lower = np.array([15, 80, 0], dtype=np.uint8)
        yellow_upper = np.array([45, 255, 255], dtype=np.uint8)

        # 흰색 범위
        white_lower = np.array([0, 0, 200], dtype=np.uint8)
        white_upper = np.array([179, 64, 255], dtype=np.uint8)

        # 마스크 계산
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        white_mask = cv2.inRange(hsv, white_lower, white_upper)

        # 두 마스크 OR 연산
        blend_mask = yellow_mask | white_mask

        # 원본 이미지에서 해당 색만 남기기
        return cv2.bitwise_and(img, img, mask=blend_mask)

    # ---------------------------------------------------------------------
    # Bird-Eye-View warp
    # ---------------------------------------------------------------------
    def img_warp(self, img):
        self.img_x, self.img_y = img.shape[1], img.shape[0]

        # src는 원본 이미지 상에서의 포인트 (사다리꼴)
        src_center_offset = [100, 158]
        src = np.array(
            [
                [0, self.img_y - 1],
                [src_center_offset[0], src_center_offset[1]],
                [self.img_x - src_center_offset[0], src_center_offset[1]],
                [self.img_x - 1, self.img_y - 1],
            ],
            dtype=np.float32,
        )

        # dst는 BEV 상에서의 직사각형 영역
        # → offset_x 만큼 좌우 여유를 둔 박스로 투영
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
        # ---- 중앙 마스크 파라미터 ----
        center_y_ratio = 0.55     # 중앙점 세로 위치(화면 높이의 55%)
        up_ratio = 0.50           # 중앙점에서 위로 지울 높이 비율
        down_ratio = 0.50         # 중앙점에서 아래로 지울 높이 비율
        half_width_ratio = 0.30   # 중앙 사각형의 좌우 반폭 비율

        # 1) 기본 이진화
        gray = cv2.cvtColor(blend_line, cv2.COLOR_BGR2GRAY)
        _, binary_line = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY)

        # 2) 중앙 사각형 마스크 처리
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

        self.nothing_pixel_left_x = np.full(
            self.nwindows, self.nothing_left_x_base, dtype=np.int32
        )
        self.nothing_pixel_right_x = np.full(
            self.nwindows, self.nothing_right_x_base, dtype=np.int32
        )

        base_y = int(self.window_height / 2)
        self.nothing_pixel_y = np.arange(
            0, self.nwindows * base_y, base_y, dtype=np.int32
        )

    # ---------------------------------------------------------------------
    # 슬라이딩 윈도우 탐색
    # ---------------------------------------------------------------------
    def window_search(self, binary_line):
        h, w = binary_line.shape

        # 1) 히스토그램
        bottom_half = binary_line[h // 2 :, :]
        histogram = np.sum(bottom_half, axis=0)

        midpoint = w // 2
        left_x_base = np.argmax(histogram[:midpoint])
        right_x_base = np.argmax(histogram[midpoint:]) + midpoint

        # 2) 초기 x 위치
        left_x_current = (
            left_x_base if left_x_base != 0 else self.nothing_left_x_base
        )
        right_x_current = (
            right_x_base
            if right_x_base != midpoint
            else self.nothing_right_x_base
        )

        # 3) 출력 이미지 준비
        out_img = (
            np.dstack((binary_line, binary_line, binary_line))
            .astype(np.uint8)
            * 255
        )

        # window parameter
        nwindows = self.nwindows
        window_height = self.window_height
        margin = 80
        min_pix = int((margin * 2 * window_height) * 0.0031)

        # 모든 nonzero 픽셀
        lane_y, lane_x = binary_line.nonzero()
        lane_y = lane_y.astype(np.int32)
        lane_x = lane_x.astype(np.int32)

        # 픽셀 index를 담을 list
        left_lane_idx_list = []
        right_lane_idx_list = []

        # 4) 윈도우 루프
        for window in range(nwindows):
            # window boundary (세로)
            win_y_low = h - (window + 1) * window_height
            win_y_high = h - window * window_height

            # 좌/우 window x 범위
            left_low = left_x_current - margin
            left_high = left_x_current + margin
            right_low = right_x_current - margin
            right_high = right_x_current + margin

            # 디버그용 사각형
            if left_x_current != 0:
                cv2.rectangle(
                    out_img,
                    (left_low, win_y_low),
                    (left_high, win_y_high),
                    (0, 255, 0),
                    2,
                )
            if right_x_current != midpoint:
                cv2.rectangle(
                    out_img,
                    (right_low, win_y_low),
                    (right_high, win_y_high),
                    (0, 0, 255),
                    2,
                )

            # window 내 픽셀 인덱스 계산
            in_window = (lane_y >= win_y_low) & (lane_y < win_y_high)

            good_left_idx = np.where(
                in_window & (lane_x >= left_low) & (lane_x < left_high)
            )[0]
            good_right_idx = np.where(
                in_window & (lane_x >= right_low) & (lane_x < right_high)
            )[0]

            left_lane_idx_list.append(good_left_idx)
            right_lane_idx_list.append(good_right_idx)

            # 픽셀 수가 충분하면 window 중심을 업데이트
            if len(good_left_idx) > min_pix:
                left_x_current = int(np.mean(lane_x[good_left_idx]))
            if len(good_right_idx) > min_pix:
                right_x_current = int(np.mean(lane_x[good_right_idx]))

        # 5) 전체 인덱스 하나로 합치기
        left_lane_idx = (
            np.concatenate(left_lane_idx_list)
            if left_lane_idx_list
            else np.array([], dtype=int)
        )
        right_lane_idx = (
            np.concatenate(right_lane_idx_list)
            if right_lane_idx_list
            else np.array([], dtype=int)
        )

        # 6) 픽셀 좌표
        left_x = lane_x[left_lane_idx]
        left_y = lane_y[left_lane_idx]
        right_x = lane_x[right_lane_idx]
        right_y = lane_y[right_lane_idx]

        # 7) fallback 처리
        if len(left_x) == 0 and len(right_x) == 0:
            left_x = self.nothing_pixel_left_x
            left_y = self.nothing_pixel_y
            right_x = self.nothing_pixel_right_x
            right_y = self.nothing_pixel_y
        else:
            if len(left_x) == 0:
                left_x = right_x - self.img_x // 2
                left_y = right_y
            elif len(right_x) == 0:
                right_x = left_x + self.img_x // 2
                right_y = left_y

        # 8) 곡선 피팅
        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)

        plot_y = np.linspace(0, h - 1, 5)
        left_fit_x = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
        right_fit_x = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]
        center_fit_x = (right_fit_x + left_fit_x) / 2.0

        left_pts = np.int32(np.column_stack((left_fit_x, plot_y)))
        right_pts = np.int32(np.column_stack((right_fit_x, plot_y)))
        center_pts = np.int32(np.column_stack((center_fit_x, plot_y)))

        cv2.polylines(out_img, [left_pts], False, (0, 0, 255), 5)
        cv2.polylines(out_img, [right_pts], False, (0, 255, 0), 5)
        cv2.polylines(out_img, [center_pts], False, (255, 0, 0), 3)

        return out_img, left_pts, right_pts, center_pts, left_x, left_y, right_x, right_y

    # ---------------------------------------------------------------------
    # 픽셀 → 미터 변환 비율
    # ---------------------------------------------------------------------
    def meter_per_pixel(self):
        # 고정된 월드 좌표 (타입 명시)
        world_warp = np.array(
            [[97, 1610], [109, 1610], [109, 1606], [97, 1606]], dtype=np.float32
        )

        # x 방향(세로) 거리
        dx_x = world_warp[0, 0] - world_warp[3, 0]
        dy_x = world_warp[0, 1] - world_warp[3, 1]
        meter_x = dx_x * dx_x + dy_x * dy_x

        # y 방향(가로) 거리
        dx_y = world_warp[0, 0] - world_warp[1, 0]
        dy_y = world_warp[0, 1] - world_warp[1, 1]
        meter_y = dx_y * dx_y + dy_y * dy_y

        meter_per_pix_x = meter_x / float(self.img_x)
        meter_per_pix_y = meter_y / float(self.img_y)

        return meter_per_pix_x, meter_per_pix_y

    # ---------------------------------------------------------------------
    # 곡률 계산
    # ---------------------------------------------------------------------
    def calc_curve(self, left_x, left_y, right_x, right_y):
        # 평가할 y (화면 맨 아래 쪽)
        y_eval = self.img_x - 1  # 원 코드 유지

        # 픽셀 → 미터 변환 계수
        meter_per_pix_x, meter_per_pix_y = self.meter_per_pixel()

        # 월드 좌표(미터 단위)로 스케일링
        left_y_m = left_y * meter_per_pix_y
        left_x_m = left_x * meter_per_pix_x
        right_y_m = right_y * meter_per_pix_y
        right_x_m = right_x * meter_per_pix_x

        # 2차 다항식 피팅
        left_fit_cr = np.polyfit(left_y_m, left_x_m, 2)
        right_fit_cr = np.polyfit(right_y_m, right_x_m, 2)

        # 공통으로 쓰는 y_eval·meter_per_pix_y 곱
        y_eval_m = y_eval * meter_per_pix_y

        # 곡률 계산
        a_l, b_l = left_fit_cr[0], left_fit_cr[1]
        a_r, b_r = right_fit_cr[0], right_fit_cr[1]

        denom_l = 2.0 * a_l
        denom_r = 2.0 * a_r

        left_curve_radius = (
            (1.0 + (2.0 * a_l * y_eval_m + b_l) ** 2) ** 1.5 / np.abs(denom_l)
        )
        right_curve_radius = (
            (1.0 + (2.0 * a_r * y_eval_m + b_r) ** 2) ** 1.5 / np.abs(denom_r)
        )

        return left_curve_radius, right_curve_radius

    # ---------------------------------------------------------------------
    # 차량 중심에서 차선 중앙까지 오프셋 계산
    # ---------------------------------------------------------------------
    def calc_vehicle_offset(self, sliding_window_img, left_x, left_y, right_x, right_y):
        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)

        h = sliding_window_img.shape[0]
        bottom_y = h - 1
        y2 = bottom_y * bottom_y

        a_l, b_l, c_l = left_fit
        a_r, b_r, c_r = right_fit

        bottom_x_left = a_l * y2 + b_l * bottom_y + c_l
        bottom_x_right = a_r * y2 + b_r * bottom_y + c_r

        img_center_x = sliding_window_img.shape[1] / 2.0
        lane_center_x = (bottom_x_left + bottom_x_right) / 2.0
        pixel_offset = img_center_x - lane_center_x

        meter_per_pix_x, _ = self.meter_per_pixel()
        vehicle_offset = pixel_offset * (2 * meter_per_pix_x)

        return vehicle_offset

    # ---------------------------------------------------------------------
    # 카메라 기반 조향각 계산 (현재는 사용 안 함)
    # ---------------------------------------------------------------------
    def cam_cal_steer(self, left_curve_radius, right_curve_radius, vehicle_offset):
        curvature = 2.0 / (left_curve_radius + right_curve_radius)
        cam_steer = atan(curvature - 0.5 * curvature) * 100  # atan(0.5 * curvature)

        if vehicle_offset > 0:
            cam_steer = -cam_steer

        return cam_steer

    # ---------------------------------------------------------------------
    # 속도/조향 명령 생성
    # ---------------------------------------------------------------------
    def ctrl_cmd(self, vehicle_offset):
        self.cmd_vel_msg.linear.x = self.speed
        self.cmd_vel_msg.angular.z = -vehicle_offset * self.trun_mutip
        return self.cmd_vel_msg

    # ---------------------------------------------------------------------
    # 콜백
    # ---------------------------------------------------------------------
    def img_CB(self, data):
        now = rospy.get_time()
        if not self.enabled:
            return
        
        self._frame_count += 1
        if self._frame_count % self.frame_skip != 0:
            return
        # 1) 이미지 변환
        img = self.bridge.compressed_imgmsg_to_cv2(data)

        # === 해상도 1/2 DOWN ===
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // 2, h // 2))

        # 2) 윈도우 파라미터
        self.nwindows = 10
        self.window_height = img.shape[0] // self.nwindows

        # 3) 파이프라인: warp → 색 검출 → 이진화
        warp_img = self.img_warp(img)
        blend_img = self.detect_color(warp_img)
        binary_img = self.img_binary(blend_img)

        # 4) nothing 초기값 설정
        if not self.nothing_flag:
            self.detect_nothing()
            self.nothing_flag = True

        # 5) 슬라이딩 윈도우로 차선 검출
        (
            sliding_window_img,
            left,
            right,
            center,
            left_x,
            left_y,
            right_x,
            right_y,
        ) = self.window_search(binary_img)

        # 6) 곡률 / 오프셋 계산
        left_curve_radius, right_curve_radius = self.calc_curve(
            left_x, left_y, right_x, right_y
        )
        vehicle_offset = self.calc_vehicle_offset(
            sliding_window_img, left_x, left_y, right_x, right_y
        )

        # 7) 제어 명령 생성
        ctrl_cmd_msg = self.ctrl_cmd(vehicle_offset)

        # 8) 일정 주기로만 cmd_vel publish (0.1s)
        if now - self.start_time >= 0.1:
            self.ctrl_pub.publish(ctrl_cmd_msg)
            self.start_time = now


if __name__ == "__main__":
    lkas = LKAS()
    rospy.spin()
