#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Twist
import enum
import copy


class DriveMode(enum.Enum):
    LANE_FOLLOW = 0
    OBSTACLE_AVOID = 1


class Mission3Phase(enum.Enum):
    BEFORE = 0        # 아직 미션3 시퀀스 시작 전
    TURNING = 1       # 고정 조향 구간
    FORCE_OBS = 2     # 장애물 회피 강제 ON 구간
    DONE = 3          # 미션3 시퀀스 끝


class V2XPhase(enum.Enum):
    WAIT_START = 0    # 아직 v2x 타이머 시작 전
    COUNTING = 1      # v2x 타이머 증가 중
    TURNING = 2       # v2x 고정 조향 구간
    DONE = 3          # v2x 시퀀스 끝


class FSMMuxNode:
    """
    FSM + MUX 노드 + lab time 2개

    입력:
      - /obstacle_state   (Bool)          : 장애물 여부 (Detect 노드 출력, sta)
      - /cmd_vel_lkas     (Twist)         : LKAS 노드 출력
      - /cmd_vel_obstacle (Twist)         : 장애물 회피 노드 출력
      - /path_            (String) [선택] : V2X 모드 (A/B/C...), 일단 상태만 저장

    출력:
      - /cmd_vel          (Twist)         : 최종 속도 명령 (로봇에 들어가는 것)
      - /drive_mode       (String)        : 현재 FSM 모드 ("LANE_FOLLOW" / "OBSTACLE_AVOID")
      - /lkas_enable      (Bool)          : LKAS 연산 on/off 플래그
      - /obstacle_enable  (Bool)          : obstacle 연산 on/off 플래그

    추가 로직:
      - lab_time_m3 : "첫 LKAS cmd 수신 시점" 부터 미션3 타이밍 제어
      - lab_time_v2x: 미션3 끝난 뒤, obstacle 회피 끝난 시점부터 v2x 타이밍 제어
    """

    def __init__(self):
        rospy.init_node("fsm_mux_node")

        # --- 파라미터 ---
        self.obstacle_topic        = rospy.get_param("~obstacle_topic", "/obstacle_state")
        self.lkas_cmd_topic        = rospy.get_param("~lkas_cmd_topic", "/cmd_vel_lkas")
        self.obstacle_cmd_topic    = rospy.get_param("~obstacle_cmd_topic", "/cmd_vel_obstacle")
        self.cmd_out_topic         = rospy.get_param("~cmd_out_topic", "/cmd_vel")
        self.mode_topic            = rospy.get_param("~mode_topic", "/drive_mode")
        self.v2x_topic             = rospy.get_param("~v2x_topic", "/path_")
        self.lkas_enable_topic     = rospy.get_param("~lkas_enable_topic", "/lkas_enable")
        self.obstacle_enable_topic = rospy.get_param("~obstacle_enable_topic", "/obstacle_enable")

        # 메인 루프 주기
        self.loop_rate = rospy.get_param("~loop_rate", 30.0)

        # ----- Mission3 / V2X 타이밍 파라미터 -----
        self.m3_phase_time       = rospy.get_param("~m3_phase_time", 40.0)   # ★ 런치가 아니라 "첫 LKAS cmd 이후" 기준
        self.m3_turn_duration    = rospy.get_param("~m3_turn_duration", 3.0)
        self.m3_turn_speed       = rospy.get_param("~m3_turn_speed", 0.16)
        self.m3_turn_yaw         = rospy.get_param("~m3_turn_yaw", -0.4)

        self.m3_force_obs_duration = rospy.get_param("~m3_force_obs_duration", 4.0)

        self.v2x_phase_time      = rospy.get_param("~v2x_phase_time", 22.0)
        self.v2x_turn_duration   = rospy.get_param("~v2x_turn_duration", 3.0)
        self.v2x_turn_speed      = rospy.get_param("~v2x_turn_speed", 0.16)
        self.v2x_turn_yaw        = rospy.get_param("~v2x_turn_yaw", 0.4)

        # --- 상태 변수들 ---
        self.current_mode = DriveMode.LANE_FOLLOW
        self.prev_mode    = self.current_mode

        # 장애물 감지 상태
        self.obstacle_state = False

        # 마지막으로 받은 cmd_vel
        self.last_lkas_cmd     = Twist()
        self.last_obstacle_cmd = Twist()
        self.have_lkas_cmd     = False
        self.have_obstacle_cmd = False

        # V2X 모드 (필요하면 나중에 FSM/출력 로직에 사용)
        self.v2x_mode = "D"

        # ----- 타이머 / 페이즈 상태 -----
        # ★ 변경: start_time을 init에서 바로 잡지 않고,
        #         첫 LKAS cmd가 들어오는 순간에 설정한다.
        self.start_time = None                 # lab_time_m3 기준 시작 시각 (첫 LKAS cmd 시점)
        self.m3_phase = Mission3Phase.BEFORE
        self.m3_turn_start_time = None
        self.m3_force_obs_start_time = None

        self.v2x_phase = V2XPhase.WAIT_START
        self.v2x_start_time = None
        self.v2x_turn_start_time = None

        # ★ 로그 주기 (초) – 자주 보고 싶으면 0.1로 세팅 (기본 0.5)
        self.log_dt = rospy.get_param("~log_dt", 0.5)
        self.last_log_time = None

        # --- Pub/Sub ---
        rospy.Subscriber(self.obstacle_topic,     Bool,   self.obstacle_cb,      queue_size=1)
        rospy.Subscriber(self.lkas_cmd_topic,     Twist,  self.lkas_cmd_cb,      queue_size=1)
        rospy.Subscriber(self.obstacle_cmd_topic, Twist,  self.obstacle_cmd_cb,  queue_size=1)
        rospy.Subscriber(self.v2x_topic,          String, self.v2x_cb,           queue_size=1)

        self.cmd_pub             = rospy.Publisher(self.cmd_out_topic,         Twist,  queue_size=1)
        self.mode_pub            = rospy.Publisher(self.mode_topic,            String, queue_size=1)
        self.lkas_enable_pub     = rospy.Publisher(self.lkas_enable_topic,     Bool,   queue_size=1)
        self.obstacle_enable_pub = rospy.Publisher(self.obstacle_enable_topic, Bool,   queue_size=1)

        # 메인 루프용 타이머
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.loop_rate), self.update)

        rospy.loginfo("[FSM_MUX] node started.")
        rospy.loginfo("[FSM_MUX] obstacle_topic=%s, lkas_cmd_topic=%s, obstacle_cmd_topic=%s",
                      self.obstacle_topic, self.lkas_cmd_topic, self.obstacle_cmd_topic)

    # ======================
    # 콜백들
    # ======================

    def obstacle_cb(self, msg: Bool):
        self.obstacle_state = msg.data

        # v2x COUNTING 중에 장애물 다시 뜨면 타이머 리셋
        if self.m3_phase == Mission3Phase.DONE and self.v2x_phase == V2XPhase.COUNTING and msg.data:
            self.v2x_start_time = rospy.get_time()
            rospy.loginfo("[FSM_MUX] v2x lab time reset (obstacle re-detected).")

    def lkas_cmd_cb(self, msg: Twist):
        self.last_lkas_cmd = msg

        # ★ 첫 LKAS cmd 들어오는 타이밍에 lab_time_m3 시작
        if not self.have_lkas_cmd:
            rospy.loginfo("[FSM_MUX] first LKAS cmd received.")
        self.have_lkas_cmd = True

        if self.start_time is None:
            self.start_time = rospy.get_time()
            self.last_log_time = self.start_time
            rospy.loginfo("[FSM_MUX] lab_time_m3 START (t_m3=0.0, from first LKAS cmd).")

    def obstacle_cmd_cb(self, msg: Twist):
        self.last_obstacle_cmd = msg
        self.have_obstacle_cmd = True

    def v2x_cb(self, msg: String):
        self.v2x_mode = msg.data

    # ======================
    #  FSM + 타이머 메인 로직
    # ======================

    def update(self, event):
        now = rospy.get_time()

        # ★ start_time이 아직 없으면 t_m3는 0으로 고정
        if self.start_time is None:
            t_m3 = 0.0
        else:
            t_m3 = now - self.start_time

        # --- 미션3 페이즈 업데이트 ---
        self.update_mission3_phase(now, t_m3)

        # --- 기본 모드 업데이트 (장애물 기반 + 미션3 강제 구간 반영) ---
        old_mode = self.current_mode
        self.update_mode_basic()
        new_mode = self.current_mode

        # --- v2x 페이즈 업데이트 (미션3 끝난 이후) ---
        self.update_v2x_phase(now, old_mode, new_mode)

        # --- 모드 / enable 플래그 publish ---
        self.publish_mode()

        # --- 최종 cmd_vel publish (필요하면 고정 조향 override) ---
        self.publish_cmd(now)

        # --- lab time 로그 (log_dt마다) ---
        if self.v2x_start_time is None:
            t_v2x = 0.0
        else:
            t_v2x = max(0.0, now - self.v2x_start_time)

        if (self.last_log_time is None) or (now - self.last_log_time >= self.log_dt):
            rospy.loginfo(
                "[FSM_MUX] t_m3=%.1f (%s), t_v2x=%.1f (%s), mode=%s, obstacle=%s",
                t_m3,
                self.m3_phase.name,
                t_v2x,
                self.v2x_phase.name,
                self.current_mode.name,
                str(self.obstacle_state),
            )
            self.last_log_time = now

    # ---------- Mission3 Phase ----------

    def update_mission3_phase(self, now, t_m3):
        # ★ 아직 lab_time_m3 시작 전이면 아무것도 안 함
        if self.start_time is None:
            return

        if self.m3_phase == Mission3Phase.BEFORE:
            if t_m3 >= self.m3_phase_time:
                self.m3_phase = Mission3Phase.TURNING
                self.m3_turn_start_time = now
                rospy.loginfo("[FSM_MUX] Mission3 TURNING start (t_m3=%.1f)", t_m3)

        elif self.m3_phase == Mission3Phase.TURNING:
            if now - self.m3_turn_start_time >= self.m3_turn_duration:
                self.m3_phase = Mission3Phase.FORCE_OBS
                self.m3_force_obs_start_time = now
                rospy.loginfo("[FSM_MUX] Mission3 FORCE_OBS start")

        elif self.m3_phase == Mission3Phase.FORCE_OBS:
            if now - self.m3_force_obs_start_time >= self.m3_force_obs_duration:
                self.m3_phase = Mission3Phase.DONE
                rospy.loginfo("[FSM_MUX] Mission3 sequence DONE")

        # DONE이면 더이상 변경 없음

    # ---------- 기본 DriveMode 업데이트 ----------

    def update_mode_basic(self):
        """
        심플 FSM + 미션3 FORCE_OBS 강제 구간 반영:
          - 미션3 FORCE_OBS 시에는 무조건 OBSTACLE_AVOID
          - 그 외에는 obstacle_state == True  -> OBSTACLE_AVOID
                          obstacle_state == False -> LANE_FOLLOW
        """
        # 미션3 강제 obstacle 구간이면 무조건 OBSTACLE_AVOID
        if self.m3_phase == Mission3Phase.FORCE_OBS:
            if self.current_mode != DriveMode.OBSTACLE_AVOID:
                rospy.loginfo("[FSM_MUX] -> OBSTACLE_AVOID (Mission3 FORCE_OBS)")
            self.current_mode = DriveMode.OBSTACLE_AVOID
            return

        # 평상시: obstacle_state 기반
        if self.obstacle_state and self.current_mode != DriveMode.OBSTACLE_AVOID:
            rospy.loginfo("[FSM_MUX] -> OBSTACLE_AVOID (obstacle_state=True)")
            self.current_mode = DriveMode.OBSTACLE_AVOID

        elif (not self.obstacle_state) and self.current_mode != DriveMode.LANE_FOLLOW:
            rospy.loginfo("[FSM_MUX] -> LANE_FOLLOW (obstacle_state=False)")
            self.current_mode = DriveMode.LANE_FOLLOW

    # ---------- V2X Phase ----------

    def update_v2x_phase(self, now, old_mode, new_mode):
        # 1) v2x 타이머 시작 조건:
        #   - 미션3 시퀀스 끝난 상태 (m3_phase == DONE)
        #   - 이전 모드: OBSTACLE_AVOID
        #   - 현재 모드: LANE_FOLLOW (장애물 회피 끝났을 때)
        if (
            self.m3_phase == Mission3Phase.DONE
            and self.v2x_phase == V2XPhase.WAIT_START
            and old_mode == DriveMode.OBSTACLE_AVOID
            and new_mode == DriveMode.LANE_FOLLOW
            and not self.obstacle_state
        ):
            self.v2x_phase = V2XPhase.COUNTING
            self.v2x_start_time = now
            rospy.loginfo("[FSM_MUX] v2x lab time START")

        # 2) COUNTING → TURNING
        if self.v2x_phase == V2XPhase.COUNTING and self.v2x_start_time is not None and not self.obstacle_state:
            t_v2x = now - self.v2x_start_time
            if t_v2x >= self.v2x_phase_time:
                self.v2x_phase = V2XPhase.TURNING
                self.v2x_turn_start_time = now
                rospy.loginfo("[FSM_MUX] v2x TURNING start (t_v2x=%.1f)", t_v2x)

        # 3) TURNING → DONE
        elif self.v2x_phase == V2XPhase.TURNING:
            if now - self.v2x_turn_start_time >= self.v2x_turn_duration:
                self.v2x_phase = V2XPhase.DONE
                rospy.loginfo("[FSM_MUX] v2x sequence DONE")

    # ---------- 모드 / enable publish ----------

    def publish_mode(self):
        # drive_mode 문자열
        mode_msg = String()
        mode_msg.data = self.current_mode.name
        self.mode_pub.publish(mode_msg)

        # LKAS enable : LANE_FOLLOW 일 때만 True
        lkas_en = Bool()
        lkas_en.data = (self.current_mode == DriveMode.LANE_FOLLOW)
        self.lkas_enable_pub.publish(lkas_en)

        # obstacle enable : OBSTACLE_AVOID 일 때만 True
        obs_en = Bool()
        obs_en.data = (self.current_mode == DriveMode.OBSTACLE_AVOID)
        self.obstacle_enable_pub.publish(obs_en)

    # ---------- cmd_vel publish (MUX + override) ----------

    def publish_cmd(self, now):
        """
        현재 DriveMode에 따라 lkas / obstacle cmd 중 하나 선택해서 /cmd_vel로 publish
        + Mission3 TURN / v2x TURN 구간에서는 고정 조향으로 override
        """
        cmd = Twist()

        # 기본 MUX
        if self.current_mode == DriveMode.LANE_FOLLOW:
            if self.have_lkas_cmd:
                cmd = copy.deepcopy(self.last_lkas_cmd)
            else:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0

        elif self.current_mode == DriveMode.OBSTACLE_AVOID:
            if self.have_obstacle_cmd:
                cmd = copy.deepcopy(self.last_obstacle_cmd)
            else:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0

        # ----- Mission3 TURNING 구간: 고정 조향 override -----
        if self.m3_phase == Mission3Phase.TURNING:
            cmd = Twist()
            cmd.linear.x = self.m3_turn_speed
            cmd.angular.z = self.m3_turn_yaw

        # ----- v2x TURNING 구간: 고정 조향 override -----
        if self.v2x_phase == V2XPhase.TURNING:
            cmd = Twist()
            cmd.linear.x = self.v2x_turn_speed
            cmd.angular.z = self.v2x_turn_yaw

        self.cmd_pub.publish(cmd)


def main():
    node = FSMMuxNode()
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass



