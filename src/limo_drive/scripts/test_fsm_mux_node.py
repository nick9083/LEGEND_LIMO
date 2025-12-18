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
    BEFORE = 0         # 아직 미션3 시퀀스 시작 전
    TURNING = 1        # 고정 조향 구간
    FORCE_OBS = 2      # mission3 DWA 강제 ON 구간
    FORCE_BASE = 3     # base obstacle 강제 ON 구간
    DONE = 4           # 미션3 시퀀스 끝


class V2XPhase(enum.Enum):
    WAIT_START = 0    # 아직 v2x 타이머 시작 전
    COUNTING = 1      # v2x 타이머 증가 중
    TURNING = 2       # v2x 고정 조향 구간 (1단계/2단계 공용 state)
    DONE = 3          # v2x 시퀀스 끝


class FSMMuxNode:
    """
    FSM + MUX 노드 + lab time 2개

    입력:
      - /obstacle_state        (Bool)          : 장애물 여부 (Detect 노드 출력, sta)
      - /cmd_vel_lkas          (Twist)         : LKAS 노드 출력
      - /cmd_vel_obstacle      (Twist)         : base obstacle 회피 노드 출력
      - /cmd_vel_obstacle_m3   (Twist)         : mission3 DWA 노드 출력
      - /path_                 (String) [선택] : V2X 모드 (A/B/C...), 일단 상태만 저장

    출력:
      - /cmd_vel               (Twist)         : 최종 속도 명령 (로봇에 들어가는 것)
      - /drive_mode            (String)        : "LANE_FOLLOW" / "OBSTACLE_AVOID"
      - /lkas_enable           (Bool)          : LKAS 연산 on/off 플래그
      - /obstacle_enable       (Bool)          : base obstacle 연산 on/off 플래그
      - /mission3_enable       (Bool)          : mission3 DWA 연산 on/off 플래그

    추가 로직:
      - lab_time_m3 : "첫 LKAS cmd 수신 시점" 부터 미션3 타이밍 제어
      - lab_time_v2x: 미션3 끝난 뒤, obstacle 회피 끝난 시점부터 v2x 타이밍 제어
    """

    def __init__(self):
        rospy.init_node("fsm_mux_node")

        # --- 파라미터 ---
        self.obstacle_topic         = rospy.get_param("~obstacle_topic", "/obstacle_state")
        self.lkas_cmd_topic         = rospy.get_param("~lkas_cmd_topic", "/cmd_vel_lkas")
        self.obstacle_cmd_topic     = rospy.get_param("~obstacle_cmd_topic", "/cmd_vel_obstacle")
        self.m3_cmd_topic           = rospy.get_param("~m3_cmd_topic", "/cmd_vel_obstacle_m3")
        self.cmd_out_topic          = rospy.get_param("~cmd_out_topic", "/cmd_vel")
        self.mode_topic             = rospy.get_param("~mode_topic", "/drive_mode")
        self.v2x_topic              = rospy.get_param("~v2x_topic", "/path_")
        self.lkas_enable_topic      = rospy.get_param("~lkas_enable_topic", "/lkas_enable")
        self.obstacle_enable_topic  = rospy.get_param("~obstacle_enable_topic", "/obstacle_enable")
        self.mission3_enable_topic  = rospy.get_param("~mission3_enable_topic", "/mission3_enable")

        # 메인 루프 주기
        self.loop_rate = rospy.get_param("~loop_rate", 30.0)

        # ----- Mission3 타이밍 파라미터 -----
        self.m3_phase_time          = rospy.get_param("~m3_phase_time", 13.4)   # 첫 LKAS cmd 이후 기준
        self.m3_turn_duration       = rospy.get_param("~m3_turn_duration", 2.5)
        self.m3_turn_speed          = rospy.get_param("~m3_turn_speed", 0.18)
        self.m3_turn_yaw            = rospy.get_param("~m3_turn_yaw", -0.4)

        # mission3 DWA 강제 구간
        self.m3_force_obs_duration  = rospy.get_param("~m3_force_obs_duration", 4.0)
        # mission3 끝난 직후 base obstacle 강제 구간
        self.m3_force_base_duration = rospy.get_param("~m3_force_base_duration", 0.0)

        # ===== v2x 두 단계 타이머 =====
        # mission5: 첫 번째 v2x 턴이 시작되는 시점
        self.mission5_phase_time = rospy.get_param("~mission5_phase_time", 4)
        # v2x: 두 번째 턴이 시작되는 시점 (기존 v2x_phase_time 유지)
        self.v2x_phase_time2    = rospy.get_param("~v2x_phase_time", 6.5)

        # 두 턴에 *각각* 쓰는 턴 지속 시간
        self.mission5_turn_duration = rospy.get_param("~mission5_turn_duration", 0)
        self.v2x_turn_duration      = rospy.get_param("~v2x_turn_duration", 1)

        # mission5(1단계 턴) 조향값
        self.mission5_turn_speed = rospy.get_param("~mission5_turn_speed", 0.0)
        self.mission5_turn_yaw   = rospy.get_param("~mission5_turn_yaw", -0.00)

        # v2x(2단계 턴) 조향값 (기존 파라미터 유지)
        self.v2x_turn_speed      = rospy.get_param("~v2x_turn_speed", 0.15)
        self.v2x_turn_yaw        = rospy.get_param("~v2x_turn_yaw", -0.18)

        # --- 상태 변수들 ---
        self.current_mode = DriveMode.LANE_FOLLOW

        # 장애물 감지 상태
        self.obstacle_state = False

        # 마지막으로 받은 cmd_vel
        self.last_lkas_cmd      = Twist()
        self.last_obstacle_cmd  = Twist()   # base obstacle
        self.last_m3_cmd        = Twist()   # mission3 DWA
        self.have_lkas_cmd      = False
        self.have_obstacle_cmd  = False
        self.have_m3_cmd        = False

        # V2X 모드 (필요하면 나중에 FSM/출력 로직에 사용)
        self.v2x_mode = "D"

        # ----- 타이머 / 페이즈 상태 -----
        self.start_time = None                 # lab_time_m3 기준 시작 시각 (첫 LKAS cmd 시점)
        self.m3_phase = Mission3Phase.BEFORE
        self.m3_turn_start_time = None
        self.m3_force_obs_start_time = None
        self.m3_force_base_start_time = None   # base 강제 구간 시작 시각

        self.v2x_phase = V2XPhase.WAIT_START
        self.v2x_start_time = None
        self.v2x_turn_start_time = None
        self.v2x_step = 0   # 0: 아직 없음, 1: 1단계 턴, 2: 2단계 턴

        # 로그 주기
        self.log_dt = rospy.get_param("~log_dt", 0.5)
        self.last_log_time = None

        # --- Pub/Sub ---
        rospy.Subscriber(self.obstacle_topic,        Bool,   self.obstacle_cb,      queue_size=1)
        rospy.Subscriber(self.lkas_cmd_topic,        Twist,  self.lkas_cmd_cb,      queue_size=1)
        rospy.Subscriber(self.obstacle_cmd_topic,    Twist,  self.obstacle_cmd_cb,  queue_size=1)
        rospy.Subscriber(self.m3_cmd_topic,          Twist,  self.m3_cmd_cb,        queue_size=1)
        rospy.Subscriber(self.v2x_topic,             String, self.v2x_cb,           queue_size=1)

        self.cmd_pub              = rospy.Publisher(self.cmd_out_topic,          Twist,  queue_size=1)
        self.mode_pub             = rospy.Publisher(self.mode_topic,             String, queue_size=1)
        self.lkas_enable_pub      = rospy.Publisher(self.lkas_enable_topic,      Bool,   queue_size=1)
        self.obstacle_enable_pub  = rospy.Publisher(self.obstacle_enable_topic,  Bool,   queue_size=1)
        self.mission3_enable_pub  = rospy.Publisher(self.mission3_enable_topic,  Bool,   queue_size=1)

        # 메인 루프용 타이머
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.loop_rate), self.update)

        rospy.loginfo("[FSM_MUX] node started.")
        rospy.loginfo("[FSM_MUX] obstacle_topic=%s, lkas_cmd_topic=%s, obstacle_cmd_topic=%s, m3_cmd_topic=%s",
                      self.obstacle_topic, self.lkas_cmd_topic,
                      self.obstacle_cmd_topic, self.m3_cmd_topic)

    # ======================
    # 콜백들
    # ======================

    def obstacle_cb(self, msg: Bool):
        self.obstacle_state = msg.data

        # v2x COUNTING 중에 장애물 다시 뜨면 타이머 리셋
        if (
            self.m3_phase == Mission3Phase.DONE
            and self.v2x_phase == V2XPhase.COUNTING
            and msg.data
        ):
            self.v2x_start_time = rospy.get_time()
            rospy.loginfo("[FSM_MUX] v2x lab time reset (obstacle re-detected).")

    def lkas_cmd_cb(self, msg: Twist):
        self.last_lkas_cmd = msg

        if not self.have_lkas_cmd:
            rospy.loginfo("[FSM_MUX] first LKAS cmd received.")
        self.have_lkas_cmd = True

        # 첫 LKAS cmd 들어오는 타이밍에 lab_time_m3 시작
        if self.start_time is None:
            self.start_time = rospy.get_time()
            self.last_log_time = self.start_time
            rospy.loginfo("[FSM_MUX] lab_time_m3 START (t_m3=0.0, from first LKAS cmd).")

    def obstacle_cmd_cb(self, msg: Twist):
        # base obstacle 회피 from base_obstacle_avoid_node
        self.last_obstacle_cmd = msg
        self.have_obstacle_cmd = True

    def m3_cmd_cb(self, msg: Twist):
        # mission3_node (DWA)에서 오는 cmd
        self.last_m3_cmd = msg
        self.have_m3_cmd = True

    def v2x_cb(self, msg: String):
        self.v2x_mode = msg.data

    # ======================
    #  FSM + 타이머 메인 로직
    # ======================

    def update(self, event):
        now = rospy.get_time()

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

        # --- 최종 cmd_vel publish (MUX + mission3/v2x override) ---
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
                rospy.loginfo("[FSM_MUX] Mission3 FORCE_OBS (mission3_node) start")

        elif self.m3_phase == Mission3Phase.FORCE_OBS:
            # mission3_node 강제 구간 끝 → base_obstacle 강제 구간으로 전이
            if now - self.m3_force_obs_start_time >= self.m3_force_obs_duration:
                self.m3_phase = Mission3Phase.FORCE_BASE
                self.m3_force_base_start_time = now
                rospy.loginfo("[FSM_MUX] Mission3 FORCE_BASE (base_obstacle) start")

        elif self.m3_phase == Mission3Phase.FORCE_BASE:
            # base_obstacle 강제 구간 끝 → DONE
            if now - self.m3_force_base_start_time >= self.m3_force_base_duration:
                self.m3_phase = Mission3Phase.DONE
                rospy.loginfo("[FSM_MUX] Mission3 sequence DONE")

    # ---------- 기본 DriveMode 업데이트 ----------

    def update_mode_basic(self):
        """
        - 미션3 FORCE_OBS 시: 무조건 OBSTACLE_AVOID (mission3_node 사용)
        - 미션3 FORCE_BASE 시: 무조건 OBSTACLE_AVOID (base_obstacle 사용)
        - 그 외: obstacle_state True → OBSTACLE_AVOID, False → LANE_FOLLOW
        """
        # 미션3 강제 구간 (DWA + base 둘 다)에서는 장애물 상태와 관계 없이
        # 항상 OBSTACLE_AVOID 모드 유지
        if self.m3_phase in (Mission3Phase.FORCE_OBS, Mission3Phase.FORCE_BASE):
            if self.current_mode != DriveMode.OBSTACLE_AVOID:
                reason = "FORCE_OBS(mission3)" if self.m3_phase == Mission3Phase.FORCE_OBS \
                         else "FORCE_BASE(base_obstacle)"
                rospy.loginfo("[FSM_MUX] -> OBSTACLE_AVOID (%s)", reason)
            self.current_mode = DriveMode.OBSTACLE_AVOID
            return

        # 평상시: obstacle_state 기반
        if self.obstacle_state and self.current_mode != DriveMode.OBSTACLE_AVOID:
            rospy.loginfo("[FSM_MUX] -> OBSTACLE_AVOID (obstacle_state=True)")
            self.current_mode = DriveMode.OBSTACLE_AVOID

        elif (not self.obstacle_state) and self.current_mode != DriveMode.LANE_FOLLOW:
            rospy.loginfo("[FSM_MUX] -> LANE_FOLLOW (obstacle_state=False)")
            self.current_mode = DriveMode.LANE_FOLLOW

    # ---------- V2X Phase (2단계 TURNING) ----------

    def update_v2x_phase(self, now, old_mode, new_mode):
        # 1) v2x 타이머 시작 조건
        if (
            self.m3_phase == Mission3Phase.DONE
            and self.v2x_phase == V2XPhase.WAIT_START
            and old_mode == DriveMode.OBSTACLE_AVOID
            and new_mode == DriveMode.LANE_FOLLOW
            and not self.obstacle_state
        ):
            self.v2x_phase = V2XPhase.COUNTING
            self.v2x_start_time = now
            self.v2x_turn_start_time = None
            self.v2x_step = 0
            rospy.loginfo("[FSM_MUX] v2x lab time START")

        # 2) COUNTING 상태에서 두 턴 시작 타이밍 관리
        if (
            self.v2x_phase == V2XPhase.COUNTING
            and self.v2x_start_time is not None
            and not self.obstacle_state
        ):
            t_v2x = now - self.v2x_start_time

            # (1) 아직 아무 턴도 안 했고, mission5 시점이 되면 1단계 턴 시작
            if self.v2x_step == 0 and t_v2x >= self.mission5_phase_time:
                self.v2x_phase = V2XPhase.TURNING
                self.v2x_turn_start_time = now
                self.v2x_step = 1
                rospy.loginfo("[FSM_MUX] v2x TURNING #1 (mission5) start (t_v2x=%.1f)", t_v2x)

            # (2) 1단계 턴은 이미 끝났고, 두 번째 타임스탬프에 도달하면 2단계 턴 시작
            elif self.v2x_step == 1 and t_v2x >= self.v2x_phase_time2:
                self.v2x_phase = V2XPhase.TURNING
                self.v2x_turn_start_time = now
                self.v2x_step = 2
                rospy.loginfo("[FSM_MUX] v2x TURNING #2 start (t_v2x=%.1f)", t_v2x)

        # 3) TURNING 상태에서 각 턴의 지속시간 관리
        elif self.v2x_phase == V2XPhase.TURNING and self.v2x_turn_start_time is not None:
            dt_turn = now - self.v2x_turn_start_time

            # 1단계 턴 종료 → 다시 COUNTING 상태로
            if self.v2x_step == 1:
                if dt_turn >= self.mission5_turn_duration:
                    self.v2x_phase = V2XPhase.COUNTING
                    self.v2x_turn_start_time = None
                    # step=1 유지 → "1단계 턴은 끝났다" 표시
                    rospy.loginfo("[FSM_MUX] v2x TURNING #1 end, back to COUNTING")

            # 2단계 턴 종료 → 전체 v2x 시퀀스 끝
            elif self.v2x_step == 2:
                if dt_turn >= self.v2x_turn_duration:
                    self.v2x_phase = V2XPhase.DONE
                    self.v2x_turn_start_time = None
                    rospy.loginfo("[FSM_MUX] v2x TURNING #2 end, sequence DONE")

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

        # mission3_enable : 미션3 FORCE_OBS 구간에만 True (DWA만 켬)
        m3_en = Bool()
        m3_en.data = (self.m3_phase == Mission3Phase.FORCE_OBS)
        self.mission3_enable_pub.publish(m3_en)

        # base obstacle_enable :
        #  - DriveMode 가 OBSTACLE_AVOID 이고
        #  - 미션3 FORCE_OBS(=DWA) 가 아닐 때 True
        #    → FORCE_BASE 구간 + 일반 obstacle 구간 모두 포함
        obs_en = Bool()
        obs_en.data = (
            self.current_mode == DriveMode.OBSTACLE_AVOID
            and self.m3_phase != Mission3Phase.FORCE_OBS
        )
        self.obstacle_enable_pub.publish(obs_en)

    # ---------- cmd_vel publish (MUX + override) ----------

    def publish_cmd(self, now):
        """
        현재 DriveMode에 따라 lkas / obstacle cmd 중 하나 선택해서 /cmd_vel로 publish
        + Mission3 TURN / v2x TURN 구간에서는 고정 조향으로 override
        + Mission3 FORCE_OBS 구간에서는 mission3_node cmd 우선 사용
        + Mission3 FORCE_BASE 구간에서는 base_obstacle cmd 강제 사용
        + v2x TURNING 은 1단계/2단계에 따라 서로 다른 조향값 사용
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
            # Mission3 FORCE_OBS 구간이면 mission3_node cmd 우선
            if self.m3_phase == Mission3Phase.FORCE_OBS:
                if self.have_m3_cmd:
                    cmd = copy.deepcopy(self.last_m3_cmd)
                elif self.have_obstacle_cmd:
                    # 혹시 mission3 cmd가 아직 없다면 base obstacle로 fallback
                    cmd = copy.deepcopy(self.last_obstacle_cmd)
                else:
                    cmd.linear.x = 0.0
                    cmd.angular.z = 0.0
            else:
                # FORCE_BASE + 일반 OBSTACLE_AVOID 둘 다 base obstacle 사용
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

        # ----- v2x TURNING 구간: 고정 조향 override (1단계/2단계) -----
        if self.v2x_phase == V2XPhase.TURNING:
            cmd = Twist()
            if self.v2x_step == 1:
                # mission5 (1단계 턴)
                cmd.linear.x = self.mission5_turn_speed
                cmd.angular.z = self.mission5_turn_yaw
            else:
                # 2단계 v2x 턴
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
