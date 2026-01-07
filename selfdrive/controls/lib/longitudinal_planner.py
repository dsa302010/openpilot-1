#!/usr/bin/env python3
"""
Longitudinal Planner - Integrated Edition (Modified v2)
功能:
1. 整合 SCDA 與 DTSC，並啟用雙向通訊 (距離傳遞)。
2. 包含踩油門取消 SCDA 的偵測邏輯。
3. 包含詳細 Debug Log。
4. [Mod] 放寬 DTSC 與 SCDA 介入時的減速變化率限制 (Slew Rate)，解決減速無感問題。
5. [Mod] 邏輯分離：DTSC 與 SCDA 變化率分開設定，同時介入時取最小值。
"""
import math
import numpy as np

import cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog

# --- DragonPilot Modules ---
try:
  from dragonpilot.selfdrive.controls.lib.acm import ACM
  from dragonpilot.selfdrive.controls.lib.aem import AEM
  from dragonpilot.selfdrive.controls.lib.dtsc import DTSC
  from dragonpilot.selfdrive.controls.lib.scda import SpeedCameraControl
  DP_MODULES_AVAILABLE = True
except ImportError:
  cloudlog.exception("DP: Critical Import Error")
  DP_MODULES_AVAILABLE = False
  class ACM: 
    def update_states(self, *args, **kwargs): pass
    def update_a_desired_trajectory(self, a): return a
  class AEM:
    def update_states(self, *args, **kwargs): pass
    def get_mode(self, mode): return mode
  class DTSC:
    def __init__(self, *args, **kwargs): pass
    def get_mpc_constraints(self, *args, **kwargs): return [], []
  class SpeedCameraControl:
    def get_target_speed(self, *args, **kwargs): 
        return {'target_speed': None, 'distance': None, 'limit': None, 'is_active': False}
    def cancel_current_camera(self): pass
    def get_statistics(self): return {}

# =============================
# Constants
# =============================
LON_MPC_STEP = 0.2
A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5

_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]

# 開啟 Log
DEBUG_LOGGING = True

class DPFlags:
  ACM = 1
  AEM = 2
  DTSC = 2 ** 2
  SCDA = 2 ** 3

def get_max_accel(v_ego):
  return np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)

def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3

def limit_accel_in_turns(v_ego, angle_steers, a_target, CP):
  a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  a_y = v_ego ** 2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
  a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))
  return [a_target[0], min(a_target[1], a_x_allowed)]

class LongitudinalPlanner:
  def __init__(self, CP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.mpc = LongitudinalMpc(dt=dt)
    self.mpc.mode = 'acc'
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True

    self.a_desired = init_a
    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.prev_accel_clip = [ACCEL_MIN, ACCEL_MAX]
    self.output_a_target = 0.0
    self.output_should_stop = False

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)
    self.solverExecutionTime = 0.0
    
    # SCDA State
    self.scda_error_count = 0
    self.scda_disabled = False
    self.scda_target_speed = None
    self.scda_distance = None      # 儲存 SCDA 距離供 DTSC 使用
    self.scda_limit = None
    self.scda_active = False       # 儲存 SCDA 是否正在介入
    
    # Init modules
    if not DP_MODULES_AVAILABLE:
      self.acm = ACM()
      self.aem = AEM()
      self.dtsc = None
      self.scda = None
      return
    
    try:
      self.acm = ACM()
      self.aem = AEM()
      self.dtsc = DTSC(aggressiveness=1.0, cp=self.CP)
      self.scda = SpeedCameraControl()
      if DEBUG_LOGGING: cloudlog.info("DP: Modules initialized")
    except Exception as e:
      cloudlog.exception(f"DP: Init error - {e}")
      self.acm = ACM()
      self.aem = AEM()
      self.dtsc = None
      self.scda = None

  @staticmethod
  def parse_model(model_msg):
    if (len(model_msg.position.x) == ModelConstants.IDX_N and
      len(model_msg.velocity.x) == ModelConstants.IDX_N and
      len(model_msg.acceleration.x) == ModelConstants.IDX_N):
      x = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.position.x)
      v = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.velocity.x)
      a = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.acceleration.x)
      j = np.zeros(len(T_IDXS_MPC))
    else:
      x = np.zeros(len(T_IDXS_MPC))
      v = np.zeros(len(T_IDXS_MPC))
      a = np.zeros(len(T_IDXS_MPC))
      j = np.zeros(len(T_IDXS_MPC))
    if len(model_msg.meta.disengagePredictions.gasPressProbs) > 1:
      throttle_prob = model_msg.meta.disengagePredictions.gasPressProbs[1]
    else:
      throttle_prob = 1.0
    return x, v, a, j, throttle_prob

  def update(self, sm, dp_flags = 0):
    mode = 'blended' if sm['selfdriveState'].experimentalMode else 'acc'
    
    # 獲取 OP 是否啟用 (ACC ON)
    acc_enabled = sm['selfdriveState'].enabled

    if dp_flags & DPFlags.AEM:
      try:
        self.aem.update_states(model_msg=sm['modelV2'], radar_msg=sm['radarState'], v_ego=sm['carState'].vEgo)
        mode = self.aem.get_mode(mode)
      except Exception: pass

    if len(sm['carControl'].orientationNED) == 3:
      accel_coast = get_coast_accel(sm['carControl'].orientationNED[1])
    else:
      accel_coast = ACCEL_MAX

    v_ego = sm['carState'].vEgo
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET

    # ============================================================
    # SCDA Logic
    # ============================================================
    self.scda_target_speed = None
    self.scda_distance = None
    self.scda_limit = None
    self.scda_active = False # [重設狀態]
    
    if (dp_flags & DPFlags.SCDA) and self.scda is not None and not self.scda_disabled:
      # 踩油門取消功能
      if sm['carState'].gasPressed:
          try: self.scda.cancel_current_camera()
          except Exception: pass

      try:
        gps = sm['gpsLocationExternal']
        is_gps_valid = (gps.flags & 1) and (gps.latitude != 0.0) and (gps.longitude != 0.0)
        
        if is_gps_valid:
          bearing = gps.bearingDeg
          if math.isnan(bearing): bearing = 0.0
          safe_v_ego = v_ego if not math.isnan(v_ego) else 0.0

          # 傳遞 acc_enabled 狀態
          scda_result = self.scda.get_target_speed(safe_v_ego, v_cruise, gps.latitude, gps.longitude, bearing, acc_enabled=acc_enabled)
          
          if isinstance(scda_result, dict):
              scda_target_ms = scda_result.get('target_speed')
              # 儲存距離供 DTSC 使用
              self.scda_distance = scda_result.get('distance') 
              self.scda_limit = scda_result.get('limit')
              # 獲取 SCDA 是否正在介入
              self.scda_active = scda_result.get('is_active', False)
          else:
              scda_target_ms = scda_result
          
          if scda_target_ms is not None and math.isfinite(scda_target_ms):
            if scda_target_ms < v_cruise:
              self.scda_target_speed = scda_target_ms
            
            v_cruise = min(v_cruise, scda_target_ms)
            self.scda_error_count = 0
            
      except Exception as e:
        self.scda_error_count += 1
        if self.scda_error_count > 5:
            self.scda_disabled = True
            if DEBUG_LOGGING: cloudlog.warning(f"DP: SCDA disabled - {e}")
    # ============================================================

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off
    force_slow_decel = sm['controlsState'].forceDecel

    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    reset_state = reset_state or not v_cruise_initialized
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    if mode == 'acc':
      accel_clip = [ACCEL_MIN, get_max_accel(v_ego)]
      steer_angle_without_offset = sm['carState'].steeringAngleDeg - sm['liveParameters'].angleOffsetDeg
      accel_clip = limit_accel_in_turns(v_ego, steer_angle_without_offset, accel_clip, self.CP)
    else:
      accel_clip = [ACCEL_MIN, ACCEL_MAX]

    if reset_state:
      self.v_desired_filter.x = v_ego
      self.a_desired = np.clip(sm['carState'].aEgo, accel_clip[0], accel_clip[1])

    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))
    x, v, a, j, throttle_prob = self.parse_model(sm['modelV2'])
    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    if not self.allow_throttle:
      clipped_accel_coast = max(accel_coast, accel_clip[0])
      clipped_accel_coast_interp = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED*2], [accel_clip[1], clipped_accel_coast])
      accel_clip[1] = min(accel_clip[1], clipped_accel_coast_interp)

    if force_slow_decel:
      v_cruise = 0.0

    self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)
    self.mpc.set_cur_state(self.v_desired_filter.x, self.a_desired)

    # ============================================================
    # DTSC Logic
    # ============================================================
    if (dp_flags & DPFlags.DTSC) and self.dtsc is not None:
      try:
        steer_angle = sm['carState'].steeringAngleDeg
        
        # 傳遞 acc_enabled 狀態
        a_min_dtsc, a_max_dtsc = self.dtsc.get_mpc_constraints(
          model_msg=sm['modelV2'], 
          v_ego=v_ego, 
          base_a_min=accel_clip[0], 
          base_a_max=accel_clip[1],
          steer_angle_deg=steer_angle,
          steer_ratio=self.CP.steerRatio,
          wheelbase=self.CP.wheelbase,
          scda_target_speed=self.scda_target_speed,
          scda_distance=self.scda_distance,
          acc_enabled=acc_enabled
        )
        
        safe_len = min(len(a_min_dtsc), self.mpc.params.shape[0])

        for i in range(safe_len):
          d_a_max = a_max_dtsc[i]
          target_min = max(accel_clip[0], a_min_dtsc[i])
          target_max = min(accel_clip[1], d_a_max)

          if target_min > target_max:
              target_min = target_max - 0.01 

          if math.isfinite(target_min) and math.isfinite(target_max):
              self.mpc.params[i, 0] = target_min
              self.mpc.params[i, 1] = target_max
              
      except Exception as e:
        if DEBUG_LOGGING: cloudlog.exception(f"DP: DTSC logic crashed - {e}")
    # ============================================================

    self.mpc.update(sm['radarState'], v_cruise, x, v, a, j, personality=sm['selfdriveState'].personality)

    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    
    if dp_flags & DPFlags.ACM:
      try:
        user_control = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
        self.acm.update_states(sm['carControl'], sm['radarState'], user_control, v_ego, v_cruise)
        self.a_desired_trajectory = self.acm.update_a_desired_trajectory(self.a_desired_trajectory)
      except Exception: pass
    
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)

    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw: cloudlog.info("FCW triggered")

    a_prev = self.a_desired
    self.a_desired = float(np.interp(self.dt, CONTROL_N_T_IDX, self.a_desired_trajectory))
    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.a_desired + a_prev) / 2.0

    action_t = self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target_mpc, output_should_stop_mpc = get_accel_from_plan(
        self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX,
        action_t=action_t, vEgoStopping=self.CP.vEgoStopping
    )
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = sm['modelV2'].action.shouldStop

    if mode == 'acc':
      output_a_target = output_a_target_mpc
      self.output_should_stop = output_should_stop_mpc
    else:
      output_a_target = min(output_a_target_mpc, output_a_target_e2e)
      self.output_should_stop = output_should_stop_e2e or output_should_stop_mpc

    # ============================================================
    # [修改] 減速變化率動態調整 (DTSC 與 SCDA 分開處理)
    # ============================================================
    default_slew_rate = 0.05
    decel_slew_rate = default_slew_rate
    
    # [設定區] 定義個別的變化率 (可在此處單獨修改數值)
    dtsc_limit = 0.15
    scda_limit = 0.15
    
    # 檢查 DTSC 狀態
    is_dtsc_active = False
    if self.dtsc is not None:
        is_dtsc_active = getattr(self.dtsc, 'active', False)

    # 判斷個別條件 (介入中 且 正在減速)
    dtsc_condition = is_dtsc_active and output_a_target < 0.0
    scda_condition = self.scda_active and output_a_target < 0.0

    # 邏輯分流
    if dtsc_condition and scda_condition:
        # 同時介入：取兩者中的最低值 (最保守/數值最小)
        decel_slew_rate = min(dtsc_limit, scda_limit)
    elif dtsc_condition:
        # 只有 DTSC
        decel_slew_rate = dtsc_limit
    elif scda_condition:
        # 只有 SCDA
        decel_slew_rate = scda_limit
    else:
        # 都沒有 (使用預設值)
        decel_slew_rate = default_slew_rate

    for idx in range(2):
      # idx 0=min, 1=max
      # 減速方向: 使用計算出的 decel_slew_rate
      # 加速方向: 固定維持 +0.05 (舒適性)
      accel_clip[idx] = np.clip(
          accel_clip[idx], 
          self.prev_accel_clip[idx] - decel_slew_rate, 
          self.prev_accel_clip[idx] + 0.05
      )
      
    self.output_a_target = np.clip(output_a_target, accel_clip[0], accel_clip[1])
    self.prev_accel_clip = accel_clip
    # ============================================================

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')
    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState', 'selfdriveState', 'radarState'])

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.status
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)

    pm.send('longitudinalPlan', plan_send)
    
  def get_diagnostics(self):
    diag = {
        "scda_disabled": self.scda_disabled,
        "scda_distance": self.scda_distance,
        "scda_limit": self.scda_limit
    }
    return diag
