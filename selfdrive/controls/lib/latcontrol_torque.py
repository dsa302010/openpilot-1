
import math
import numpy as np

from cereal import log
from opendbc.car.lateral import FRICTION_THRESHOLD, get_friction
from opendbc.car.interfaces import LatControlInputs
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY, CV
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.common.pid import PIDController
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_CTRL

# At higher speeds (25+mph) we can assume:
# Lateral acceleration achieved by a specific car correlates to
# torque applied to the steering rack. It does not correlate to
# wheel slip, or to speed.

# This controller applies torque to achieve desired lateral
# accelerations. To compensate for the low speed effects we
# use a LOW_SPEED_FACTOR in the error. Additionally, there is
# friction in the steering wheel that needs to be overcome to
# move it at all, this is compensated for too.

# -------------------------------------------------------------------
# [V5.1 Taiwan Optimization] 台灣道路環境專用版
# -------------------------------------------------------------------
# 設計目標：
# 1. 消除市區直行時的左右拉扯 (Ping-pong)。
# 2. 解決 50km/h 上下過彎容易推頭的問題 (強化中段)。
# 3. 提升 100km/h+ 高速公路抗側風與穩定性 (強化尾段)。
# 4. 保持起步時的舒適性，避免方向盤死硬。

# X軸: 車速節點 (km/h)
# 這是切換扭力增益的時機點
LOW_SPEED_X_KPH = [0, 9, 18, 36, 54, 72, 108]

# -----------------------------------------------------------
# 模式 A: ACC 開啟 (自動駕駛中)
# -----------------------------------------------------------
# Y軸: 扭力補償增益 (Torque Gain)
# 數值定義：數值越大 = 方向盤越重/鎖定感越強；數值越小 = 方向盤越輕。
#
# [參數解析]
# 0   (180): [強增益] 確保靜止與剛起步時方向盤抓地，避免游移。
# 9   (100): [中增益] 起步後快速釋放力道，讓轉向變線性，不突兀。
# 18  ( 75): [適中]   提升低速轉彎的循跡性，避免迴轉時手感過軟無力。
# 36  ( 75): [適中]   市區主力區間，維持足夠抓地力以防畫龍，但不至於死硬。
# 54  ( 80): [中高]   進入快速道路前段，增加阻尼感，預備進入高架。
# 72  ( 85): [高增益] 快速道路/匝道，對抗離心力與高架側風。
# 108 ( 35): [低衰減] 高速巡航補償，比原版(25)高，解決大車旁飄浮感。
LOW_SPEED_Y_ACC_ON = [190, 100, 80, 80, 80, 85, 30]

# -----------------------------------------------------------
# 模式 B: ACC 關閉 (手動駕駛/滑行)
# -----------------------------------------------------------
# Y軸: 扭力補償增益 (Torque Gain)
# 整體設定較輕 (數值較低)，減少電腦介入，保留駕駛手感。
# 9km/h 處降至 65，確保大迴轉後方向盤能順暢回正。
LOW_SPEED_Y_ACC_OFF = [160, 75, 80, 80, 75, 75, 25]

# 自動將您填寫的公里速轉換為 Openpilot 運算用的 m/s (請勿更動此行)
LOW_SPEED_X = [x * CV.KPH_TO_MS for x in LOW_SPEED_X_KPH]

# 濾波器頻率: 1.2Hz (過濾掉方向盤的細微抖動)
LP_FILTER_CUTOFF_HZ = 1.2

# -------------------------------------------------------------------

class LatControlTorque(LatControl):
  def __init__(self, CP, CI):
    super().__init__(CP, CI)
    self.torque_params = CP.lateralTuning.torque.as_builder()
    self.pid = PIDController(self.torque_params.kp, self.torque_params.ki,
                             k_f=self.torque_params.kf, pos_limit=self.steer_max, neg_limit=-self.steer_max)
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg

    # SP Optimization: 初始化濾波器 (用於 D 項優化)
    self.measurement_rate_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * LP_FILTER_CUTOFF_HZ), DT_CTRL)
    self.previous_measurement = 0.0

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction

  def update(self, active, CS, VM, params, steer_limited_by_controls, desired_curvature, curvature_limited):
    pid_log = log.ControlsState.LateralTorqueState.new_message()
    if not active:
      output_torque = 0.0
      pid_log.active = False
      # Reset filter when not active
      self.previous_measurement = 0.0
      self.measurement_rate_filter.x = 0.0
    else:
      actual_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
      roll_compensation = params.roll * ACCELERATION_DUE_TO_GRAVITY
      curvature_deadzone = abs(VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), CS.vEgo, 0.0))
      desired_lateral_accel = desired_curvature * CS.vEgo ** 2

      # desired rate is the desired rate of change in the setpoint, not the absolute desired curvature
      # desired_lateral_jerk = desired_curvature_rate * CS.vEgo ** 2
      actual_lateral_accel = actual_curvature * CS.vEgo ** 2
      lateral_accel_deadzone = curvature_deadzone * CS.vEgo ** 2

      # -------------------------------------------------------------
      # V5.1 Logic: 根據 ACC 狀態切換增益表
      # 注意：直接使用數值插值，不進行平方運算 (No **2)
      # -------------------------------------------------------------
      if CS.cruiseState.enabled:
          target_y_table = LOW_SPEED_Y_ACC_ON
      else:
          target_y_table = LOW_SPEED_Y_ACC_OFF
      
      low_speed_factor = np.interp(CS.vEgo, LOW_SPEED_X, target_y_table)
      # -------------------------------------------------------------
      
      setpoint = desired_lateral_accel + low_speed_factor * desired_curvature
      measurement = actual_lateral_accel + low_speed_factor * actual_curvature

      # SP Optimization: Calculate and filter the measurement rate
      measurement_rate = self.measurement_rate_filter.update((measurement - self.previous_measurement) / DT_CTRL)
      self.previous_measurement = measurement

      gravity_adjusted_lateral_accel = desired_lateral_accel - roll_compensation
      torque_from_setpoint = self.torque_from_lateral_accel(LatControlInputs(setpoint, roll_compensation, CS.vEgo, CS.aEgo), self.torque_params,
                                                            gravity_adjusted=False)
      torque_from_measurement = self.torque_from_lateral_accel(LatControlInputs(measurement, roll_compensation, CS.vEgo, CS.aEgo), self.torque_params,
                                                               gravity_adjusted=False)
      pid_log.error = float(torque_from_setpoint - torque_from_measurement)
      ff = self.torque_from_lateral_accel(LatControlInputs(gravity_adjusted_lateral_accel, roll_compensation, CS.vEgo, CS.aEgo), self.torque_params,
                                          gravity_adjusted=True)
      ff += get_friction(desired_lateral_accel - actual_lateral_accel, lateral_accel_deadzone, FRICTION_THRESHOLD, self.torque_params)

      freeze_integrator = steer_limited_by_controls or CS.steeringPressed or CS.vEgo < 5
      
      # SP Optimization: Pass filtered rate to PID
      error_rate_torque = -measurement_rate * self.torque_params.latAccelFactor

      output_torque = self.pid.update(pid_log.error,
                                      error_rate=error_rate_torque, 
                                      feedforward=ff,
                                      speed=CS.vEgo,
                                      freeze_integrator=freeze_integrator)

      pid_log.active = True
      pid_log.p = float(self.pid.p)
      pid_log.i = float(self.pid.i)
      pid_log.d = float(self.pid.d)
      pid_log.f = float(self.pid.f)
      pid_log.output = float(-output_torque)
      pid_log.actualLateralAccel = float(measurement)
      pid_log.desiredLateralAccel = float(setpoint)
      pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited_by_controls, curvature_limited))

    # TODO left is positive in this convention
    return -output_torque, 0.0, pid_log
