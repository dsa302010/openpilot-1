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

# 在較高速度下（例如 40 km/h 以上），我們可以假設：
# 車輛產生的橫向加速度與施加在轉向齒條上的扭矩相關。
# 它與車輪滑移或速度無直接關係。

# 這個控製器施加扭矩以達到預期的橫向加速度。
# 為了補償低速效應，我們使用 LOW_SPEED_FACTOR 在低速時增加誤差增益。
# 此外，方向盤存在靜摩擦力，需要克服它才能移動，這部分也有補償。

# -------------------------------------------------------------------
# [V4 最終版] 全公里制 (km/h) 設定 - 雙模式 (ACC ON/OFF)
# 含 DP 原版參數對照
# -------------------------------------------------------------------

# X軸: 車速節點 (直接填寫公里/小時)
# 0   = 靜止
# 10  = 迴轉/剛起步 (解決不回正的關鍵點)
# 36  = 市區慢速 (原版 10 m/s)
# 72  = 快速道路 (原版 20 m/s)
# 108 = 高速公路 (原版 30 m/s)
LOW_SPEED_X_KPH = [0, 10, 36, 72, 108]

# -----------------------------------------------------------
# 模式 A: ACC 開啟 (自動駕駛中) - 咬地力強，直行不晃動
# -----------------------------------------------------------
# Y軸: 放大倍率 (數值越大 = 抓越緊)
#
# [原版參數對照]
# DP 原版原始值: [15, N/A, 13, 10,  5]
# DP 實際運算值: [225, N/A, 169, 100, 25] (原始值取平方)
#
# [V4 優化值說明]
# 0   km/h (180): [起步] 比原版(225) 稍柔和，但仍保留 80% 力道，確保起步轉向有力。
# 10  km/h ( 90): [迴轉] 新增節點！快速降增益，讓迴轉後方向盤能自動回正 (解決黏滯感)。
# 36  km/h ( 65): [市區] 關鍵！比原版(169) 降低約 60%，徹底消除直行左右晃動，但仍比 SP/原版硬。
# 72  km/h ( 30): [快速] 比原版(100) 降低，提升高速巡航的舒適度與容錯率。
# 108 km/h ( 10): [高速] 比原版( 25) 降低，回歸穩定安全，避免高速過敏。
LOW_SPEED_Y_ACC_ON = [180, 90, 70, 80, 20]

# -----------------------------------------------------------
# 模式 B: ACC 關閉 (自己開/純維持) - 手感輕盈
# -----------------------------------------------------------
# 整體力道約為 ACC 模式的 60%~70%，讓您超車或滑行時手感更自然。
LOW_SPEED_Y_ACC_OFF = [140, 65, 65, 75, 20]

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

    # SP Optimization: 初始化濾波器 (用於 D 項)
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
      # V4 Logic: 根據 ACC 狀態切換增益表
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
      pid_log.actualLateralAccel = float(actual_lateral_accel)
      pid_log.desiredLateralAccel = float(desired_lateral_accel)
      pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited_by_controls, curvature_limited))

    # TODO left is positive in this convention
    return -output_torque, 0.0, pid_log
