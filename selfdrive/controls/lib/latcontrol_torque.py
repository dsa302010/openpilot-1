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
LOW_SPEED_X_KPH = [0, 9, 18, 36, 54, 72, 108]

# -----------------------------------------------------------
# 模式 A: ACC 開啟 (自動駕駛中)
# -----------------------------------------------------------
# [參數解析]
# 0   (180): 維持高抓地力，確保靜止與剛起步時方向盤鎖定。
# 9   (100): [優化] 比原版(90)稍高，讓起步加速過程的力道銜接更線性，不突兀。
# 18  ( 75): [優化] 提升低速轉彎的循跡性，避免迴轉時手感過軟。
# 36  ( 75): [優化] 市區主力區間，由 70 提升至 75，解決微彎道路抓不住線的問題。
# 54  ( 80): [優化] 銜接快速道路前段，增加阻尼感，預備進入高架。
# 72  ( 85): [優化] 快速道路/匝道，對抗離心力與高架側風。
# 108 ( 35): [關鍵] 高速公路巡航，由 25 大幅提升至 35，解決大車旁飄浮感。
LOW_SPEED_Y_ACC_ON = [180, 100, 75, 75, 80, 85, 30]

# -----------------------------------------------------------
# 模式 B: ACC 關閉 (手動駕駛/滑行)
# -----------------------------------------------------------
# 維持輕盈手感，方便隨時介入，僅提升高速尾段以策安全。
LOW_SPEED_Y_ACC_OFF = [140, 65, 65, 65, 65, 75, 30]

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
      actual_lateral_
