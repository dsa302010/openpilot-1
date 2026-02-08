import time
import numpy as np
from cereal import log
from openpilot.common.swaglog import cloudlog

# ==============================================================================
# [移植注意] 引入 MPC 函式庫
# ==============================================================================
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  COMFORT_BRAKE, STOP_DISTANCE, get_safe_obstacle_distance, 
  get_stopped_equivalence_factor, get_T_FOLLOW
)

# =========================================================
# [參數設定區] ACM & Soft Hold & TTC Limit 參數
# =========================================================

# --- 1. ACM 滑行速度區間設定 (單位：km/h) ---
SPEED_OFFSET_MIN_KPH = 2.0 
SPEED_OFFSET_MAX_FLAT_KPH = 10.0
SPEED_OFFSET_MAX_DOWNHILL_KPH = 5.0

# --- 2. 坡度邏輯設定 (單位：弧度 Radians) ---
PITCH_UPHILL_THRESHOLD = 0.015    
PITCH_DOWNHILL_THRESHOLD = -0.030 

# --- 3. Soft Hold (防點頭) 參數 ---
SOFT_HOLD_ACCEL = -0.00       # 強制加速度上限 (0.0=滑行)
SOFT_HOLD_RANGE_MIN = 0.76    # 觸發下限：76% 安全距離
SOFT_HOLD_RANGE_MAX = 1.00    # 觸發上限：100% 安全距離

# --- 4. Soft Stop (防頓挫) 參數 ---
SOFT_STOP_SPEED_MAX = 5.0     # 啟用速度：< 5 m/s (18 km/h)
SOFT_STOP_MAX_DECEL = -1.50   # 最大煞車力道限制
SOFT_STOP_RANGE_CRITICAL = 0.60 # 緊急界線：距離剩 60% 時取消限制

# --- 5. [新增] 動態 TTC 限制參數 (台灣路況優化版) ---
# 說明：針對市區防插隊與高速舒適性進行分層設定
# 
# (A) TTC 觸發門檻 (Dynamic Threshold)
# 低速 (<40kph) 用 2.0s -> 跟緊一點防止插隊
# 高速 (>90kph) 用 2.5s -> 提早反應比較舒適
TTC_THRESHOLD_BP_SPEED = [11.1, 25.0]  # [m/s] 40kph, 90kph
TTC_THRESHOLD_VALS     = [2.0,  2.5]   # [s]   對應的 TTC 秒數

# (B) 加速度限制值 (Dynamic Limit)
# 低速 (<36kph) 允許 1.0 m/s^2 -> 塞車時起步要靈活
# 高速 (>90kph) 限制 0.4 m/s^2 -> 高速接近前車時要非常溫柔
LIMIT_BP_SPEED         = [10.0, 25.0]  # [m/s] 36kph, 90kph
LIMIT_ACCEL_VALS       = [1.5,  0.5]   # [m/s^2] 對應的最大加速度

# --- 6. 其他常數 ---
TTC_BP = [10., 30.]
TTC_V  = [2.0, 3.0]
EMERGENCY_TTC = 2.0
EMERGENCY_RELATIVE_SPEED = 10.0
EMERGENCY_DECEL_THRESHOLD = -1.5
LEAD_COOLDOWN_TIME = 0.5
SPEED_BP = [0., 10., 20., 30.]
MIN_DIST_V = [15., 20., 25., 30.]

class ACM:
  def __init__(self):
    self.enabled = False
    self._is_in_coast_window = False
    self._has_lead = False
    self._active_prev = False
    self._last_lead_time = 0.0

    self.active = False
    self.just_disabled = False
    
    self.current_ttc_threshold = 3.0
    self.current_pitch = 0.0
    self.current_max_offset = 0.0 

    self.personality = log.LongitudinalPersonality.standard

  # ============================================================================
  # 邏輯區塊 1: 狀態更新與 ACM 啟用判斷
  # ============================================================================
  def _check_emergency_conditions(self, lead, v_ego, current_time):
    if not lead or not lead.status:
      return False

    self.lead_ttc = lead.dRel / max(v_ego, 0.1)
    relative_speed = v_ego - lead.vLead
    min_dist_for_speed = np.interp(v_ego, SPEED_BP, MIN_DIST_V)

    if lead.dRel < min_dist_for_speed and (
        self.lead_ttc < EMERGENCY_TTC or
        relative_speed > EMERGENCY_RELATIVE_SPEED):

      self._last_lead_time = current_time
      if self.active:
        cloudlog.warning(f"ACM emergency disable: dRel={lead.dRel:.1f}m, TTC={self.lead_ttc:.1f}s")
      return True

    return False

  def _update_lead_status(self, lead, v_ego, current_time):
    if lead and lead.status:
      self.lead_ttc = lead.dRel / max(v_ego, 0.1)
      self.current_ttc_threshold = np.interp(v_ego, TTC_BP, TTC_V)

      if self.lead_ttc < self.current_ttc_threshold:
        self._has_lead = True
        self._last_lead_time = current_time
      else:
        self._has_lead = False
    else:
      self._has_lead = False
      self.lead_ttc = float('inf')

  def _check_cooldown(self, current_time):
    time_since_lead = current_time - self._last_lead_time
    return time_since_lead < LEAD_COOLDOWN_TIME

  def _should_activate(self, user_ctrl_lon, v_ego, v_cruise, in_cooldown, pitch):
    if pitch > PITCH_UPHILL_THRESHOLD:
        self._is_in_coast_window = False
        return False

    if pitch < PITCH_DOWNHILL_THRESHOLD:
        self.current_max_offset = SPEED_OFFSET_MAX_DOWNHILL_KPH 
    else:
        self.current_max_offset = SPEED_OFFSET_MAX_FLAT_KPH     

    lower_bound = v_cruise - (SPEED_OFFSET_MIN_KPH / 3.6)
    upper_bound = v_cruise + (self.current_max_offset / 3.6)
    
    self._is_in_coast_window = lower_bound < v_ego < upper_bound

    return (not user_ctrl_lon and
            not self._has_lead and
            not in_cooldown and
            self._is_in_coast_window)

  def update_states(self, cc, rs, user_ctrl_lon, v_ego, v_cruise, personality=log.LongitudinalPersonality.standard):
    self.personality = personality 
    
    if not self.enabled or len(cc.orientationNED) != 3:
      self.active = False
      return

    self.current_pitch = cc.orientationNED[1]
    current_time = time.monotonic()
    lead = rs.leadOne

    if self._check_emergency_conditions(lead, v_ego, current_time):
      self.active = False
      self._active_prev = self.active
      return

    self._update_lead_status(lead, v_ego, current_time)
    in_cooldown = self._check_cooldown(current_time)
    
    self.active = self._should_activate(user_ctrl_lon, v_ego, v_cruise, in_cooldown, self.current_pitch)

    self.just_disabled = self._active_prev and not self.active
    if self.active and not self._active_prev:
      pitch_deg = self.current_pitch * 57.2958
      cloudlog.info(f"ACM ON: v={v_ego*3.6:.0f}, pitch={pitch_deg:.1f}deg, Max+{self.current_max_offset:.0f}kph")
    elif self.just_disabled:
      cloudlog.info("ACM OFF")

    self._active_prev = self.active

  # ============================================================================
  # 邏輯區塊 2: 軌跡修正核心 (Soft Hold / Soft Stop / TTC Limit)
  # ============================================================================

  # [新增] 動態 TTC 加速度限制 (Dynamic TTC Limit)
  def _apply_ttc_limit(self, a_desired_trajectory, lead, v_ego):
    """
    動態 TTC 限制邏輯：
    1. 僅在有前車且正在接近 (vRel < 0) 時生效。
    2. 根據當前車速動態調整 TTC 門檻 (2.0s ~ 2.5s)。
    3. 根據當前車速動態調整 最大加速度限制 (1.0 ~ 0.4 m/s^2)。
    """
    if lead.status and lead.vRel < -0.1:
        # 計算真實 TTC (Distance / Closing Speed)
        closing_speed = -lead.vRel
        real_ttc = lead.dRel / max(closing_speed, 0.1)

        # 1. 計算當前速度下的 TTC 門檻 (市區2.0s <--> 高速2.5s)
        current_ttc_threshold = np.interp(v_ego, TTC_THRESHOLD_BP_SPEED, TTC_THRESHOLD_VALS)

        if real_ttc < current_ttc_threshold:
            # 2. 計算當前速度下的 加速度上限 (市區1.0 <--> 高速0.4)
            current_accel_limit = np.interp(v_ego, LIMIT_BP_SPEED, LIMIT_ACCEL_VALS)
            
            # 3. 執行限制 (取 min, 不影響原本的煞車請求)
            a_desired_trajectory = np.minimum(a_desired_trajectory, current_accel_limit)
            
    return a_desired_trajectory

  def _apply_soft_hold(self, a_desired_trajectory, v_ego, lead):
    """
    對 MPC 輸出的軌跡進行修正，實現舒適跟車與煞停。
    """
    if not lead.status:
      return a_desired_trajectory

    # 坡度保護
    if self.current_pitch > PITCH_UPHILL_THRESHOLD or self.current_pitch < PITCH_DOWNHILL_THRESHOLD:
      return a_desired_trajectory

    # 起步加速保護
    if lead.vRel > 0.1 and lead.vLead > 0.2:
        return a_desired_trajectory

    # 前車靜止保護
    if lead.vLead < 0.5: 
        return a_desired_trajectory

    t_follow = get_T_FOLLOW(self.personality)
    desired_dist = get_safe_obstacle_distance(v_ego, t_follow)
    
    lead_obstacle_dist = lead.dRel + get_stopped_equivalence_factor(lead.vLead)

    if desired_dist < 0.1:
      ratio = 10.0
    else:
      ratio = lead_obstacle_dist / desired_dist

    # 邏輯 A: Soft Hold (76% - 100%) -> 強制滑行
    if SOFT_HOLD_RANGE_MIN < ratio < SOFT_HOLD_RANGE_MAX:
      a_desired_trajectory = np.minimum(a_desired_trajectory, SOFT_HOLD_ACCEL)

    # 邏輯 B: Soft Stop (< 18km/h) -> 限制最大煞車
    if (v_ego < SOFT_STOP_SPEED_MAX) and (ratio > SOFT_STOP_RANGE_CRITICAL):
        a_desired_trajectory = np.maximum(a_desired_trajectory, SOFT_STOP_MAX_DECEL)

    return a_desired_trajectory

  # ============================================================================
  # 邏輯區塊 3: 軌跡修正入口
  # ============================================================================
  def update_a_desired_trajectory(self, a_desired_trajectory, v_ego=0.0, lead=None):
    
    traj = a_desired_trajectory

    # --- 階段 1: ACM 原生邏輯 (無前車、遠距離滑行) ---
    if self.active:
      min_accel = np.min(traj)
      if min_accel < EMERGENCY_DECEL_THRESHOLD:
        cloudlog.warning(f"ACM aborting: MPC requested {min_accel:.2f} m/s² braking")
        self.active = False
      else:
        # ACM 運作中：將微減速 (-1.0 ~ 0) 全部抹平為 0 (滑行)
        modified_trajectory = np.copy(traj)
        for i in range(len(modified_trajectory)):
          if -1.0 < modified_trajectory[i] < 0:
            modified_trajectory[i] = 0.0
        traj = modified_trajectory
    
    # --- 階段 2: Soft Hold & Stop & TTC Limit 邏輯 (跟車模式) ---
    if lead is not None:
        # 1. 執行 Soft Hold / Soft Stop (針對距離與靜止)
        traj = self._apply_soft_hold(traj, v_ego, lead)
        
        # 2. [新增] 執行 Dynamic TTC Limit (針對接近過程)
        traj = self._apply_ttc_limit(traj, lead, v_ego)
    
    return traj
