"""
Copyright (c) 2025, Rick Lan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the \"Software\"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, and/or sublicense,
for non-commercial purposes only, subject to the following conditions:

- The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.
- Commercial use (e.g. use in a product, service, or activity intended to
  generate revenue) is prohibited without explicit written permission from
  the copyright holder.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import time
import numpy as np
from openpilot.common.swaglog import cloudlog

# ACM (Active Coasting Management) 參數組態

# 基本安全參數
SPEED_RATIO = 0.98  # 原始參數，因使用查表法而被覆蓋，保留作為參考
TTC_THRESHOLD = 3.0  # 秒 - 當與前車碰撞時間 (TTC) 低於此值時，ACM 正常停用

# 緊急停用閾值 (最高優先級) - 立即停用 ACM
EMERGENCY_TTC = 2.0  # 秒 - 判斷為緊急情況的 TTC 門檻
EMERGENCY_RELATIVE_SPEED = 10.0  # m/s (~36 km/h) - 判斷為緊急情況的相對接近速度門檻

# [已修改] 滑行允許的最大減速度閾值
# 如果 MPC 請求的減速超過此值 (例如 -0.6)，代表有明確煞車意圖 (如 DTSC 入彎或跟車)，ACM 將停用
# 原本為 -1.5，改為 -0.5 以避免干擾 DTSC 的輕微煞車
COAST_MAX_DECEL = -0.5  # m/s²

# [新增] 上坡偵測閾值 (單位: radians)
# 0.02 rad 約等於 1.15 度或 2% 坡度
# 超過此坡度時，ACM 將不啟動以維持爬坡動力
UPHILL_THRESHOLD = 0.04

# 前車偵測冷卻
LEAD_COOLDOWN_TIME = 0.5  # 秒 - 偵測到前車後的短暫冷卻期，用於處理感測器瞬時故障

# --- 動態最小距離組態 ---
# 根據車速調整最小安全距離（用於緊急停用判斷）
# 速度斷點 (m/s)
SPEED_BP = [0., 10., 15., 20., 25., 30.]  # (0, 36, 54, 72, 90, 108 km/h)
# 對應的最小距離 (m) - 低速更寬鬆，高速更保守
MIN_DIST_V = [15., 20., 22.5, 25., 27.5, 30.]  
# --- 動態最小距離組態結束 ---

# --- 動態 SPEED_RATIO 組態 ---
# 根據車速調整滑行啟動的巡航速度比例（SPEED_RATIO）
# 速度斷點 (m/s)
SPEED_RATIO_BP = [0., 10., 15., 20., 25., 30.]  # (0, 36, 54, 72, 90, 108 km/h)  
# 對應的 SPEED_RATIO 值 (越低越早啟動滑行)
SPEED_RATIO_V = [0.93, 0.94, 0.95, 0.96, 0.97, 0.98] 
# --- 動態 SPEED_RATIO 組態結束 ---


class ACM:
  def __init__(self):
    self.enabled = False             # ACM 功能是否啟用
    self._is_speed_over_cruise = False # 當前速度是否高於巡航目標 * SPEED_RATIO
    self._has_lead = False           # 是否有近距離前車
    self._active_prev = False        # 前一幀的 active 狀態
    self._last_lead_time = 0.0       # 最後一次偵測到前車的時間

    self.active = False              # ACM 當前是否啟用 (滑行邏輯是否生效)
    self.just_disabled = False       # ACM 是否剛被停用

  def _check_emergency_conditions(self, lead, v_ego, current_time):
    """檢查需要立即停用 ACM 的緊急條件。"""
    if not lead or not lead.status:
      return False

    # 簡單 TTC 計算 (使用本車速度)
    self.lead_ttc = lead.dRel / max(v_ego, 0.1)
    relative_speed = v_ego - lead.vLead  # 正值表示本車正在接近前車

    # 計算速度適應的最小距離閾值
    min_dist_for_speed = np.interp(v_ego, SPEED_BP, MIN_DIST_V)

    # 緊急停用條件：必須同時滿足 距離過近 AND (TTC極短 OR 接近速度過快)
    if lead.dRel < min_dist_for_speed and (
        self.lead_ttc < EMERGENCY_TTC or
        relative_speed > EMERGENCY_RELATIVE_SPEED):

      self._last_lead_time = current_time
      if self.active:  # 只有在實際停用的時候才發出警告
        cloudlog.warning(f"ACM emergency disable: dRel={lead.dRel:.1f}m, TTC={self.lead_ttc:.1f}s, relSpeed={relative_speed:.1f}m/s")
      return True

    return False

  def _update_lead_status(self, lead, v_ego, current_time):
    """更新前車狀態，用於正常停用判斷。"""
    if lead and lead.status:
      self.lead_ttc = lead.dRel / max(v_ego, 0.1)

      # 正常停用條件：TTC < 3.0秒
      if self.lead_ttc < TTC_THRESHOLD:
        self._has_lead = True
        self._last_lead_time = current_time
      else:
        self._has_lead = False
    else:
      self._has_lead = False
      self.lead_ttc = float('inf')

  def _check_cooldown(self, current_time):
    """檢查是否還在前車偵測後的冷卻期內。"""
    time_since_lead = current_time - self._last_lead_time
    return time_since_lead < LEAD_COOLDOWN_TIME

  def _should_activate(self, user_ctrl_lon, v_ego, v_cruise, in_cooldown):
    """判斷 ACM 是否應該被啟用 (所有必要條件)。"""
    
    # 1. 根據當前車速計算動態 SPEED_RATIO
    current_speed_ratio = np.interp(v_ego, SPEED_RATIO_BP, SPEED_RATIO_V)

    # 2. 判斷速度是否超過巡航設定值的動態比例
    self._is_speed_over_cruise = v_ego > (v_cruise * current_speed_ratio)

    # 3. 必須同時滿足以下所有條件才能啟用
    return (not user_ctrl_lon and     # 駕駛沒有縱向控制
            not self._has_lead and    # 沒有近距離前車
            not in_cooldown and       # 不在冷卻期內
            self._is_speed_over_cruise) # 車速超過目標比例

  def update_states(self, cc, rs, user_ctrl_lon, v_ego, v_cruise):
    """更新 ACM 的內部狀態。"""
    if not self.enabled or len(cc.orientationNED) != 3:
      self.active = False
      return
    
    # [新增] 上坡檢查
    # orientationNED[1] 是 Pitch (俯仰角)。正值通常代表車頭朝上(上坡)。
    # 如果坡度大於閾值 (約 4%)，強制停用 ACM 以維持動力。
    pitch = cc.orientationNED[1]
    if pitch > UPHILL_THRESHOLD:
      self.active = False
      # 如果之前是啟用的，記錄一下是因為上坡而停用 (可選，避免log過多先省略)
      self._active_prev = self.active
      return

    current_time = time.monotonic()
    lead = rs.leadOne

    # 優先檢查緊急停用條件
    if self._check_emergency_conditions(lead, v_ego, current_time):
      self.active = False
      self._active_prev = self.active
      return

    # 更新正常前車狀態
    self._update_lead_status(lead, v_ego, current_time)

    # 檢查冷卻期
    in_cooldown = self._check_cooldown(current_time)

    # 決定 ACM 最終狀態
    self.active = self._should_activate(user_ctrl_lon, v_ego, v_cruise, in_cooldown)

    # 記錄狀態變化 (供調試用)
    self.just_disabled = self._active_prev and not self.active
    if self.active and not self._active_prev:
      # 記錄啟動時使用的動態 SPEED_RATIO
      current_speed_ratio = np.interp(v_ego, SPEED_RATIO_BP, SPEED_RATIO_V)
      cloudlog.info(f"ACM activated (Ratio={current_speed_ratio:.2f}): v_ego={v_ego*3.6:.1f} km/h, v_cruise={v_cruise*3.6:.1f} km/h")
    elif self.just_disabled:
      cloudlog.info("ACM deactivated")

    self._active_prev = self.active

  def update_a_desired_trajectory(self, a_desired_trajectory):
    """
    修改加速度軌跡以允許滑行 (Coasting)。
    SAFETY: 檢查是否有強烈煞車請求 (DTSC, 跟車等) 並中止 ACM。
    """
    if not self.active:
      return a_desired_trajectory

    # 安全檢查：
    # 如果 MPC 請求的減速度超過閾值 (例如 < -0.5 m/s²)，
    # 這意味著需要實際的煞車力道 (可能來自 DTSC 入彎需求或前車減速)。
    # 在這種情況下，我們必須停用滑行，讓 MPC 執行煞車。
    min_accel = np.min(a_desired_trajectory)
    
    # [已修改] 使用更靈敏的 COAST_MAX_DECEL (-0.5) 取代原本的緊急閾值 (-1.5)
    # 這能確保 DTSC 的輕微煞車指令不會被 ACM 吃掉。
    if min_accel < COAST_MAX_DECEL:
      # 只有當減速非常劇烈時才寫 log，避免一般入彎頻繁刷屏
      if min_accel < -1.5: 
        cloudlog.warning(f"ACM aborting: MPC requested {min_accel:.2f} m/s² braking")
      
      self.active = False  # 立即停用
      return a_desired_trajectory  # 返回未修改的原軌跡

    # 核心滑行邏輯：
    # 只抑制 "極輕微" 的負加速度 (例如 -0.5 到 0.0 之間)，這通常是空氣阻力補償或微調。
    # 將其修正為 0.0 可以讓車輛真正滑行。
    modified_trajectory = np.copy(a_desired_trajectory)
    for i in range(len(modified_trajectory)):
      # [已修改] 範圍縮小至 COAST_MAX_DECEL
      if COAST_MAX_DECEL < modified_trajectory[i] < 0:
        modified_trajectory[i] = 0.0
      
    return modified_trajectory
