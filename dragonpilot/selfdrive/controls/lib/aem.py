"""
AEM (Automatic Experimental Mode) - Final Clean Version (No Blinker)
Copyright (c) 2025, Modified for DragonPilot
"""

import numpy as np
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants

# ==============================================================================
#                               CONFIG (參數設定區)
# ==============================================================================
class Config:
    # --- 閾值設定 ---
    TTC_EMERGENCY     = 0.9   # [秒] TTC 緊急碰撞時間

    # --- 距離參數 (單位：公尺 m) ---
    EMERGENCY_DIST_CITY    = 20.0  # [m] 市區緊急煞停
    EMERGENCY_DIST_HIGHWAY = 40.0  # [m] 高速緊急煞停

    # 視覺牆防護
    RADAR_MISS_DIST        = 20.0  # [m] 視覺牆防護
    RADAR_MISS_SPEED       = 10.0  # [km/h] 最低作動速度

    LEAD_CLOSE_DIST        = 15.0  # [m] 貼車防撞
    SLOW_LEAD_DIST_MAX     = 100.0 # [m] 慢車偵測

    # --- 彎道救援參數 ---
    BAILOUT_LAT_G      = 2.5   # [G] 側向力救車門檻
    BAILOUT_LAT_ERROR  = 0.35  # [m] 車道偏離救車門檻
    BAILOUT_SPEED_MIN  = 30.0  # [km/h]

    # --- 速度定義 ---
    HIGHWAY_SPEED  = 80.0
    SLOW_LEAD_DIFF = 30.0

    # --- 靈敏度曲線 (平衡版) ---
    SENSITIVITY_BP   = [0.,  60., 70., 90.]
    SENSITIVITY_VALS = [1.0, 1.0, 0.7, 0.6]

    # --- 減速模型 ---
    SLOW_DOWN_BP   = [0., 5.,  10., 15., 20., 25., 30.]
    SLOW_DOWN_DIST = [5., 20., 40., 60., 80., 100., 120.]

    # --- 模式定義 ---
    MODE_ACC = 'acc'
    MODE_BLENDED = 'blended'

# ==============================================================================
#                         UTILITY CLASSES (工具類別)
# ==============================================================================

class SmoothKalmanFilter:
  def __init__(self, initial_value=0, alpha=1.0, smoothing_factor=0.85):
    self.x = initial_value
    self.P = 1.0
    self.R = 0.2
    self.Q = 0.01
    self.alpha = alpha
    self.smoothing_factor = smoothing_factor
    self.initialized = False

  def add_data(self, measurement):
    if not self.initialized:
      self.x = measurement
      self.initialized = True
      return

    self.P = self.alpha * self.P + self.Q
    K = self.P / (self.P + self.R)
    effective_K = K * (1.0 - self.smoothing_factor) + self.smoothing_factor * 0.1
    innovation = measurement - self.x
    self.x = self.x + effective_K * innovation
    self.P = (1 - effective_K) * self.P

  def get_value(self):
    return self.x if self.initialized else 0.0

class ModeTransitionManager:
  def __init__(self):
    self.current_mode = Config.MODE_ACC
    self.mode_confidence = {Config.MODE_ACC: 1.0, Config.MODE_BLENDED: 0.0}
    self.lock_timer = 0
    self.frame_count = 0

  def request_mode(self, mode, confidence=1.0, emergency=False):
    if emergency:
      self.current_mode = mode
      self.lock_timer = 20
      self.mode_confidence[Config.MODE_BLENDED] = 1.0
      self.mode_confidence[Config.MODE_ACC] = 0.0
      return

    if self.lock_timer > 0:
      return

    target_conf = min(1.0, self.mode_confidence[mode] + 0.05 * confidence)
    self.mode_confidence[mode] = target_conf

    for m in self.mode_confidence:
      if m != mode:
        self.mode_confidence[m] = max(0.0, self.mode_confidence[m] - 0.05)

    threshold = 0.75 if mode != self.current_mode else 0.4
    if self.mode_confidence[mode] > threshold:
        self.current_mode = mode

  def update(self):
    if self.lock_timer > 0:
      self.lock_timer -= 1
    self.mode_confidence[Config.MODE_BLENDED] *= 0.95
    self.mode_confidence[Config.MODE_ACC] = 1.0 - self.mode_confidence[Config.MODE_BLENDED]
    self.frame_count += 1

  def get_mode(self):
    return self.current_mode

# ==============================================================================
#                               CORE LOGIC (核心邏輯)
# ==============================================================================

class AEM:
    def __init__(self):
        self._mode_manager = ModeTransitionManager()
        self._slow_down_filter = SmoothKalmanFilter(alpha=1.01, smoothing_factor=0.9)
        self._urgency = 0.0

    def get_mode(self, current_mode_str):
        return self._mode_manager.get_mode()

    # [移除] left_blinker, right_blinker 參數
    def update_states(self, model_msg, radar_msg, v_ego):
        """主邏輯更新"""
        if not (len(model_msg.position.x) == ModelConstants.IDX_N and 
                len(model_msg.position.z) == ModelConstants.IDX_N):
            return

        v_kph = v_ego * 3.6
        path_x = model_msg.position.x 
        path_z = model_msg.position.z 
        model_end_dist = path_z[ModelConstants.IDX_N - 1]

        # 計算曲率
        curvature_val = 0.0
        try:
            mid_idx = 15
            if path_z[mid_idx] > 1.0:
                curvature_val = abs(path_x[mid_idx]) / (path_z[mid_idx]**2)
        except:
            curvature_val = 0.0

        # 計算偏差
        current_lat_error = 0.0
        try:
            current_lat_error = np.mean(np.abs(path_x[0:5]))
        except:
            pass

        # 計算舒適減速
        self._calculate_slow_down(model_end_dist, min(1.0, curvature_val * 500.0), v_ego, v_kph)

        # 決策 (不再傳入方向燈參數)
        self._make_decision(radar_msg, v_kph, model_end_dist, curvature_val, current_lat_error)

        self._mode_manager.update()

    def _calculate_slow_down(self, model_end_dist, curvature, v_ego, v_kph):
        """計算舒適減速 Urgency"""
        base_expected = np.interp(v_ego, Config.SLOW_DOWN_BP, Config.SLOW_DOWN_DIST)
        sensitivity = np.interp(v_kph, Config.SENSITIVITY_BP, Config.SENSITIVITY_VALS)

        curve_penalty = 1.0 - (curvature * 0.2)
        expected_distance = base_expected * sensitivity * curve_penalty

        urgency = 0.0

        # City Boost Logic
        if v_kph < 55.0:
            if model_end_dist < expected_distance:
                shortage = expected_distance - model_end_dist
                shortage_ratio = shortage / expected_distance
                urgency = np.clip(shortage_ratio * 2.0, 0.0, 1.0)
        else:
            if model_end_dist < (expected_distance * 0.85):
                shortage = expected_distance - model_end_dist
                shortage_ratio = shortage / expected_distance
                urgency = np.clip(shortage_ratio * 1.5, 0.0, 1.0)

        self._slow_down_filter.add_data(urgency)
        self._urgency = self._slow_down_filter.get_value()

    def _make_decision(self, radar_msg, v_kph, model_end_dist, curvature_val, current_lat_error):
        """分層決策"""

        # [優先級 0] 彎道救援
        if self._check_bailout(v_kph, curvature_val, current_lat_error):
             self._mode_manager.request_mode(Config.MODE_BLENDED, confidence=1.0, emergency=True)
             return

        # [優先級 1] 危險 (Emergency)
        if self._check_danger(radar_msg, v_kph, model_end_dist):
            self._mode_manager.request_mode(Config.MODE_BLENDED, confidence=1.0, emergency=True)
            return

        # [優先級 2] 舒適減速
        if self._urgency > 0.6:
            self._mode_manager.request_mode(Config.MODE_BLENDED, confidence=self._urgency)
            return

        # [優先級 3] 預設 ACC
        self._mode_manager.request_mode(Config.MODE_ACC, confidence=0.8)

    # 判斷邏輯
    def _check_bailout(self, v_kph, curvature_val, current_lat_error):
        if v_kph < Config.BAILOUT_SPEED_MIN: return False

        v_ego = v_kph / 3.6
        approx_k = 2.0 * curvature_val
        estimated_lat_g = (v_ego ** 2) * approx_k

        # 條件 A: 側向力極大 (失控)
        if estimated_lat_g > Config.BAILOUT_LAT_G: return True

        # 條件 B: 彎中偏離 (推頭)
        if estimated_lat_g > 0.9 and current_lat_error > Config.BAILOUT_LAT_ERROR: return True

        return False

    def _check_danger(self, radar_msg, v_kph, model_end_dist):
        """危險情境檢查"""
        if radar_msg is None: return False
        lead = radar_msg.leadOne
        v_ego = v_kph / 3.6

        if not lead.status:
            if v_kph > Config.RADAR_MISS_SPEED and model_end_dist < Config.RADAR_MISS_DIST:
                return True
            return False

        d_lead = lead.dRel
        v_lead = lead.vLead

        thresh_dist = Config.EMERGENCY_DIST_HIGHWAY if v_kph > Config.HIGHWAY_SPEED else Config.EMERGENCY_DIST_CITY
        if d_lead < thresh_dist:
            if (v_ego > v_lead) or (d_lead < Config.LEAD_CLOSE_DIST):
                return True

        if v_ego > v_lead:
            ttc = d_lead / (v_ego - v_lead)
            if ttc < Config.TTC_EMERGENCY:
                return True

        return False
