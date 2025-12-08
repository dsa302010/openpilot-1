"""
AEM (Automatic Experimental Mode) - Final Fixed Version
Copyright (c) 2025, Rick Lan

修正說明：
1. 修正 update_states 縮排問題，確保程式能讀取到此函式。
2. 包含：70km/h 跳水靈敏度、TTC 防護、動態緊急距離。
"""

import time
import numpy as np
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL


# ==============================================================================
#                               CONFIG (參數設定區)
# ==============================================================================

class Config:
    # --- 時間參數 (秒) ---
    FIRST_PARAMETER = 2.0  # 第一參數 (一般舒適減速)
    SECOND_PARAMETER = 2.0  # 第二參數 (緊急/危險情境)

    # --- 閾值設定 ---
    TRIGGER_THRESHOLD = 0.5   # 濾波器啟動門檻
    TTC_EMERGENCY     = 1.0   # TTC 緊急碰撞時間 (秒)

    # --- 距離參數 (公尺) ---
    EMERGENCY_DIST_CITY    = 50.0 # 市區緊急煞停距離
    EMERGENCY_DIST_HIGHWAY = 30.0 # 高速緊急煞停距離
    RADAR_MISS_DIST        = 20.0 # 無雷達防護距離
    LEAD_CLOSE_DIST        = 25.0 # 跟車過近防撞
    SLOW_LEAD_DIST_MAX     = 120.0 # 慢車偵測距離

    # --- 速度定義 (km/h) ---
    HIGHWAY_SPEED = 70.0  # 高速判定
    SLOW_LEAD_DIFF = 36.0 # 慢車速差門檻

    # --- 靈敏度曲線 (70km/h 跳水) ---
    SENSITIVITY_BP   = [0.,  69.9, 70.0, 200.]
    SENSITIVITY_VALS = [1.0, 1.0,  0.8,  0.8]

    # --- 減速模型 ---
    SLOW_DOWN_BP   = [0., 2.78, 5.56, 8.34, 11.12, 13.89, 15.28, 22.22, 30.0]
    SLOW_DOWN_DIST = [10, 30., 50., 70., 80., 90., 120., 140., 160.]


# ==============================================================================
#                               CORE LOGIC (核心邏輯)
# ==============================================================================

class AEM:
    def __init__(self):
        self._active = False
        self._cooldown_end_time = 0.0
        self._trigger_filter = FirstOrderFilter(0, 0.5, DT_MDL)

    def _activate(self, duration):
        """啟動 AEM 並設定冷卻時間"""
        self._active = True
        self._cooldown_end_time = time.monotonic() + duration

    def get_mode(self, mode):
        """外部呼叫介面：決定駕駛模式"""
        if time.monotonic() < self._cooldown_end_time:
            mode = 'blended'
        else:
            self._active = False
        return mode

    # 重要：此函式必須縮排在 class AEM 裡面
    def update_states(self, model_msg, radar_msg, v_ego):
        """主更新迴圈"""
        # 1. 數據完整性檢查
        if not (len(model_msg.position.x) == ModelConstants.IDX_N and 
                len(model_msg.orientation.x) == ModelConstants.IDX_N):
            self._trigger_filter.update(0)
            return

        # 2. 基礎變數準備
        v_kph = v_ego * 3.6
        model_dist = np.clip(model_msg.position.x[ModelConstants.IDX_N - 1], 0.0, np.inf)
        
        # 計算基礎觸發距離
        base_trigger_dist = np.interp(v_ego, Config.SLOW_DOWN_BP, Config.SLOW_DOWN_DIST)
        sensitivity = self._get_sensitivity(v_kph)
        trigger_dist = base_trigger_dist * sensitivity

        # 彎道適配
        curvature = np.mean(np.abs(np.diff(model_msg.orientation.x))) / DT_MDL
        if curvature > 0.1:
            trigger_dist *= 0.8

        # --- 判斷邏輯開始 ---

        # [優先權 1] 危險情境強制介入 (TTC / 慢車 / 跟車過近 / 無雷達)
        if self._check_lead_danger(radar_msg, model_dist, v_ego, v_kph, sensitivity):
            self._activate(Config.SECOND_PARAMETER) 
            self._trigger_filter.update(1.0) 
            return

        # [優先權 2] 緊急距離旁路
        dynamic_emerg_dist = self._get_dynamic_emergency_dist(v_kph)
        if model_dist < dynamic_emerg_dist:
            self._activate(Config.SECOND_PARAMETER) 
            self._trigger_filter.update(1.0)
            return

        # [優先權 3] 一般舒適減速
        input_score = 0.0
        if model_dist < trigger_dist:
            raw_score = (trigger_dist - model_dist) / trigger_dist
            input_score = np.clip(raw_score, 0.0, 1.0)

        self._trigger_filter.update(input_score)

        if self._trigger_filter.x >= Config.TRIGGER_THRESHOLD:
            self._activate(Config.FIRST_PARAMETER)

    # 輔助函式 (Helper Functions)
    def _get_sensitivity(self, v_kph):
        return np.interp(v_kph, Config.SENSITIVITY_BP, Config.SENSITIVITY_VALS)

    def _get_dynamic_emergency_dist(self, v_kph):
        if v_kph > Config.HIGHWAY_SPEED:
            return Config.EMERGENCY_DIST_HIGHWAY
        return Config.EMERGENCY_DIST_CITY

    def _check_lead_danger(self, radar_msg, model_dist, v_ego, v_kph, sensitivity):
        if not radar_msg.leadOne.status:
            return model_dist < Config.RADAR_MISS_DIST

        v_lead = radar_msg.leadOne.vLead
        speed_diff = v_ego - v_lead

        # TTC 緊急防護
        if speed_diff > 0:
            ttc = model_dist / speed_diff
            if ttc < Config.TTC_EMERGENCY:
                return True

        # 跟車過近
        if model_dist < Config.LEAD_CLOSE_DIST:
            return True

        # 慢車偵測
        current_slow_max = Config.SLOW_LEAD_DIST_MAX
        if v_kph > Config.HIGHWAY_SPEED:
            current_slow_max *= 0.8
        
        slow_diff_threshold = Config.SLOW_LEAD_DIFF / 3.6 
        
        if speed_diff > slow_diff_threshold and model_dist < current_slow_max:
            return True

        return False
