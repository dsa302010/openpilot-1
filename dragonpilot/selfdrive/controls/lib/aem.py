"""
AEM (Automatic Experimental Mode) - Balanced Fixed Version (Traditional Chinese)
Copyright (c) 2025, Modified for DragonPilot

版本說明：
1. 修復座標軸錯誤 (X軸橫向/Z軸縱向)，解決市區紅綠燈誤煞車問題。
2. 採用平衡型高速邏輯：70km/h (0.7), 90km/h (0.6)，高速巡航更穩定。
3. 增加詳細中文註解與單位標示。
"""

import numpy as np
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants

# ==============================================================================
#                               CONFIG (參數設定區)
# ==============================================================================
class Config:
    # --- 閾值設定 ---
    TTC_EMERGENCY     = 0.8   # [秒] TTC 緊急碰撞時間 (低於此值強制觸發)
    
    # --- 距離參數 (單位：公尺 m) ---
    EMERGENCY_DIST_CITY    = 30.0  # [m] 市區跟車緊急煞停距離 (放寬以避免過敏)
    EMERGENCY_DIST_HIGHWAY = 50.0  # [m] 高速跟車緊急煞停距離
    RADAR_MISS_DIST        = 10.0  # [m] 無雷達時，視覺牆的最小防護距離 (過濾雜訊後)
    LEAD_CLOSE_DIST        = 15.0  # [m] 極限貼車距離 (低於此值無視速差強制介入)
    SLOW_LEAD_DIST_MAX     = 100.0 # [m] 慢車偵測的最遠距離

    # --- 速度定義 (單位：km/h) ---
    HIGHWAY_SPEED  = 80.0  # [km/h] 判定為高速公路的門檻
    SLOW_LEAD_DIFF = 30.0  # [km/h] 與前車的速差門檻 (我比前車快多少)

    # --- 靈敏度曲線 (平衡版高速邏輯) ---
    # 格式：[速度節點 km/h]
    # 說明：
    # 0-60 km/h: 1.0 (市區全開，紅燈/路口完整減速)
    # 70 km/h:   0.7 (高速過渡區，過濾 30% 雜訊)
    # 90 km/h+:  0.6 (高速巡航區，過濾 40% 雜訊，以 ACC 為主)
    SENSITIVITY_BP   = [0.,  60., 70., 90.]
    SENSITIVITY_VALS = [1.0, 1.0, 0.7, 0.6]

    # --- 減速模型 (預期煞停距離表) ---
    # BP: 車速 [m/s], DIST: 距離 [m]
    # 轉換參考: 30 m/s = 108 km/h
    SLOW_DOWN_BP   = [0., 5.,  10., 15., 20., 25., 30.]
    SLOW_DOWN_DIST = [5., 20., 40., 60., 80., 100., 120.]
    
    # --- 模式定義字串 ---
    MODE_ACC = 'acc'          # 一般 ACC 模式
    MODE_BLENDED = 'blended'  # 實驗模式 (End-to-End Longitudinal)

# ==============================================================================
#                         UTILITY CLASSES (工具類別)
# ==============================================================================

class SmoothKalmanFilter:
  """
  簡易卡爾曼濾波器 (Kalman Filter)
  用途：將急迫性數值 (Urgency) 平滑化，避免數值跳動造成模式頻繁切換。
  """
  def __init__(self, initial_value=0, alpha=1.0, smoothing_factor=0.85):
    self.x = initial_value
    self.P = 1.0   # 估計誤差協方差
    self.R = 0.2   # 測量雜訊 (數值越大，濾波效果越強，反應越慢)
    self.Q = 0.01  # 過程雜訊
    self.alpha = alpha
    self.smoothing_factor = smoothing_factor
    self.initialized = False

  def add_data(self, measurement):
    """輸入新的測量值"""
    if not self.initialized:
      self.x = measurement
      self.initialized = True
      return
    
    # 預測步驟
    self.P = self.alpha * self.P + self.Q
    # 更新步驟
    K = self.P / (self.P + self.R)
    # 混合平滑因子
    effective_K = K * (1.0 - self.smoothing_factor) + self.smoothing_factor * 0.1
    innovation = measurement - self.x
    self.x = self.x + effective_K * innovation
    self.P = (1 - effective_K) * self.P

  def get_value(self):
    """取得濾波後的數值"""
    return self.x if self.initialized else 0.0

class ModeTransitionManager:
  """
  模式切換管理器 (State Machine)
  用途：管理 ACC 與 Experimental 模式之間的切換，包含遲滯 (Hysteresis) 與冷卻邏輯。
  """
  def __init__(self):
    self.current_mode = Config.MODE_ACC
    # 信心指數：1.0 代表非常有信心，0.0 代表無信心
    self.mode_confidence = {Config.MODE_ACC: 1.0, Config.MODE_BLENDED: 0.0}
    self.lock_timer = 0       # 緊急鎖定計時器 (Frame)
    self.frame_count = 0

  def request_mode(self, mode, confidence=1.0, emergency=False):
    """請求切換模式"""
    # [邏輯 1] 緊急情況：立即切換並鎖定
    if emergency:
      self.current_mode = mode
      self.lock_timer = 20  # 鎖定約 1 秒 (20 frames)
      self.mode_confidence[Config.MODE_BLENDED] = 1.0
      self.mode_confidence[Config.MODE_ACC] = 0.0
      return

    # 若處於鎖定冷卻期，忽略一般請求
    if self.lock_timer > 0:
      return

    # [邏輯 2] 信心指數累積
    target_conf = min(1.0, self.mode_confidence[mode] + 0.05 * confidence)
    self.mode_confidence[mode] = target_conf
    
    # 對立模式信心扣除
    for m in self.mode_confidence:
      if m != mode:
        self.mode_confidence[m] = max(0.0, self.mode_confidence[m] - 0.05)

    # [邏輯 3] 切換門檻 (遲滯 Hysteresis)
    # 若要切換到新模式，需要較高信心 (0.75)；維持現有模式只需較低信心 (0.4)
    threshold = 0.75 if mode != self.current_mode else 0.4
    
    if self.mode_confidence[mode] > threshold:
        self.current_mode = mode

  def update(self):
    """每幀更新狀態"""
    if self.lock_timer > 0:
      self.lock_timer -= 1
      
    # 自然衰減：若無人請求，信心會慢慢回到 ACC
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
        # 初始化濾波器，alpha=1.01 略微增加預測趨勢
        self._slow_down_filter = SmoothKalmanFilter(alpha=1.01, smoothing_factor=0.9)
        self._urgency = 0.0

    def get_mode(self, current_mode_str):
        """外部呼叫介面：取得當前建議模式"""
        return self._mode_manager.get_mode()

    def update_states(self, model_msg, radar_msg, v_ego):
        """主邏輯更新迴圈"""
        # 1. 數據完整性檢查
        if not (len(model_msg.position.x) == ModelConstants.IDX_N and 
                len(model_msg.position.z) == ModelConstants.IDX_N):
            return

        v_kph = v_ego * 3.6 # 轉換單位：m/s -> km/h
        
        # [修正 1] 座標系修正
        # model.position.x = 橫向偏移 (Lateral)
        # model.position.z = 縱向距離 (Longitudinal / Distance)
        path_x = model_msg.position.x 
        path_z = model_msg.position.z 
        model_end_dist = path_z[ModelConstants.IDX_N - 1] # 取得模型預測的最遠距離

        # [修正 2] 彎道計算 (使用路徑幾何，防止紅綠燈抖動誤判)
        curvature_factor = 0.0
        try:
            # 取約 2 秒處 (index 15) 的點來計算曲率
            mid_idx = 15
            if path_z[mid_idx] > 1.0:
                # 簡單曲率估算：橫向偏移量 / 距離平方
                curvature_val = abs(path_x[mid_idx]) / (path_z[mid_idx]**2)
                curvature_factor = min(1.0, curvature_val * 500.0) 
        except:
            curvature_factor = 0.0

        # 2. 計算舒適減速需求
        self._calculate_slow_down(model_end_dist, curvature_factor, v_ego, v_kph)

        # 3. 執行分層決策
        self._make_decision(radar_msg, v_kph, model_end_dist)

        # 4. 更新狀態機
        self._mode_manager.update()

    def _calculate_slow_down(self, model_end_dist, curvature, v_ego, v_kph):
        """計算是否需要舒適減速 (Urgency Score)"""
        
        # 步驟 A: 查表取得基礎預期距離
        base_expected = np.interp(v_ego, Config.SLOW_DOWN_BP, Config.SLOW_DOWN_DIST)
        
        # 步驟 B: 應用高速靈敏度曲線 (Fix 8)
        # 70km/h -> 0.7, 90km/h -> 0.6
        sensitivity = np.interp(v_kph, Config.SENSITIVITY_BP, Config.SENSITIVITY_VALS)
        
        # 步驟 C: 彎道修正 (大彎道時縮短預期距離，避免誤判)
        curve_penalty = 1.0 - (curvature * 0.2)
        
        # 計算最終觸發門檻
        expected_distance = base_expected * sensitivity * curve_penalty

        urgency = 0.0
        # 緩衝區設計：只有當「模型預測終點」小於「期望距離的 85%」才開始計算
        # 這能避免數值在臨界點跳動
        if model_end_dist < (expected_distance * 0.85):
            shortage = expected_distance - model_end_dist
            shortage_ratio = shortage / expected_distance
            # 計算急迫性分數 (0.0 ~ 1.0)
            urgency = np.clip(shortage_ratio * 1.5, 0.0, 1.0)

        # 平滑化輸出
        self._slow_down_filter.add_data(urgency)
        self._urgency = self._slow_down_filter.get_value()

    def _make_decision(self, radar_msg, v_kph, model_end_dist):
        """分層決策邏輯 (Priority Logic)"""
        
        # [優先級 1] 危險 (Emergency) -> 強制介入，信心 1.0
        if self._check_danger(radar_msg, v_kph, model_end_dist):
            self._mode_manager.request_mode(Config.MODE_BLENDED, confidence=1.0, emergency=True)
            return

        # [優先級 2] 舒適減速 (Comfort Stop) -> 根據急迫性平滑介入
        # 註：高速時因靈敏度係數 (0.6) 壓低了預期距離，此處較難觸發，除非真的很近
        if self._urgency > 0.6:
            self._mode_manager.request_mode(Config.MODE_BLENDED, confidence=self._urgency)
            return

        # [優先級 3] 預設狀態 -> 請求 ACC
        self._mode_manager.request_mode(Config.MODE_ACC, confidence=0.8)

    def _check_danger(self, radar_msg, v_kph, model_end_dist):
        """危險情境檢查 (Safety Net)"""
        if radar_msg is None:
            return False
            
        lead = radar_msg.leadOne
        v_ego = v_kph / 3.6 # m/s
        
        # 1. 視覺防護牆檢查 (無雷達車但模型看到牆/靜止物)
        # 條件：無雷達目標 + 速度 > 20km/h + 模型預測路徑 < 10m
        if not lead.status:
            if v_kph > 20.0 and model_end_dist < Config.RADAR_MISS_DIST:
                return True
            return False

        # 以下為有前車的情境
        d_lead = lead.dRel   # 前車距離 [m]
        v_lead = lead.vLead  # 前車速度 [m/s]
        
        # 2. 絕對距離防護
        # 根據市區/高速選用不同距離 (30m / 50m)
        thresh_dist = Config.EMERGENCY_DIST_HIGHWAY if v_kph > Config.HIGHWAY_SPEED else Config.EMERGENCY_DIST_CITY
        if d_lead < thresh_dist:
            # 只有當「我比前車快」或者「距離極近 (<15m)」才觸發
            # 避免正常排隊跟車時誤觸發
            if (v_ego > v_lead) or (d_lead < Config.LEAD_CLOSE_DIST):
                return True

        # 3. TTC (Time To Collision) 防護
        # 只有當 TTC < 0.5秒 時才視為緊急，1.0秒以上交給 ACC 處理
        if v_ego > v_lead:
            ttc = d_lead / (v_ego - v_lead)
            if ttc < Config.TTC_EMERGENCY:
                return True

        return False
