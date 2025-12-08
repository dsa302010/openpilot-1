"""
Dynamic Turn Speed Controller (DTSC) - Smooth Hybrid Edition (Final)
整合與優化:
1. DTSC v18 (順滑邏輯) + DTSC v20 (強化舵角輔助)
2. SCC-V Abort Logic (誤判防護)
3. 5-Point Speed-Dependent Limit (5點式速度限制):
   - 0-18 km/h: 1.8G (極致防護)
   - 90 km/h+:  2.8G (高速全開)
4. Temporal Low-Pass Filter (時間平滑濾波，消除頓挫)

Checked & Optimized by Gemini
"""

import numpy as np
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.common.swaglog import cloudlog

# =============================
# 基本參數
# =============================
# Model 輸出的時間軸
MODEL_T_IDXS = ModelConstants.T_IDXS

# --- 彎道 Lateral G 安全限制 ---
# 這是基礎參考值，實際運作會完全由下方的 5-Point Curve 決定
BASE_LAT_ACC = 2.8
SAFETY_SPEED_FACTOR = 0.95

# --- [關鍵設定] 5點式速度依賴限製表 ---
# 解決「低速急彎衝出」與「高速過度減速」的矛盾
# 格式: [速度 m/s], [對應的 Lat G 限制]
# 5.0 m/s  = 18 km/h
# 15.0 m/s = 54 km/h
# 25.0 m/s = 90 km/h (超過此速度將鎖定為 2.8G)
LAT_LIMIT_BP = [5.0, 10.0, 15.0, 20.0, 25.0]
LAT_LIMIT_V  = [1.8,  1.8,  2.4, 2.65,  2.8]

# --- Low-Pass Filter 平滑係數 ---
# 範圍 0.0 ~ 1.0 (0.3 代表新數據佔 30%，舊數據佔 70%)
# 設為 0.3 可有效過濾模型跳動，同時保持足夠的反應速度
LPF_ALPHA = 0.3

# ---  Pre-deceleration（平滑提前煞車）設定 ---
ENTERING_SMOOTH_DECEL_BP = np.array([0.8, 1.4, 2.2])      # 預測 lat_acc
ENTERING_SMOOTH_DECEL_V  = np.array([-0.5, -1.0, -1.8])   # 對應 decel（越負煞越強）

# --- 減速度限制（單位 m/s²）---
MAX_COMFORT_DECEL = -2.0   # 舒適減速極限
EMERGENCY_DECEL   = -4.5   # 緊急減速極限

# MIN_CURVE_DISTANCE：若彎道在此距離之內 → 直接視為急彎模式
MIN_CURVE_DISTANCE = 5.0

# MAX_EXIT_ACCEL：出彎時的最大加速度限制 (避免S彎猛加速)
MAX_EXIT_ACCEL = 0.7

# --- 強化版舵角輔助參數  ---
STEER_ASSIST_ANGLE_THRESHOLD = 10.0  # (度) 超過 10 度即啟動輔助，涵蓋路口轉彎
STEER_SPEED_SCALE = 1.0              # 舵角計算出的速度縮放係數
STEER_AGGRESSIVENESS = 1.0           # 針對舵角的 Lat G 容許值倍率

# --- 巷道誤判防護參數 (整合 SCC-V) ---
PERSISTENCE_MIN_FRAC = 0.5
CURVATURE_MIN_FOR_PERSIST = 0.01
SHORT_DIST_IGNORE = 3.5
STEER_ANGLE_FOR_SHORT = 8.0
SCCV_ABORT_PRED_LAT_ACC_TH = 1.1 

# 前方彎道檢查參數
FUTURE_CURVE_THRESHOLD = 0.01

# =============================
# 工具函式
# =============================
def clamp(x, low, high):
    return max(low, min(high, x))

def interp_clamped(x, bp, fp):
    """Safe interpolation with clamp."""
    if x <= bp[0]:
        return fp[0]
    if x >= bp[-1]:
        return fp[-1]
    return float(np.interp(x, bp, fp))

# =============================
# DTSC 主類別
# =============================
class DTSC:
    def __init__(self, aggressiveness=1.0):
        # 限制 aggressiveness 範圍
        self.aggressiveness = clamp(aggressiveness, 0.5, 1.8)
        self.active = False
        self._hysteresis_state = False
        
        # 用於儲存上一幀的 Lat Limit 陣列，做平滑濾波用
        self.filtered_lat_limits = None
        
        cloudlog.info(f"DTSC Final: Initialized with aggressiveness {self.aggressiveness:.2f}")

    def set_aggressiveness(self, value):
        self.aggressiveness = clamp(value, 0.5, 1.8)

    def _is_model_valid(self, model_msg):
        try:
            return (len(model_msg.position.x) == ModelConstants.IDX_N and
                    len(model_msg.velocity.x) == ModelConstants.IDX_N and
                    len(model_msg.orientationRate.z) == ModelConstants.IDX_N)
        except Exception:
            return False

    def _compute_model_arrays(self, model_msg):
        v_arr = np.array(model_msg.velocity.x)
        pos_arr = np.array(model_msg.position.x)
        yaw_arr = np.array(model_msg.orientationRate.z)

        # Interpolate to MPC timeline
        v_pred = np.interp(T_IDXS_MPC, MODEL_T_IDXS, v_arr)
        pos = np.interp(T_IDXS_MPC, MODEL_T_IDXS, pos_arr)
        yaw = np.interp(T_IDXS_MPC, MODEL_T_IDXS, yaw_arr)

        rel_pos = pos - pos[0]
        rel_pos = np.maximum(rel_pos, 0.0)
        return v_pred, rel_pos, yaw

    def _compute_safe_speeds(self, v_pred, yaw_rates, steer_angle_deg, steer_ratio, wheelbase):
        """
        計算安全速度：結合模型曲率 + 5點式速度限制 + 時間平滑 + 舵角輔助
        """
        # 1. 計算原始的動態限制 (Raw Dynamic Limits)
        # 根據速度 v_pred 查表，獲得當下速度允許的 Lat G
        # 90km/h (25m/s) 以上會自動取 2.8G
        raw_lat_limits = np.interp(v_pred, LAT_LIMIT_BP, LAT_LIMIT_V) * self.aggressiveness
        
        # 2. 時間平滑濾波 (Temporal Low-Pass Filter)
        if self.filtered_lat_limits is None:
            self.filtered_lat_limits = raw_lat_limits
        else:
            # LPF 公式: New = alpha * Raw + (1 - alpha) * Old
            self.filtered_lat_limits = (LPF_ALPHA * raw_lat_limits) + \
                                       ((1.0 - LPF_ALPHA) * self.filtered_lat_limits)
        
        # 使用平滑後的限制值 (確保不低於 1.0G 以免數學異常)
        current_lat_limits = np.maximum(self.filtered_lat_limits, 1.0)
        
        # 3. 模型曲率計算
        # 限制分母最小速度為 1.0，防止靜止時除以零
        v_clip = np.clip(v_pred, 1.0, 100.0)
        curvatures = np.abs(yaw_rates / v_clip)
        
        # 計算安全速度 (Physics Formula: v = sqrt(a_lat / k))
        safe_speeds_model = np.sqrt(current_lat_limits / (curvatures + 1e-6)) * SAFETY_SPEED_FACTOR

        # 4. 強化版舵角輔助 (Steering Assist)
        final_safe_speeds = safe_speeds_model.copy()

        abs_steer = abs(steer_angle_deg)
        if abs_steer > STEER_ASSIST_ANGLE_THRESHOLD and steer_ratio > 0 and wheelbase > 0:
            steer_rad = np.radians(abs_steer)
            steer_curvature = steer_rad / (steer_ratio * wheelbase)
            
            if steer_curvature > 1e-6:
                # 舵角計算同樣受 current_lat_limits 約束，確保低速邏輯一致
                lat_acc_limit_steer = current_lat_limits * STEER_AGGRESSIVENESS
                
                raw_safe_speed_steer = np.sqrt(lat_acc_limit_steer / steer_curvature)
                safe_speed_steer_val = raw_safe_speed_steer * SAFETY_SPEED_FACTOR * STEER_SPEED_SCALE
                
                # 取兩者最小值 (模型 vs 實際舵角)
                final_safe_speeds = np.minimum(safe_speeds_model, safe_speed_steer_val)

        return final_safe_speeds, curvatures

    def _compute_sp_decel(self, predicted_lat_acc_max):
        """SP 順滑減速邏輯 (查表法)"""
        if predicted_lat_acc_max <= ENTERING_SMOOTH_DECEL_BP[0]:
            return 0.0
        decel = interp_clamped(predicted_lat_acc_max, ENTERING_SMOOTH_DECEL_BP, ENTERING_SMOOTH_DECEL_V)
        return clamp(decel, EMERGENCY_DECEL, 0.0)

    def _compute_dtsc_decel(self, v_ego, v_pred, rel_pos, safe_speeds):
        """DTSC 主減速計算邏輯 (物理公式法)"""
        speed_excess = v_pred - safe_speeds
        if np.all(speed_excess <= 0.0):
            return 0.0, None, None

        # 找出最需要減速的關鍵點
        critical_idx = int(np.argmax(speed_excess))
        critical_rel_dist = rel_pos[critical_idx]

        if critical_rel_dist < 1e-3:
            critical_rel_dist = 1e-3

        # 物理公式: v^2 - u^2 = 2as
        decel_by_distance = (safe_speeds[critical_idx] ** 2 - v_ego ** 2) / (2.0 * critical_rel_dist)
        decel_by_distance = min(decel_by_distance, 0.0)

        if critical_rel_dist <= MIN_CURVE_DISTANCE:
            decel_by_distance = max(decel_by_distance, EMERGENCY_DECEL)
            mode = 'EMERGENCY'
        else:
            decel_by_distance = max(decel_by_distance, MAX_COMFORT_DECEL)
            mode = 'COMFORT'

        return decel_by_distance, critical_idx, mode

    def get_mpc_constraints(self, model_msg, v_ego, base_a_min, base_a_max,
                            steer_angle_deg=0.0,
                            steer_ratio=14.3, 
                            wheelbase=2.7):
        """
        回傳 MPC 的加速度限制 (a_min, a_max)
        """
        horizon_len = len(T_IDXS_MPC)
        
        # 初始化輸出陣列
        a_min = np.ones(horizon_len) * (base_a_min if np.isscalar(base_a_min) else base_a_min[0])
        a_max = np.array(base_a_max) if not np.isscalar(base_a_max) else np.ones(horizon_len) * base_a_max

        if not self._is_model_valid(model_msg):
            # 若模型無效，重設濾波器狀態，避免殘留錯誤數值
            self.filtered_lat_limits = None 
            return a_min, a_max

        # 1. 處理模型數據
        v_pred, rel_pos, yaw_rates = self._compute_model_arrays(model_msg)
        
        # 2. 計算預測的側向加速度 (用於 SP 預減速)
        predicted_lat_accels = np.abs(v_pred * yaw_rates)
        predicted_lat_acc_max = float(np.max(predicted_lat_accels))

        # 3. 計算安全速度 (含 5點式限制 + LPF + 舵角)
        safe_speeds, curvatures = self._compute_safe_speeds(
            v_pred, yaw_rates, steer_angle_deg, steer_ratio, wheelbase)

        # 4. 計算兩種減速需求
        sp_decel = self._compute_sp_decel(predicted_lat_acc_max)
        dt_decel, critical_idx, dt_mode = self._compute_dtsc_decel(v_ego, v_pred, rel_pos, safe_speeds)

        # 5. 誤判防護 (SCC-V Abort Logic)
        speed_excess = v_pred - safe_speeds
        mask_curve = curvatures > CURVATURE_MIN_FOR_PERSIST
        mask_speed = speed_excess > 0.01
        mask = np.logical_and(mask_speed, mask_curve)
        
        frac_problem = float(np.sum(mask)) / len(mask) if len(mask) > 0 else 0
        persistence_ok = frac_problem >= PERSISTENCE_MIN_FRAC
        critical_dist = rel_pos[critical_idx] if critical_idx is not None else 999.0

        # 中止邏輯 A: 預測側向力過低 (表示彎不急)
        if predicted_lat_acc_max < SCCV_ABORT_PRED_LAT_ACC_TH:
            dt_decel = sp_decel = 0.0
            dt_mode = None
        # 中止邏輯 B: 短距離且模型信心不足
        elif not persistence_ok and critical_dist < SHORT_DIST_IGNORE:
            if abs(steer_angle_deg) < STEER_ANGLE_FOR_SHORT:
                dt_decel = sp_decel = 0.0
                dt_mode = None

        # 6. 整合最終減速度 (取兩者最安全值)
        final_required_decel = 0.0
        
        if dt_mode == "EMERGENCY":
            final_required_decel = dt_decel
        else:
            sp = min(sp_decel, 0.0)
            dt = min(dt_decel, 0.0)
            final_required_decel = min(sp, dt)
            final_required_decel = clamp(final_required_decel, EMERGENCY_DECEL, 0.0)

        # 7. Hysteresis (滯後) 狀態管理
        if final_required_decel < 0.0:
            self._hysteresis_state = True
            self.active = True
        else:
            if self._hysteresis_state:
                self._hysteresis_state = False
            self.active = False

        # 8. 套用到 MPC Constraints
        if self.active and final_required_decel < 0.0:
            critical_distance = rel_pos[critical_idx] if critical_idx is not None else np.max(rel_pos)
            critical_distance = max(critical_distance, 1e-3)

            # 檢查出彎後是否還有彎道
            has_future_curve = any(
                rel_pos[i] > critical_distance and curvatures[i] > FUTURE_CURVE_THRESHOLD
                for i in range(horizon_len)
            )

            for i in range(horizon_len):
                # 入彎前與彎中: 套用減速
                if rel_pos[i] <= critical_distance + 1e-6:
                    a_max[i] = min(a_max[i], final_required_decel)
                else:
                    # 出彎後: 若有後續彎道則限制加速
                    if has_future_curve:
                        a_max[i] = min(a_max[i], MAX_EXIT_ACCEL)

        return a_min, a_max
