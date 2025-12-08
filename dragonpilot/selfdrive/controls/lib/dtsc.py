"""
Dynamic Turn Speed Controller (DTSC) - Refined Final Edition
整合項目:
1. DTSC v18 (順滑邏輯) + DTSC v20 (強化舵角輔助)
2. SCC-V Abort Logic (誤判防護 - 修正門檻版)
3. 5-Point Speed-Dependent Limit (5點式速度限制)
4. Temporal Low-Pass Filter (智能重置版)
5. Hysteresis Timer (滯後計時器)

Optimized for:
- City: 防止過度減速 (Min Speed Floor)
- Highway: 只擋緊急閃避 (Safe Limits)
- Mountain/Ramp: 流暢度提升 (Tuned Thresholds)
"""

import numpy as np
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.common.swaglog import cloudlog

# =============================
# 基本參數
# =============================
MODEL_T_IDXS = ModelConstants.T_IDXS
DT_MPC = 0.05  # MPC 運行頻率約 20Hz

# --- 彎道 Lateral G 安全限制 ---
BASE_LAT_ACC = 2.8
SAFETY_SPEED_FACTOR = 0.95

# --- [關鍵設定] 5點式速度依賴限製表 (m/s²) ---
# 0-18 km/h: 1.7 m/s² (防護)
# 90 km/h+:  2.8 m/s² (寬容)
LAT_LIMIT_BP = [5.0, 10.0, 15.0, 20.0, 25.0]
LAT_LIMIT_V  = [1.7, 1.7, 2.4, 2.6, 2.8]

# --- Low-Pass Filter 平滑係數 ---
LPF_ALPHA = 0.3

# ---  Pre-deceleration（平滑提前煞車）設定 ---
ENTERING_SMOOTH_DECEL_BP = np.array([0.8, 1.4, 2.2])
ENTERING_SMOOTH_DECEL_V  = np.array([-0.5, -1.0, -1.8])

# --- 減速度限制（單位 m/s²）---
MAX_COMFORT_DECEL = -2.0
EMERGENCY_DECEL   = -4.5

# MIN_CURVE_DISTANCE
MIN_CURVE_DISTANCE = 5.0

# MAX_EXIT_ACCEL (出彎限制加速)
MAX_EXIT_ACCEL = 0.7

# --- 強化版舵角輔助參數 (最終微調) ---
STEER_ASSIST_ANGLE_THRESHOLD = 10.0
STEER_SPEED_SCALE = 1.05          # [優化] 1.0 -> 1.05，讓山路爬坡稍微輕快一點
STEER_AGGRESSIVENESS = 1.0        # 保持 1.0，與模型共用 G Limit
MIN_STEER_SPEED_FLOOR = 5.0       # [新增] 5.0 m/s (~18 km/h)，防止市區轉彎壓到個位數速度

# --- 巷道誤判防護參數 (SCC-V) ---
PERSISTENCE_MIN_FRAC = 0.5
CURVATURE_MIN_FOR_PERSIST = 0.01
SHORT_DIST_IGNORE = 3.5
STEER_ANGLE_FOR_SHORT = 8.0
SCCV_ABORT_PRED_LAT_ACC_TH = 0.5  # [修正] 1.1 -> 0.5，恢復正常彎道偵測能力

# --- 前方彎道與直線檢查參數 ---
FUTURE_CURVE_THRESHOLD = 0.015    # [優化] 0.03 -> 0.015，確保能偵測到高速公路匝道
STRAIGHT_ROAD_CURVATURE = 0.002   # 直線判定門檻 (用於重置 LPF)

# --- Hysteresis (滯後) 設定 ---
HYSTERESIS_TIME = 0.5             # 減速需求消失後，維持 Active 0.5秒

# =============================
# 工具函式
# =============================
def clamp(x, low, high):
    return max(low, min(high, x))

def interp_clamped(x, bp, fp):
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
        self.aggressiveness = clamp(aggressiveness, 0.5, 1.8)
        self.active = False
        
        # Hysteresis 計時器
        self.hysteresis_timer = 0.0
        
        # LPF 狀態
        self.filtered_lat_limits = None
        
        cloudlog.info(f"DTSC Final Refined: Initialized with aggressiveness {self.aggressiveness:.2f}")

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
        # 1. 查表獲得當下速度允許的 Lat G (m/s²)
        raw_lat_limits = np.interp(v_pred, LAT_LIMIT_BP, LAT_LIMIT_V) * self.aggressiveness
        
        # 2. 時間平滑濾波 (Temporal LPF)
        if self.filtered_lat_limits is None:
            self.filtered_lat_limits = raw_lat_limits
        else:
            self.filtered_lat_limits = (LPF_ALPHA * raw_lat_limits) + \
                                       ((1.0 - LPF_ALPHA) * self.filtered_lat_limits)
        
        # 確保除數安全
        current_lat_limits = np.maximum(self.filtered_lat_limits, 1.0)
        
        # 3. 模型曲率計算
        v_clip = np.clip(v_pred, 1.0, 100.0)
        curvatures = np.abs(yaw_rates / v_clip)
        
        # 模型安全速度
        safe_speeds_model = np.sqrt(current_lat_limits / (curvatures + 1e-6)) * SAFETY_SPEED_FACTOR

        # 4. 強化版舵角輔助 (Fix applied)
        final_safe_speeds = safe_speeds_model.copy()

        abs_steer = abs(steer_angle_deg)
        if abs_steer > STEER_ASSIST_ANGLE_THRESHOLD and steer_ratio > 0 and wheelbase > 0:
            steer_rad = np.radians(abs_steer)
            steer_curvature = steer_rad / (steer_ratio * wheelbase)
            
            if steer_curvature > 1e-6:
                # [關鍵修正] 舵角共用模型 G Limit，不重複懲罰
                lat_acc_limit_steer = current_lat_limits
                
                raw_safe_speed_steer = np.sqrt(lat_acc_limit_steer / steer_curvature)
                safe_speed_steer_val = raw_safe_speed_steer * SAFETY_SPEED_FACTOR * STEER_SPEED_SCALE
                
                # [關鍵修正] 最低速度保護 (防止市區過慢)
                safe_speed_steer_val = np.maximum(safe_speed_steer_val, MIN_STEER_SPEED_FLOOR)
                
                # 取兩者最小值 (模型 vs 實際舵角)
                final_safe_speeds = np.minimum(safe_speeds_model, safe_speed_steer_val)

        return final_safe_speeds, curvatures

    def _compute_sp_decel(self, predicted_lat_acc_max):
        """SP 順滑減速邏輯"""
        if predicted_lat_acc_max <= ENTERING_SMOOTH_DECEL_BP[0]:
            return 0.0
        decel = interp_clamped(predicted_lat_acc_max, ENTERING_SMOOTH_DECEL_BP, ENTERING_SMOOTH_DECEL_V)
        return clamp(decel, EMERGENCY_DECEL, 0.0)

    def _compute_dtsc_decel(self, v_ego, v_pred, rel_pos, safe_speeds):
        """DTSC 主減速計算邏輯"""
        speed_excess = v_pred - safe_speeds
        if np.all(speed_excess <= 0.0):
            return 0.0, None, None

        # 找出最需要減速的關鍵點
        critical_idx = int(np.argmax(speed_excess))
        critical_rel_dist = rel_pos[critical_idx]
        critical_rel_dist = max(critical_rel_dist, 1e-3)

        # 物理公式: v^2 - u^2 = 2as -> a = (v^2 - u^2) / 2s
        decel_by_distance = (safe_speeds[critical_idx] ** 2 - v_ego ** 2) / (2.0 * critical_rel_dist)
        decel_by_distance = min(decel_by_distance, 0.0)

        # 區分舒適與緊急模式
        if critical_rel_dist <= MIN_CURVE_DISTANCE:
            mode = 'EMERGENCY'
            # 緊急模式：不做舒適限制，但後續會有總體 clamp
        else:
            mode = 'COMFORT'
            decel_by_distance = max(decel_by_distance, MAX_COMFORT_DECEL)

        return decel_by_distance, critical_idx, mode

    def get_mpc_constraints(self, model_msg, v_ego, base_a_min, base_a_max,
                            steer_angle_deg=0.0,
                            steer_ratio=14.3, 
                            wheelbase=2.7):
        """
        回傳 MPC 的加速度限制 (a_min, a_max)
        """
        horizon_len = len(T_IDXS_MPC)
        a_min = np.ones(horizon_len) * (base_a_min if np.isscalar(base_a_min) else base_a_min[0])
        a_max = np.array(base_a_max) if not np.isscalar(base_a_max) else np.ones(horizon_len) * base_a_max

        if not self._is_model_valid(model_msg):
            self.filtered_lat_limits = None 
            return a_min, a_max

        # 1. 處理模型數據
        v_pred, rel_pos, yaw_rates = self._compute_model_arrays(model_msg)
        predicted_lat_accels = np.abs(v_pred * yaw_rates)
        predicted_lat_acc_max = float(np.max(predicted_lat_accels))

        # [智能重置] 若當前是直路，重置 LPF 記憶
        if predicted_lat_acc_max < STRAIGHT_ROAD_CURVATURE or np.max(np.abs(yaw_rates)) < 0.01:
             self.filtered_lat_limits = None

        # 2. 計算安全速度
        safe_speeds, curvatures = self._compute_safe_speeds(
            v_pred, yaw_rates, steer_angle_deg, steer_ratio, wheelbase)

        # 3. 計算減速需求
        sp_decel = self._compute_sp_decel(predicted_lat_acc_max)
        dt_decel, critical_idx, dt_mode = self._compute_dtsc_decel(v_ego, v_pred, rel_pos, safe_speeds)

        # 4. SCC-V Abort Logic (誤判防護)
        speed_excess = v_pred - safe_speeds
        mask_curve = curvatures > CURVATURE_MIN_FOR_PERSIST
        mask_speed = speed_excess > 0.01
        mask = np.logical_and(mask_speed, mask_curve)
        
        frac_problem = float(np.sum(mask)) / len(mask) if len(mask) > 0 else 0
        persistence_ok = frac_problem >= PERSISTENCE_MIN_FRAC
        critical_dist = rel_pos[critical_idx] if critical_idx is not None else 999.0

        # [修正] 門檻 0.5G (原 1.1G)
        if predicted_lat_acc_max < SCCV_ABORT_PRED_LAT_ACC_TH:
            dt_decel = sp_decel = 0.0
            dt_mode = None
        elif not persistence_ok and critical_dist < SHORT_DIST_IGNORE:
            if abs(steer_angle_deg) < STEER_ANGLE_FOR_SHORT:
                dt_decel = sp_decel = 0.0
                dt_mode = None

        # 5. 整合最終減速度
        final_required_decel = 0.0
        if dt_mode == "EMERGENCY":
            final_required_decel = dt_decel
        else:
            sp = min(sp_decel, 0.0)
            dt = min(dt_decel, 0.0)
            final_required_decel = min(sp, dt)
        
        # 最終統一限制 (Clamp)
        final_required_decel = clamp(final_required_decel, EMERGENCY_DECEL, 0.0)

        # 6. Hysteresis Timer (滯後計時器) 管理
        if final_required_decel < -0.1:  # 有實質減速需求
            self.hysteresis_timer = HYSTERESIS_TIME
            self.active = True
        else:
            if self.hysteresis_timer > 0:
                self.hysteresis_timer -= DT_MPC
                self.active = True
            else:
                self.active = False
                self.hysteresis_timer = 0

        # 7. 套用到 MPC Constraints
        if self.active:
            # 在 Hysteresis 期間若無實質需求，不強制減速，但保持 active 狀態
            pass_decel = final_required_decel if final_required_decel < 0 else 0.0

            critical_distance = rel_pos[critical_idx] if critical_idx is not None else np.max(rel_pos)
            critical_distance = max(critical_distance, 1e-3)

            # [修正] 門檻 0.015 (原 0.03)，確保匝道適用
            has_future_curve = any(
                rel_pos[i] > critical_distance and curvatures[i] > FUTURE_CURVE_THRESHOLD
                for i in range(horizon_len)
            )

            for i in range(horizon_len):
                if rel_pos[i] <= critical_distance + 1e-6:
                    if pass_decel < 0:
                        a_max[i] = min(a_max[i], pass_decel)
                else:
                    if has_future_curve:
                        a_max[i] = min(a_max[i], MAX_EXIT_ACCEL)

        return a_min, a_max
