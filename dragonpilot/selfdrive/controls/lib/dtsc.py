"""
Dynamic Turn Speed Controller (DTSC) - Refined Final Edition (v4)
更新項目:
1. [修復] SyntaxError: 修正了第 217 行斷行導致的語法錯誤
2. [新增] 支援 CarParams (CP) 傳入：解決 Hardcoded 參數問題，自動適應不同車種
3. [包含] v2/v3 的所有邏輯優化 (LPF Reset 0.2, Hysteresis, Min Speed Floor)

Fixed Syntax Error & Fully Optimized.
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

# --- 強化版舵角輔助參數 ---
STEER_ASSIST_ANGLE_THRESHOLD = 10.0
STEER_SPEED_SCALE = 1.05
STEER_AGGRESSIVENESS = 1.0
MIN_STEER_SPEED_FLOOR = 5.0 

# --- 巷道誤判防護參數 (SCC-V) ---
PERSISTENCE_MIN_FRAC = 0.5
CURVATURE_MIN_FOR_PERSIST = 0.01
SHORT_DIST_IGNORE = 3.5
STEER_ANGLE_FOR_SHORT = 8.0
SCCV_ABORT_PRED_LAT_ACC_TH = 0.5

# --- 前方彎道與直線檢查參數 ---
FUTURE_CURVE_THRESHOLD = 0.015
# [修正] 提高重設門檻至 0.2 m/s²，避免過度敏感重設
LPF_RESET_LAT_ACC_THRESHOLD = 0.2

# --- Hysteresis (滯後) 設定 ---
HYSTERESIS_TIME = 0.5

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
    # [新增] 這裡加入了 cp=None，允許外部傳入車輛參數
    def __init__(self, aggressiveness=1.0, cp=None):
        self.aggressiveness = clamp(aggressiveness, 0.5, 1.8)
        self.active = False
        self.hysteresis_timer = 0.0
        self.filtered_lat_limits = None
        
        # [關鍵優化] 自動讀取車輛參數
        if cp is not None:
            self.steer_ratio = cp.steerRatio
            self.wheelbase = cp.wheelbase
            cloudlog.info(f"DTSC Final v4: Loaded CarParams - SR:{self.steer_ratio:.2f}, WB:{self.wheelbase:.2f}")
        else:
            # 備用預設值 (若未傳入 CP)
            self.steer_ratio = 14.3
            self.wheelbase = 2.7
            cloudlog.warning("DTSC Final v4: Warning! Using hardcoded params (SR:14.3, WB:2.7). Pass CP to fix.")
        
        cloudlog.info(f"DTSC Final v4: Initialized with aggressiveness {self.aggressiveness:.2f}")

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

        v_pred = np.interp(T_IDXS_MPC, MODEL_T_IDXS, v_arr)
        pos = np.interp(T_IDXS_MPC, MODEL_T_IDXS, pos_arr)
        yaw = np.interp(T_IDXS_MPC, MODEL_T_IDXS, yaw_arr)

        rel_pos = pos - pos[0]
        rel_pos = np.maximum(rel_pos, 0.0)
        return v_pred, rel_pos, yaw

    def _compute_safe_speeds(self, v_pred, yaw_rates, steer_angle_deg, steer_ratio, wheelbase):
        # 1. 查表獲得當下速度允許的 Lat G
        raw_lat_limits = np.interp(v_pred, LAT_LIMIT_BP, LAT_LIMIT_V) * self.aggressiveness
        
        # 2. 時間平滑濾波 (Temporal LPF)
        if self.filtered_lat_limits is None:
            self.filtered_lat_limits = raw_lat_limits
        else:
            self.filtered_lat_limits = (LPF_ALPHA * raw_lat_limits) + \
                                       ((1.0 - LPF_ALPHA) * self.filtered_lat_limits)
        
        current_lat_limits = np.maximum(self.filtered_lat_limits, 1.0)
        
        # 3. 模型曲率計算
        v_clip = np.clip(v_pred, 1.0, 100.0)
        curvatures = np.abs(yaw_rates / v_clip)
        
        safe_speeds_model = np.sqrt(current_lat_limits / (curvatures + 1e-6)) * SAFETY_SPEED_FACTOR

        # 4. 強化版舵角輔助
        final_safe_speeds = safe_speeds_model.copy()
        abs_steer = abs(steer_angle_deg)
        if abs_steer > STEER_ASSIST_ANGLE_THRESHOLD and steer_ratio > 0 and wheelbase > 0:
            steer_rad = np.radians(abs_steer)
            steer_curvature = steer_rad / (steer_ratio * wheelbase)
            
            if steer_curvature > 1e-6:
                lat_acc_limit_steer = current_lat_limits
                raw_safe_speed_steer = np.sqrt(lat_acc_limit_steer / steer_curvature)
                safe_speed_steer_val = raw_safe_speed_steer * SAFETY_SPEED_FACTOR * STEER_SPEED_SCALE
                safe_speed_steer_val = np.maximum(safe_speed_steer_val, MIN_STEER_SPEED_FLOOR)
                final_safe_speeds = np.minimum(safe_speeds_model, safe_speed_steer_val)

        return final_safe_speeds, curvatures

    def _compute_sp_decel(self, predicted_lat_acc_max):
        if predicted_lat_acc_max <= ENTERING_SMOOTH_DECEL_BP[0]:
            return 0.0
        decel = interp_clamped(predicted_lat_acc_max, ENTERING_SMOOTH_DECEL_BP, ENTERING_SMOOTH_DECEL_V)
        return clamp(decel, EMERGENCY_DECEL, 0.0)

    def _compute_dtsc_decel(self, v_ego, v_pred, rel_pos, safe_speeds):
        speed_excess = v_pred - safe_speeds
        if np.all(speed_excess <= 0.0):
            return 0.0, None, None

        critical_idx = int(np.argmax(speed_excess))
        critical_rel_dist = rel_pos[critical_idx]
        critical_rel_dist = max(critical_rel_dist, 1e-3)

        decel_by_distance = (safe_speeds[critical_idx] ** 2 - v_ego ** 2) / (2.0 * critical_rel_dist)
        decel_by_distance = min(decel_by_distance, 0.0)

        if critical_rel_dist <= MIN_CURVE_DISTANCE:
            mode = 'EMERGENCY'
        else:
            mode = 'COMFORT'
            decel_by_distance = max(decel_by_distance, MAX_COMFORT_DECEL)

        return decel_by_distance, critical_idx, mode

    def get_mpc_constraints(self, model_msg, v_ego, base_a_min, base_a_max,
                            steer_angle_deg=0.0,
                            steer_ratio=None, 
                            wheelbase=None):
        """
        [優化] 參數現在預設為 None，會自動使用 __init__ 讀取到的正確車輛參數
        """
        # 如果呼叫時沒有傳入參數，就使用初始化時從 CP 抓到的值
        current_steer_ratio = steer_ratio if steer_ratio is not None else self.steer_ratio
        current_wheelbase = wheelbase if wheelbase is not None else self.wheelbase
        
        horizon_len = len(T_IDXS_MPC)
        a_min = np.ones(horizon_len) * (base_a_min if np.isscalar(base_a_min) else base_a_min[0])
        a_max = np.array(base_a_max) if not np.isscalar(base_a_max) else np.ones(horizon_len) * base_a_max

        if not self._is_model_valid(model_msg):
            self.filtered_lat_limits = None 
            return a_min, a_max

        v_pred, rel_pos, yaw_rates = self._compute_model_arrays(model_msg)
        predicted_lat_accels = np.abs(v_pred * yaw_rates)
        predicted_lat_acc_max = float(np.max(predicted_lat_accels))

        if predicted_lat_acc_max < LPF_RESET_LAT_ACC_THRESHOLD:
             self.filtered_lat_limits = None

        safe_speeds, curvatures = self._compute_safe_speeds(
            v_pred, yaw_rates, steer_angle_deg, current_steer_ratio, current_wheelbase)

        sp_decel = self._compute_sp_decel(predicted_lat_acc_max)
        dt_decel, critical_idx, dt_mode = self._compute_dtsc_decel(v_ego, v_pred, rel_pos, safe_speeds)

        speed_excess = v_pred - safe_speeds
        mask_curve = curvatures > CURVATURE_MIN_FOR_PERSIST
        mask_speed = speed_excess > 0.01
        mask = np.logical_and(mask_speed, mask_curve)
        
        frac_problem = float(np.sum(mask)) / len(mask) if len(mask) > 0 else 0
        persistence_ok = frac_problem >= PERSISTENCE_MIN_FRAC
        critical_dist = rel_pos[critical_idx] if critical_idx is not None else 999.0

        if predicted_lat_acc_max < SCCV_ABORT_PRED_LAT_ACC_TH:
            dt_decel = sp_decel = 0.0
            dt_mode = None
        # [修復] 這裡已經合併為同一行，解決 SyntaxError
        elif not persistence_ok and critical_dist < SHORT_DIST_IGNORE:
            if abs(steer_angle_deg) < STEER_ANGLE_FOR_SHORT:
                dt_decel = sp_decel = 0.0
                dt_mode = None

        final_required_decel = 0.0
        if dt_mode == "EMERGENCY":
            final_required_decel = dt_decel
        else:
            sp = min(sp_decel, 0.0)
            dt = min(dt_decel, 0.0)
            final_required_decel = min(sp, dt)
        
        final_required_decel = clamp(final_required_decel, EMERGENCY_DECEL, 0.0)

        if final_required_decel < -0.1:
            self.hysteresis_timer = HYSTERESIS_TIME
            self.active = True
        else:
            if self.hysteresis_timer > 0:
                self.hysteresis_timer -= DT_MPC
                self.active = True
            else:
                self.active = False
                self.hysteresis_timer = 0

        if self.active:
            pass_decel = final_required_decel if final_required_decel < 0 else 0.0
            critical_distance = rel_pos[critical_idx] if critical_idx is not None else np.max(rel_pos)
            critical_distance = max(critical_distance, 1e-3)

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
