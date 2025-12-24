"""
Dynamic Turn Speed Controller (DTSC) - Optimized Edition (Sensitive Fix)
"""

import numpy as np
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.common.swaglog import cloudlog

# =============================
# 基本參數
# =============================
MODEL_T_IDXS = ModelConstants.T_IDXS
DT_MPC = 0.05

# --- 彎道 Lateral G 安全限制 ---
BASE_LAT_ACC = 2.8 # 
SAFETY_SPEED_FACTOR = 0.95

# --- 5點式速度依賴限製表 (m/s²) ---
LAT_LIMIT_BP = [5.0, 10.0, 15.0, 20.0, 25.0]
LAT_LIMIT_V  = [2.0, 2.1, 2.4, 2.7, 2.8]

# --- Low-Pass Filter ---
LPF_ALPHA = 0.3
LPF_RESET_LAT_ACC_THRESHOLD = 0.15 

# --- 雙模組 Pre-deceleration 設定 ---
CITY_DECEL_BP = np.array([0.8, 1.0, 2.0])
CITY_DECEL_V  = np.array([-0.8, -2.0, -3.5])

HIGHWAY_DECEL_BP = np.array([1.3, 1.8, 2.5])
HIGHWAY_DECEL_V  = np.array([-0.3, -0.8, -1.8])

TRANSITION_BP = [15.0, 20.0]
TRANSITION_VALS = [0.0, 1.0]

# --- 減速度限制 ---
MAX_COMFORT_DECEL = -2.5 # [修正] 放寬舒適減速
EMERGENCY_DECEL   = -4.5

MIN_CURVE_DISTANCE = 5.0
MAX_EXIT_ACCEL = 0.5

STEER_ASSIST_ANGLE_THRESHOLD = 20.0
STEER_SPEED_SCALE = 1.0
STEER_AGGRESSIVENESS = 1.0
MIN_STEER_SPEED_FLOOR = 5.0

PERSISTENCE_MIN_FRAC = 0.5
CURVATURE_MIN_FOR_PERSIST = 0.01
SHORT_DIST_IGNORE = 3.5
STEER_ANGLE_FOR_SHORT = 8.0

# [關鍵修正] 降低門檻，避免輕微彎道被過濾
SCCV_ABORT_PRED_LAT_ACC_TH = 0.5 

FUTURE_CURVE_THRESHOLD = 0.015
HYSTERESIS_TIME = 0.5

SCDA_PRIORITY_MARGIN = 1.15  
SCDA_MIN_RELAXATION = -3.5   

DEBUG_LOGGING = True

# =============================
# 工具函式
# =============================
def clamp(x, low, high):
    return max(low, min(high, x))

def interp_clamped(x, bp, fp):
    if x <= bp[0]: return fp[0]
    if x >= bp[-1]: return fp[-1]
    return float(np.interp(x, bp, fp))

# =============================
# DTSC 主類別
# =============================
class DTSC:
    def __init__(self, aggressiveness=1.0, cp=None):
        self.aggressiveness = clamp(aggressiveness, 0.5, 1.8)
        self.active = False
        self.hysteresis_timer = 0.0
        self.filtered_lat_limits = None
        self.lpf_reset_counter = 0 
        
        if cp is not None:
            self.steer_ratio = cp.steerRatio
            self.wheelbase = cp.wheelbase
        else:
            self.steer_ratio = 14.3
            self.wheelbase = 2.7
        
        if DEBUG_LOGGING:
            cloudlog.info(f"DTSC v10.0: Init. Blend Range: {TRANSITION_BP[0]}-{TRANSITION_BP[1]} m/s")

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
        raw_lat_limits = np.interp(v_pred, LAT_LIMIT_BP, LAT_LIMIT_V) * self.aggressiveness
        
        if self.filtered_lat_limits is None:
            self.filtered_lat_limits = raw_lat_limits
        else:
            self.filtered_lat_limits = (LPF_ALPHA * raw_lat_limits) + \
                                       ((1.0 - LPF_ALPHA) * self.filtered_lat_limits)
        
        current_lat_limits = np.maximum(self.filtered_lat_limits, 1.0)
        
        v_clip = np.clip(v_pred, 1.0, 100.0)
        curvatures = np.abs(yaw_rates / v_clip)
        
        safe_speeds_model = np.sqrt(current_lat_limits / (curvatures + 1e-6)) * SAFETY_SPEED_FACTOR

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

    def _compute_sp_decel(self, predicted_lat_acc_max, v_ego):
        if predicted_lat_acc_max <= CITY_DECEL_BP[0]:
            decel_city = 0.0
        else:
            decel_city = interp_clamped(predicted_lat_acc_max, CITY_DECEL_BP, CITY_DECEL_V)
            
        if predicted_lat_acc_max <= HIGHWAY_DECEL_BP[0]:
            decel_hwy = 0.0
        else:
            decel_hwy = interp_clamped(predicted_lat_acc_max, HIGHWAY_DECEL_BP, HIGHWAY_DECEL_V)
            
        blend_factor = np.interp(v_ego, TRANSITION_BP, TRANSITION_VALS)
        final_decel = (decel_city * (1.0 - blend_factor)) + (decel_hwy * blend_factor)
        return clamp(final_decel, EMERGENCY_DECEL, 0.0)

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
                            steer_angle_deg=0.0, steer_ratio=None, wheelbase=None, scda_target_speed=None):
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
            self.lpf_reset_counter += 1
            if self.lpf_reset_counter > 10:
                self.filtered_lat_limits = None
                self.lpf_reset_counter = 0
        else:
            self.lpf_reset_counter = 0

        safe_speeds, curvatures = self._compute_safe_speeds(
            v_pred, yaw_rates, steer_angle_deg, current_steer_ratio, current_wheelbase)

        sp_decel = self._compute_sp_decel(predicted_lat_acc_max, v_ego)
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

        # SCDA 優先級處理
        scda_required_decel = 0.0
        if scda_target_speed is not None and scda_target_speed < v_ego:
            max_distance = np.max(rel_pos)
            if max_distance > 1.0:
                scda_required_decel = (scda_target_speed ** 2 - v_ego ** 2) / (2.0 * max_distance)
                scda_required_decel = clamp(scda_required_decel, EMERGENCY_DECEL, 0.0)
                
                if scda_required_decel < final_required_decel:
                    scda_required_decel *= SCDA_PRIORITY_MARGIN

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
                # [修正] 如果當前需要減速，直接應用到所有點，確保 MPC 響應
                if pass_decel < 0:
                    a_max[i] = min(a_max[i], pass_decel)

                if rel_pos[i] <= critical_distance + 1e-6:
                    if pass_decel < 0:
                        if scda_required_decel < pass_decel:
                            effective_decel = max(scda_required_decel, SCDA_MIN_RELAXATION)
                            a_max[i] = min(a_max[i], effective_decel)
                else:
                    if has_future_curve:
                        a_max[i] = min(a_max[i], MAX_EXIT_ACCEL)

        return a_min, a_max
