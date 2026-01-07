"""
Dynamic Turn Speed Controller (DTSC) - Smart Log Edition
功能更新:
1. 智慧 Log: 只有在「真正介入減速」(G < -0.1) 時才寫入檔案，平時完全靜默。
2. 頻率控制: 介入期間每 0.5 秒寫一次，避免拖慢系統。
3. [新增] Log 開關控制: 支援 Master Switch 與 ACC On 檢查。
"""
import time
import numpy as np
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.common.swaglog import cloudlog

# =============================
# 基本參數
# =============================
MODEL_T_IDXS = ModelConstants.T_IDXS
DT_MPC = 0.05

# --- [新增] Log 總開關 ---
MASTER_LOG_ENABLED = True  # True: 開啟寫入檔案, False: 完全停用

# --- 彎道安全參數 ---
BASE_LAT_ACC = 2.8
SAFETY_SPEED_FACTOR = 0.95
LAT_LIMIT_BP = [5.0, 10.0, 15.0, 20.0, 25.0]
LAT_LIMIT_V  = [2.0, 2.1, 2.2, 2.3, 2.3]

# --- LPF ---
LPF_ALPHA = 0.3
LPF_RESET_LAT_ACC_THRESHOLD = 0.15 

# --- 減速模型 ---
CITY_DECEL_BP = np.array([0.8, 1.0, 2.0])
CITY_DECEL_V  = np.array([-0.8, -2.0, -3.5])
HIGHWAY_DECEL_BP = np.array([1.3, 1.8, 2.5])
HIGHWAY_DECEL_V  = np.array([-0.3, -0.8, -1.8])
TRANSITION_BP = [15.0, 20.0]
TRANSITION_VALS = [0.0, 1.0]

# --- 限制閾值 ---
MAX_COMFORT_DECEL = -2.0
EMERGENCY_DECEL   = -4.5
MIN_CURVE_DISTANCE = 5.0
MAX_EXIT_ACCEL = 0.5

# --- 轉向輔助 ---
STEER_ASSIST_ANGLE_THRESHOLD = 20.0
STEER_SPEED_SCALE = 1.0
MIN_STEER_SPEED_FLOOR = 5.0

# --- 誤判防護 ---
PERSISTENCE_MIN_FRAC = 0.5
CURVATURE_MIN_FOR_PERSIST = 0.01
SHORT_DIST_IGNORE = 3.5
STEER_ANGLE_FOR_SHORT = 8.0
SCCV_ABORT_PRED_LAT_ACC_TH = 0.7
FUTURE_CURVE_THRESHOLD = 0.015
HYSTERESIS_TIME = 0.5

# --- SCDA 協調參數 ---
SCDA_PRIORITY_MARGIN = 1.15  

# --- [Log] 寫入檔案函式 ---
def write_file_log(msg):
    """
    將 Log 寫入 /data/media/0/dtsc_log.txt
    只有在真正運作時才會呼叫此函式
    """
    try:
        with open("/data/media/0/dtsc_log.txt", "a") as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(f"[{timestamp}] {msg}\n")
    except Exception:
        pass

def clamp(x, low, high):
    return max(low, min(high, x))

def interp_clamped(x, bp, fp):
    if x <= bp[0]: return fp[0]
    if x >= bp[-1]: return fp[-1]
    return float(np.interp(x, bp, fp))

class DTSC:
    def __init__(self, aggressiveness=1.0, cp=None):
        self.aggressiveness = clamp(aggressiveness, 0.5, 1.8)
        self.active = False
        self.hysteresis_timer = 0.0
        self.filtered_lat_limits = None
        self.lpf_reset_counter = 0 
        
        # Log 頻率計時器
        self.last_log_time = 0.0

        if cp is not None:
            self.steer_ratio = cp.steerRatio
            self.wheelbase = cp.wheelbase
        else:
            self.steer_ratio = 14.3
            self.wheelbase = 2.7

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
            self.filtered_lat_limits = (LPF_ALPHA * raw_lat_limits) + ((1.0 - LPF_ALPHA) * self.filtered_lat_limits)
        
        current_lat_limits = np.maximum(self.filtered_lat_limits, 1.0)
        v_clip = np.clip(v_pred, 1.0, 100.0)
        curvatures = np.abs(yaw_rates / v_clip)
        safe_speeds_model = np.sqrt(current_lat_limits / (curvatures + 1e-6)) * SAFETY_SPEED_FACTOR
        final_safe_speeds = safe_speeds_model.copy()
        
        abs_steer = abs(steer_angle_deg)
        if abs_steer > STEER_ASSIST_ANGLE_THRESHOLD:
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
        if predicted_lat_acc_max <= CITY_DECEL_BP[0]: decel_city = 0.0
        else: decel_city = interp_clamped(predicted_lat_acc_max, CITY_DECEL_BP, CITY_DECEL_V)
        if predicted_lat_acc_max <= HIGHWAY_DECEL_BP[0]: decel_hwy = 0.0
        else: decel_hwy = interp_clamped(predicted_lat_acc_max, HIGHWAY_DECEL_BP, HIGHWAY_DECEL_V)
        blend_factor = np.interp(v_ego, TRANSITION_BP, TRANSITION_VALS)
        return clamp((decel_city * (1.0 - blend_factor)) + (decel_hwy * blend_factor), EMERGENCY_DECEL, 0.0)

    def _compute_dtsc_decel(self, v_ego, v_pred, rel_pos, safe_speeds):
        speed_excess = v_pred - safe_speeds
        if np.all(speed_excess <= 0.0): return 0.0, None, None
        critical_idx = int(np.argmax(speed_excess))
        critical_rel_dist = max(rel_pos[critical_idx], 1e-3)
        decel_by_distance = (safe_speeds[critical_idx] ** 2 - v_ego ** 2) / (2.0 * critical_rel_dist)
        decel_by_distance = min(decel_by_distance, 0.0)
        mode = 'EMERGENCY' if critical_rel_dist <= MIN_CURVE_DISTANCE else 'COMFORT'
        if mode == 'COMFORT': decel_by_distance = max(decel_by_distance, MAX_COMFORT_DECEL)
        return decel_by_distance, critical_idx, mode

    def get_mpc_constraints(self, model_msg, v_ego, base_a_min, base_a_max,
                            steer_angle_deg=0.0, steer_ratio=None, wheelbase=None, 
                            scda_target_speed=None, scda_distance=None, acc_enabled=False):
        
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

        safe_speeds, curvatures = self._compute_safe_speeds(v_pred, yaw_rates, steer_angle_deg, steer_ratio or self.steer_ratio, wheelbase or self.wheelbase)
        sp_decel = self._compute_sp_decel(predicted_lat_acc_max, v_ego)
        dt_decel, critical_idx, dt_mode = self._compute_dtsc_decel(v_ego, v_pred, rel_pos, safe_speeds)

        speed_excess = v_pred - safe_speeds
        mask_curve = curvatures > CURVATURE_MIN_FOR_PERSIST
        mask_speed = speed_excess > 0.01
        mask = np.logical_and(mask_speed, mask_curve)
        persistence_ok = (float(np.sum(mask)) / len(mask) if len(mask) > 0 else 0) >= PERSISTENCE_MIN_FRAC
        
        if predicted_lat_acc_max < SCCV_ABORT_PRED_LAT_ACC_TH or (not persistence_ok and (rel_pos[critical_idx] if critical_idx else 999) < SHORT_DIST_IGNORE):
            dt_decel = sp_decel = 0.0
            dt_mode = None

        final_required_decel = dt_decel if dt_mode == "EMERGENCY" else min(min(sp_decel, 0.0), min(dt_decel, 0.0))
        final_required_decel = clamp(final_required_decel, EMERGENCY_DECEL, 0.0)

        # [SCDA 整合邏輯]
        scda_required_decel = 0.0
        scda_active = False
        if scda_target_speed is not None and scda_target_speed < v_ego:
            calc_dist = scda_distance if scda_distance is not None else np.max(rel_pos)
            calc_dist = max(calc_dist, 1.0)
            
            scda_required_decel = (scda_target_speed ** 2 - v_ego ** 2) / (2.0 * calc_dist)
            scda_required_decel = clamp(scda_required_decel, EMERGENCY_DECEL, 0.0)
            scda_active = True
            
            if scda_required_decel < final_required_decel:
                scda_required_decel *= SCDA_PRIORITY_MARGIN

        # 啟動條件: DTSC 想介入 OR SCDA 想介入
        if final_required_decel < -0.1 or (scda_active and scda_required_decel < -0.1):
            self.hysteresis_timer = HYSTERESIS_TIME
            self.active = True
        elif self.hysteresis_timer > 0:
            self.hysteresis_timer -= DT_MPC
            self.active = True
        else:
            self.active = False
            self.hysteresis_timer = 0

        if self.active:
            pass_decel = final_required_decel if final_required_decel < 0 else 0.0
            critical_distance = rel_pos[critical_idx] if critical_idx is not None else np.max(rel_pos)
            
            # --- [智慧 Log 邏輯] ---
            # 頻率限制: 每 0.5 秒
            current_time = time.monotonic()
            if current_time - self.last_log_time > 0.5:
                # 判定是誰導致的減速
                reason = "DTSC"
                actual_val = pass_decel
                
                if scda_active and scda_required_decel < pass_decel:
                    reason = "SCDA"
                    actual_val = scda_required_decel
                
                # [關鍵] 條件: 實際有減速 + ACC 開啟 + 總開關開啟
                if actual_val < -0.1 and MASTER_LOG_ENABLED and acc_enabled:
                    log_msg = f"DTSC介入: 來源={reason} | 車速={v_ego*3.6:.0f} | 減速G={actual_val:.2f} | 彎道G={predicted_lat_acc_max:.2f}"
                    write_file_log(log_msg)
                    self.last_log_time = current_time
            # ---------------------

            has_future_curve = any(rel_pos[i] > critical_distance and curvatures[i] > FUTURE_CURVE_THRESHOLD for i in range(horizon_len))

            for i in range(horizon_len):
                if rel_pos[i] <= critical_distance + 1e-6:
                    final_limit = pass_decel
                    if scda_active and scda_required_decel < final_limit:
                        final_limit = scda_required_decel
                    
                    if final_limit < 0:
                        a_max[i] = min(a_max[i], final_limit)
                else:
                    if has_future_curve:
                        a_max[i] = min(a_max[i], MAX_EXIT_ACCEL)

        return a_min, a_max
