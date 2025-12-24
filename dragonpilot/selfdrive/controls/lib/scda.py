#!/usr/bin/env python3
"""
Speed Camera Detection & Adjustment (SCDA) - Aggressive Edition
測速照相偵測與調整 - 激進介入版

修改紀錄:
3. 放寬緊急煞車閾值，允許更強的減速力道。
4. 強制開啟調試日誌 (DEBUG_LOGGING)，方便追蹤介入情況。
"""
import os
import csv
import math
import time
import numpy as np
from openpilot.common.swaglog import cloudlog

# =========================================
# 常數定義
# =========================================
MS_TO_KPH = 3.6
KPH_TO_MS = 1. / 3.6

# --- 安全參數 (激進設定) ---
# 最大舒適減速度 (參考用，主要由縱向控制處理平滑度)
MAX_COMFORTABLE_DECEL = -3.0  
# 緊急煞車閾值: 低於此值會發出警告，但仍允許介入 (OpenPilot 硬體極限約 -4.5 ~ -5.0)
EMERGENCY_DECEL_THRESHOLD = -4.5  

# --- 調試開關 ---
# 設為 True 以便在 logcat 查看詳細的介入與拒絕原因
DEBUG_LOGGING = True 

class SpeedCameraControl:
  def __init__(self):
    self.cameras = np.empty((0, 3))
    
    # --- 參數設定 ---
    
    # 1. 角度過濾參數 (Degrees)
    # [速度(kph), 角度範圍]
    # 速度越快，視野越窄，專注於前方；低速時視野較寬
    self.angle_bp = [0., 60., 80., 110.]
    self.angle_vals = [20., 20., 22., 25.]
    
    # 2. 減速啟動半徑 (Meters)
    # [速度(kph), 啟動距離]
    # 速度越快，需要越早開始減速 (例如 110kph 時在 300m 外開始減速)
    self.limit_radius_bp = [0., 40., 60., 80., 105., 110.]
    self.limit_radius_vals = [100., 150., 150., 200., 250., 300.]
    
    # 3. 搜尋半徑 (Meters)
    # 僅載入此半徑內的相機進行運算，節省效能
    self.search_bp = [0., 79., 105.]
    self.search_vals = [500., 500., 600.]
    
    # 4. 通過保持距離 (Meters)
    # 通過相機後多少距離內仍保持限速，避免立刻急加速
    self.center_hold_dist = 50.0 
    
    # --- 內部狀態變數 ---
    self.last_load_time = 0.0
    self.last_log_time = 0.0 
    
    # 忽略機制: 暫時忽略的相機 ID (例如踩油門後)
    self.ignore_index = -1       
    # 當前生效的相機 ID
    self.active_index = -1       
    
    # 統計數據
    self.intervention_count = 0
    self.rejection_count = 0

    self._load_cameras()

  def _load_cameras(self):
    """
    讀取測速照相 CSV 檔案
    格式需求: Latitude, Longitude, Limit (或小寫)
    """
    # 支援兩個路徑，優先使用使用者上傳路徑
    paths = [
        os.path.join(os.path.dirname(__file__), 'NPA_TD1.csv'),
        '/data/openpilot/dragonpilot/selfdrive/controls/lib/NPA_TD1.csv'
    ]
    csv_path = next((p for p in paths if os.path.exists(p)), None)

    if not csv_path:
      if DEBUG_LOGGING:
          cloudlog.warning("SCDA 錯誤: 找不到 NPA_TD1.csv 檔案!")
      return

    try:
      current_mtime = os.path.getmtime(csv_path)
      # 如果檔案沒更新且已經載入過，就不重新讀取
      if current_mtime == self.last_load_time and self.cameras.shape[0] > 0:
        return
      
      self.last_load_time = current_mtime
      cams = []
      with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
          try:
            # 相容大小寫欄位名稱
            lat = r.get('Latitude') or r.get('latitude')
            lon = r.get('Longitude') or r.get('longitude')
            spd = r.get('Limit') or r.get('limit')
            
            if lat and lon and spd:
                cams.append([float(lat), float(lon), float(spd)])
          except ValueError:
            continue
            
      self.cameras = np.asarray(cams)
      
      # 重設狀態
      self.ignore_index = -1 
      self.active_index = -1

      if len(self.cameras) > 0:
          cloudlog.warning(f"SCDA: 載入成功,共 {len(self.cameras)} 支相機。")
          
    except Exception as e:
      cloudlog.error(f"SCDA: 讀取失敗 - {e}")
      self.cameras = np.empty((0, 3))

  def cancel_current_camera(self):
    """
    當使用者踩下油門時呼叫此方法。
    將當前作用中的相機加入忽略清單，直到遠離該點。
    """
    if self.active_index != -1:
        if self.ignore_index != self.active_index:
            cloudlog.warning(f"SCDA: 使用者介入(踩油門), 已忽略相機 ID {self.active_index}")
        self.ignore_index = self.active_index

  @staticmethod
  def _haversine(lat1, lon1, lat2, lon2):
    """計算兩點經緯度之間的直線距離 (公尺)"""
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 6371000 * 2 * np.arctan2(np.sqrt(np.maximum(a, 0)), np.sqrt(np.maximum(1-a, 0)))

  @staticmethod
  def _bearing(lat1, lon1, lat2, lon2):
    """計算從點1到點2的方位角 (0-360度)"""
    dlon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360

  def _is_safe_to_intervene(self, v_ego_ms, cam_limit_kph, distance):
    """
    [核心邏輯] 判斷是否應該介入減速
    Returns: (是否安全, 需求減速度, 原因字串)
    """
    cam_limit_ms = cam_limit_kph * KPH_TO_MS
    
    # 1. 速度檢查: 如果已經低於限速，不需要介入
    if v_ego_ms <= cam_limit_ms:
        return True, 0.0, "already_below_limit"
    
    # 防止距離為 0 導致除法錯誤
    dist_safe = max(distance, 1.0)
    
    # 2. 物理計算: 需要多少減速度才能在目標點前降到限速?
    # 公式: a = (v_f^2 - v_i^2) / (2 * d)
    required_decel = (cam_limit_ms ** 2 - v_ego_ms ** 2) / (2.0 * dist_safe)
    
    # 3. 極限過濾: 過濾 GPS 飄移或數據錯誤導致的「不可能的減速」
    # -5.5 m/s² 是普通輪胎的物理極限邊緣，超過這個值代表距離數據可能有誤
    if required_decel < -5.5:
        self.rejection_count += 1
        return False, required_decel, "impossible_decel"
    
    # 4. 緊急煞車檢查: 
    # 如果需要的減速度很大 (例如 -4.0)，記錄警告，但 "返回 True" 允許介入。
    # 這樣 LongController 仍會盡力減速。
    if required_decel < EMERGENCY_DECEL_THRESHOLD:
        if DEBUG_LOGGING:
            cloudlog.warning(f"SCDA: 觸發緊急煞車需求 a={required_decel:.2f}")
        return True, required_decel, "emergency_braking"
    
    # 5. 通過所有檢查: 允許介入
    return True, required_decel, "safe"

  def get_target_speed(self, v_ego_ms, v_cruise_ms, lat, lon, bearing_deg):
    """
    SCDA 主迴圈
    輸入: 車速, 巡航設定, GPS座標, 車輛方位
    輸出: 目標速度與狀態字典
    """
    # 基本數據有效性檢查
    if not math.isfinite(v_ego_ms) or not math.isfinite(lat) or not math.isfinite(lon):
      return {'target_speed': v_cruise_ms, 'distance': None, 'limit': None, 'is_active': False}

    current_time = time.monotonic()
    should_log = (current_time - self.last_log_time) > 1.0

    v_ego_kph = v_ego_ms * MS_TO_KPH
    v_cruise_kph = v_cruise_ms * MS_TO_KPH
    
    candidates = [v_cruise_kph]

    if self.cameras.size == 0:
      return {'target_speed': v_cruise_ms, 'distance': None, 'limit': None, 'is_active': False}

    # 1. 根據車速動態計算搜尋範圍
    search_radius = np.interp(v_ego_kph, self.search_bp, self.search_vals)
    limit_radius = np.interp(v_ego_kph, self.limit_radius_bp, self.limit_radius_vals)
    base_angle = np.interp(v_ego_kph, self.angle_bp, self.angle_vals)

    # 2. 粗略篩選: 找出方框內的相機 (比 Haversine 快)
    deg_diff = (search_radius / 111000) * 1.5
    mask = (np.abs(self.cameras[:, 0] - lat) < deg_diff) & (np.abs(self.cameras[:, 1] - lon) < deg_diff)
    nearby_cams = self.cameras[mask]
    nearby_indices = np.arange(self.cameras.shape[0])[mask]

    # 重設每幀狀態
    self.active_index = -1
    
    # 檢查是否應該解除「忽略」狀態 (若已遠離被忽略的相機)
    if self.ignore_index != -1 and self.ignore_index not in nearby_indices:
         ignored_cam = self.cameras[self.ignore_index]
         dist_to_ignored = self._haversine(lat, lon, ignored_cam[0], ignored_cam[1])
         if dist_to_ignored > search_radius + 100: 
             self.ignore_index = -1

    if nearby_cams.size == 0:
      return {'target_speed': v_cruise_ms, 'distance': None, 'limit': None, 'is_active': False}

    closest_log_info = None 
    min_dist_found = 9999.0
    active_camera_limit = None

    # 3. 詳細比對迴圈
    for i, cam in enumerate(nearby_cams):
      original_idx = nearby_indices[i]
      
      # 跳過被忽略的相機 (踩油門後)
      if original_idx == self.ignore_index:
          continue
          
      cam_lat, cam_lon, cam_limit = cam
      dist = self._haversine(lat, lon, cam_lat, cam_lon)
      
      # 距離過濾
      if dist > search_radius:
        continue

      # 角度過濾: 近距離 (150m內) 放寬角度限制，避免轉彎時丟失目標
      if dist <= 150.0:
          allowed_angle = 30.0 
      else:
          allowed_angle = base_angle
      
      if math.isnan(bearing_deg): 
          continue
      
      # 計算方位角差
      cam_bearing = self._bearing(lat, lon, cam_lat, cam_lon)
      diff_angle = abs(bearing_deg - cam_bearing)
      if diff_angle > 180: diff_angle = 360 - diff_angle
      
      # 判斷相機是在「前方」還是「後方但很近」(剛通過)
      is_front = diff_angle <= allowed_angle
      is_behind = (180 - diff_angle) <= allowed_angle
      
      # 只處理前方的相機，或剛通過還在範圍內的相機
      if not (is_front or (is_behind and dist < limit_radius)):
        continue

      # 4. 安全介入檢查 (已移除時間限制)
      is_safe, required_decel, reason = self._is_safe_to_intervene(v_ego_ms, cam_limit, dist)
      
      if not is_safe:
        # 如果因為物理極限被拒絕，記錄最近的一個供 Debug
        if dist < min_dist_found:
            closest_log_info = {
                "status": f"已拒絕({reason})", 
                "dist": dist, 
                "limit": cam_limit, 
                "req_a": required_decel
            }
        continue

      # 5. 計算目標速度 (線性插值)
      # 超過減速半徑: 維持巡航速度
      # 進入減速半徑: 線性從巡航速度降至限速
      if dist > limit_radius:
        target = v_cruise_kph
      else:
        target = np.interp(dist, [self.center_hold_dist, limit_radius], [cam_limit, v_cruise_kph])
      
      if math.isfinite(target):
        candidates.append(target)
        # 追蹤最近且生效的相機
        if dist < min_dist_found:
            min_dist_found = dist
            active_camera_limit = cam_limit
            self.active_index = original_idx 
            self.intervention_count += 1
            closest_log_info = {
                "dist": dist, 
                "limit": cam_limit, 
                "status": "介入中", 
                "target": target, 
                "req_a": required_decel
            }

    # 6. 決策: 取所有候選速度的最小值
    final_target_kph = min(candidates)
    final_target_kph = min(final_target_kph, v_cruise_kph)

    # 7. 日誌輸出
    if should_log and closest_log_info and closest_log_info["dist"] < search_radius:
        self.last_log_time = current_time
        status = closest_log_info["status"]
        limit = closest_log_info["limit"]
        dist = closest_log_info["dist"]
        # 只有在真的需要減速，或被拒絕時才印出 Log
        if final_target_kph < v_cruise_kph - 1.0 or "已拒絕" in status:
            cloudlog.warning(f"SCDA {status}: 限速{limit:.0f}|距離{dist:.0f}m|目標{final_target_kph:.1f}|Req A:{closest_log_info.get('req_a', 0):.2f}")

    is_active = (min_dist_found < 9999.0 and final_target_kph < v_cruise_kph)
    
    return {
        'target_speed': final_target_kph * KPH_TO_MS,
        'distance': min_dist_found if is_active else None,
        'limit': active_camera_limit if is_active else None,
        'is_active': is_active
    }
  
  def get_statistics(self):
    """回傳統計數據供診斷用"""
    return {
        "interventions": self.intervention_count, 
        "rejections": self.rejection_count, 
        "cameras_loaded": len(self.cameras)
    }
