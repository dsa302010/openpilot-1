#!/usr/bin/env python3
"""
Speed Camera Detection & Adjustment (SCDA) - Physics Based Edition
測速照相偵測 - 純物理計算版

核心邏輯變更:
1. [移除] 時間緩衝檢查 (Time Buffer): 這是多餘的邏輯。
2. [保留] 物理減速檢查 (Required Decel): 這是唯一的真理。
   只要需求減速度在物理極限內 (-6.0 m/s²)，無論時間剩多少，系統都會強制介入。
3. [優化] 針對「高於限速」的情況，給予最直接的減速反應。
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

# --- 安全閾值 ---
# 緊急煞車閾值: 低於此值 (例如 -4.5) 代表需要急煞
# 我們記錄它，但不阻止它，因為避免罰單優先
EMERGENCY_DECEL_THRESHOLD = -4.5  

# --- 調試開關 ---
DEBUG_LOGGING = True 

class SpeedCameraControl:
  def __init__(self):
    self.cameras = np.empty((0, 3))
    
    # --- 參數設定 ---
    
    # 1. 角度過濾參數 [速度(kph), 角度範圍]
    # 高速時視角窄(專注前方)，低速時視角寬(避免漏抓)
    self.angle_bp = [0., 60., 80., 110.]
    self.angle_vals = [20., 20., 22., 25.]
    
    # 2. 減速啟動半徑 [速度(kph), 距離(m)]
    # 速度越快，越早在遠處開始反應
    self.limit_radius_bp = [0., 40., 60., 80., 105., 110.]
    self.limit_radius_vals = [100., 150., 150., 200., 250., 300.]
    
    # 3. 搜尋半徑 (只載入範圍內的相機)
    self.search_bp = [0., 79., 105.]
    self.search_vals = [500., 500., 600.]
    
    # 4. 通過後保持距離 (防止過了相機馬上急加速)
    self.center_hold_dist = 50.0 
    
    # --- 狀態變數 ---
    self.last_load_time = 0.0
    self.last_log_time = 0.0 
    self.ignore_index = -1       
    self.active_index = -1       
    
    # 統計用
    self.intervention_count = 0
    self.rejection_count = 0

    self._load_cameras()

  def _load_cameras(self):
    """讀取 NPA_TD1.csv"""
    paths = [
        os.path.join(os.path.dirname(__file__), 'NPA_TD1.csv'),
        '/data/openpilot/dragonpilot/selfdrive/controls/lib/NPA_TD1.csv'
    ]
    csv_path = next((p for p in paths if os.path.exists(p)), None)

    if not csv_path:
      if DEBUG_LOGGING:
          cloudlog.warning("SCDA 錯誤: 找不到 CSV 檔案!")
      return

    try:
      current_mtime = os.path.getmtime(csv_path)
      if current_mtime == self.last_load_time and self.cameras.shape[0] > 0:
        return
      
      self.last_load_time = current_mtime
      cams = []
      with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
          try:
            lat = r.get('Latitude') or r.get('latitude')
            lon = r.get('Longitude') or r.get('longitude')
            spd = r.get('Limit') or r.get('limit')
            if lat and lon and spd:
                cams.append([float(lat), float(lon), float(spd)])
          except ValueError:
            continue
            
      self.cameras = np.asarray(cams)
      self.ignore_index = -1 
      self.active_index = -1
      if len(self.cameras) > 0:
          cloudlog.warning(f"SCDA: 載入成功, {len(self.cameras)} 支相機。")
    except Exception as e:
      cloudlog.error(f"SCDA: 讀取失敗 - {e}")
      self.cameras = np.empty((0, 3))

  def cancel_current_camera(self):
    """使用者踩油門時呼叫，忽略當前相機"""
    if self.active_index != -1:
        self.ignore_index = self.active_index

  @staticmethod
  def _haversine(lat1, lon1, lat2, lon2):
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 6371000 * 2 * np.arctan2(np.sqrt(np.maximum(a, 0)), np.sqrt(np.maximum(1-a, 0)))

  @staticmethod
  def _bearing(lat1, lon1, lat2, lon2):
    dlon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360

  def _is_safe_to_intervene(self, v_ego_ms, cam_limit_kph, distance):
    """
    [核心邏輯] 純物理檢查
    """
    cam_limit_ms = cam_limit_kph * KPH_TO_MS
    
    # 1. 沒超速就不管
    if v_ego_ms <= cam_limit_ms:
        return True, 0.0, "already_below_limit"
    
    dist_safe = max(distance, 1.0)
    
    # 2. 計算物理需求減速度 (Required Deceleration)
    # 這是判斷「能不能煞得住」的唯一標準
    required_decel = (cam_limit_ms ** 2 - v_ego_ms ** 2) / (2.0 * dist_safe)
    
    # 3. 物理極限過濾
    # 如果算出需要 -6.0 m/s² (0.6G) 以上的減速，代表距離太近根本煞不住
    # 或是 GPS 飄移導致距離錯誤，這時候介入只會造成危險
    if required_decel < -6.0:
        self.rejection_count += 1
        return False, required_decel, "impossible_decel"
    
    # 4. 緊急煞車標記
    # 如果需要急煞 (-4.5 ~ -6.0)，標記為 Emergency 但允許執行
    # 既然超速了，就算急煞也要試著降速
    if required_decel < EMERGENCY_DECEL_THRESHOLD:
        return True, required_decel, "emergency_braking"
    
    return True, required_decel, "safe"

  def get_target_speed(self, v_ego_ms, v_cruise_ms, lat, lon, bearing_deg):
    if not math.isfinite(v_ego_ms) or not math.isfinite(lat) or not math.isfinite(lon):
      return {'target_speed': v_cruise_ms, 'distance': None, 'limit': None, 'is_active': False}

    current_time = time.monotonic()
    should_log = (current_time - self.last_log_time) > 1.0

    v_ego_kph = v_ego_ms * MS_TO_KPH
    v_cruise_kph = v_cruise_ms * MS_TO_KPH
    
    candidates = [v_cruise_kph]

    if self.cameras.size == 0:
      return {'target_speed': v_cruise_ms, 'distance': None, 'limit': None, 'is_active': False}

    # 1. 根據車速決定搜尋參數
    search_radius = np.interp(v_ego_kph, self.search_bp, self.search_vals)
    limit_radius = np.interp(v_ego_kph, self.limit_radius_bp, self.limit_radius_vals)
    base_angle = np.interp(v_ego_kph, self.angle_bp, self.angle_vals)

    # 2. 粗篩選
    deg_diff = (search_radius / 111000) * 1.5
    mask = (np.abs(self.cameras[:, 0] - lat) < deg_diff) & (np.abs(self.cameras[:, 1] - lon) < deg_diff)
    nearby_cams = self.cameras[mask]
    nearby_indices = np.arange(self.cameras.shape[0])[mask]

    self.active_index = -1
    
    # 遠離忽略點後自動重設
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

    # 3. 逐一檢查相機
    for i, cam in enumerate(nearby_cams):
      original_idx = nearby_indices[i]
      if original_idx == self.ignore_index: continue
          
      cam_lat, cam_lon, cam_limit = cam
      dist = self._haversine(lat, lon, cam_lat, cam_lon)
      
      if dist > search_radius: continue

      # 近距離放寬角度，避免最後時刻丟失目標
      if dist <= 150.0: allowed_angle = 30.0 
      else: allowed_angle = base_angle
      
      if math.isnan(bearing_deg): continue
      
      cam_bearing = self._bearing(lat, lon, cam_lat, cam_lon)
      diff_angle = abs(bearing_deg - cam_bearing)
      if diff_angle > 180: diff_angle = 360 - diff_angle
      
      is_front = diff_angle <= allowed_angle
      is_behind = (180 - diff_angle) <= allowed_angle
      
      if not (is_front or (is_behind and dist < limit_radius)): continue

      # [關鍵] 呼叫純物理檢查
      is_safe, required_decel, reason = self._is_safe_to_intervene(v_ego_ms, cam_limit, dist)
      
      if not is_safe:
        if dist < min_dist_found:
            closest_log_info = {
                "status": f"已拒絕({reason})", "dist": dist, "limit": cam_limit, "req_a": required_decel
            }
        continue

      # 計算目標速度
      if dist > limit_radius:
        target = v_cruise_kph
      else:
        # 線性插值: 越靠近目標，速度限制越嚴格
        target = np.interp(dist, [self.center_hold_dist, limit_radius], [cam_limit, v_cruise_kph])
      
      if math.isfinite(target):
        candidates.append(target)
        if dist < min_dist_found:
            min_dist_found = dist
            active_camera_limit = cam_limit
            self.active_index = original_idx 
            self.intervention_count += 1
            closest_log_info = {
                "dist": dist, "limit": cam_limit, "status": "介入中", "target": target, "req_a": required_decel
            }

    final_target_kph = min(candidates)
    final_target_kph = min(final_target_kph, v_cruise_kph)

    # Log 輸出 (僅在有需要介入或異常時)
    if should_log and closest_log_info and closest_log_info["dist"] < search_radius:
        self.last_log_time = current_time
        status = closest_log_info["status"]
        limit = closest_log_info["limit"]
        dist = closest_log_info["dist"]
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
    return {
        "interventions": self.intervention_count, 
        "rejections": self.rejection_count, 
        "cameras_loaded": len(self.cameras)
    }
