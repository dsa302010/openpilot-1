#!/usr/bin/env python3
"""
SCDA (12) - Log Enhanced Edition
基於: 您上傳的 scda (12).py
新增: write_file_log 功能，將紀錄直接存檔到 /data/media/0/scda_log.txt
功能: 
1. 安全門檻(+15) & 物理防急煞(-4.5)
2. 踩油門取消 & 通過後緩加速
3. 動態保持距離 (25/50/75m)
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

# --- 安全參數 ---
SAFETY_OFFSET = 15.0       # 安全門檻: +15km/h
EMERGENCY_DECEL = -4.5     # 急煞門檻: -4.5m/s^2

# --- [新增] 寫入 Log 檔案函式 ---
def write_file_log(msg):
    try:
        # 寫入到 /data/media/0/scda_log.txt (一般使用者可見的儲存空間)
        with open("/data/media/0/scda_log.txt", "a") as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(f"[{timestamp}] {msg}\n")
    except Exception:
        pass

class SpeedCameraControl:
  def __init__(self):
    self.cameras = np.empty((0, 3))
    
    # --- 動態參數 (scda 12 設定) ---
    self.angle_bp = [0., 60., 80.]
    self.angle_vals = [20., 20., 15.]
    
    self.limit_radius_bp = [0., 40., 60., 80., 105., 110.]
    self.limit_radius_vals = [100., 100., 100., 150., 200., 300.]
    
    self.search_bp = [0., 79., 105.]
    self.search_vals = [500., 500., 600.]
    
    self.last_load_time = 0.0
    self.last_log_time = 0.0 
    
    self.ignore_index = -1       
    self.active_index = -1       
    
    self._load_cameras()

  def _load_cameras(self):
    paths = [
        os.path.join(os.path.dirname(__file__), 'NPA_TD1.csv'),
        '/data/openpilot/dragonpilot/selfdrive/controls/lib/NPA_TD1.csv'
    ]
    csv_path = next((p for p in paths if os.path.exists(p)), None)

    if not csv_path:
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
          msg = f"SCDA: Loaded {len(self.cameras)} cameras."
          cloudlog.warning(msg)
          write_file_log(msg) # 記錄載入成功
    except Exception:
      self.cameras = np.empty((0, 3))

  def cancel_current_camera(self):
    if self.active_index != -1:
        if self.ignore_index != self.active_index:
            msg = f"SCDA: 使用者踩油門，已取消相機 ID {self.active_index}"
            cloudlog.warning(msg)
            write_file_log(msg) # 記錄踩油門取消
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

  def get_target_speed(self, v_ego_ms, v_cruise_ms, lat, lon, bearing_deg):
    default_ret = {'target_speed': v_cruise_ms, 'distance': None, 'limit': None, 'is_active': False}

    if not math.isfinite(v_ego_ms) or not math.isfinite(lat) or not math.isfinite(lon):
      return default_ret

    current_time = time.monotonic()
    should_log = (current_time - self.last_log_time) > 1.0

    v_ego_kph = v_ego_ms * MS_TO_KPH
    v_cruise_kph = v_cruise_ms * MS_TO_KPH
    
    candidates = [v_cruise_kph]

    if self.cameras.size == 0:
      return default_ret

    # 1. 動態參數計算
    search_radius = np.interp(v_ego_kph, self.search_bp, self.search_vals)
    limit_radius = np.interp(v_ego_kph, self.limit_radius_bp, self.limit_radius_vals)
    base_angle = np.interp(v_ego_kph, self.angle_bp, self.angle_vals)

    # [scda 12 特色] 動態保持距離
    if v_ego_kph >= 80.0:
        current_hold_dist = 75.0
    elif v_ego_kph >= 60.0:
        current_hold_dist = 50.0
    else:
        current_hold_dist = 25.0

    # 2. 粗篩選
    deg_diff = (search_radius / 111000) * 1.5
    mask = (np.abs(self.cameras[:, 0] - lat) < deg_diff) & (np.abs(self.cameras[:, 1] - lon) < deg_diff)
    nearby_cams = self.cameras[mask]
    nearby_indices = np.arange(self.cameras.shape[0])[mask]

    self.active_index = -1
    
    if self.ignore_index != -1 and self.ignore_index not in nearby_indices:
         ignored_cam = self.cameras[self.ignore_index]
         dist_to_ignored = self._haversine(lat, lon, ignored_cam[0], ignored_cam[1])
         if dist_to_ignored > search_radius + 100: 
             self.ignore_index = -1

    if nearby_cams.size == 0:
      return default_ret

    closest_log_info = None 
    min_dist_found = 9999.0
    active_camera_limit = None

    for i, cam in enumerate(nearby_cams):
      original_idx = nearby_indices[i]
      if original_idx == self.ignore_index: continue
          
      cam_lat, cam_lon, cam_limit = cam
      dist = self._haversine(lat, lon, cam_lat, cam_lon)
      
      if dist > search_radius: continue

      allowed_angle = 20.0 if dist <= 150.0 else base_angle
      if math.isnan(bearing_deg): continue
      cam_bearing = self._bearing(lat, lon, cam_lat, cam_lon)
      diff_angle = abs(bearing_deg - cam_bearing)
      if diff_angle > 180: diff_angle = 360 - diff_angle
      
      is_front = diff_angle <= allowed_angle
      is_behind = (180 - diff_angle) <= allowed_angle
      
      if not (is_front or (is_behind and dist < limit_radius)): 
          continue

      # --- 核心過濾 ---
      
      # 1. 安全門檻 (+15)
      if v_ego_kph > (cam_limit + SAFETY_OFFSET):
          if dist < 200 and dist < min_dist_found:
              closest_log_info = {"status": f"忽略(門檻>{SAFETY_OFFSET})", "dist": dist, "limit": cam_limit}
          continue

      # 2. 物理急煞 (-4.5)
      cam_limit_ms = cam_limit * KPH_TO_MS
      if v_ego_ms > cam_limit_ms and is_front: 
          required_decel = (cam_limit_ms**2 - v_ego_ms**2) / (2 * max(dist, 1.0))
          if required_decel < EMERGENCY_DECEL:
              if dist < min_dist_found:
                  closest_log_info = {"status": f"忽略(急煞{required_decel:.1f})", "dist": dist, "limit": cam_limit}
              continue

      # 計算目標速度 (包含動態保持距離)
      if dist > limit_radius:
        target = v_cruise_kph
      else:
        target = np.interp(dist, [current_hold_dist, limit_radius], [cam_limit, v_cruise_kph])
      
      if math.isfinite(target):
        candidates.append(target)
        if dist < min_dist_found:
            min_dist_found = dist
            active_camera_limit = cam_limit
            self.active_index = original_idx 
            
            status_str = "介入中" if is_front else "通過|回速中"
            
            closest_log_info = {
                "dist": dist, 
                "limit": cam_limit, 
                "status": status_str, 
                "target": target
            }

    final_target_kph = min(candidates)
    final_target_kph = min(final_target_kph, v_cruise_kph)

    # --- Log 輸出 (雙重備份: CloudLog + 文字檔) ---
    if should_log and closest_log_info and closest_log_info["dist"] < search_radius:
        self.last_log_time = current_time
        status = closest_log_info["status"]
        limit = closest_log_info["limit"]
        dist = closest_log_info["dist"]
        
        is_intervening = (final_target_kph < v_cruise_kph - 1.0)
        
        if is_intervening or "忽略" in status or "回速" in status:
            log_msg = f"SCDA {status}: 限速{limit:.0f} | 距離{dist:.0f}m | 目標{final_target_kph:.0f}kph"
            
            # 1. 輸出到系統 Logcat
            cloudlog.warning(log_msg)
            
            # 2. [新增] 輸出到文字檔 /data/media/0/scda_log.txt
            write_file_log(log_msg)

    is_active = (min_dist_found < 9999.0 and final_target_kph < v_cruise_kph)
    
    return {
        'target_speed': final_target_kph * KPH_TO_MS,
        'distance': min_dist_found if is_active else None,
        'limit': active_camera_limit if is_active else None,
        'is_active': is_active
    }
  
  def get_statistics(self):
    return {"cameras_loaded": len(self.cameras)}
