#!/usr/bin/env python3
import os
import csv
import math
import time
import numpy as np
from openpilot.common.swaglog import cloudlog

# 定義轉換常數
MS_TO_KPH = 3.6
KPH_TO_MS = 1. / 3.6

class SpeedCameraControl:
  def __init__(self):
    self.cameras = np.empty((0, 3))
    
    # 調整角度參數：0-60km/h 用 20度，80km/h 以上用 15度
    self.angle_bp = [0., 60., 80.]
    self.angle_vals = [20., 20., 15.]
    
    # 減速啟動半徑 (meters)：針對台灣高速公路優化
    self.limit_radius_bp = [0., 40., 60., 80., 105., 110.]
    self.limit_radius_vals = [100., 100., 100., 150., 200., 300.]
    
    # 搜尋半徑 (meters)：確保在減速點前就能偵測到相機
    self.search_bp = [0., 79., 105.]
    self.search_vals = [500., 500., 600.]
    
    # --- 修改點 1: 設定中心維持距離為 20 公尺 ---
    self.center_hold_dist = 20.0 
    
    self.last_load_time = 0.0
    self.last_log_time = 0.0 
    self._load_cameras()

  def _load_cameras(self):
    paths = [
        os.path.join(os.path.dirname(__file__), 'NPA_TD1.csv'),
        '/data/openpilot/dragonpilot/selfdrive/controls/lib/NPA_TD1.csv'
    ]
    csv_path = next((p for p in paths if os.path.exists(p)), None)

    if not csv_path:
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
      if len(self.cameras) > 0:
          cloudlog.warning(f"SCDA: 載入成功，共 {len(self.cameras)} 支相機。")
          
    except Exception as e:
      cloudlog.error(f"SCDA: 讀取失敗 - {e}")
      self.cameras = np.empty((0, 3))

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
    if not math.isfinite(v_ego_ms) or not math.isfinite(lat) or not math.isfinite(lon):
      return v_cruise_ms

    current_time = time.monotonic()
    should_log = (current_time - self.last_log_time) > 1.0

    v_ego_kph = v_ego_ms * MS_TO_KPH
    v_cruise_kph = v_cruise_ms * MS_TO_KPH
    
    candidates = [v_cruise_kph]

    if self.cameras.size == 0:
      return v_cruise_ms

    # 動態參數計算
    search_radius = np.interp(v_ego_kph, self.search_bp, self.search_vals)
    limit_radius = np.interp(v_ego_kph, self.limit_radius_bp, self.limit_radius_vals)
    base_angle = np.interp(v_ego_kph, self.angle_bp, self.angle_vals)

    # 範圍過濾
    deg_diff = (search_radius / 111000) * 1.5
    mask = (np.abs(self.cameras[:, 0] - lat) < deg_diff) & (np.abs(self.cameras[:, 1] - lon) < deg_diff)
    nearby_cams = self.cameras[mask]

    if nearby_cams.size == 0:
      return v_cruise_ms

    closest_log_info = None 
    min_dist_found = 9999.0

    for cam in nearby_cams:
      cam_lat, cam_lon, cam_limit = cam
      dist = self._haversine(lat, lon, cam_lat, cam_lon)
      
      # 初步過濾：如果距離超過搜尋範圍則跳過
      # 注意：如果是後方(離去)，我們稍後會給予更大的寬容度，但在這裡先用 search_radius 過濾大方向
      if dist > search_radius:
        continue

      # 視角計算
      allowed_angle = 20.0 if dist > 150 else base_angle
      if math.isnan(bearing_deg): continue
      
      cam_bearing = self._bearing(lat, lon, cam_lat, cam_lon)
      diff_angle = abs(bearing_deg - cam_bearing)
      if diff_angle > 180: diff_angle = 360 - diff_angle
      
      # 判定前方或後方
      is_front = diff_angle <= allowed_angle
      is_behind = (180 - diff_angle) <= allowed_angle
      
      # --- 修改點 2: 設定離去(加速)時的參數 ---
      # 如果在相機後方 (is_behind)，我們將作用半徑擴大 1.5 倍
      # 這意味著從 25m 到 (limit_radius * 1.5) 的距離內會進行線性加速，坡度較緩
      departure_factor = 1.5 if is_behind else 1.0
      effective_radius = limit_radius * departure_factor

      # 如果既不是前方，也不是後方有效範圍內，則跳過
      if not (is_front or (is_behind and dist < effective_radius)):
        continue

      # 安全門檻：防急煞 (僅針對前方，若已經通過相機正在加速，則放寬限制)
      if is_front and v_ego_kph > cam_limit + 15:
        if dist < min_dist_found:
            closest_log_info = {"status": "速差過大(防急煞)", "dist": dist, "limit": cam_limit}
        continue

      # --- 修改點 3: 計算目標速度的核心邏輯 ---
      target = v_cruise_kph
      
      if dist <= self.center_hold_dist:
        # 情境 A: 距離小於 50m (包含接近中與剛通過) -> 強制維持限速
        target = cam_limit
      elif dist > effective_radius:
        # 情境 B: 超過有效半徑 -> 恢復巡航速度
        target = v_cruise_kph
      else:
        # 情境 C: 介於 50m 與 有效半徑之間 -> 線性插值
        # 接近時 (is_front): 從 effective_radius 減速到 50m 處
        # 離去時 (is_behind): 從 50m 處加速到 effective_radius (因為半徑較大，斜率較平緩)
        target = np.interp(dist, [self.center_hold_dist, effective_radius], [cam_limit, v_cruise_kph])
      
      if math.isfinite(target):
        candidates.append(target)
        if dist < min_dist_found:
            min_dist_found = dist
            closest_log_info = {
                "dist": dist, 
                "limit": cam_limit, 
                "status": "介入中" if is_front else "緩回速中",
                "target": target
            }

    # 決策：取最小值
    final_target_kph = min(candidates)
    final_target_kph = min(final_target_kph, v_cruise_kph)

    # --- Log 輸出 ---
    if should_log and closest_log_info and closest_log_info["dist"] < search_radius:
        self.last_log_time = current_time
        status = closest_log_info["status"]
        limit = closest_log_info["limit"]
        dist = closest_log_info["dist"]
        # 只要目標速度低於巡航速度，就顯示 Log
        if final_target_kph < v_cruise_kph - 1.0:
             cloudlog.warning(f"SCDA {status}: 限速{limit:.0f} | 距離{dist:.0f}m | 目標{final_target_kph:.1f}kph")

    return final_target_kph * KPH_TO_MS
