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
    # 這是「基礎角度」，用於遠距離偵測
    self.angle_bp = [0., 60., 80.]
    self.angle_vals = [20., 20., 15.]
    
    # 減速啟動半徑 (meters)：針對台灣高速公路優化
    # 時速 110km/h 對應 300 公尺，提供更平滑的減速段差
    self.limit_radius_bp = [0., 40., 60., 80., 105., 110.]
    self.limit_radius_vals = [100., 100., 100., 150., 200., 300.]
    
    # 搜尋半徑 (meters)：確保在減速點前就能偵測到相機
    self.search_bp = [0., 79., 105.]
    self.search_vals = [200., 300., 500.]
    
    self.center_hold_dist = 25.0 # 抵達相機前 20 公尺維持限速
    
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
    
    # 這裡計算基礎角度 (高速時通常為 15度)
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
      
      if dist > search_radius:
        continue

      # --- 視角計算：明確的雙重檢查邏輯 ---
      # 說明：
      # 1. 距離 > 100m：使用 base_angle (高速時較嚴格，如 15度)，避免遠處誤抓。
      # 2. 距離 <= 100m：強制放寬至 20度 進行雙重確認。
      #    - 若角度 < 15：持續運作。
      #    - 若角度 > 15：視為誤判，下面邏輯會將其過濾並 continue。
      
      if dist <= 100.0:
          allowed_angle = 20.0
      else:
          allowed_angle = base_angle
      
      if math.isnan(bearing_deg): continue
      
      cam_bearing = self._bearing(lat, lon, cam_lat, cam_lon)
      diff_angle = abs(bearing_deg - cam_bearing)
      if diff_angle > 180: diff_angle = 360 - diff_angle
      
      # 判定前方或後方
      is_front = diff_angle <= allowed_angle
      is_behind = (180 - diff_angle) <= allowed_angle
      
      # 若在 100m 內且 diff_angle > 20，is_front 為 False -> 執行 continue -> 取消減速
      if not (is_front or (is_behind and dist < limit_radius)):
        continue

      # 安全門檻：防急煞
      if v_ego_kph > cam_limit + 15:
        if dist < min_dist_found:
            closest_log_info = {"status": "速差過大(防急煞)", "dist": dist, "limit": cam_limit}
        continue

      # 計算目標速度
      if dist > limit_radius:
        target = v_cruise_kph
      else:
        # 使用線性插值，在 50m 到 limit_radius 之間平滑變動
        target = np.interp(dist, [self.center_hold_dist, limit_radius], [cam_limit, v_cruise_kph])
      
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
        if final_target_kph < v_cruise_kph - 1.0:
             cloudlog.warning(f"SCDA {status}: 限速{limit:.0f} | 距離{dist:.0f}m | 目標{final_target_kph:.1f}kph")

    return final_target_kph * KPH_TO_MS
