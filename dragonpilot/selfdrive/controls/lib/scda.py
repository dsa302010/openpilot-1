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
    
    # [修改點 1] 調整角度參數
    # 0-60km/h 用 20度，80km/h 以上用 15度
    self.angle_bp = [0., 60., 80.]
    self.angle_vals = [20., 20., 15.]
    
    # deceleration start radius (meters)
    self.limit_radius_bp = [0., 40., 60., 80., 105.]
    self.limit_radius_vals = [100., 100., 100., 150., 200.]
    
    # search radius (meters)
    self.search_bp = [0., 79., 80.]
    self.search_vals = [500., 500., 500.]
    
    self.center_hold_dist = 20.0
    
    self.last_load_time = 0.0
    self.last_log_time = 0.0 # 用於限制 Log 輸出頻率
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
            cams.append([float(r['Latitude']), float(r['Longitude']), float(r['Limit'])])
          except ValueError:
            continue
      self.cameras = np.asarray(cams)
      cloudlog.warning(f"SCDA: CSV 讀取成功，共載入 {len(self.cameras)} 支相機。")
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
    # 基礎輸入檢查
    if not math.isfinite(v_ego_ms) or not math.isfinite(lat) or not math.isfinite(lon):
      return v_cruise_ms

    current_time = time.monotonic()
    should_log = (current_time - self.last_log_time) > 1.0  # 每1秒只紀錄一次 Log

    v_ego_kph = v_ego_ms * MS_TO_KPH
    v_cruise_kph = v_cruise_ms * MS_TO_KPH
    
    candidates = [v_cruise_kph]

    if self.cameras.size == 0:
      return v_cruise_ms

    search_radius = np.interp(v_ego_kph, self.search_bp, self.search_vals)
    limit_radius = np.interp(v_ego_kph, self.limit_radius_bp, self.limit_radius_vals)
    base_angle = np.interp(v_ego_kph, self.angle_bp, self.angle_vals)

    # 快速過濾
    deg_diff = (search_radius / 111000) * 1.5
    mask = (np.abs(self.cameras[:, 0] - lat) < deg_diff) & (np.abs(self.cameras[:, 1] - lon) < deg_diff)
    nearby_cams = self.cameras[mask]

    if nearby_cams.size == 0:
      return v_cruise_ms

    # 用於 Log 的變數 (只記錄最近的一支)
    closest_log_info = None 
    min_dist_found = 9999.0

    for cam in nearby_cams:
      cam_lat, cam_lon, cam_limit = cam
      
      dist = self._haversine(lat, lon, cam_lat, cam_lon)
      if dist > search_radius:
        continue

      # 記錄最近的相機資訊供除錯
      if dist < min_dist_found:
        min_dist_found = dist
        # 先暫存基本資訊，後面檢查通過與否再更新狀態
        closest_log_info = {
            "dist": dist, 
            "limit": cam_limit, 
            "status": "檢查中", 
            "angle_diff": 0.0,
            "allowed_angle": 0.0
        }

      # 視角優化：遠窄近寬
      allowed_angle = 15.0 if dist > 150 else base_angle
      
      if closest_log_info and dist == min_dist_found:
          closest_log_info["allowed_angle"] = allowed_angle

      if math.isnan(bearing_deg): continue
      
      cam_bearing = self._bearing(lat, lon, cam_lat, cam_lon)
      diff_angle = abs(bearing_deg - cam_bearing)
      if diff_angle > 180: diff_angle = 360 - diff_angle
      
      if closest_log_info and dist == min_dist_found:
          closest_log_info["angle_diff"] = diff_angle

      if diff_angle > allowed_angle:
        if closest_log_info and dist == min_dist_found:
            closest_log_info["status"] = "角度過大"
        continue

      # 安全檢查：防止急煞
      if v_ego_kph > cam_limit + 20:
        if closest_log_info and dist == min_dist_found:
            closest_log_info["status"] = "速差過大(防急煞)"
        continue

      # 計算目標速度
      if dist > limit_radius:
        target = v_cruise_kph
        if closest_log_info and dist == min_dist_found:
            closest_log_info["status"] = "待命(距離外)"
      else:
        target = np.interp(dist, [self.center_hold_dist, limit_radius], [cam_limit, v_cruise_kph])
        if closest_log_info and dist == min_dist_found:
            closest_log_info["status"] = "介入中"
            closest_log_info["target"] = target
      
      if math.isfinite(target):
        candidates.append(target)

    # 決策：取最小值
    final_target_kph = min(candidates)
    final_target_kph = min(final_target_kph, v_cruise_kph)

    # --- 中文 Log 輸出區塊 (每秒一次) ---
    if should_log and closest_log_info and closest_log_info["dist"] < 500:
        self.last_log_time = current_time
        status = closest_log_info["status"]
        limit = closest_log_info["limit"]
        dist = closest_log_info["dist"]
        
        # 如果正在介入 (目標速度 < 巡航速度)
        if final_target_kph < v_cruise_kph - 1.0:
             cloudlog.warning(f"SCDA 介入: 限速{limit:.0f} | 距離{dist:.0f}m | 目標{final_target_kph:.1f}kph")
        
        # 如果被忽略 (角度或速差) 且距離很近 (<300m) 才顯示警告，方便除錯
        elif status in ["角度過大", "速差過大(防急煞)"] and dist < 300:
            diff = closest_log_info.get("angle_diff", 0)
            allowed = closest_log_info.get("allowed_angle", 0)
            if status == "角度過大":
                cloudlog.warning(f"SCDA 忽略: 角度{diff:.1f}° > 允許{allowed:.1f}° (距離{dist:.0f}m)")
            else:
                cloudlog.warning(f"SCDA 忽略: 速差過大 (車速{v_ego_kph:.0f} > 限速{limit:.0f}+20)")
        
        # 正常待命中 (除錯用，可視情況開啟，目前設為 info)
        # else:
        #    cloudlog.info(f"SCDA 待命: 最近相機 {dist:.0f}m (限速{limit})")

    return final_target_kph * KPH_TO_MS
