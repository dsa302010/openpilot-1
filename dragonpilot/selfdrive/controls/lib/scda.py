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
    
    # [修改點 1] 調整角度參數 (市區縮小至 20 度，高速 15 度)
    # 原本: [0., 79., 80.] -> [30., 30., 15.]
    # 修改: 0-60km/h 用 20度，80km/h 以上用 15度
    self.angle_bp = [0., 60., 80.]
    self.angle_vals = [20., 20., 15.]
    
    # deceleration start radius (meters)
    self.limit_radius_bp = [0., 40., 60., 80., 105.]
    self.limit_radius_vals = [100., 100., 100., 150., 200.]
    
    # search radius (meters)
    self.search_bp = [0., 79., 80.]
    self.search_vals = [500., 500., 500.]
    
    self.center_hold_dist = 20.0
    # self.active_cam 被廢除，不再需要記憶單一相機
    
    self.last_load_time = 0.0
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
            cams.append([float(r['Latitude']), float(r['Longitude']), float(r['Limit'])])
          except ValueError:
            continue
      self.cameras = np.asarray(cams)
      cloudlog.info(f"SCDA: Loaded {len(self.cameras)} cameras.")
    except Exception:
      cloudlog.exception("SCDA: Failed to load cameras")
      self.cameras = np.empty((0, 3))

  @staticmethod
  def _haversine(lat1, lon1, lat2, lon2):
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    # 增加 epsilon 避免 sqrt(0) 或負數誤差
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
    """
    全域競價邏輯：掃描所有相機，取最低需求速度
    回傳：目標速度 (m/s)，若無目標則回傳 v_cruise_ms
    """
    # 基礎輸入檢查
    if not math.isfinite(v_ego_ms) or not math.isfinite(lat) or not math.isfinite(lon):
      return v_cruise_ms # 安全回傳

    v_ego_kph = v_ego_ms * MS_TO_KPH
    v_cruise_kph = v_cruise_ms * MS_TO_KPH
    
    # 1. 預設候選速度為當前巡航速度 (如果沒相機，就維持原速)
    candidates = [v_cruise_kph]

    if self.cameras.size == 0:
      return v_cruise_ms

    # 計算動態半徑與角度
    search_radius = np.interp(v_ego_kph, self.search_bp, self.search_vals)
    limit_radius = np.interp(v_ego_kph, self.limit_radius_bp, self.limit_radius_vals)
    base_angle = np.interp(v_ego_kph, self.angle_bp, self.angle_vals)

    # 2. 快速過濾：只取方框範圍內的相機 (減少運算量)
    deg_diff = (search_radius / 111000) * 1.5
    lat_diff = np.abs(self.cameras[:, 0] - lat)
    lon_diff = np.abs(self.cameras[:, 1] - lon)
    mask = (lat_diff < deg_diff) & (lon_diff < deg_diff)
    
    nearby_cams = self.cameras[mask]
    if nearby_cams.size == 0:
      return v_cruise_ms

    # 3. 詳細迴圈檢查 (不再只取最近的一個，而是檢查每一個)
    for cam in nearby_cams:
      cam_lat, cam_lon, cam_limit = cam
      
      # 計算距離
      dist = self._haversine(lat, lon, cam_lat, cam_lon)
      if dist > search_radius:
        continue

      # [修改點 2] 視角優化：遠窄近寬
      # 如果距離 > 150m (遠處)，強制將角度縮窄至 15 度，避免抓到隔壁路
      # 如果距離 < 150m (近處)，使用 base_angle (20度)，允許轉彎時稍微偏一點
      allowed_angle = 15.0 if dist > 150 else base_angle

      # 計算角度差
      if math.isnan(bearing_deg): continue
      cam_bearing = self._bearing(lat, lon, cam_lat, cam_lon)
      diff_angle = abs(bearing_deg - cam_bearing)
      if diff_angle > 180: diff_angle = 360 - diff_angle
      
      if diff_angle > allowed_angle:
        continue

      # [修改點 3] 安全檢查：防止急煞 (Anti-Hard Brake)
      # 如果當前車速比速限高出 20km/h 以上，視為誤判或危險介入，忽略此相機
      if v_ego_kph > cam_limit + 20:
        continue

      # 4. 計算此相機建議的速度
      if dist > limit_radius:
        # 還在減速區外，這支相機允許你開 v_cruise
        target = v_cruise_kph
      else:
        # 進入減速區，開始線性遞減
        target = np.interp(dist, [self.center_hold_dist, limit_radius], [cam_limit, v_cruise_kph])
      
      if math.isfinite(target):
        candidates.append(target)

    # 5. [修改點 4] 全域最小值決策 (Global Minimum)
    # 取出所有候選速度中最慢的一個
    final_target_kph = min(candidates)

    # 雙重確認回傳值不超過原本設定的巡航速度 (避免加速)
    final_target_kph = min(final_target_kph, v_cruise_kph)

    return final_target_kph * KPH_TO_MS