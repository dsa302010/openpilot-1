#!/usr/bin/env python3
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

class SpeedCameraControl:
  def __init__(self):
    self.cameras = np.empty((0, 3))
    
    # --- 參數設定 ---
    
    # 1. 角度過濾參數 (Degrees)
    # 邏輯：速度越快，視角越窄，避免抓到隔壁道路的相機
    # [速度節點 kph] -> [允許角度]
    self.angle_bp = [0., 60., 80.]
    self.angle_vals = [20., 20., 20.]
    
    # 2. 減速啟動半徑 (Meters)
    # 邏輯：速度越快，需要越早開始減速 (例如 110kph 時在 300m 處開始)
    # [速度節點 kph] -> [啟動距離 m]
    self.limit_radius_bp = [0., 40., 60., 80., 105., 110.]
    self.limit_radius_vals = [100., 150., 150., 200., 250., 300.]
    
    # 3. 搜尋半徑 (Meters)
    # 邏輯：這是程式「看到」相機的最遠距離，需比減速半徑大
    self.search_bp = [0., 79., 105.]
    self.search_vals = [500., 500., 600.]
    
    # 4. 通過保持距離
    # 在抵達相機前 50 公尺維持限速，避免持聽筒早加速
    self.center_hold_dist = 50.0 
    
    # --- 狀態變數 ---
    self.last_load_time = 0.0
    self.last_log_time = 0.0 
    
    # [新增] 忽略機制變數
    # ignore_index: 當前被使用者(踩油門)強制忽略的相機 ID
    # active_index: 目前系統正在參照(最優先)的相機 ID
    self.ignore_index = -1       
    self.active_index = -1       

    self._load_cameras()

  def _load_cameras(self):
    """讀取測速照相 CSV 檔案"""
    paths = [
        os.path.join(os.path.dirname(__file__), 'NPA_TD1.csv'),
        '/data/openpilot/dragonpilot/selfdrive/controls/lib/NPA_TD1.csv'
    ]
    # 尋找存在的路徑
    csv_path = next((p for p in paths if os.path.exists(p)), None)

    if not csv_path:
      cloudlog.warning("SCDA 錯誤: 找不到 CSV 檔案!")
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
            # 支援大小寫欄位名稱
            lat = r.get('Latitude') or r.get('latitude')
            lon = r.get('Longitude') or r.get('longitude')
            spd = r.get('Limit') or r.get('limit')
            
            if lat and lon and spd:
                cams.append([float(lat), float(lon), float(spd)])
          except ValueError:
            continue
            
      self.cameras = np.asarray(cams)
      
      # 重新載入檔案時，重設忽略狀態，避免 ID 對應錯誤
      self.ignore_index = -1 
      self.active_index = -1

      if len(self.cameras) > 0:
          cloudlog.warning(f"SCDA: 載入成功，共 {len(self.cameras)} 支相機。")
          
    except Exception as e:
      cloudlog.error(f"SCDA: 讀取失敗 - {e}")
      # 失敗時清空陣列，避免錯誤數據
      self.cameras = np.empty((0, 3))

  def cancel_current_camera(self):
    """
    [外部呼叫] 當使用者踩下油門時呼叫此方法。
    功能：將當前正在作用中的相機加入忽略清單。
    """
    if self.active_index != -1:
        if self.ignore_index != self.active_index:
            cloudlog.warning(f"SCDA: 使用者踩油門，已忽略相機 ID {self.active_index}")
        self.ignore_index = self.active_index

  @staticmethod
  def _haversine(lat1, lon1, lat2, lon2):
    """計算兩點經緯度之間的距離 (公尺)"""
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

  def get_target_speed(self, v_ego_ms, v_cruise_ms, lat, lon, bearing_deg):
    """
    主要邏輯函式
    輸入：當前車速、巡航設定速度、GPS座標、車輛方位角
    輸出：目標速度 (m/s)
    """
    # 基本檢查
    if not math.isfinite(v_ego_ms) or not math.isfinite(lat) or not math.isfinite(lon):
      return v_cruise_ms

    current_time = time.monotonic()
    should_log = (current_time - self.last_log_time) > 1.0

    v_ego_kph = v_ego_ms * MS_TO_KPH
    v_cruise_kph = v_cruise_ms * MS_TO_KPH
    
    candidates = [v_cruise_kph]

    if self.cameras.size == 0:
      return v_cruise_ms

    # 1. 根據車速取得動態參數
    search_radius = np.interp(v_ego_kph, self.search_bp, self.search_vals)
    limit_radius = np.interp(v_ego_kph, self.limit_radius_bp, self.limit_radius_vals)
    base_angle = np.interp(v_ego_kph, self.angle_bp, self.angle_vals)

    # 2. 粗略篩選 (大幅降低運算量)
    # 使用經緯度差值做第一層過濾
    deg_diff = (search_radius / 111000) * 1.5
    
    # 建立索引陣列，確保過濾後還知道原始 ID (用於忽略功能)
    all_indices = np.arange(self.cameras.shape[0])
    
    mask = (np.abs(self.cameras[:, 0] - lat) < deg_diff) & (np.abs(self.cameras[:, 1] - lon) < deg_diff)
    nearby_cams = self.cameras[mask]
    nearby_indices = all_indices[mask] # 保留原始索引

    # 每一幀重設 active_index，稍後重新判定
    self.active_index = -1
    
    # --- 忽略清單維護邏輯 ---
    # 如果已經遠離了被忽略的相機，則解除忽略狀態 (讓回程時還能偵測)
    if self.ignore_index != -1:
        # 如果被忽略的相機已經不在附近的清單裡 -> 解除忽略
        if self.ignore_index not in nearby_indices:
             # 雙重確認：計算實際距離
             ignored_cam = self.cameras[self.ignore_index]
             dist_to_ignored = self._haversine(lat, lon, ignored_cam[0], ignored_cam[1])
             # 距離大於搜尋半徑 + 100m 緩衝區 -> 重設
             if dist_to_ignored > search_radius + 100: 
                 self.ignore_index = -1
    # -----------------------

    if nearby_cams.size == 0:
      return v_cruise_ms

    closest_log_info = None 
    min_dist_found = 9999.0

    # 3. 詳細比對迴圈
    # 使用 enumerate 配合 nearby_indices 來獲取正確 ID
    for i, cam in enumerate(nearby_cams):
      original_idx = nearby_indices[i]
      
      # [關鍵] 如果此相機在忽略清單中，直接跳過
      if original_idx == self.ignore_index:
          continue
          
      cam_lat, cam_lon, cam_limit = cam
      dist = self._haversine(lat, lon, cam_lat, cam_lon)
      
      # 距離過濾
      if dist > search_radius:
        continue

      # 角度過濾邏輯：
      # 距離 > 150m：使用 base_angle (較嚴格，防止抓到遠處平行道路)
      # 距離 <= 150m：放寬至 25度 (防止接近相機時因 GPS 漂移而丟失目標)
      if dist <= 150.0:
          allowed_angle = 25.0
      else:
          allowed_angle = base_angle
      
      if math.isnan(bearing_deg): continue
      
      cam_bearing = self._bearing(lat, lon, cam_lat, cam_lon)
      diff_angle = abs(bearing_deg - cam_bearing)
      if diff_angle > 180: diff_angle = 360 - diff_angle
      
      # 判定是前方還是後方
      is_front = diff_angle <= allowed_angle
      is_behind = (180 - diff_angle) <= allowed_angle
      
      # 邏輯：必須是前方，或者是後方但在極近距離內 (防止剛過相機就急加速)
      # 這裡的 limit_radius 當作後方緩衝區有點大，但保留原邏輯
      if not (is_front or (is_behind and dist < limit_radius)):
        continue

      # 防急煞邏輯：如果目前車速遠大於限速 (超過 15kph)，且距離很近
      # 這裡選擇不介入，避免在高速公路上因為誤判平面道路相機而急煞
      if v_ego_kph > cam_limit + 15:
        if dist < min_dist_found:
            closest_log_info = {"status": "速差過大(防急煞)", "dist": dist, "limit": cam_limit}
        continue

      # 4. 計算目標速度
      if dist > limit_radius:
        # 在減速區外，維持巡航速度
        target = v_cruise_kph
      else:
        # 在減速區內，進行線性插值
        # 距離 limit_radius 時 -> v_cruise
        # 距離 center_hold_dist (50m) 時 -> cam_limit
        target = np.interp(dist, [self.center_hold_dist, limit_radius], [cam_limit, v_cruise_kph])
      
      if math.isfinite(target):
        candidates.append(target)
        if dist < min_dist_found:
            min_dist_found = dist
            # 標記當前生效的相機 ID
            self.active_index = original_idx 
            closest_log_info = {
                "dist": dist, 
                "limit": cam_limit, 
                "status": "介入中" if is_front else "緩回速中",
                "target": target
            }

    # 5. 決策：取所有候選速度的最小值
    final_target_kph = min(candidates)
    # 確保不會超過原本的巡航設定
    final_target_kph = min(final_target_kph, v_cruise_kph)

    # 6. Log 輸出
    if should_log and closest_log_info and closest_log_info["dist"] < search_radius:
        self.last_log_time = current_time
        status = closest_log_info["status"]
        limit = closest_log_info["limit"]
        dist = closest_log_info["dist"]
        # 只有當目標速度真的比巡航速度低時才顯示，避免洗版
        if final_target_kph < v_cruise_kph - 1.0:
             cloudlog.warning(f"SCDA {status}: 限速{limit:.0f} | 距離{dist:.0f}m | 目標{final_target_kph:.1f}kph")

    return final_target_kph * KPH_TO_MS
