#!/usr/bin/env python3
"""
Speed Camera Detection & Adjustment (SCDA) - Optimized Edition
主要改進:
1. [關鍵] 返回距離資訊 - 讓 DTSC 可以判斷緊急程度
2. 改進防急煞邏輯 - 基於物理計算而非固定速差
3. 增加安全裕度檢查 - 確保有足夠距離減速
4. 優化角度過濾 - 更智能的動態調整
5. 增加調試日誌 - 方便追蹤問題
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

# --- [新增] 安全參數 ---
MAX_COMFORTABLE_DECEL = -2.5  # m/s² - 舒適減速度上限
EMERGENCY_DECEL_THRESHOLD = -3.5  # m/s² - 緊急煞車閾值
SAFETY_TIME_BUFFER = 2.0  # 秒 - 安全時間緩衝

# --- [新增] 調試開關 ---
DEBUG_LOGGING = False  # 設為 True 查看詳細 Log

class SpeedCameraControl:
  def __init__(self):
    self.cameras = np.empty((0, 3))
    
    # --- 參數設定 ---
    
    # 1. 角度過濾參數 (Degrees)
    # [改進] 增加高速時的視角範圍,因為高速公路視野更開闊
    self.angle_bp = [0., 60., 80., 110.]
    self.angle_vals = [20., 20., 22., 25.]
    
    # 2. 減速啟動半徑 (Meters)
    # 邏輯:速度越快,需要越早開始減速
    self.limit_radius_bp = [0., 40., 60., 80., 105., 110.]
    self.limit_radius_vals = [100., 150., 150., 200., 250., 300.]
    
    # 3. 搜尋半徑 (Meters)
    self.search_bp = [0., 79., 105.]
    self.search_vals = [500., 500., 600.]
    
    # 4. 通過保持距離
    self.center_hold_dist = 50.0 
    
    # --- 狀態變數 ---
    self.last_load_time = 0.0
    self.last_log_time = 0.0 
    
    # 忽略機制變數
    self.ignore_index = -1       
    self.active_index = -1       
    
    # [新增] 性能統計
    self.intervention_count = 0
    self.rejection_count = 0

    self._load_cameras()

  def _load_cameras(self):
    """讀取測速照相 CSV 檔案"""
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
      
      # 重新載入檔案時,重設忽略狀態
      self.ignore_index = -1 
      self.active_index = -1

      if len(self.cameras) > 0:
          cloudlog.warning(f"SCDA: 載入成功,共 {len(self.cameras)} 支相機。")
          
    except Exception as e:
      cloudlog.error(f"SCDA: 讀取失敗 - {e}")
      self.cameras = np.empty((0, 3))

  def cancel_current_camera(self):
    """
    [外部呼叫] 當使用者踩下油門時呼叫此方法。
    功能:將當前正在作用中的相機加入忽略清單。
    """
    if self.active_index != -1:
        if self.ignore_index != self.active_index:
            cloudlog.warning(f"SCDA: 使用者踩油門,已忽略相機 ID {self.active_index}")
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

  def _is_safe_to_intervene(self, v_ego_ms, cam_limit_kph, distance):
    """
    [新增] 安全介入檢查 - 基於物理計算
    
    判斷是否有足夠距離以舒適方式減速到限速
    
    返回: (is_safe, required_decel, reason)
    """
    cam_limit_ms = cam_limit_kph * KPH_TO_MS
    
    # 如果已經低於限速,不需要檢查
    if v_ego_ms <= cam_limit_ms:
        return True, 0.0, "already_below_limit"
    
    # 計算所需減速度: v_f^2 = v_i^2 + 2*a*d
    required_decel = (cam_limit_ms ** 2 - v_ego_ms ** 2) / (2.0 * distance)
    
    # 檢查是否需要緊急煞車
    if required_decel < EMERGENCY_DECEL_THRESHOLD:
        self.rejection_count += 1
        return False, required_decel, "emergency_brake_needed"
    
    # 檢查是否超過舒適減速度 (給予警告但仍介入)
    if required_decel < MAX_COMFORTABLE_DECEL:
        if DEBUG_LOGGING:
            cloudlog.warning(f"SCDA: 減速較激進 a={required_decel:.2f} m/s²")
        return True, required_decel, "aggressive_decel"
    
    # 檢查時間緩衝 (是否有足夠時間反應)
    time_available = distance / v_ego_ms if v_ego_ms > 0 else 0
    if time_available < SAFETY_TIME_BUFFER:
        self.rejection_count += 1
        return False, required_decel, "insufficient_time"
    
    return True, required_decel, "safe"

  def get_target_speed(self, v_ego_ms, v_cruise_ms, lat, lon, bearing_deg):
    """
    主要邏輯函式
    輸入:當前車速、巡航設定速度、GPS座標、車輛方位角
    輸出:字典包含 {target_speed, distance, limit, is_active}
    
    [重要變更] 現在返回字典而非單一速度值,以支援 DTSC 協調
    """
    # 基本檢查
    if not math.isfinite(v_ego_ms) or not math.isfinite(lat) or not math.isfinite(lon):
      return {
          'target_speed': v_cruise_ms,
          'distance': None,
          'limit': None,
          'is_active': False
      }

    current_time = time.monotonic()
    should_log = (current_time - self.last_log_time) > 1.0

    v_ego_kph = v_ego_ms * MS_TO_KPH
    v_cruise_kph = v_cruise_ms * MS_TO_KPH
    
    candidates = [v_cruise_kph]

    if self.cameras.size == 0:
      return {
          'target_speed': v_cruise_ms,
          'distance': None,
          'limit': None,
          'is_active': False
      }

    # 1. 根據車速取得動態參數
    search_radius = np.interp(v_ego_kph, self.search_bp, self.search_vals)
    limit_radius = np.interp(v_ego_kph, self.limit_radius_bp, self.limit_radius_vals)
    base_angle = np.interp(v_ego_kph, self.angle_bp, self.angle_vals)

    # 2. 粗略篩選
    deg_diff = (search_radius / 111000) * 1.5
    
    all_indices = np.arange(self.cameras.shape[0])
    
    mask = (np.abs(self.cameras[:, 0] - lat) < deg_diff) & (np.abs(self.cameras[:, 1] - lon) < deg_diff)
    nearby_cams = self.cameras[mask]
    nearby_indices = all_indices[mask]

    # 每一幀重設 active_index
    self.active_index = -1
    
    # --- 忽略清單維護邏輯 ---
    if self.ignore_index != -1:
        if self.ignore_index not in nearby_indices:
             ignored_cam = self.cameras[self.ignore_index]
             dist_to_ignored = self._haversine(lat, lon, ignored_cam[0], ignored_cam[1])
             if dist_to_ignored > search_radius + 100: 
                 self.ignore_index = -1
                 if DEBUG_LOGGING:
                     cloudlog.debug("SCDA: 忽略清單已清除")

    if nearby_cams.size == 0:
      return {
          'target_speed': v_cruise_ms,
          'distance': None,
          'limit': None,
          'is_active': False
      }

    closest_log_info = None 
    min_dist_found = 9999.0
    active_camera_limit = None  # [新增] 記錄生效相機的限速

    # 3. 詳細比對迴圈
    for i, cam in enumerate(nearby_cams):
      original_idx = nearby_indices[i]
      
      # 如果此相機在忽略清單中,直接跳過
      if original_idx == self.ignore_index:
          continue
          
      cam_lat, cam_lon, cam_limit = cam
      dist = self._haversine(lat, lon, cam_lat, cam_lon)
      
      # 距離過濾
      if dist > search_radius:
        continue

      # 角度過濾邏輯
      if dist <= 150.0:
          allowed_angle = 25.0
      else:
          allowed_angle = base_angle
      
      if math.isnan(bearing_deg): 
          continue
      
      cam_bearing = self._bearing(lat, lon, cam_lat, cam_lon)
      diff_angle = abs(bearing_deg - cam_bearing)
      if diff_angle > 180: 
          diff_angle = 360 - diff_angle
      
      is_front = diff_angle <= allowed_angle
      is_behind = (180 - diff_angle) <= allowed_angle
      
      if not (is_front or (is_behind and dist < limit_radius)):
        continue

      # [改進] 使用物理計算的防急煞邏輯
      is_safe, required_decel, reason = self._is_safe_to_intervene(v_ego_ms, cam_limit, dist)
      
      if not is_safe:
        if dist < min_dist_found:
            closest_log_info = {
                "status": f"已拒絕({reason})", 
                "dist": dist, 
                "limit": cam_limit,
                "required_decel": required_decel
            }
        continue

      # 4. 計算目標速度
      if dist > limit_radius:
        target = v_cruise_kph
      else:
        target = np.interp(dist, [self.center_hold_dist, limit_radius], [cam_limit, v_cruise_kph])
      
      if math.isfinite(target):
        candidates.append(target)
        if dist < min_dist_found:
            min_dist_found = dist
            active_camera_limit = cam_limit  # [新增] 記錄限速
            self.active_index = original_idx 
            self.intervention_count += 1
            closest_log_info = {
                "dist": dist, 
                "limit": cam_limit, 
                "status": "介入中" if is_front else "緩回速中",
                "target": target,
                "required_decel": required_decel
            }

    # 5. 決策:取所有候選速度的最小值
    final_target_kph = min(candidates)
    final_target_kph = min(final_target_kph, v_cruise_kph)

    # 6. Log 輸出
    if should_log and closest_log_info and closest_log_info["dist"] < search_radius:
        self.last_log_time = current_time
        status = closest_log_info["status"]
        limit = closest_log_info["limit"]
        dist = closest_log_info["dist"]
        
        if final_target_kph < v_cruise_kph - 1.0 or "已拒絕" in status:
            log_msg = f"SCDA {status}: 限速{limit:.0f} | 距離{dist:.0f}m"
            if "target" in closest_log_info:
                log_msg += f" | 目標{closest_log_info['target']:.1f}kph"
            if "required_decel" in closest_log_info and abs(closest_log_info["required_decel"]) > 0.1:
                log_msg += f" | 需求a={closest_log_info['required_decel']:.2f}m/s²"
            cloudlog.warning(log_msg)

    # [關鍵變更] 返回完整資訊字典
    is_active = (min_dist_found < 9999.0 and final_target_kph < v_cruise_kph)
    
    return {
        'target_speed': final_target_kph * KPH_TO_MS,
        'distance': min_dist_found if is_active else None,
        'limit': active_camera_limit if is_active else None,
        'is_active': is_active
    }
  
  def get_statistics(self):
    """
    [新增] 取得統計資訊
    """
    return {
        "interventions": self.intervention_count,
        "rejections": self.rejection_count,
        "cameras_loaded": len(self.cameras)
    }
