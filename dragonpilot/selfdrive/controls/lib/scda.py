#!/usr/bin/env python3
"""
SCDA (12) - Log Enhanced & Fully Commented Edition
功能全開版:
1. 安全門檻 (+20km/h): 防止高架/平面誤判
2. 物理防護 (-4.5m/s²): 防止數據錯誤導致急煞
3. 踩油門取消: 使用者介入後暫時忽略該相機
4. 通過後緩加速: 根據距離線性恢復速度
5. 動態保持距離: 依車速調整通過後的鎖定距離 (25/50/75m)
6. 雙重 Log: 支援 logcat 與 /data/media/0/scda_log.txt
7. [新增] Log 開關控制: 支援 Master Switch 與 ACC On 檢查。
"""
import os
import csv
import math
import time
import numpy as np
from openpilot.common.swaglog import cloudlog

# =========================================
# 1. 常數定義區 (Constants)
# =========================================
MS_TO_KPH = 3.6           # [轉換係數] m/s -> km/h
KPH_TO_MS = 1. / 3.6      # [轉換係數] km/h -> m/s

# --- [新增] Log 總開關 ---
MASTER_LOG_ENABLED = True  # True: 開啟寫入檔案, False: 完全停用

# --- [設定] 安全參數 ---
SAFETY_OFFSET = 20.0       # [參數] 安全忽略門檻 (km/h)
                           # 作用：若車速 > (限速 + 20)，視為高架/平面誤判，直接忽略。
                           
EMERGENCY_DECEL = -4.5     # [參數] 物理急煞門檻 (m/s^2)
                           # 作用：若算出來需要煞車力道 > -4.5 (如 -6.0)，視為數據錯誤，忽略之。

# --- [新增] 寫入 Log 檔案函式 ---
def write_file_log(msg):
    """
    將 Log 寫入到手機儲存空間，方便用檔案管理員查看
    路徑: /data/media/0/scda_log.txt
    """
    try:
        with open("/data/media/0/scda_log.txt", "a") as f:
            # 加上當下時間戳記
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(f"[{timestamp}] {msg}\n")
    except Exception:
        pass

class SpeedCameraControl:
  def __init__(self):
    # 初始化相機陣列 [緯度, 經度, 限速]
    self.cameras = np.empty((0, 3))
    
    # --- 動態參數 (依車速改變行為) ---
    
    # 1. 視角判斷 (Angle)
    # 邏輯: 速度越快(>80)，視角越窄(15度)，避免誤抓隔壁道路
    self.angle_bp = [0., 60., 80.]
    self.angle_vals = [20., 20., 15.]
    
    # 2. 減速啟動半徑 (Limit Radius)
    # 邏輯: 速度越快(110kph)，要在越遠的地方(300m)開始減速，煞車才平穩
    self.limit_radius_bp = [0., 40., 60., 80., 105., 110.]
    self.limit_radius_vals = [100., 100., 100., 150., 200., 300.]
    
    # 3. 搜尋半徑 (Search Radius)
    # 邏輯: 只載入車輛附近的相機進行運算，節省效能
    self.search_bp = [0., 79., 105.]
    self.search_vals = [500., 500., 600.]
    
    # 記錄時間 (用於熱重載與 Log 頻率控制)
    self.last_load_time = 0.0
    self.last_log_time = 0.0 
    
    # --- 狀態追蹤 ---
    self.ignore_index = -1       # 被踩油門取消的相機 ID
    self.active_index = -1       # 當前正在作用的相機 ID
    
    # 啟動時載入檔案
    self._load_cameras()

  def _load_cameras(self):
    """
    讀取 NPA_TD1.csv 檔案並轉換為 numpy 格式
    """
    # 定義檔案路徑 (優先讀取當前目錄)
    paths = [
        os.path.join(os.path.dirname(__file__), 'NPA_TD1.csv'),
        '/data/openpilot/dragonpilot/selfdrive/controls/lib/NPA_TD1.csv'
    ]
    csv_path = next((p for p in paths if os.path.exists(p)), None)

    if not csv_path:
      return

    try:
      # 檢查檔案是否更新，沒更新就不重複讀取
      current_mtime = os.path.getmtime(csv_path)
      if current_mtime == self.last_load_time and self.cameras.shape[0] > 0:
        return
      
      self.last_load_time = current_mtime
      cams = []
      
      # 開啟 CSV 讀取資料
      with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
          try:
            # 取得經緯度與限速
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
          msg = f"SCDA: 載入成功，共 {len(self.cameras)} 支相機。"
          cloudlog.warning(msg)
          if MASTER_LOG_ENABLED:
              write_file_log(msg) # 寫入文字檔 Log
    except Exception:
      self.cameras = np.empty((0, 3))

  def cancel_current_camera(self):
    """
    [外部呼叫] 當使用者踩油門時呼叫
    功能: 將當前作用中的相機加入忽略名單
    """
    if self.active_index != -1:
        # 如果是新的取消動作，寫入 Log
        if self.ignore_index != self.active_index:
            msg = f"SCDA: 使用者踩油門，已取消相機 ID {self.active_index}"
            cloudlog.warning(msg)
            if MASTER_LOG_ENABLED:
                write_file_log(msg)
        # 設定忽略 ID
        self.ignore_index = self.active_index

  # --- 數學計算工具 ---
  @staticmethod
  def _haversine(lat1, lon1, lat2, lon2):
    """計算兩點間的直線距離 (單位: 公尺)"""
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 6371000 * 2 * np.arctan2(np.sqrt(np.maximum(a, 0)), np.sqrt(np.maximum(1-a, 0)))

  @staticmethod
  def _bearing(lat1, lon1, lat2, lon2):
    """計算方位角 (0-360度)"""
    dlon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360

  def get_target_speed(self, v_ego_ms, v_cruise_ms, lat, lon, bearing_deg, acc_enabled=False):
    """
    [核心邏輯] 計算目標速度
    回傳: 包含 target_speed 的字典
    """
    default_ret = {'target_speed': v_cruise_ms, 'distance': None, 'limit': None, 'is_active': False}

    # 檢查 GPS 訊號有效性
    if not math.isfinite(v_ego_ms) or not math.isfinite(lat) or not math.isfinite(lon):
      return default_ret

    # 控制 Log 頻率 (每秒一次)
    current_time = time.monotonic()
    should_log = (current_time - self.last_log_time) > 1.0

    v_ego_kph = v_ego_ms * MS_TO_KPH
    v_cruise_kph = v_cruise_ms * MS_TO_KPH
    
    candidates = [v_cruise_kph]

    if self.cameras.size == 0:
      return default_ret

    # 1. 取得動態參數
    search_radius = np.interp(v_ego_kph, self.search_bp, self.search_vals)
    limit_radius = np.interp(v_ego_kph, self.limit_radius_bp, self.limit_radius_vals)
    base_angle = np.interp(v_ego_kph, self.angle_bp, self.angle_vals)

    # [新增] 動態保持距離 (Dynamic Hold Distance)
    # 依車速決定通過相機後要保持限速多遠，才開始加速
    if v_ego_kph >= 80.0:
        current_hold_dist = 75.0  # 高速: 保持 75m (更保守)
    elif v_ego_kph >= 60.0:
        current_hold_dist = 50.0  # 中速: 保持 50m
    else:
        current_hold_dist = 25.0  # 低速: 保持 25m (快點離開)

    # 2. 粗篩選 (方框過濾，比 Haversine 快)
    deg_diff = (search_radius / 111000) * 1.5
    mask = (np.abs(self.cameras[:, 0] - lat) < deg_diff) & (np.abs(self.cameras[:, 1] - lon) < deg_diff)
    nearby_cams = self.cameras[mask]
    nearby_indices = np.arange(self.cameras.shape[0])[mask]

    self.active_index = -1
    
    # 檢查是否遠離了被忽略的相機 (若是則重設忽略狀態)
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

    # 3. 詳細比對迴圈
    for i, cam in enumerate(nearby_cams):
      original_idx = nearby_indices[i]
      
      # 跳過忽略名單中的相機
      if original_idx == self.ignore_index: continue
          
      cam_lat, cam_lon, cam_limit = cam
      dist = self._haversine(lat, lon, cam_lat, cam_lon)
      
      # 距離過濾
      if dist > search_radius: continue

      # 視角過濾 (近距離 <=150m 時放寬至 20度)
      allowed_angle = 30.0 if dist <= 150.0 else base_angle
      if math.isnan(bearing_deg): continue
      cam_bearing = self._bearing(lat, lon, cam_lat, cam_lon)
      diff_angle = abs(bearing_deg - cam_bearing)
      if diff_angle > 180: diff_angle = 360 - diff_angle
      
      # 位置判定: 前方(is_front) 或 剛通過(is_behind)
      is_front = diff_angle <= allowed_angle
      is_behind = (180 - diff_angle) <= allowed_angle
      
      # [緩加速條件] 
      # 必須是「在前方」或者「在後方但還在減速半徑內」
      if not (is_front or (is_behind and dist < limit_radius)): 
          continue

      # ==========================================================
      # 核心過濾邏輯
      # ==========================================================

      # 1. 安全門檻檢查 (+15 km/h)
      # 目的: 防止高架誤讀平面相機
      # 邏輯: 車速 > (限速 + 15) -> 忽略
      if v_ego_kph > (cam_limit + SAFETY_OFFSET):
          if dist < 200 and dist < min_dist_found:
              closest_log_info = {"status": f"忽略(門檻>{SAFETY_OFFSET})", "dist": dist, "limit": cam_limit}
          continue

      # 2. 物理急煞檢查 (-4.5 m/s²)
      # 目的: 防止距離數據跳變導致危險急煞
      # 邏輯: 需求減速度 < -4.5 -> 忽略
      cam_limit_ms = cam_limit * KPH_TO_MS
      if v_ego_ms > cam_limit_ms and is_front: 
          required_decel = (cam_limit_ms**2 - v_ego_ms**2) / (2 * max(dist, 1.0))
          if required_decel < EMERGENCY_DECEL:
              if dist < min_dist_found:
                  closest_log_info = {"status": f"忽略(急煞{required_decel:.1f})", "dist": dist, "limit": cam_limit}
              continue

      # ==========================================================

      # 計算目標速度
      if dist > limit_radius:
        # 距離還遠，保持巡航速度
        target = v_cruise_kph
      else:
        # [動態緩加速核心]
        # 使用線性插值 (Linear Interpolation)
        # 0 ~ hold_dist: 鎖定在 cam_limit
        # hold_dist ~ limit_radius: 線性上升至 v_cruise_kph
        target = np.interp(dist, [current_hold_dist, limit_radius], [cam_limit, v_cruise_kph])
      
      if math.isfinite(target):
        candidates.append(target)
        
        # 記錄最近的一個有效相機
        if dist < min_dist_found:
            min_dist_found = dist
            active_camera_limit = cam_limit
            self.active_index = original_idx 
            
            # 狀態顯示文字
            status_str = "介入中" if is_front else "通過|回速中"
            
            closest_log_info = {
                "dist": dist, 
                "limit": cam_limit, 
                "status": status_str, 
                "target": target
            }

    # 取所有候選速度的最小值
    final_target_kph = min(candidates)
    final_target_kph = min(final_target_kph, v_cruise_kph)

    # --- Log 輸出 (雙重備份) ---
    if should_log and closest_log_info and closest_log_info["dist"] < search_radius:
        self.last_log_time = current_time
        status = closest_log_info["status"]
        limit = closest_log_info["limit"]
        dist = closest_log_info["dist"]
        
        # 判斷是否真的有作用 (目標速度低於巡航速度)
        is_intervening = (final_target_kph < v_cruise_kph - 1.0)
        
        if is_intervening or "忽略" in status or "回速" in status:
            log_msg = f"SCDA {status}: 限速{limit:.0f} | 距離{dist:.0f}m | 目標{final_target_kph:.0f}kph"
            
            # 1. 輸出到 logcat (系統日誌) - 保持總是輸出以利除錯
            cloudlog.warning(log_msg)
            
            # 2. 輸出到文字檔 (方便查看) - [新增] 只有在 Master Switch 和 ACC On 時寫入
            if MASTER_LOG_ENABLED and acc_enabled:
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
