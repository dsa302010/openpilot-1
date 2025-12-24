#!/usr/bin/env python3
"""
SCDA - Dynamic Hold Edition
包含: 
1. 安全門檻(+15) & 物理防急煞
2. 踩油門取消 & 通過後緩加速
3. [新增] 動態保持距離 (依車速改變通過後的限速維持長度)
"""
import os
import csv
import math
import time
import numpy as np
from openpilot.common.swaglog import cloudlog

# =========================================
# 常數定義 (Constants)
# =========================================
MS_TO_KPH = 3.6           # [轉換係數] 公尺/秒 (m/s) 轉 公里/小時 (km/h)
KPH_TO_MS = 1. / 3.6      # [轉換係數] 公里/小時 (km/h) 轉 公尺/秒 (m/s)

# --- [設定] 安全參數 ---
SAFETY_OFFSET = 15.0       # [參數] 安全忽略門檻 (km/h)。
                           # 作用：若當前車速比限速高出 15km/h 以上，視為駕駛有意超速或數據錯誤，不進行介入。
                           
EMERGENCY_DECEL = -4.5     # [參數] 物理急煞門檻 (m/s^2)。
                           # 作用：若計算出需要的減速度小於此值 (例如 -5.0)，代表需要極猛烈煞車，
                           # 為避免危險 (後車追撞或乘客不適)，系統將選擇忽略此相機。

class SpeedCameraControl:
  def __init__(self):
    # 初始化相機數據容器，格式為 numpy array，每列包含 [緯度, 經度, 限速]
    self.cameras = np.empty((0, 3))
    
    # --- 動態參數設定 (依車速改變行為) ---
    # 視角判斷 (Angle) 的車速節點 (Breakpoints) [km/h]
    self.angle_bp = [0., 60., 80.]
    # 對應上述車速的視角容許值 (Values) [度]
    # 作用：低速時容許較大角度誤差，高速時需要更精準的角度以避免誤判隔壁道路
    self.angle_vals = [20., 20., 15.]
    
    # 減速/限制生效半徑 (Limit Radius) 的車速節點 [km/h]
    self.limit_radius_bp = [0., 40., 60., 80., 105., 110.]
    # 對應的生效距離 [公尺]
    # 作用：車速越快，需要越早開始減速 (半徑越大)；通過後也需要越長距離恢復速度
    self.limit_radius_vals = [100., 100., 100., 150., 200., 300.]
    
    # 搜尋半徑 (Search Radius) 的車速節點 [km/h]
    self.search_bp = [0., 79., 105.]
    # 對應的搜尋範圍 [公尺]
    # 作用：決定程式要掃描周圍多遠的相機資料
    self.search_vals = [500., 500., 600.]
    
    # 記錄最後一次載入檔案的時間，用於熱重載 (Hot Reload)
    self.last_load_time = 0.0
    # 記錄最後一次輸出的時間，避免 Log 洗版
    self.last_log_time = 0.0 
    
    # --- 忽略清單機制 ---
    self.ignore_index = -1       # [狀態] 被使用者強制取消 (踩油門) 的相機索引 ID
    self.active_index = -1       # [狀態] 當前正在介入控制中的相機索引 ID
    
    # 初始化時嘗試載入相機資料
    self._load_cameras()

  def _load_cameras(self):
    """
    [方法] 載入測速相機 CSV 檔案
    邏輯：檢查指定路徑，若檔案存在且有更新，則讀取經緯度與限速。
    """
    # 定義可能的檔案路徑 (優先讀取當前目錄，其次讀取系統目錄)
    paths = [
        os.path.join(os.path.dirname(__file__), 'NPA_TD1.csv'),
        '/data/openpilot/dragonpilot/selfdrive/controls/lib/NPA_TD1.csv'
    ]
    # 找出第一個存在的路徑
    csv_path = next((p for p in paths if os.path.exists(p)), None)

    # 若找不到檔案則直接返回
    if not csv_path:
      return

    try:
      # 取得檔案最後修改時間
      current_mtime = os.path.getmtime(csv_path)
      # 如果檔案修改時間未變，且記憶體中已有資料，則不重複載入
      if current_mtime == self.last_load_time and self.cameras.shape[0] > 0:
        return
      
      # 更新最後載入時間
      self.last_load_time = current_mtime
      cams = []
      
      # 開啟 CSV 檔案進行讀取
      with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
          try:
            # 嘗試取得經緯度與限速欄位 (相容大小寫)
            lat = r.get('Latitude') or r.get('latitude')
            lon = r.get('Longitude') or r.get('longitude')
            spd = r.get('Limit') or r.get('limit')
            
            # 若資料完整，則加入列表
            if lat and lon and spd:
                cams.append([float(lat), float(lon), float(spd)])
          except ValueError:
            continue # 若資料格式錯誤則跳過該行
            
      # 將列表轉換為 Numpy Array 以提升計算效能
      self.cameras = np.asarray(cams)
      
      # 重設忽略與當前索引，因為陣列順序可能改變
      self.ignore_index = -1 
      self.active_index = -1
      
      # 若成功載入，寫入 Log
      if len(self.cameras) > 0:
          cloudlog.warning(f"SCDA: Loaded {len(self.cameras)} cameras.")
    except Exception:
      # 發生任何錯誤時，清空相機列表以防錯誤控制
      self.cameras = np.empty((0, 3))

  def cancel_current_camera(self):
    """
    [功能] 踩油門取消當下目標
    外部呼叫此方法時，將當前 active_index 加入忽略名單
    """
    # 只有在當前有鎖定相機時才執行
    if self.active_index != -1:
        # 如果這個相機還沒被忽略過，則記錄 Log
        if self.ignore_index != self.active_index:
            cloudlog.warning(f"SCDA: 使用者踩油門，已取消相機 ID {self.active_index}")
        # 將當前相機 ID 設定為忽略 ID
        self.ignore_index = self.active_index

  @staticmethod
  def _haversine(lat1, lon1, lat2, lon2):
    """
    [靜態方法] 半正矢公式 (Haversine Formula)
    作用：計算地球表面兩點 (經緯度) 之間的最短距離 (大圓距離)。
    回傳單位：公尺 (m)
    """
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    # 6371000 為地球平均半徑 (公尺)
    return 6371000 * 2 * np.arctan2(np.sqrt(np.maximum(a, 0)), np.sqrt(np.maximum(1-a, 0)))

  @staticmethod
  def _bearing(lat1, lon1, lat2, lon2):
    """
    [靜態方法] 方位角計算
    作用：計算從點1到點2的方位角 (0-360度，正北為0，順時針)。
    """
    dlon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    # math.atan2 計算弧度，轉為角度後轉正值
    return (math.degrees(math.atan2(y, x)) + 360) % 360

  def get_target_speed(self, v_ego_ms, v_cruise_ms, lat, lon, bearing_deg):
    """
    [核心方法] 計算目標速度
    輸入:
      v_ego_ms: 當前車速 (m/s)
      v_cruise_ms: 巡航設定速度 (m/s)
      lat, lon: 車輛當前經緯度
      bearing_deg: 車輛行駛方向 (度)
    輸出:
      Dict 包含目標速度、距離、限速資訊
    """
    # 預設回傳值：不改變速度 (目標 = 巡航速度)，無啟動狀態
    default_ret = {'target_speed': v_cruise_ms, 'distance': None, 'limit': None, 'is_active': False}

    # 檢查輸入數據有效性 (防止 NaN 導致崩潰)
    if not math.isfinite(v_ego_ms) or not math.isfinite(lat) or not math.isfinite(lon):
      return default_ret

    # 取得系統時間，用於控制 Log 頻率
    current_time = time.monotonic()
    should_log = (current_time - self.last_log_time) > 1.0

    # 單位轉換：m/s -> km/h
    v_ego_kph = v_ego_ms * MS_TO_KPH
    v_cruise_kph = v_cruise_ms * MS_TO_KPH
    
    # 候選目標速度列表，預設包含當前巡航速度
    candidates = [v_cruise_kph]

    # 若無相機資料，直接回傳預設值
    if self.cameras.size == 0:
      return default_ret

    # ---------------------------------------------------
    # 1. 根據車速取得動態參數 (Interpolation)
    # ---------------------------------------------------
    # 依車速計算搜尋半徑 (車越快看越遠)
    search_radius = np.interp(v_ego_kph, self.search_bp, self.search_vals)
    # 依車速計算減速生效半徑 (車越快越早減速)
    limit_radius = np.interp(v_ego_kph, self.limit_radius_bp, self.limit_radius_vals)
    # 依車速計算基本容許角度
    base_angle = np.interp(v_ego_kph, self.angle_bp, self.angle_vals)

    # [新增功能] 動態設定通過後的保持距離 (Center Hold Dist)
    # 邏輯：根據車速分級，設定通過相機後要「維持限速」多久才開始加速
    if v_ego_kph >= 80.0:
        current_hold_dist = 75.0  # 高速：保持 75m
    elif v_ego_kph >= 60.0:
        current_hold_dist = 50.0  # 中速：保持 50m
    else:
        current_hold_dist = 25.0  # 低速：保持 25m

    # ---------------------------------------------------
    # 2. 粗篩選 (Broad Phase Filtering)
    # ---------------------------------------------------
    # 計算經緯度差異閾值，用於快速過濾 (1度約=111km，111000m)
    # *1.5 是為了保留緩衝區
    deg_diff = (search_radius / 111000) * 1.5
    # 建立遮罩：只選取經緯度在大概範圍內的相機
    mask = (np.abs(self.cameras[:, 0] - lat) < deg_diff) & (np.abs(self.cameras[:, 1] - lon) < deg_diff)
    nearby_cams = self.cameras[mask]            # 篩選出的相機資料
    nearby_indices = np.arange(self.cameras.shape[0])[mask] # 篩選出的相機原始索引

    # 重設當前啟動索引
    self.active_index = -1
    
    # [機制] 檢查是否已經遠離被忽略的點
    # 如果目前有被忽略的點 (ignore_index != -1)，且該點不在附近的篩選範圍內，
    # 或者距離已經超過搜尋半徑+100m，則將其從忽略名單中移除 (重設)，以便下次經過時能再次生效。
    if self.ignore_index != -1 and self.ignore_index not in nearby_indices:
         ignored_cam = self.cameras[self.ignore_index]
         dist_to_ignored = self._haversine(lat, lon, ignored_cam[0], ignored_cam[1])
         if dist_to_ignored > search_radius + 100: 
             self.ignore_index = -1

    # 若粗篩後無相機，回傳預設值
    if nearby_cams.size == 0:
      return default_ret

    # 用於記錄最近相機的 Log 資訊
    closest_log_info = None 
    min_dist_found = 9999.0
    active_camera_limit = None

    # ---------------------------------------------------
    # 3. 精確計算與篩選 (Loop through candidates)
    # ---------------------------------------------------
    for i, cam in enumerate(nearby_cams):
      original_idx = nearby_indices[i]
      # 若此相機在忽略名單中，直接跳過
      if original_idx == self.ignore_index: continue
          
      cam_lat, cam_lon, cam_limit = cam
      # 計算精確距離 (Haversine)
      dist = self._haversine(lat, lon, cam_lat, cam_lon)
      
      # 若距離超出搜尋半徑，跳過
      if dist > search_radius: continue

      # --- 角度判斷 ---
      # 若距離非常近 (<=150m)，放寬角度限制至 20 度 (避免GPS漂移導致丟失目標)
      # 否則使用依車速計算的 base_angle
      allowed_angle = 20.0 if dist <= 150.0 else base_angle
      
      if math.isnan(bearing_deg): continue
      # 計算車輛到相機的方位角
      cam_bearing = self._bearing(lat, lon, cam_lat, cam_lon)
      # 計算角度差 (取絕對值)
      diff_angle = abs(bearing_deg - cam_bearing)
      # 處理 0/360 度交界問題
      if diff_angle > 180: diff_angle = 360 - diff_angle
      
      # 判定相對位置
      is_front = diff_angle <= allowed_angle            # 相機在車輛前方
      is_behind = (180 - diff_angle) <= allowed_angle   # 相機在車輛後方 (背對)
      
      # [緩加速條件判斷]
      # 有效條件：1. 相機在前方 (還沒過) OR 2. 相機在後方但距離還在限制半徑內 (剛過，還沒走遠)
      if not (is_front or (is_behind and dist < limit_radius)): 
          continue

      # --- 安全門檻檢查 (+15) ---
      # 若當前車速已經超過限速 15km/h，判定為駕駛有意超速，不介入
      if v_ego_kph > (cam_limit + SAFETY_OFFSET):
          # 僅記錄最近且距離小於200m的忽略事件
          if dist < 200 and dist < min_dist_found:
              closest_log_info = {"status": f"忽略(門檻>{SAFETY_OFFSET})", "dist": dist, "limit": cam_limit}
          continue

      # --- 物理急煞檢查 ---
      # 只在相機位於前方 (接近中) 時檢查
      cam_limit_ms = cam_limit * KPH_TO_MS
      if v_ego_ms > cam_limit_ms and is_front: 
          # 計算需要的減速度: a = (v_f^2 - v_i^2) / 2d
          required_decel = (cam_limit_ms**2 - v_ego_ms**2) / (2 * max(dist, 1.0))
          # 若需要減速度小於急煞門檻 (例如 -5 < -4.5)，則忽略
          if required_decel < EMERGENCY_DECEL:
              if dist < min_dist_found:
                  closest_log_info = {"status": f"忽略(急煞{required_decel:.1f})", "dist": dist, "limit": cam_limit}
              continue

      # --- 計算該相機的目標速度 ---
      if dist > limit_radius:
        # 若距離還很遠 (大於生效半徑)，目標為巡航速度 (不減速)
        target = v_cruise_kph
      else:
        # [關鍵邏輯] 動態保持距離與緩加速
        # 使用 np.interp 進行線性插值
        # X軸 (距離): [current_hold_dist, limit_radius]
        # Y軸 (速度): [cam_limit, v_cruise_kph]
        #
        # 意義：
        # 1. 當 dist < current_hold_dist (剛通過或極近): 目標速度 = cam_limit (維持限速)
        # 2. 當 dist 介於 hold 與 limit_radius 之間: 目標速度從 cam_limit 線性上升至 v_cruise_kph
        # 3. 當 dist > limit_radius: 上面 if 已處理，回歸巡航速度
        target = np.interp(dist, [current_hold_dist, limit_radius], [cam_limit, v_cruise_kph])
      
      # 若計算出的目標有效
      if math.isfinite(target):
        candidates.append(target)
        
        # 追蹤最近的一個有效相機，用於 UI 顯示與 Log
        if dist < min_dist_found:
            min_dist_found = dist
            active_camera_limit = cam_limit
            self.active_index = original_idx # 標記當前生效的相機 ID
            
            # 設定 Log 顯示狀態
            status_str = "介入中" if is_front else "通過|回速中"
            
            closest_log_info = {
                "dist": dist, 
                "limit": cam_limit, 
                "status": status_str, 
                "target": target
            }

    # ---------------------------------------------------
    # 4. 決策與輸出
    # ---------------------------------------------------
    # 從所有候選速度中選最低的 (最安全的)
    final_target_kph = min(candidates)
    # 確保不會超過原本設定的巡航速度
    final_target_kph = min(final_target_kph, v_cruise_kph)

    # Log 輸出邏輯
    if should_log and closest_log_info and closest_log_info["dist"] < search_radius:
        self.last_log_time = current_time
        status = closest_log_info["status"]
        limit = closest_log_info["limit"]
        dist = closest_log_info["dist"]
        
        # 判斷是否真的在進行速度控制 (目標比巡航低 1kph 以上)
        is_intervening = (final_target_kph < v_cruise_kph - 1.0)
        
        # 只有在介入中、被忽略或回速中時才印出 Log
        if is_intervening or "忽略" in status or "回速" in status:
            cloudlog.warning(f"SCDA {status}: 限速{limit:.0f} | 距離{dist:.0f}m | 目標{final_target_kph:.0f}kph")

    # 最終判斷是否為「啟動」狀態 (有找到相機且目標速度低於巡航速度)
    is_active = (min_dist_found < 9999.0 and final_target_kph < v_cruise_kph)
    
    # 回傳最終結果
    return {
        'target_speed': final_target_kph * KPH_TO_MS, # 轉回 m/s
        'distance': min_dist_found if is_active else None,
        'limit': active_camera_limit if is_active else None,
        'is_active': is_active
    }
  
  def get_statistics(self):
    """回傳統計資訊，用於除錯"""
    return {"cameras_loaded": len(self.cameras)}
