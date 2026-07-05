#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
generate_trace_v3.py — 雜訊範圍掃描 (Noise-Regime Sweep)
============================================================
研究問題:三階 defer 機制的收益是否取決於觀測不確定性?

v3 變更:
- NOISE_SIGMA 掃描 {0.02, 0.05, 0.10, 0.15, 0.20}
- 持續性驗證向量化(快約 50 倍)
- 內建對齊比較摘要:每個 (sigma, N) 直接印出 MGL vs Binary
  的對齊後平均差距與 normalized AUC,一眼看出收益區間
- 門檻網格改為 0.10–0.55(原 0.5 以上是死區)
- 輸出 noise_sweep.csv(完整)供後續正式分析
其餘機制與 v2 相同:真實時間冗餘性驗證、promote 計入完整成本。
============================================================
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd
import cv2

# ==========================================
# 參數設定
# ==========================================
VIDEO_PATH = "VIRAT_S_050201_05_000890_000944.mp4"
SALIENCE_PATH = "salience_virat.npy"   # 前處理抽出的 salience;存在則優先讀檔,與 mp4 脫鉤
TILE_SIZE = 64
SEEDS = 10
THRESHOLDS = np.linspace(0.10, 0.55, 10)
MAX_FRAMES = 500

C_b, C_p = 0.05, 0.02
WATCHLIST_META_COST = C_b * 0.05

GT_TAU = 0.25
NOISE_SIGMA_LIST = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
N_PERSIST_LIST = [2, 3, 5]
ADAPTIVE_DEFER_DISABLE = True
DEFER_DISABLE_TAU = 0.25


def process_video_to_p_values(video_path):
    """
    將影片轉換為p值矩陣
    
    Args:
        video_path (str): 影片路徑
        
    Returns:
        numpy.ndarray: p值矩陣 [時間, 瓦片數]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"無法開啟影片: {video_path}")
        
    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("無法讀取第一幀")
        
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    target_shape = (prev_gray.shape[1], prev_gray.shape[0])
    h, w = prev_gray.shape
    ty_n, tx_n = h // TILE_SIZE, w // TILE_SIZE
    print(f"🎬 影片載入: {w}x{h}, tiles {tx_n}x{ty_n}")

    all_p, fc = [], 0
    max_sad = (TILE_SIZE * TILE_SIZE * 255) * 0.1
    
    while True:
        ret, frame = cap.read()
        if not ret or fc >= MAX_FRAMES:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape != prev_gray.shape:
            gray = cv2.resize(gray, target_shape)
            
        diff = cv2.absdiff(gray, prev_gray).astype(np.float64)
        
        # 向量化 tile SAD
        d = diff[:ty_n*TILE_SIZE, :tx_n*TILE_SIZE]
        d = d.reshape(ty_n, TILE_SIZE, tx_n, TILE_SIZE).sum(axis=(1, 3))
        all_p.append(np.clip(d / max_sad, 0, 1).ravel())
        
        prev_gray = gray
        fc += 1
    
    cap.release()
    return np.array(all_p)  # [T, n_tiles]


def _sha256_of(arr):
    """對陣列原始位元組+形狀+dtype 取 SHA256(與 preprocess_salience 相同演算法)。"""
    m = hashlib.sha256()
    m.update(str(arr.shape).encode())
    m.update(str(arr.dtype).encode())
    m.update(np.ascontiguousarray(arr).tobytes())
    return m.hexdigest()


def load_salience(salience_path=SALIENCE_PATH, video_path=VIDEO_PATH):
    """
    取得 p_matrix(salience)。優先讀 preprocess_salience.py 抽出的 .npy(與 mp4 脫鉤);
    找不到才 fallback 現算 mp4。若同名 .meta.json 存在,核對 SHA256 確保檔案未損毀/竄改。
    讀檔版與現算版位元一致(已由 preprocess_salience.py verify 證明),故承重數字不變。
    """
    if os.path.exists(salience_path):
        p = np.load(salience_path)
        print(f"📂 讀取 salience: {salience_path}  shape={p.shape} dtype={p.dtype}")
        meta_path = os.path.splitext(salience_path)[0] + ".meta.json"
        if os.path.exists(meta_path):
            meta = json.load(open(meta_path, encoding="utf-8"))
            h = _sha256_of(p)
            if h == meta.get("sha256"):
                print(f"🔑 SHA256 核對通過:{h[:16]}…")
            else:
                raise SystemExit(
                    f"❌ salience SHA256 與 meta 不符,檔案可能損毀:\n"
                    f"   file={h}\n   meta={meta.get('sha256')}")
        return p
    if not os.path.exists(video_path):
        raise SystemExit(f"❌ 找不到 salience({salience_path})也找不到影片({video_path})")
    print(f"⚠ 找不到 {salience_path},fallback 現算 mp4:{video_path}")
    return process_video_to_p_values(video_path)


def persistence_promote_vec(above, defer_mask, n_persist):
    """
    向量化持續性驗證：defer[t] 且 above[t+1..t+n] 全為真 → promote
    
    Args:
        above (numpy.ndarray): 布林矩陣顯示各時間點是否高於門檻 [時間, 瓦片數]
        defer_mask (numpy.ndarray): 布林矩陣顯示要延遲的瓦片 [時間, 瓦片數]
        n_persist (int): 持續驗證的幀數
        
    Returns:
        numpy.ndarray: promote 記錄 [時間, 瓦片數]
    """
    T = above.shape[0]
    S = np.vstack([np.zeros((1, above.shape[1]), dtype=np.int32),
                   np.cumsum(above.astype(np.int32), axis=0)])  # [T+1, n]
    
    promote = np.zeros_like(defer_mask)
    
    for t in range(T - 1):
        end = min(t + 1 + n_persist, T)
        need = end - (t + 1)
        ok = (S[end] - S[t+1]) == need
        promote[t] = defer_mask[t] & ok
        
    return promote


def aligned_summary(df):
    """
    快速對齊比較：以 Binary 各操作點預算為基準，內插 MGL recall
    
    Args:
        df (pandas.DataFrame): 資料框包含各策略的運行結果
        
    Returns:
        pandas.DataFrame: 對齊後的平均差距
    """
    out = []
    
    for seed in sorted(df.seed.unique()):
        # 獲取 Binary 策略數據並排序
        b = df[(df.strategy == "Binary") & (df.seed == seed)].sort_values("total_bytes_mbit")
        
        # 獲取 MGL 策略數據並聚合平均 recall
        m = (df[(df.strategy == "MGL") & (df.seed == seed)]
             .groupby("total_bytes_mbit", as_index=False)["recall"].mean()
             .sort_values("total_bytes_mbit"))
        
        if len(m) < 2:
            continue
            
        lo, hi = m.total_bytes_mbit.min(), m.total_bytes_mbit.max()
        
        for _, r in b.iterrows():
            x = r.total_bytes_mbit
            if lo <= x <= hi and x > 0:
                pred = np.interp(x, m.total_bytes_mbit, m.recall)
                out.append({
                    "seed": seed, 
                    "budget": x,
                    "delta_pp": (pred - r.recall) * 100
                })
                
    return pd.DataFrame(out)


def auc_gain(df):
    """
    計算 AUC 增益
    
    Args:
        df (pandas.DataFrame): 資料框包含各策略的運行結果
        
    Returns:
        float: MGL 相對於 Binary 的 AUC 增益
    """
    gmax = df.total_bytes_mbit.max()
    res = {}
    
    for st in ["Binary", "MGL"]:
        vals = []
        
        for sd in sorted(df.seed.unique()):
            sub = df[(df.strategy == st) & (df.seed == sd)].sort_values("total_bytes_mbit")
            vals.append(np.trapezoid(np.clip(sub.recall, 0, 1),
                                     sub.total_bytes_mbit / gmax))
            
        res[st] = np.mean(vals)
        
    return res["MGL"] - res["Binary"]


if __name__ == "__main__":
    # 取得 salience:優先讀檔(與 mp4 脫鉤),找不到才現算 mp4
    p_matrix = load_salience()
    gt_mask = p_matrix > GT_TAU
    total_gt = max(1, int(gt_mask.sum()))
    print(f"🚀 GT salient tiles = {total_gt} | 掃描 σ={NOISE_SIGMA_LIST}, N={N_PERSIST_LIST}\n")

    all_rows = []
    
    # 遍歷所有雜訊參數和種子
    for sigma in NOISE_SIGMA_LIST:
        for seed in range(SEEDS):
            rng = np.random.default_rng(seed)
            noisy_p = np.clip(p_matrix + rng.normal(0, sigma, p_matrix.shape), 0, 1)

            # 遍歷所有門檻值
            for t_idx, tau_U in enumerate(THRESHOLDS):
                bin_mask = noisy_p >= tau_U
                
                all_rows.append({
                    "noise_sigma": sigma, 
                    "n_persist": 0,
                    "strategy": "Binary", 
                    "seed": seed, 
                    "threshold_id": t_idx,
                    "threshold_val": float(tau_U),
                    "total_bytes_mbit": float(bin_mask.sum() * (C_b + C_p)),
                    "recall": float((bin_mask & gt_mask).sum() / total_gt)
                })

                # 計算 MGL 策略的邏輯
                tau_B = tau_U * 0.5
                compute_mask = bin_mask
                defer_off = ADAPTIVE_DEFER_DISABLE and (tau_U < DEFER_DISABLE_TAU)
                defer_mask = (np.zeros_like(noisy_p, dtype=bool) if defer_off
                              else (noisy_p >= tau_B) & (noisy_p < tau_U))
                above_B = noisy_p >= tau_B

                # 處理不同 persist 值
                for n_persist in N_PERSIST_LIST:
                    promote = (np.zeros_like(defer_mask) if defer_off
                               else persistence_promote_vec(above_B, defer_mask, n_persist))
                    
                    bytes_ = (compute_mask.sum() * (C_b + C_p)
                              + promote.sum() * (C_b + C_p)
                              + defer_mask.sum() * WATCHLIST_META_COST)
                              
                    rec = ((compute_mask & gt_mask).sum()
                           + (promote & gt_mask).sum()) / total_gt
                    
                    all_rows.append({
                        "noise_sigma": sigma, 
                        "n_persist": n_persist,
                        "strategy": "MGL", 
                        "seed": seed, 
                        "threshold_id": t_idx,
                        "threshold_val": float(tau_U),
                        "total_bytes_mbit": float(bytes_),
                        "recall": float(np.clip(rec, 0, 1))
                    })
        print(f"  σ={sigma} 完成")

    # 保存結果到 CSV
    df = pd.DataFrame(all_rows)
    df.to_csv("noise_sweep.csv", index=False)
    print("\n💾 noise_sweep.csv 已輸出\n")

    # ===== 對齊比較摘要 =====
    print("===== 對齊比較摘要 (MGL − Binary) =====")
    print(f"{'sigma':>6} {'N':>3} {'低預算Δpp':>12} {'全域Δpp':>10} {'AUC增益':>10}")
    
    for sigma in NOISE_SIGMA_LIST:
        for n in N_PERSIST_LIST:
            sub = df[(df.noise_sigma == sigma)
                     & ((df.strategy == "Binary") | (df.n_persist == n))]
            
            al = aligned_summary(sub)
            
            if al.empty:
                continue
                
            low = al[al.budget < 10].delta_pp.mean()
            allm = al.delta_pp.mean()
            ag = auc_gain(sub)
            
            print(f"{sigma:>6} {n:>3} {low:>+12.2f} {allm:>+10.2f} {ag:>+10.4f}")
    
    print("\n判讀:若 Δpp / AUC增益 隨 σ 上升由負轉正,即為『高不確定性收益區間』的證據。")