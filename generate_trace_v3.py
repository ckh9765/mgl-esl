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
import numpy as np
import pandas as pd
import cv2

# ==========================================
# 參數
# ==========================================
VIDEO_PATH = "VIRAT_S_050201_05_000890_000944.mp4"
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


def persistence_promote_vec(above, defer_mask, n_persist):
    """向量化持續性驗證:defer[t] 且 above[t+1..t+n] 全為真 → promote。
    above: bool [T, n];影片尾端不足 n 幀者以可得幀驗證,無後續幀則 drop。"""
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
    """快速對齊比較:以 Binary 各操作點預算為基準,內插 MGL recall。"""
    out = []
    for seed in sorted(df.seed.unique()):
        b = df[(df.strategy == "Binary") & (df.seed == seed)].sort_values("total_bytes_mbit")
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
                out.append({"seed": seed, "budget": x,
                            "delta_pp": (pred - r.recall) * 100})
    return pd.DataFrame(out)


def auc_gain(df):
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
    if not os.path.exists(VIDEO_PATH):
        raise SystemExit(f"❌ 找不到影片:{VIDEO_PATH}")

    p_matrix = process_video_to_p_values(VIDEO_PATH)
    gt_mask = p_matrix > GT_TAU
    total_gt = max(1, int(gt_mask.sum()))
    print(f"🚀 GT salient tiles = {total_gt} | 掃描 σ={NOISE_SIGMA_LIST}, N={N_PERSIST_LIST}\n")

    all_rows = []
    for sigma in NOISE_SIGMA_LIST:
        for seed in range(SEEDS):
            rng = np.random.default_rng(seed)
            noisy_p = np.clip(p_matrix + rng.normal(0, sigma, p_matrix.shape), 0, 1)

            for t_idx, tau_U in enumerate(THRESHOLDS):
                bin_mask = noisy_p >= tau_U
                all_rows.append({
                    "noise_sigma": sigma, "n_persist": 0,
                    "strategy": "Binary", "seed": seed, "threshold_id": t_idx,
                    "threshold_val": float(tau_U),
                    "total_bytes_mbit": float(bin_mask.sum() * (C_b + C_p)),
                    "recall": float((bin_mask & gt_mask).sum() / total_gt)})

                tau_B = tau_U * 0.5
                compute_mask = bin_mask
                defer_off = ADAPTIVE_DEFER_DISABLE and (tau_U < DEFER_DISABLE_TAU)
                defer_mask = (np.zeros_like(noisy_p, dtype=bool) if defer_off
                              else (noisy_p >= tau_B) & (noisy_p < tau_U))
                above_B = noisy_p >= tau_B

                for n_persist in N_PERSIST_LIST:
                    promote = (np.zeros_like(defer_mask) if defer_off
                               else persistence_promote_vec(above_B, defer_mask, n_persist))
                    bytes_ = (compute_mask.sum() * (C_b + C_p)
                              + promote.sum() * (C_b + C_p)
                              + defer_mask.sum() * WATCHLIST_META_COST)
                    rec = ((compute_mask & gt_mask).sum()
                           + (promote & gt_mask).sum()) / total_gt
                    all_rows.append({
                        "noise_sigma": sigma, "n_persist": n_persist,
                        "strategy": "MGL", "seed": seed, "threshold_id": t_idx,
                        "threshold_val": float(tau_U),
                        "total_bytes_mbit": float(bytes_),
                        "recall": float(np.clip(rec, 0, 1))})
        print(f"  σ={sigma} 完成")

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
