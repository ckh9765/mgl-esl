#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
sensitivity_defer_disable.py — DEFER_DISABLE_TAU 敏感度檢查
============================================================
目的:回應審稿疑慮「defer-disable 門檻 0.25 與 GT 門檻 0.25 相同,
是否為 test set leakage?」

方法:把 DEFER_DISABLE_TAU 換成 {0.15, 0.20, 0.25, 0.30, 關閉},
其餘協定與 generate_trace_v3.py 完全相同(N=2,7 個雜訊等級,
10 seeds,10 個門檻),觀察收益窗口的兩個特徵是否移動:
  - σ*(AUC 增益由負轉正的位置)
  - 峰值位置與峰值大小

判讀:
  - 若各設定下 σ* 與峰值幾乎不動 → 0.25 只是預算充裕區的
    工程性開關,與 GT 門檻的重合為巧合,窗口結構對此參數不敏感
    → 論文加一句敏感度聲明即可拆彈
  - 若窗口大幅位移 → 存在真實的耦合,投稿前必須處理

用法:
  python sensitivity_defer_disable.py [影片路徑]
  (預設讀同目錄的 VIRAT_S_050201_05_000890_000944.mp4)

輸出:sensitivity_defer_disable.csv + 終端機判決表
============================================================
"""

import os
import sys
import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:
    raise SystemExit("需要 opencv:pip install opencv-python(或 conda install opencv)")

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "VIRAT_S_050201_05_000890_000944.mp4"
TILE_SIZE = 64
SEEDS = 10
THRESHOLDS = np.linspace(0.10, 0.55, 10)
MAX_FRAMES = 500
C_b, C_p = 0.05, 0.02
WATCHLIST_META_COST = C_b * 0.05
GT_TAU = 0.25
NOISE_SIGMA_LIST = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
N_PERSIST = 2                                   # 論文主結果的 N
DD_TAU_LIST = [0.15, 0.20, 0.25, 0.30, None]    # None = 完全不關閉 defer


def process_video_to_p_values(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"無法開啟影片: {video_path}")
    ret, prev_frame = cap.read()
    if not ret:
        raise SystemExit("無法讀取第一幀")
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
        d = diff[:ty_n*TILE_SIZE, :tx_n*TILE_SIZE]
        d = d.reshape(ty_n, TILE_SIZE, tx_n, TILE_SIZE).sum(axis=(1, 3))
        all_p.append(np.clip(d / max_sad, 0, 1).ravel())
        prev_gray = gray
        fc += 1
    cap.release()
    return np.array(all_p)


def persistence_promote_vec(above, defer_mask, n_persist):
    T = above.shape[0]
    S = np.vstack([np.zeros((1, above.shape[1]), dtype=np.int32),
                   np.cumsum(above.astype(np.int32), axis=0)])
    promote = np.zeros_like(defer_mask)
    for t in range(T - 1):
        end = min(t + 1 + n_persist, T)
        need = end - (t + 1)
        ok = (S[end] - S[t+1]) == need
        promote[t] = defer_mask[t] & ok
    return promote


def auc_gain_per_seed(df, gmax):
    """回傳各 seed 的 (MGL−Binary) normalized AUC 差。"""
    gains = []
    for sd in sorted(df.seed.unique()):
        out = {}
        for st in ["Binary", "MGL"]:
            s = df[(df.strategy == st) & (df.seed == sd)] \
                .sort_values('total_bytes_mbit')
            out[st] = np.trapezoid(np.clip(s.recall, 0, 1),
                                   s.total_bytes_mbit / gmax)
        gains.append(out["MGL"] - out["Binary"])
    return np.array(gains)


if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        raise SystemExit(f"❌ 找不到影片:{VIDEO_PATH}\n用法:python {sys.argv[0]} <影片路徑>")

    p_matrix = process_video_to_p_values(VIDEO_PATH)
    gt_mask = p_matrix > GT_TAU
    total_gt = max(1, int(gt_mask.sum()))
    print(f"🚀 GT={total_gt} | DD_TAU 掃描 {DD_TAU_LIST} | N={N_PERSIST}\n")

    rows = []
    for dd_tau in DD_TAU_LIST:
        tag = "off" if dd_tau is None else f"{dd_tau:.2f}"
        for sigma in NOISE_SIGMA_LIST:
            for seed in range(SEEDS):
                rng = np.random.default_rng(seed)
                noisy_p = np.clip(p_matrix + rng.normal(0, sigma, p_matrix.shape), 0, 1)
                for tau_U in THRESHOLDS:
                    # Binary
                    bm = noisy_p >= tau_U
                    rows.append(dict(dd_tau=tag, noise_sigma=sigma, strategy="Binary",
                        seed=seed, threshold_val=float(tau_U),
                        total_bytes_mbit=float(bm.sum()*(C_b+C_p)),
                        recall=float((bm & gt_mask).sum()/total_gt)))
                    # MGL
                    tau_B = tau_U * 0.5
                    cm = noisy_p >= tau_U
                    defer_off = (dd_tau is not None) and (tau_U < dd_tau)
                    dm = (np.zeros_like(noisy_p, dtype=bool) if defer_off
                          else (noisy_p >= tau_B) & (noisy_p < tau_U))
                    pm = (np.zeros_like(dm) if defer_off
                          else persistence_promote_vec(noisy_p >= tau_B, dm, N_PERSIST))
                    bytes_ = (cm.sum()*(C_b+C_p) + pm.sum()*(C_b+C_p)
                              + dm.sum()*WATCHLIST_META_COST)
                    rec = ((cm & gt_mask).sum() + (pm & gt_mask).sum()) / total_gt
                    rows.append(dict(dd_tau=tag, noise_sigma=sigma, strategy="MGL",
                        seed=seed, threshold_val=float(tau_U),
                        total_bytes_mbit=float(bytes_),
                        recall=float(np.clip(rec, 0, 1))))
        print(f"  DD_TAU={tag} 完成")

    df = pd.DataFrame(rows)
    df.to_csv("sensitivity_defer_disable.csv", index=False)
    print("\n💾 sensitivity_defer_disable.csv 已輸出\n")

    # ===== 判決表 =====
    print("===== 收益窗口 vs DEFER_DISABLE_TAU(N=2, 增益為 MGL−Binary AUC)=====")
    header = f"{'σ':>6} |" + "".join(f"  DD={t if t else 'off':>5}" for t in
                                     ["0.15","0.20","0.25","0.30","off"])
    print(header)
    summary = {}
    for sigma in NOISE_SIGMA_LIST:
        line = f"{sigma:>6} |"
        for tag in ["0.15","0.20","0.25","0.30","off"]:
            d = df[(df.dd_tau == tag) & (df.noise_sigma == sigma)]
            gmax = d.total_bytes_mbit.max()
            g = auc_gain_per_seed(d, gmax)
            summary.setdefault(tag, []).append((sigma, g.mean()))
            line += f" {g.mean():>+7.4f}"
        print(line)

    print("\n===== 窗口特徵 =====")
    print(f"{'DD_TAU':>7} {'σ*(轉正)':>10} {'峰值位置':>9} {'峰值大小':>9}")
    for tag in ["0.15","0.20","0.25","0.30","off"]:
        arr = summary[tag]
        cross = next((s for s, g in arr if g > 0), None)
        peak_s, peak_g = max(arr, key=lambda x: x[1])
        print(f"{tag:>7} {str(cross):>10} {peak_s:>9} {peak_g:>+9.4f}")

    print("\n判讀:各 DD_TAU 下 σ* 與峰值位置若基本一致 → 窗口結構對此參數")
    print("     不敏感,0.25 與 GT 門檻重合為巧合,論文加一句聲明即可。")
    print("     若明顯位移 → 存在耦合,回報結果再議。")
