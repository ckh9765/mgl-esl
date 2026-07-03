#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
final_robustness_suite.py — 投稿前最終穩健性實驗(三合一)
============================================================
回應最終審稿意見的三項關鍵質疑,一次跑完:

[A] AR(1) 時間相關雜訊(生死實驗)
    質疑:q^N 論證依賴逐幀獨立雜訊;真實雜訊有時間相關性,
    相關性會削弱持續性驗證的辨別力,收益窗口可能消失。
    方法:η_t = ρ·η_{t−1} + √(1−ρ²)·ε_t,ε~N(0,σ²)
    (邊際變異數固定為 σ²,確保與 i.i.d. 情況公平對照)
    掃 ρ ∈ {0, 0.3, 0.5, 0.7}。
    判讀:窗口(正增益區)在 ρ>0 下是否存活、如何縮放。

[B] GT_TAU 敏感度
    質疑:GT 門檻 0.25 與 defer-disable 門檻重合,GT 門檻
    本身也該掃。掃 GT_TAU ∈ {0.20, 0.25, 0.30}。

[C] Watchlist 中繼成本比
    質疑:成本常數任意。解析事實:C_b+C_p 為三策略共同乘數,
    等比縮放不改變 normalized AUC;唯一結構性參數是中繼成本比。
    掃 META_RATIO ∈ {0.02, 0.05, 0.10}(× C_b)。

其餘協定與 generate_trace_v3 相同(N=2、10 seeds、10 門檻)。
輸出:三張判決表 + robustness_suite.csv
用法:python final_robustness_suite.py [影片路徑]
============================================================
"""

import os
import sys
import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:
    raise SystemExit("需要 opencv:pip install opencv-python")

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "VIRAT_S_050201_05_000890_000944.mp4"
TILE_SIZE = 64
SEEDS = 10
THRESHOLDS = np.linspace(0.10, 0.55, 10)
MAX_FRAMES = 500
C_b, C_p = 0.05, 0.02
GT_TAU_DEFAULT = 0.25
META_DEFAULT = 0.05
NOISE_SIGMA_LIST = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
N_PERSIST = 2
DD_TAU = 0.25

RHO_LIST = [0.0, 0.3, 0.5, 0.7]          # [A]
GT_LIST = [0.20, 0.25, 0.30]             # [B]
META_LIST = [0.02, 0.05, 0.10]           # [C]


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"無法開啟影片: {video_path}")
    ret, prev = cap.read()
    if not ret:
        raise SystemExit("無法讀取第一幀")
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    tshape = (prev_gray.shape[1], prev_gray.shape[0])
    h, w = prev_gray.shape
    ty, tx = h // TILE_SIZE, w // TILE_SIZE
    print(f"🎬 影片載入: {w}x{h}, tiles {tx}x{ty}")
    all_p, fc = [], 0
    max_sad = (TILE_SIZE * TILE_SIZE * 255) * 0.1
    while True:
        ret, frame = cap.read()
        if not ret or fc >= MAX_FRAMES:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if g.shape != prev_gray.shape:
            g = cv2.resize(g, tshape)
        d = cv2.absdiff(g, prev_gray).astype(np.float64)[:ty*TILE_SIZE, :tx*TILE_SIZE]
        d = d.reshape(ty, TILE_SIZE, tx, TILE_SIZE).sum(axis=(1, 3))
        all_p.append(np.clip(d / max_sad, 0, 1).ravel())
        prev_gray = g
        fc += 1
    cap.release()
    return np.array(all_p)


def gen_noise_ar1(shape, sigma, rho, rng):
    """AR(1) 時間相關雜訊,邊際變異數 = σ²。ρ=0 即 i.i.d.。"""
    T, n = shape
    eta = np.empty(shape)
    eta[0] = rng.normal(0, sigma, n)
    if rho == 0:
        eta[1:] = rng.normal(0, sigma, (T-1, n))
        return eta
    innov_scale = sigma * np.sqrt(1 - rho**2)
    for t in range(1, T):
        eta[t] = rho * eta[t-1] + rng.normal(0, innov_scale, n)
    return eta


def promote_vec(above, defer_mask, n_persist):
    T = above.shape[0]
    S = np.vstack([np.zeros((1, above.shape[1]), dtype=np.int32),
                   np.cumsum(above.astype(np.int32), axis=0)])
    out = np.zeros_like(defer_mask)
    for t in range(T - 1):
        end = min(t + 1 + n_persist, T)
        need = end - (t + 1)
        out[t] = defer_mask[t] & ((S[end] - S[t+1]) == need)
    return out


def run_condition(p_matrix, gt_mask, total_gt, sigma, seed, rho, meta_ratio):
    """單一 (σ, seed, ρ, meta) 條件,回傳 Binary/MGL 兩策略的曲線列。"""
    rng = np.random.default_rng(seed)
    eta = gen_noise_ar1(p_matrix.shape, sigma, rho, rng)
    noisy = np.clip(p_matrix + eta, 0, 1)
    meta_cost = C_b * meta_ratio
    rows = []
    for tau_U in THRESHOLDS:
        bm = noisy >= tau_U
        rows.append(("Binary", float(tau_U),
                     float(bm.sum()*(C_b+C_p)),
                     float((bm & gt_mask).sum()/total_gt)))
        tau_B = tau_U * 0.5
        cm = noisy >= tau_U
        off = tau_U < DD_TAU
        dm = (np.zeros_like(noisy, dtype=bool) if off
              else (noisy >= tau_B) & (noisy < tau_U))
        pm = (np.zeros_like(dm) if off
              else promote_vec(noisy >= tau_B, dm, N_PERSIST))
        by = cm.sum()*(C_b+C_p) + pm.sum()*(C_b+C_p) + dm.sum()*meta_cost
        rc = ((cm & gt_mask).sum() + (pm & gt_mask).sum()) / total_gt
        rows.append(("MGL", float(tau_U), float(by), float(np.clip(rc, 0, 1))))
    return rows


def gain_table(df):
    """df: 單一實驗組。回傳 {σ: mean paired AUC gain}。"""
    out = {}
    for sg in sorted(df.noise_sigma.unique()):
        d = df[df.noise_sigma == sg]
        gmax = d.total_bytes_mbit.max()
        gains = []
        for sd in sorted(d.seed.unique()):
            a = {}
            for st in ["Binary", "MGL"]:
                s = d[(d.strategy == st) & (d.seed == sd)].sort_values('total_bytes_mbit')
                a[st] = np.trapezoid(np.clip(s.recall, 0, 1), s.total_bytes_mbit/gmax)
            gains.append(a["MGL"] - a["Binary"])
        out[sg] = float(np.mean(gains))
    return out


def print_verdict(title, col_labels, tables):
    print(f"\n===== {title} =====")
    print(f"{'σ':>6} |" + "".join(f" {c:>9}" for c in col_labels))
    for sg in NOISE_SIGMA_LIST:
        print(f"{sg:>6} |" + "".join(f" {tables[c].get(sg, float('nan')):>+9.4f}"
                                     for c in col_labels))
    print(f"{'σ*':>6} |" + "".join(
        f" {str(next((s for s in NOISE_SIGMA_LIST if tables[c].get(s,-1)>0), None)):>9}"
        for c in col_labels))
    print(f"{'峰值σ':>6} |" + "".join(
        f" {max(tables[c], key=tables[c].get):>9}" for c in col_labels))
    print(f"{'峰值':>6} |" + "".join(
        f" {max(tables[c].values()):>+9.4f}" for c in col_labels))


if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        raise SystemExit(f"❌ 找不到影片:{VIDEO_PATH}")

    p_matrix = process_video(VIDEO_PATH)
    all_rows = []

    # ===== [A] AR(1) 相關雜訊(GT=0.25, meta=0.05)=====
    gt = p_matrix > GT_TAU_DEFAULT
    tg = max(1, int(gt.sum()))
    print(f"\n[A] AR(1) 相關雜訊掃描 ρ={RHO_LIST}(GT tiles={tg})")
    tabA = {}
    for rho in RHO_LIST:
        rows = []
        for sg in NOISE_SIGMA_LIST:
            for sd in range(SEEDS):
                for st, tv, by, rc in run_condition(p_matrix, gt, tg, sg, sd, rho, META_DEFAULT):
                    rows.append(dict(part="A", param=f"rho={rho}", noise_sigma=sg,
                                     strategy=st, seed=sd, threshold_val=tv,
                                     total_bytes_mbit=by, recall=rc))
        df = pd.DataFrame(rows)
        all_rows.append(df)
        tabA[f"ρ={rho}"] = gain_table(df)
        print(f"  ρ={rho} 完成")
    print_verdict("[A] MGL−Binary AUC 增益 vs 雜訊時間相關性 ρ",
                  [f"ρ={r}" for r in RHO_LIST], tabA)

    # ===== [B] GT_TAU 敏感度(i.i.d., meta=0.05)=====
    print(f"\n[B] GT_TAU 掃描 {GT_LIST}")
    tabB = {}
    for g in GT_LIST:
        gt_m = p_matrix > g
        tg_m = max(1, int(gt_m.sum()))
        rows = []
        for sg in NOISE_SIGMA_LIST:
            for sd in range(SEEDS):
                for st, tv, by, rc in run_condition(p_matrix, gt_m, tg_m, sg, sd, 0.0, META_DEFAULT):
                    rows.append(dict(part="B", param=f"GT={g}", noise_sigma=sg,
                                     strategy=st, seed=sd, threshold_val=tv,
                                     total_bytes_mbit=by, recall=rc))
        df = pd.DataFrame(rows)
        all_rows.append(df)
        tabB[f"GT={g}"] = gain_table(df)
        print(f"  GT_TAU={g} 完成 (GT tiles={tg_m})")
    print_verdict("[B] MGL−Binary AUC 增益 vs GT 門檻",
                  [f"GT={g}" for g in GT_LIST], tabB)

    # ===== [C] 中繼成本比(i.i.d., GT=0.25)=====
    print(f"\n[C] META_RATIO 掃描 {META_LIST}")
    tabC = {}
    for m in META_LIST:
        rows = []
        for sg in NOISE_SIGMA_LIST:
            for sd in range(SEEDS):
                for st, tv, by, rc in run_condition(p_matrix, gt, tg, sg, sd, 0.0, m):
                    rows.append(dict(part="C", param=f"meta={m}", noise_sigma=sg,
                                     strategy=st, seed=sd, threshold_val=tv,
                                     total_bytes_mbit=by, recall=rc))
        df = pd.DataFrame(rows)
        all_rows.append(df)
        tabC[f"m={m}"] = gain_table(df)
        print(f"  META_RATIO={m} 完成")
    print_verdict("[C] MGL−Binary AUC 增益 vs watchlist 中繼成本比",
                  [f"m={m}" for m in META_LIST], tabC)

    pd.concat(all_rows).to_csv("robustness_suite.csv", index=False)
    print("\n💾 robustness_suite.csv 已輸出")
    print("""
判讀指南:
[A] 生死判決——看各 ρ 欄的正增益區:
    ρ=0.3~0.5 仍有正窗口 → 核心結論在中度相關下存活,入稿補強
    ρ≥0.3 全負 → 窗口僅存在於低相關雜訊,Limitations 須明確限定
[B] 各 GT 欄的窗口結構(σ*、峰值位置)一致 → GT 門檻選擇非結果驅動
[C] 各 meta 欄結構一致 → 成本模型結論穩健(C_b+C_p 為共同乘數,
    解析上不影響 normalized AUC;此處驗證唯一結構性參數)
""")
