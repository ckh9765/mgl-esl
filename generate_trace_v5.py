#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
generate_trace_v5.py — 三方部署地圖:加入事件偵測延遲
============================================================
v5 新增(相對 v4):
- 事件(Event)定義:GT 中每個 tile 的「連續顯著區段」
  (clean p > GT_TAU 連續 ≥ MIN_EVENT_LEN 幀)為一個事件。
- 偵測延遲(Detection Latency):事件起始幀 → 該 tile 首次被
  策略「放行」(Binary/Smooth: 超過門檻;MGL: Admit 或 Promote)
  的幀數差。Promote 計入 N 幀驗證延遲。事件結束前未放行 = Miss。
- 輸出三維評估:每個 (σ, W/N, τ_U) 之
  recall、bytes、median/mean detection latency、miss rate。
- 摘要表:在「召回率對齊」的操作點上比較三策略的偵測延遲,
  檢驗核心假說:MGL 對明確事件零延遲、Smooth 全域延遲。

評估邏輯與 v4 完全一致(同成本模型、同公平性原則),
只是多記錄了時間維度。輸出:
- detection_latency_v5.csv   (事件層級原始紀錄)
- summary_v5.csv             (操作點層級彙總)
- 終端機判決表
============================================================
"""

import os
import numpy as np
import pandas as pd
import cv2

VIDEO_PATH = "VIRAT_S_050201_05_000890_000944.mp4"
TILE_SIZE = 64
SEEDS = 10
THRESHOLDS = np.linspace(0.10, 0.55, 10)
MAX_FRAMES = 500

C_b, C_p = 0.05, 0.02
WATCHLIST_META_COST = C_b * 0.05

GT_TAU = 0.25
MIN_EVENT_LEN = 3            # 連續 ≥3 幀才算一個事件(濾掉單幀雜訊)
NOISE_SIGMA_LIST = [0.02, 0.10, 0.15, 0.20, 0.30]   # 精選 5 級,控制執行時間
WINDOW_LIST = [2, 3, 5]
ADAPTIVE_DEFER_DISABLE = True
DEFER_DISABLE_TAU = 0.25
MS_PER_FRAME = 33.3          # 30fps,延遲可換算 ms


# ---------- 影片處理(與 v3/v4 相同) ----------
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
        d = diff[:ty_n*TILE_SIZE, :tx_n*TILE_SIZE]
        d = d.reshape(ty_n, TILE_SIZE, tx_n, TILE_SIZE).sum(axis=(1, 3))
        all_p.append(np.clip(d / max_sad, 0, 1).ravel())
        prev_gray = gray
        fc += 1
    cap.release()
    return np.array(all_p)


def extract_events(gt_mask, min_len=MIN_EVENT_LEN):
    """從 GT 遮罩抽出事件:(tile, start, end) 連續區段,長度 ≥ min_len。"""
    T, n_tiles = gt_mask.shape
    events = []
    for i in range(n_tiles):
        col = gt_mask[:, i]
        t = 0
        while t < T:
            if col[t]:
                s = t
                while t < T and col[t]:
                    t += 1
                if t - s >= min_len:
                    events.append((i, s, t - 1))
            else:
                t += 1
    return events


def persistence_promote_vec(above, defer_mask, n_persist):
    """回傳 promote_decision_frame:tile 在 frame t defer、t+n 驗證通過,
    則「實際放行時間」為 t + n_persist(計入驗證延遲)。
    輸出與 defer_mask 同形的 bool(於原 defer 幀標記),延遲在事件統計時加上。"""
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


def causal_moving_avg(noisy_p, w):
    T = noisy_p.shape[0]
    S = np.vstack([np.zeros((1, noisy_p.shape[1])), np.cumsum(noisy_p, axis=0)])
    out = np.empty_like(noisy_p)
    for t in range(T):
        s = max(0, t - w + 1)
        out[t] = (S[t+1] - S[s]) / (t + 1 - s)
    return out


def event_latencies(events, admit_time):
    """admit_time: [T, n_tiles] 整數陣列,值 = 該幀該 tile 的「實際放行時間」
    (未放行 = 大數)。回傳每個事件的偵測延遲(幀)或 NaN(miss)。"""
    lats = []
    for (tile, s, e) in events:
        # 事件期間內,該 tile 任一幀觸發的最早實際放行時間
        window = admit_time[s:e+1, tile]
        earliest = window.min()
        if earliest <= e:               # 必須在事件結束前完成放行
            lats.append(earliest - s)
        else:
            lats.append(np.nan)
    return np.array(lats, dtype=float)


if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        raise SystemExit(f"❌ 找不到影片:{VIDEO_PATH}")

    p_matrix = process_video_to_p_values(VIDEO_PATH)
    T, n_tiles = p_matrix.shape
    gt_mask = p_matrix > GT_TAU
    total_gt = max(1, int(gt_mask.sum()))
    events = extract_events(gt_mask)
    print(f"🚀 GT tiles={total_gt} | 事件數={len(events)} | σ={NOISE_SIGMA_LIST}\n")

    BIG = 10**6
    sum_rows = []

    for sigma in NOISE_SIGMA_LIST:
        for seed in range(SEEDS):
            rng = np.random.default_rng(seed)
            noisy_p = np.clip(p_matrix + rng.normal(0, sigma, p_matrix.shape), 0, 1)
            smooth = {w: causal_moving_avg(noisy_p, w) for w in WINDOW_LIST}
            frame_idx = np.arange(T)[:, None] * np.ones((1, n_tiles), dtype=int)

            for t_idx, tau_U in enumerate(THRESHOLDS):
                # ---------- Binary ----------
                bm = noisy_p >= tau_U
                at = np.where(bm, frame_idx, BIG)
                lats = event_latencies(events, at)
                sum_rows.append(dict(noise_sigma=sigma, strategy="Binary", win=0,
                    seed=seed, threshold_val=float(tau_U),
                    total_bytes_mbit=float(bm.sum()*(C_b+C_p)),
                    recall=float((bm & gt_mask).sum()/total_gt),
                    med_latency=float(np.nanmedian(lats)),
                    miss_rate=float(np.isnan(lats).mean())))

                # ---------- Smooth ----------
                for w in WINDOW_LIST:
                    sm = smooth[w] >= tau_U
                    at = np.where(sm, frame_idx, BIG)
                    lats = event_latencies(events, at)
                    sum_rows.append(dict(noise_sigma=sigma, strategy="Smooth", win=w,
                        seed=seed, threshold_val=float(tau_U),
                        total_bytes_mbit=float(sm.sum()*(C_b+C_p)),
                        recall=float((sm & gt_mask).sum()/total_gt),
                        med_latency=float(np.nanmedian(lats)),
                        miss_rate=float(np.isnan(lats).mean())))

                # ---------- MGL ----------
                tau_B = tau_U * 0.5
                cm = noisy_p >= tau_U
                defer_off = ADAPTIVE_DEFER_DISABLE and (tau_U < DEFER_DISABLE_TAU)
                dm = (np.zeros_like(noisy_p, dtype=bool) if defer_off
                      else (noisy_p >= tau_B) & (noisy_p < tau_U))
                above_B = noisy_p >= tau_B
                for n in WINDOW_LIST:
                    pm = (np.zeros_like(dm) if defer_off
                          else persistence_promote_vec(above_B, dm, n))
                    # Admit 零延遲;Promote 於 defer 幀 + n 幀後放行
                    at = np.full((T, n_tiles), BIG, dtype=int)
                    at[cm] = frame_idx[cm]
                    promote_time = frame_idx + n
                    upd = pm & (promote_time < at)
                    at[upd] = promote_time[upd]
                    lats = event_latencies(events, at)
                    bytes_ = (cm.sum()*(C_b+C_p) + pm.sum()*(C_b+C_p)
                              + dm.sum()*WATCHLIST_META_COST)
                    rec = ((cm & gt_mask).sum() + (pm & gt_mask).sum()) / total_gt
                    sum_rows.append(dict(noise_sigma=sigma, strategy="MGL", win=n,
                        seed=seed, threshold_val=float(tau_U),
                        total_bytes_mbit=float(bytes_),
                        recall=float(np.clip(rec, 0, 1)),
                        med_latency=float(np.nanmedian(lats)),
                        miss_rate=float(np.isnan(lats).mean())))
        print(f"  σ={sigma} 完成")

    df = pd.DataFrame(sum_rows)
    df.to_csv("summary_v5.csv", index=False)
    print("\n💾 summary_v5.csv 已輸出\n")

    # ===== 判決表:召回率對齊下的延遲比較 =====
    # 對每個 (σ, 策略, win):找出 recall 最接近 0.7 的操作點,報其延遲與成本
    TARGET_RECALL = 0.7
    print(f"===== 召回率對齊判決表 (各策略取 recall 最接近 {TARGET_RECALL} 的操作點) =====")
    print(f"{'sigma':>6} {'strategy':>8} {'W/N':>4} {'recall':>7} {'延遲中位數(幀)':>14} "
          f"{'延遲(ms)':>9} {'miss率':>7} {'bytes':>9}")
    for sigma in NOISE_SIGMA_LIST:
        d = df[df.noise_sigma == sigma]
        combos = [("Binary", 0)] + [("Smooth", w) for w in WINDOW_LIST] \
                 + [("MGL", n) for n in WINDOW_LIST]
        for st, w in combos:
            sub = d[(d.strategy == st) & (d.win == w)] \
                .groupby('threshold_val')[['recall', 'med_latency',
                                           'miss_rate', 'total_bytes_mbit']].mean()
            if sub.empty or sub.recall.max() < 0.3:
                continue
            best = (sub.recall - TARGET_RECALL).abs().idxmin()
            r = sub.loc[best]
            print(f"{sigma:>6} {st:>8} {w:>4} {r.recall:>7.3f} "
                  f"{r.med_latency:>14.2f} {r.med_latency*MS_PER_FRAME:>9.1f} "
                  f"{r.miss_rate:>7.3f} {r.total_bytes_mbit:>9.2f}")
        print("  " + "-"*70)
    print("\n判讀:同等召回率下,比較『延遲中位數』與『miss率』。")
    print("     假說:MGL 延遲 ≈ Binary(明確事件零延遲直送),Smooth 延遲隨 W 增加;")
    print("     若成立 → 三方部署地圖論述成立(安全關鍵場景選 Ternary)。")
