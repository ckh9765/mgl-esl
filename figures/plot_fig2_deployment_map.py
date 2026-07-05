#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
plot_fig2_deployment_map.py
============================================================
讀取 summary_v5.csv(由 generate_trace_v5.py 產生),重現論文 Fig. 2:
三方部署地圖 — (a) 頻寬效率 AUC vs σ、(b) 召回率對齊下的事件漏失率 vs σ。

可重現鏈:
  generate_trace_v5.py → summary_v5.csv → 本腳本 → Fig3_deployment_map.png

(註:檔名沿用論文中的 Fig3_deployment_map.png,因論文編號演變過;
 在最終投稿稿中它是 Fig. 2。)

用法:
  python plot_fig2_deployment_map.py [summary_v5.csv 路徑]
============================================================
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "summary_v5.csv"
OUT_PATH = "Fig3_deployment_map.png"
WIN_LIST = [2, 3, 5]
TARGET_RECALL = 0.7


def auc_per(df, st, w):
    gmax = df.total_bytes_mbit.max()
    vals = []
    for sd in sorted(df.seed.unique()):
        s = df[(df.strategy == st) & (df.win == w) & (df.seed == sd)] \
            .sort_values("total_bytes_mbit")
        vals.append(np.trapezoid(np.clip(s.recall, 0, 1),
                                 s.total_bytes_mbit / gmax))
    return np.array(vals)


def aligned_miss(df, st, w, target=TARGET_RECALL):
    """每個 seed 取 recall 最接近 target 的操作點之 miss_rate,回平均(%)。"""
    vals = []
    for sd in sorted(df.seed.unique()):
        sub = df[(df.strategy == st) & (df.win == w) & (df.seed == sd)]
        g = sub.groupby("threshold_val")[["recall", "miss_rate"]].mean()
        if g.empty or g.recall.max() < 0.4:
            continue
        best = (g.recall - target).abs().idxmin()
        vals.append(g.loc[best, "miss_rate"])
    return np.mean(vals) * 100 if vals else np.nan


def main():
    df = pd.read_csv(CSV_PATH)
    sigmas = sorted(df.noise_sigma.unique())

    plt.rcParams.update({
        "font.size": 10, "font.family": "serif", "axes.grid": True,
        "grid.alpha": 0.3, "mathtext.fontset": "cm",
    })
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7))

    # ---- (a) 頻寬效率:各策略取最佳視窗 ----
    ax = axes[0]
    rows = []
    for sg in sigmas:
        d = df[df.noise_sigma == sg]
        a_bin = auc_per(d, "Binary", 0).mean()
        a_mgl = auc_per(d, "MGL", 2).mean()   # v8: 固定 N=2,公平比較
        a_smo = max(auc_per(d, "Smooth", w).mean() for w in WIN_LIST)
        rows.append((sg, a_bin, a_mgl, a_smo))
    arr = np.array(rows)
    ax.plot(arr[:, 0], arr[:, 1], "o--", color="#555555", label="Binary", ms=4)
    ax.plot(arr[:, 0], arr[:, 2], "^-", color="#D62728", label="MGL ($N$=2)", ms=5)
    ax.plot(arr[:, 0], arr[:, 3], "s-.", color="#1F77B4", label="Smooth (best $W$)", ms=4)
    ax.set_xlabel(r"Observation noise $\sigma$")
    ax.set_ylabel("Normalized AUC (recall--budget)")
    ax.set_title("(a) Bandwidth efficiency", fontsize=10)
    ax.legend(fontsize=8)

    # ---- (b) 事件漏失率 @ recall≈0.7 ----
    ax = axes[1]
    rows = []
    for sg in sigmas:
        d = df[df.noise_sigma == sg]
        binary = aligned_miss(d, "Binary", 0)
        mgl = aligned_miss(d, "MGL", 2)
        smo = [aligned_miss(d, "Smooth", w) for w in WIN_LIST]
        rows.append((sg, binary, mgl, min(smo), max(smo)))
    rb = np.array(rows)
    ax.plot(rb[:, 0], rb[:, 1], "o--", color="#555555", label="Binary", ms=4)
    ax.plot(rb[:, 0], rb[:, 2], "^-", color="#D62728", label="MGL ($N$=2)", ms=5)
    ax.fill_between(rb[:, 0], rb[:, 3], rb[:, 4], color="#1F77B4", alpha=0.18)
    ax.plot(rb[:, 0], rb[:, 3], "s-.", color="#1F77B4", label="Smooth ($W$=2--5)", ms=4)
    ax.plot(rb[:, 0], rb[:, 4], "s-.", color="#1F77B4", ms=4, alpha=0.5)
    ax.set_xlabel(r"Observation noise $\sigma$")
    ax.set_ylabel(r"Event miss rate (%) @ recall$\approx$0.7")
    ax.set_title("(b) Transient-event coverage", fontsize=10)
    ax.legend(fontsize=8)

    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    print(f"✅ 已輸出 {OUT_PATH}")


if __name__ == "__main__":
    main()
