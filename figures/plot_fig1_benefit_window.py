#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
plot_fig1_benefit_window.py
============================================================
讀取 noise_sweep.csv(由 generate_trace_v3.py 產生),重現論文 Fig. 1:
三階(MGL)對二元(Binary)的配對 AUC 增益收益窗口,含 95% CI。

可重現鏈:
  generate_trace_v3.py → noise_sweep.csv → 本腳本 → Fig1_benefit_window.png

用法:
  python plot_fig1_benefit_window.py [noise_sweep.csv 路徑]
  (不給路徑則預設讀同目錄的 noise_sweep.csv)
============================================================
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "noise_sweep.csv"
OUT_PATH = "Fig1_benefit_window.png"
N_LIST = [2, 3, 5]
COLORS = {2: "#D62728", 3: "#1F77B4", 5: "#7F7F7F"}


def auc_per_seed(df_b, df_m, gmax):
    """單一 seed 的 normalized AUC 配對差 (MGL - Binary)。"""
    out = {}
    for tag, sub in [("B", df_b), ("M", df_m)]:
        s = sub.sort_values("total_bytes_mbit")
        out[tag] = trapz(np.clip(s.recall, 0, 1),
                                s.total_bytes_mbit / gmax)
    return out["M"] - out["B"]


def main():
    df = pd.read_csv(CSV_PATH)
    sigmas = sorted(df.noise_sigma.unique())

    plt.rcParams.update({
        "font.size": 10, "font.family": "serif", "axes.grid": True,
        "grid.alpha": 0.3, "mathtext.fontset": "cm",
    })
    fig, ax = plt.subplots(figsize=(6, 4.2))

    for n in N_LIST:
        means, cis = [], []
        for sg in sigmas:
            d = df[df.noise_sigma == sg]
            gmax = d.total_bytes_mbit.max()
            gains = []
            for sd in sorted(d.seed.unique()):
                b = d[(d.strategy == "Binary") & (d.seed == sd)]
                m = d[(d.strategy == "MGL") & (d.n_persist == n) & (d.seed == sd)]
                gains.append(auc_per_seed(b, m, gmax))
            g = np.array(gains)
            means.append(g.mean())
            cis.append(stats.t.ppf(0.975, len(g) - 1) * g.std(ddof=1) / np.sqrt(len(g)))
        ax.errorbar(sigmas, means, yerr=cis, marker="o", ms=4, capsize=3,
                    color=COLORS[n], label=f"$N_{{persist}}$ = {n}")

    ax.axhline(0, color="k", lw=0.8)
    ax.axvspan(0.075, 0.30, alpha=0.06, color="green")
    ax.annotate("benefit window", xy=(0.18, 0.027), fontsize=9,
                color="darkgreen", ha="center")
    ax.set_xlabel(r"Observation noise $\sigma$")
    ax.set_ylabel("Normalized AUC gain (MGL $-$ Binary)")
    ax.set_title("Benefit Window of Ternary Memory-Side Governance")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    print(f"✅ 已輸出 {OUT_PATH}")

    # 同時印出關鍵數值供核對
    print("\n關鍵數值核對 (N=2):")
    for sg in sigmas:
        d = df[df.noise_sigma == sg]
        gmax = d.total_bytes_mbit.max()
        gains = [auc_per_seed(d[(d.strategy=="Binary")&(d.seed==sd)],
                              d[(d.strategy=="MGL")&(d.n_persist==2)&(d.seed==sd)],
                              gmax) for sd in sorted(d.seed.unique())]
        g = np.array(gains)
        ci = stats.t.ppf(0.975, len(g)-1) * g.std(ddof=1) / np.sqrt(len(g))
        print(f"  σ={sg}: {g.mean():+.4f} ± {ci:.4f}")


if __name__ == "__main__":
    main()
