#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analytic_window_model.py — 收益窗口的解析驗證模型
雙族群高斯模型,套用與論文完全相同的協定(門檻掃描/成本模型/
defer-disable/AUC),驗證窗口的區間結構是否從機制湧現。
對應論文 revision 的理論段落;詳見 理論推導_收益窗口.md。
族群直方圖為 trace-motivated 假設(需在論文標注),非量測值。
"""
import numpy as np
from scipy.stats import norm
Q = lambda x: 1 - norm.cdf(x)

gamma = 316/240000                                # GT tile-frame 比例(trace 事實)
GT_masses = [(0.28,0.5),(0.35,0.3),(0.45,0.2)]    # 假設:GT 顯著值粗直方圖
BG_masses = [(0.02,0.8),(0.06,0.15),(0.10,0.05)]  # 假設:背景粗直方圖
C_b, C_p, META = 0.05, 0.02, 0.05*0.05            # 論文成本模型
THRESHOLDS = np.linspace(0.10, 0.55, 10)          # 論文掃描
DEFER_OFF_TAU = 0.25                              # 論文 adaptive 規則

def curves(sigma, N):
    rows_b, rows_m = [], []
    for tau_U in THRESHOLDS:
        tau_B = tau_U*0.5
        def probs(p):
            A = Q((tau_U-p)/sigma)
            D = max(Q((tau_B-p)/sigma) - A, 0)
            r = Q((tau_B-p)/sigma)
            return A, D, r
        rec_b = sum(w*Q((tau_U-p)/sigma) for p,w in GT_masses)
        adm = gamma*rec_b + (1-gamma)*sum(w*Q((tau_U-p)/sigma) for p,w in BG_masses)
        rows_b.append((adm*(C_b+C_p), rec_b))
        off = tau_U < DEFER_OFF_TAU
        rec_m, traffic, meta = 0, 0, 0
        for p,w in GT_masses:
            A,D,r = probs(p); rec_m += w*(A + (0 if off else D*r**N))
        for grp, wt in [(GT_masses,gamma),(BG_masses,1-gamma)]:
            for p,w in grp:
                A,D,r = probs(p)
                traffic += wt*w*(A + (0 if off else D*r**N))
                meta    += 0 if off else wt*w*D
        rows_m.append((traffic*(C_b+C_p)+meta*META, rec_m))
    return np.array(rows_b), np.array(rows_m)

def auc(rows, gmax):
    r = rows[np.argsort(rows[:,0])]
    return np.trapezoid(np.clip(r[:,1],0,1), r[:,0]/gmax)

if __name__ == "__main__":
    print("解析模型 ΔAUC (MGL−Binary);對照實驗:σ=0.02→−0.019, σ*≈0.07, 峰值0.15→+0.029")
    print(f"{'σ':>6} {'N=2':>9} {'N=3':>9} {'N=5':>9}")
    for s in [0.02,0.05,0.07,0.10,0.15,0.20,0.25,0.30]:
        vals=[]
        for N in [2,3,5]:
            b,m = curves(s,N); g = max(b[:,0].max(), m[:,0].max())
            vals.append(auc(m,g)-auc(b,g))
        print(f"{s:>6} {vals[0]:>+9.4f} {vals[1]:>+9.4f} {vals[2]:>+9.4f}")
    print("\n註:模型重現區間結構(負→轉正→峰→衰減、低σ時N大者虧少);")
    print("N=5 恆負須加入有限事件長度(N>L 結構性丟棄)方能解釋——見推導文件第三節。")
