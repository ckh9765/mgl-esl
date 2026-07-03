#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
run_all.py — 一鍵重現 MGL 論文的所有數據與圖
============================================================
本腳本「不修改」任何現有程式,只依序呼叫它們,把論文需要的
數據與圖一次全部產出。已驗證的模擬器與繪圖邏輯保持原封不動。

完整可重現鏈:
  generate_trace_v3.py → noise_sweep.csv  → plot_fig1 → Fig1_benefit_window.png
  generate_trace_v5.py → summary_v5.csv   → plot_fig2 → Fig3_deployment_map.png

用法:
  python run_all.py              # 完整跑:重新產數據 + 畫圖
  python run_all.py --plots-only # 只畫圖(若 CSV 已存在,跳過耗時的模擬)

需求:
  - 與本腳本同目錄需有:generate_trace_v3.py, generate_trace_v5.py,
    plot_fig1_benefit_window.py, plot_fig2_deployment_map.py
  - 影片 VIRAT_S_050201_05_000890_000944.mp4 在模擬器讀取的位置
  - conda 環境:mgl(含 numpy/pandas/scipy/matplotlib/opencv)
============================================================
"""

import subprocess
import sys
import os
import time

# (描述, 腳本檔名, 預期產出檔, 是否為耗時模擬)
STEPS = [
    ("模擬:雜訊掃描 (Fig.1 數據)", "generate_trace_v3.py", "noise_sweep.csv", True),
    ("模擬:事件覆蓋 (Fig.2 數據)", "generate_trace_v5.py", "summary_v5.csv",  True),
    ("繪圖:Fig.1 收益窗口",        "plot_fig1_benefit_window.py", "Fig1_benefit_window.png", False),
    ("繪圖:Fig.2 部署地圖",        "plot_fig2_deployment_map.py", "Fig3_deployment_map.png", False),
]


def run_step(desc, script, expected, py):
    print(f"\n{'='*60}\n▶ {desc}\n  ({script})\n{'='*60}")
    if not os.path.exists(script):
        print(f"  ❌ 找不到腳本:{script} — 跳過")
        return False
    t0 = time.time()
    result = subprocess.run([py, script], capture_output=False)
    dt = time.time() - t0
    if result.returncode != 0:
        print(f"  ❌ {script} 執行失敗(return code {result.returncode})")
        return False
    if expected and not os.path.exists(expected):
        print(f"  ⚠ 執行完成但找不到預期產出:{expected}")
        return False
    print(f"  ✅ 完成({dt:.1f}s),產出:{expected}")
    return True


def main():
    plots_only = "--plots-only" in sys.argv
    py = sys.executable  # 用當前 python(確保是 mgl 環境)

    print("MGL 論文圖一鍵重現")
    print(f"Python:{py}")
    print(f"模式:{'只畫圖(跳過模擬)' if plots_only else '完整重現(模擬+畫圖)'}")

    ok, skip = 0, 0
    for desc, script, expected, is_sim in STEPS:
        if plots_only and is_sim:
            if os.path.exists(expected):
                print(f"\n⏭ 跳過模擬步驟(--plots-only,且 {expected} 已存在)")
                skip += 1
                continue
            else:
                print(f"\n⚠ --plots-only 但缺少 {expected},仍需先跑模擬")
        if run_step(desc, script, expected, py):
            ok += 1

    print(f"\n{'='*60}\n總結:{ok} 步成功" + (f",{skip} 步跳過" if skip else ""))
    print("論文圖:Fig1_benefit_window.png、Fig3_deployment_map.png")
    print("="*60)


if __name__ == "__main__":
    main()
