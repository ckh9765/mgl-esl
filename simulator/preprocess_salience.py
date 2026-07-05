#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
preprocess_salience.py — MGL salience 前處理獨立腳本
============================================================
目的:
  把原本內建在 generate_trace_v3_refactored.py / generate_trace_v5.py
  裡的 process_video_to_p_values() 「1:1」抽成一支獨立前處理,
  讓 salience 與 mp4 脫鉤:mp4 留本機,只上傳抽好的小 salience 檔。

嚴格保證(與 ESL 可比協定一致,勿改):
  - TILE_SIZE   = 64            (tile 64×64)
  - MAX_FRAMES  = 500
  - max_sad     = (64*64*255) * 0.1
  - 灰階 absdiff、向量化 tile SAD、np.clip(d/max_sad, 0, 1)
  - 輸出形狀 [T, n_tiles] 的 float64 矩陣(與模擬器內部 p_matrix 完全相同)

本腳本「不改動任何計算」。process_video_to_p_values() 的函式體與
generate_trace_v3_refactored.py 逐行相同(v5 的版本邏輯亦完全一致)。

用法:
  # 1) 抽出 salience,輸出 .npy(位元精確、canonical 檔)
  python preprocess_salience.py extract VIRAT_S_050201_05_000890_000944.mp4 \
      -o salience_virat.npy

  # 2) 自檢:確認抽出的 salience 與『兩支模擬器內建版本』位元一致
  python preprocess_salience.py verify VIRAT_S_050201_05_000890_000944.mp4

  # (選用)同時輸出人可讀的壓縮 csv(僅供目視檢查,非位元基準)
  python preprocess_salience.py extract <mp4> -o salience.npy --also-csv

輸出檔:
  <name>.npy        ← 位元精確的 salience 矩陣(float64,[T, n_tiles])。這是基準檔。
  <name>.meta.json  ← 形狀/dtype/SHA256/參數,供追溯。
  <name>.csv.gz     ← (加 --also-csv 才產)每列一幀、每欄一 tile,full precision 文字。
============================================================
"""

import os
import sys
import json
import gzip
import hashlib
import argparse

import numpy as np
import cv2

# ==========================================
# 參數設定 —— 必須與 v3/v5 完全相同,勿改
# ==========================================
TILE_SIZE = 64
MAX_FRAMES = 500


def process_video_to_p_values(video_path):
    """
    將影片轉換為 p 值矩陣（salience）。

    ★ 本函式與 generate_trace_v3_refactored.py 內建版本逐行相同，
      generate_trace_v5.py 的版本邏輯亦完全一致（僅 print/註解排版不同）。
      請勿改動任何一行運算，否則會破壞與 ESL 的可比性與承重數字複現。

    Args:
        video_path (str): 影片路徑

    Returns:
        numpy.ndarray: p 值矩陣 [時間, 瓦片數]，dtype=float64
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
        d = diff[:ty_n * TILE_SIZE, :tx_n * TILE_SIZE]
        d = d.reshape(ty_n, TILE_SIZE, tx_n, TILE_SIZE).sum(axis=(1, 3))
        all_p.append(np.clip(d / max_sad, 0, 1).ravel())

        prev_gray = gray
        fc += 1

    cap.release()
    return np.array(all_p)  # [T, n_tiles]


# ==========================================
# 工具函式
# ==========================================
def array_sha256(arr):
    """對陣列的原始位元組 + 形狀 + dtype 取 SHA256，做為位元指紋。"""
    m = hashlib.sha256()
    m.update(str(arr.shape).encode())
    m.update(str(arr.dtype).encode())
    m.update(np.ascontiguousarray(arr).tobytes())
    return m.hexdigest()


def save_salience(arr, out_path, also_csv=False):
    """存 .npy(位元基準)+ .meta.json;選用 .csv.gz。"""
    base, ext = os.path.splitext(out_path)
    if ext.lower() != ".npy":
        base = out_path
        out_path = base + ".npy"

    np.save(out_path, arr)

    meta = {
        "source_function": "process_video_to_p_values",
        "identical_to": ["generate_trace_v3_refactored.py", "generate_trace_v5.py"],
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": array_sha256(arr),
        "params": {
            "TILE_SIZE": TILE_SIZE,
            "MAX_FRAMES": MAX_FRAMES,
            "max_sad": (TILE_SIZE * TILE_SIZE * 255) * 0.1,
            "grayscale_absdiff": True,
            "clip_range": [0, 1],
        },
    }
    meta_path = base + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"💾 已存 {out_path}  shape={arr.shape} dtype={arr.dtype}")
    print(f"🔑 SHA256 = {meta['sha256']}")
    print(f"🧾 已存 {meta_path}")

    if also_csv:
        csv_path = base + ".csv.gz"
        # full precision,%r 等效的 repr;用 %.17g 保證 float64 round-trip
        with gzip.open(csv_path, "wt", encoding="utf-8") as f:
            np.savetxt(f, arr, delimiter=",", fmt="%.17g")
        print(f"🧾 已存 {csv_path}(僅供目視,非位元基準)")

    return out_path, meta


# ==========================================
# 子命令:extract
# ==========================================
def cmd_extract(args):
    if not os.path.exists(args.video):
        raise SystemExit(f"❌ 找不到影片:{args.video}")

    arr = process_video_to_p_values(args.video)
    out_path, meta = save_salience(arr, args.out, also_csv=args.also_csv)

    # 立刻做一次 round-trip 完整性檢查:存回來的 .npy 必須位元一致
    reloaded = np.load(out_path)
    ok = np.array_equal(arr, reloaded) and (arr.dtype == reloaded.dtype)
    print("\n[round-trip 檢查] 重新載入 .npy 與原矩陣位元一致:",
          "✅ 通過" if ok else "❌ 失敗")
    if not ok:
        raise SystemExit("❌ round-trip 不一致,請勿使用此檔。")


# ==========================================
# 子命令:verify —— 位元一致性自檢
# ==========================================
def cmd_verify(args):
    """
    自檢:對同一支 mp4,分別用
      (a) 本腳本 preprocess_salience.process_video_to_p_values
      (b) generate_trace_v3_refactored.process_video_to_p_values
      (c) generate_trace_v5.process_video_to_p_values
    三者輸出必須『位元完全一致』(np.array_equal 且 SHA256 相同)。
    import 這兩支模擬器不會觸發其掃描(都在 if __name__=='__main__' 保護下)。
    """
    if not os.path.exists(args.video):
        raise SystemExit(f"❌ 找不到影片:{args.video}")

    # 讓 import 找得到同目錄的兩支模擬器
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    results = {}

    print("=" * 56)
    print("[a] preprocess_salience(本腳本)")
    a = process_video_to_p_values(args.video)
    results["preprocess_salience"] = a

    ok_all = True
    for modname in ("generate_trace_v3_refactored", "generate_trace_v5"):
        print("-" * 56)
        print(f"[+] import {modname} 並執行其內建 process_video_to_p_values")
        try:
            mod = __import__(modname)
        except Exception as e:
            print(f"⚠️  無法 import {modname}:{e}(略過此模組)")
            ok_all = False
            continue
        if not hasattr(mod, "process_video_to_p_values"):
            print(f"⚠️  {modname} 沒有 process_video_to_p_values(略過)")
            ok_all = False
            continue
        b = mod.process_video_to_p_values(args.video)
        results[modname] = b

    # 逐一與基準 (a) 比對
    print("=" * 56)
    ref = results["preprocess_salience"]
    ref_hash = array_sha256(ref)
    print(f"基準 preprocess_salience : shape={ref.shape} dtype={ref.dtype}")
    print(f"                         SHA256={ref_hash}")
    print("-" * 56)
    for name, arr in results.items():
        if name == "preprocess_salience":
            continue
        same_shape = arr.shape == ref.shape
        same_dtype = arr.dtype == ref.dtype
        bit_equal = same_shape and np.array_equal(arr, ref)
        h = array_sha256(arr)
        hash_equal = (h == ref_hash)
        status = "✅ 位元一致" if (bit_equal and hash_equal) else "❌ 不一致"
        print(f"{name:35s} {status}")
        print(f"    shape={arr.shape} dtype={arr.dtype} SHA256={h}")
        if not (bit_equal and hash_equal):
            ok_all = False
            # 若不一致,印出最大絕對差,協助定位
            if same_shape:
                d = np.abs(arr.astype(np.float64) - ref.astype(np.float64))
                print(f"    max|Δ|={d.max():.3e}  非零元素數={int((d>0).sum())}")

    print("=" * 56)
    if ok_all:
        print("✅ 全部位元一致:抽出的 salience 與兩支模擬器內建版本完全相同。")
        print("   之後模擬器改讀此 salience 檔,可保證與現有 CSV/圖位元可複現。")
        sys.exit(0)
    else:
        print("❌ 有模組不一致或無法比對,請勿用抽出的 salience 取代內建版本。")
        sys.exit(1)


def build_parser():
    p = argparse.ArgumentParser(
        description="MGL salience 前處理:1:1 抽出 process_video_to_p_values,並提供位元一致性自檢。")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="抽出 salience 並存成 .npy(+meta,選用 csv.gz)")
    pe.add_argument("video", help="輸入 mp4 路徑")
    pe.add_argument("-o", "--out", default="salience.npy", help="輸出 .npy 路徑(預設 salience.npy)")
    pe.add_argument("--also-csv", action="store_true", help="同時輸出人可讀的 .csv.gz(僅目視)")
    pe.set_defaults(func=cmd_extract)

    pv = sub.add_parser("verify", help="對同一 mp4 比對本腳本 vs v3 vs v5 是否位元一致")
    pv.add_argument("video", help="輸入 mp4 路徑")
    pv.set_defaults(func=cmd_verify)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
