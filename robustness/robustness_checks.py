#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
robustness_checks.py — 跨場景結果的統計嚴謹性檢查
============================================================
回應審視意見,對 multiscene 結果補上不確定性量化,避免點估計過度宣稱:
  --boot     : seed-level bootstrap CI(峰值增益、峰值σ 分布、σ*、N=2 相對次佳N 優勢)
  --coverage : 事件覆蓋率的 bootstrap CI(事件<5 標記不可解讀)
  --stats    : 每場景 salience 分布共變數(mean/var/kurtosis/lag-1 時間自相關/活躍%)
  --downsample <mp4> : 降採樣控制(同場景不同解析度),鑑別「峰值σ 位移是否為解析度假影」

單一真相來源:import multiscene_generalization(其掃描邏輯 import 自 v3/v5)。
所有結果可由 salience_*.npy 重跑。B 預設 2000。

用法:
  python robustness_checks.py salience_out --ref salience_virat.npy --boot --coverage --stats
  python robustness_checks.py --downsample video_file/VIRAT_S_050201_05_000890_000944.mp4 --ref salience_virat.npy
============================================================
"""
import os, sys, glob, argparse, importlib.util
from collections import Counter
import numpy as np

def _load(mod):
    here=os.path.dirname(os.path.abspath(__file__))
    spec=importlib.util.spec_from_file_location(mod, os.path.join(here,mod+".py"))
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
M=_load("multiscene_generalization")
TILE=64; MAXF=500; MAXSAD=(TILE*TILE*255)*0.1


def per_seed_gain(ndf, n):
    out={}
    for sg in sorted(ndf.noise_sigma.unique()):
        d=ndf[ndf.noise_sigma==sg]; gmax=d.total_bytes_mbit.max(); g=[]
        for sd in sorted(d.seed.unique()):
            b=d[(d.strategy=="Binary")&(d.seed==sd)].sort_values("total_bytes_mbit")
            mv=d[(d.strategy=="MGL")&(d.n_persist==n)&(d.seed==sd)].sort_values("total_bytes_mbit")
            g.append(np.trapezoid(np.clip(mv.recall,0,1),mv.total_bytes_mbit/gmax)
                     -np.trapezoid(np.clip(b.recall,0,1),b.total_bytes_mbit/gmax))
        out[sg]=np.array(g)
    return out


def boot_scene(name, path, B, rng):
    p=np.load(path); ndf,_=M.build_noise_df(p)
    gm={n:per_seed_gain(ndf,n) for n in [2,3,5]}
    sig=sorted(gm[2]); ns=len(next(iter(gm[2].values()))); G2=np.array([gm[2][x] for x in sig])
    pk=[]; pg=[]; n2=[]; ss=[]
    for _ in range(B):
        idx=rng.integers(0,ns,ns); mn=G2[:,idx].mean(1); i=int(np.argmax(mn))
        pk.append(sig[i]); pg.append(mn[i])
        n2.append(mn[i]-max(gm[3][sig[i]][idx].mean(), gm[5][sig[i]][idx].mean()))
        for j in range(len(sig)-1):
            if mn[j]<0<=mn[j+1]:
                ss.append(sig[j]+(0-mn[j])*(sig[j+1]-sig[j])/(mn[j+1]-mn[j])); break
    pg=np.array(pg); n2=np.array(n2); ss=np.array(ss)
    ci=lambda a:(np.percentile(a,2.5),np.percentile(a,97.5))
    print(f"== {name} ==")
    print(f"  峰值σ 分布: {[(round(k,3),v) for k,v in Counter(pk).most_common(3)]}")
    print(f"  峰值增益 {pg.mean():.4f} CI[{ci(pg)[0]:.4f},{ci(pg)[1]:.4f}] 含0={'是' if ci(pg)[0]<=0 else '否'}")
    if len(ss): print(f"  σ* 有零交叉{100*len(ss)/B:.0f}% 中位{np.median(ss):.3f} CI[{ci(ss)[0]:.3f},{ci(ss)[1]:.3f}]")
    else: print("  σ* 無零交叉")
    print(f"  N=2優勢 {n2.mean():+.4f} CI[{ci(n2)[0]:+.4f},{ci(n2)[1]:+.4f}] >0={100*(n2>0).mean():.0f}%")


def coverage_ci(name, path, B, rng):
    p=np.load(path); edf,ne=M.build_event_summary(p,2)
    if ne==0: print(f"== {name} 事件=0 → 覆蓋率不可定義"); return
    per=[]
    for sd in sorted(edf.seed.unique()):
        d=edf[edf.seed==sd]; vals=[]
        for sg in sorted(d.noise_sigma.unique()):
            g=d[d.noise_sigma==sg].groupby("threshold_val")[["recall","miss_rate"]].mean()
            if g.empty or g.recall.max()<0.4: continue
            vals.append(g.loc[(g.recall-0.7).abs().idxmin(),"miss_rate"])
        if vals: per.append(1-np.mean(vals))
    per=np.array(per)*100
    bs=[per[rng.integers(0,len(per),len(per))].mean() for _ in range(B)]
    flag=" ⚠事件過少,不可解讀" if ne<5 else ""
    print(f"== {name} 事件={ne} 覆蓋率 {per.mean():.1f}% CI[{np.percentile(bs,2.5):.1f},{np.percentile(bs,97.5):.1f}]{flag}")


def sal_stats(name, path):
    p=np.load(path); x=p.ravel(); mu=x.mean(); sd=x.std()
    kurt=float(((x-mu)**4).mean()/sd**4-3) if sd>0 else float("nan")
    a,b=p[:-1].ravel(),p[1:].ravel()
    ac=float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else float("nan")
    print(f"  {name:30s} tiles={p.shape[1]:3d} active%={100*(p>0.25).mean():5.2f} "
          f"mean={mu:.4f} var={p.var():.5f} kurt={kurt:7.1f} autocorr={ac:.3f}")


def extract_resized(path, target):
    import cv2
    cap=cv2.VideoCapture(path); ret,prev=cap.read()
    pg=cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
    if target: pg=cv2.resize(pg,target)
    h,w=pg.shape; ty,tx=h//TILE,w//TILE; allp=[]; fc=0
    while True:
        ret,fr=cap.read()
        if not ret or fc>=MAXF: break
        g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
        if target: g=cv2.resize(g,target)
        d=cv2.absdiff(g,pg).astype(np.float64)[:ty*TILE,:tx*TILE]
        d=d.reshape(ty,TILE,tx,TILE).sum(axis=(1,3))
        allp.append(np.clip(d/MAXSAD,0,1).ravel()); pg=g; fc+=1
    cap.release(); return np.array(allp)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="salience_*.npy 或資料夾")
    ap.add_argument("--ref", default="salience_virat.npy")
    ap.add_argument("--boot", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--downsample", help="mp4 路徑;做同場景降採樣控制")
    ap.add_argument("-B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args=ap.parse_args()
    rng=np.random.default_rng(args.seed)

    files=[]
    for p in args.paths:
        if os.path.isdir(p): files+=sorted(glob.glob(os.path.join(p,"salience_*.npy")))
        elif p.endswith(".npy"): files.append(p)
    if os.path.exists(args.ref) and args.ref not in files: files=[args.ref]+files
    files=list(dict.fromkeys(files))

    if args.downsample:
        print("="*60+"\n降採樣控制實驗(同場景,只變解析度)\n"+"="*60)
        p1080=extract_resized(args.downsample, None)
        p720=extract_resized(args.downsample, (1280,720))
        np.save("salience_downsample720.npy", p720)
        if os.path.exists(args.ref):
            print("原生抽取 == 基準:", np.array_equal(p1080, np.load(args.ref)))
        print(f"1080p tiles={p1080.shape[1]} GT={int((p1080>0.25).sum())}  "
              f"720p tiles={p720.shape[1]} GT={int((p720>0.25).sum())}")
        boot_scene("原生 1080p", args.ref if os.path.exists(args.ref) else None or "salience_downsample720.npy", args.B, rng)
        boot_scene("降採樣 720p", "salience_downsample720.npy", args.B, rng)
        return

    if args.stats:
        print("="*60+"\nsalience 分布共變數\n"+"="*60)
        for f in files: sal_stats(os.path.basename(f).replace("salience_","").replace(".npy",""), f)
    if args.boot:
        print("="*60+"\nbootstrap CI(峰值/σ*/N=2優勢)\n"+"="*60)
        for f in files: boot_scene(os.path.basename(f).replace("salience_","").replace(".npy",""), f, args.B, rng)
    if args.coverage:
        print("="*60+"\n事件覆蓋率 bootstrap CI\n"+"="*60)
        for f in files: coverage_ci(os.path.basename(f).replace("salience_","").replace(".npy",""), f, args.B, rng)


if __name__ == "__main__":
    main()
