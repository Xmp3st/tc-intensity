# -*- coding: utf-8 -*-
"""
seam_diagnostic.py —— 检验「补全边界接缝伪影」假设。

假设：benchmark 只衡量填补值的数值精度；但下游 CNN 还受**填补引入的空间伪影**影响。
 - `griddata` 用周围 IR1 做空间插值 → 填补值是周围 IR1 的自然延续（无接缝）。
 - `crosschannel` 用 WV 的值 → 填补值来自另一通道，其空间纹理与周围 IR1 不同
   → 在缺测区边界可能出现「接缝」/台阶。

指标：seam_ratio = 边界环带的梯度幅值 / 帧内背景梯度幅值（按通道 σ 归一化）。
  seam_ratio ≈ 1 → 无缝（与背景一致）；越大 → 接缝越明显。

用法：python scripts/seam_diagnostic.py [--n 300]
"""
import argparse
import json
import os
import sys

import numpy as np
import h5py
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from tcintens.data.band_health import (detect_missing_region, inpaint_regions,
                                       inpaint_regions_crosschannel)

H5 = "data/wpac_96.h5"
LOW = [150.0, 150.0]
CH = ["IR1", "WV"]


def seam_ratio(img, mask):
    """边界环带梯度 / 背景梯度（各自按该帧 σ 归一化后取比值）。"""
    m = np.asarray(mask, bool)
    if not m.any() or m.all():
        return None
    valid = ~m
    if valid.sum() < 50:
        return None
    gx = ndimage.sobel(img, axis=1)
    gy = ndimage.sobel(img, axis=0)
    g = np.hypot(gx, gy)
    ring = ndimage.binary_dilation(m, iterations=1) & ~m     # 紧贴边界的外侧 1 圈
    bg = ndimage.binary_dilation(m, iterations=4)            # 排除区（边界外 4 圈）
    far = ~bg
    if ring.sum() == 0 or far.sum() < 50:
        return None
    return float(g[ring].mean() / (g[far].mean() + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="每种通道采样的帧数")
    ap.add_argument("--stride", type=int, default=37)
    args = ap.parse_args()

    methods = {
        "griddata": lambda arr, m: inpaint_regions(arr, m, method="griddata"),
        "harmonic": lambda arr, m: inpaint_regions(arr, m, method="harmonic", iters=100),
        "crosschannel": None,   # 需要多通道，单独处理
    }

    res = {c: {k: [] for k in ["griddata", "harmonic", "crosschannel"]} for c in CH}
    got = {c: 0 for c in CH}

    with h5py.File(H5, "r") as hf:
        mat = hf["matrix"]
        N = mat.shape[0]
        for s in range(0, N, 512):
            e = min(s + 512, N)
            x = mat[s:e][..., :3].astype(np.float64)             # (B,96,96,3)
            for k in range(x.shape[0]):
                if min(got.values()) >= args.n:
                    break
                frame3 = np.ascontiguousarray(x[k].transpose(2, 0, 1))   # (3,96,96)
                for ci in range(2):
                    if got[CH[ci]] >= args.n:
                        continue
                    m = detect_missing_region(frame3[ci][None], 0.0, LOW[ci],
                                              min_blob=20, dilate=1)[0]
                    if m.sum() < 30:
                        continue
                    # 真实帧：仅在「该通道缺测」的位置评估；另一通道缺失位置需排除
                    m_other = detect_missing_region(frame3[1 - ci][None], 0.0,
                                                    LOW[1 - ci], min_blob=20, dilate=1)[0]
                    two = np.zeros((2, 96, 96), bool)
                    two[ci] = m
                    # 1) 空间方法：只传该通道
                    one = frame3[ci][None]
                    for name in ("griddata", "harmonic"):
                        if name == "crosschannel":
                            continue
                        f = inpaint_regions(one, m[None], method=name,
                                            iters=100 if name == "harmonic" else 100)[0]
                        r = seam_ratio(f, m)
                        if r is not None:
                            res[CH[ci]][name].append(r)
                    # 2) 跨通道：需两通道一起，参考通道用 IR1/WV（排除 PMW）
                    try:
                        fc = inpaint_regions_crosschannel(frame3[:2], two, deg=2,
                                                          ref_idx=[0, 1])[ci]
                        r = seam_ratio(fc, m)
                        if r is not None:
                            res[CH[ci]]["crosschannel"].append(r)
                    except Exception:
                        pass
                    got[CH[ci]] += 1
            if min(got.values()) >= args.n:
                break

    out = {}
    print("=" * 74)
    print("接缝诊断：seam_ratio = 边界环带梯度 / 背景梯度（1.0 = 与背景一致，越大越有接缝）")
    print("=" * 74)
    for c in CH:
        print(f"\n【{c}】样本 {got[c]} 帧")
        print(f"  {'方法':<14}{'中位数':>9}{'均值':>9}{'p90':>9}{'>1.5占比':>11}")
        out[c] = {}
        for name in ("griddata", "harmonic", "crosschannel"):
            a = np.array(res[c][name])
            if a.size == 0:
                continue
            out[c][name] = {"n": int(a.size), "median": float(np.median(a)),
                            "mean": float(a.mean()), "p90": float(np.percentile(a, 90)),
                            "frac_gt15": float((a > 1.5).mean())}
            print(f"  {name:<14}{np.median(a):>9.3f}{a.mean():>9.3f}"
                  f"{np.percentile(a,90):>9.3f}{(a>1.5).mean()*100:>10.1f}%")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/seam_diagnostic.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已写出: outputs/seam_diagnostic.json")


if __name__ == "__main__":
    main()
