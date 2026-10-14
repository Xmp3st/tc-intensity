# -*- coding: utf-8 -*-
"""
verify_feather.py —— 验证「边界羽化」是否既消除接缝、又保住补全精度。

对比 crosschannel 在 feather = 0/2/3/5 下：
  (a) 合成 benchmark 补全 RMSE（有真值）—— 精度不能明显变差；
  (b) 真实帧接缝比 seam_ratio       —— 应回落到 griddata/harmonic 水平。
"""
import os
import sys

import numpy as np
import h5py
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from tcintens.data.band_health import (detect_missing_region,
                                       inpaint_regions_crosschannel)

FEATHERS = (0, 2, 3, 5)


def bench(masks, ci, feather, n=150):
    z = np.load("data/inpaint_bench_cache.npz")
    clean = z["clean"]
    rng = np.random.default_rng(0)
    se, nn = 0.0, 0
    for i in range(min(n, masks.shape[0])):
        m = masks[i]
        if not m.any():
            continue
        k = rng.integers(clean.shape[0])
        truth = clean[k, :, :, ci].astype(np.float64)
        other = clean[k, :, :, 1 - ci].astype(np.float64)
        arr2 = np.stack([truth, other]) if ci == 0 else np.stack([other, truth])
        tgt = 0 if ci == 0 else 1
        mask2 = np.zeros((2,) + truth.shape, bool)
        mask2[tgt] = m
        f = inpaint_regions_crosschannel(arr2, mask2, deg=2, ref_idx=[0, 1],
                                         feather=feather)
        e = f[tgt][m] - truth[m]
        se += float((e ** 2).sum())
        nn += e.size
    return float(np.sqrt(se / nn))


def seam(img, m):
    if not m.any() or (~m).sum() < 50:
        return None
    g = np.hypot(ndimage.sobel(img, axis=1), ndimage.sobel(img, axis=0))
    ring = ndimage.binary_dilation(m, iterations=1) & ~m
    far = ~ndimage.binary_dilation(m, iterations=4)
    if ring.sum() == 0 or far.sum() < 50:
        return None
    return float(g[ring].mean() / (g[far].mean() + 1e-9))


def seam_real(feather, n=300, low=150.0):
    out = []
    got = 0
    with h5py.File("data/wpac_96.h5", "r") as hf:
        mat = hf["matrix"]
        for s in range(0, mat.shape[0], 512):
            x = mat[s:s + 512][..., :3].astype(np.float64)
            for k in range(x.shape[0]):
                if got >= n:
                    break
                fr = np.ascontiguousarray(x[k].transpose(2, 0, 1))
                m = detect_missing_region(fr[0][None], 0.0, low, min_blob=20,
                                          dilate=1)[0]
                if m.sum() < 30:
                    continue
                two = np.zeros((2, 96, 96), bool)
                two[0] = m
                fc = inpaint_regions_crosschannel(fr[:2], two, deg=2,
                                                  ref_idx=[0, 1],
                                                  feather=feather)[0]
                r = seam(fc, m)
                if r is not None:
                    out.append(r)
                got += 1
            if got >= n:
                break
    return np.array(out), got


def main():
    z = np.load("data/inpaint_bench_cache.npz")
    mi, mw = z["masks_IR1"], z["masks_WV"]

    print("=== (a) 合成 benchmark 补全 RMSE (K, 越低越好) ===")
    print(f"{'feather':>9}{'IR1':>10}{'WV':>10}")
    for f in FEATHERS:
        print(f"{f:>9}{bench(mi, 0, f):>10.3f}{bench(mw, 1, f):>10.3f}")

    print()
    print("=== (b) 真实帧接缝比 seam_ratio (IR1, 越低越好) ===")
    for f in FEATHERS:
        a, got = seam_real(f)
        print(f"  feather={f}: n={got}  中位 {np.median(a):.3f} | 均值 {a.mean():.3f}"
              f" | p90 {np.percentile(a, 90):.3f} | >1.5 {(a > 1.5).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
