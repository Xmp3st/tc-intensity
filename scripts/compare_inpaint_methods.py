# -*- coding: utf-8 -*-
"""compare_inpaint_methods.py —— 在真实缺失帧上对比 griddata / harmonic / crosschannel。

与 benchmark_inpaint.py 互补：
  * benchmark 用「真实掩码 + 干净帧」给出**有真值的定量误差**；
  * 本脚本用**真实缺失帧**看修复结果的物理合理性（亮温范围、空间连续性、
    是否残留 0/低值）、以及三种方法在同样输入上的差异。

输出
  outputs/inpaint_real_compare.png   多案例并排对比图
  outputs/inpaint_real_stats.json    各方法的修复值统计
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tcintens.data.band_health import (  # noqa: E402
    detect_missing_region, harmonic_fill, inpaint_regions,
    inpaint_regions_crosschannel)

H5 = "data/wpac_96.h5"
OUT_PNG = "outputs/inpaint_real_compare.png"
OUT_JSON = "outputs/inpaint_real_stats.json"
CH = ["IR1", "WV", "PMW"]
LOW = [150.0, 150.0, -999.0]


def find_cases(n_ir1=2, n_wv=2, want_ir1_area=(0.005, 0.10), want_wv_area=(0.20, 0.95)):
    """挑若干代表性真实缺失帧：IR1 细条带、WV 大块。"""
    import h5py
    cases = []

    def _mk(arr, mask):
        o = {"griddata": inpaint_regions(arr, mask, method="griddata"),
             "harmonic": inpaint_regions(arr, mask, method="harmonic", iters=100),
             "crosschannel": inpaint_regions_crosschannel(
                 arr, mask, deg=2, ref_idx=[0, 1])}
        return o

    with h5py.File(H5, "r") as hf:
        mat = hf["matrix"]
        N = mat.shape[0]
        got_ir1 = got_wv = 0
        for s in range(0, N, 256):
            e = min(s + 256, N)
            x = mat[s:e][..., :3].astype(np.float32)
            for k in range(e - s):
                arr = np.ascontiguousarray(x[k].transpose(2, 0, 1))
                # 与 pipeline 一致：IR1/WV 一起检测。否则参考通道自身的 0 值
                # （未标记的缺测）会混进回归拟合，导致预测发散。
                full = np.zeros(arr.shape, dtype=bool)
                for ci2 in (0, 1):
                    full[ci2] = detect_missing_region(
                        arr[ci2:ci2 + 1], sentinel=0.0, low_thr=LOW[ci2],
                        min_blob=20, dilate=1)[0]
                for ci, (name, target, want) in enumerate([
                        ("IR1", "ir1", want_ir1_area),
                        ("WV", "wv", want_wv_area)]):
                    got = got_ir1 if target == "ir1" else got_wv
                    if got >= (n_ir1 if target == "ir1" else n_wv):
                        continue
                    m = full[ci]
                    a = float(m.mean())
                    if not (want[0] <= a <= want[1]):
                        continue
                    if target == "ir1":
                        got_ir1 += 1
                    else:
                        got_wv += 1
                    cases.append({"idx": s + k, "ch": name, "ci": ci,
                                  "area": a, "arr": arr.copy(), "mask": m.copy(),
                                  "filled": _mk(arr, full)})
            if got_ir1 >= n_ir1 and got_wv >= n_wv:
                break
    return cases


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cases = find_cases()
    stats = {}
    print(f"找到 {len(cases)} 个案例\n")
    for cs in cases:
        m, ci, arr = cs["mask"], cs["ci"], cs["arr"]
        key = f"idx{cs['idx']}_{cs['ch']}"
        rec = {"area": round(cs["area"], 4),
               "raw_in_hole": {"min": float(arr[ci][m].min()),
                               "max": float(arr[ci][m].max()),
                               "mean": round(float(arr[ci][m].mean()), 2)}}
        print(f"[{key}] 缺测 {cs['area']*100:.2f}%  原始缺测区 "
              f"min={rec['raw_in_hole']['min']:.1f} max={rec['raw_in_hole']['max']:.1f}")
        for name, out in cs["filled"].items():
            v = out[ci][m]
            other_ci = 1 - ci if ci < 2 else 0
            nonhole = ~m
            rec[name] = {
                "hole_min": round(float(v.min()), 2),
                "hole_max": round(float(v.max()), 2),
                "hole_mean": round(float(v.mean()), 2),
                "still_below_thr": int((v < (LOW[ci] if LOW[ci] > -900 else 0)).sum()),
                "nan": int(np.isnan(out).sum()),
                "max_abs_dev_from_valid_mean": round(
                    float(np.abs(v - arr[ci][nonhole].mean()).max()), 2),
            }
            print(f"    {name:<13} 修复后 hole=[{rec[name]['hole_min']:.1f},"
                  f"{rec[name]['hole_max']:.1f}] mean={rec[name]['hole_mean']:.1f} "
                  f"| 残留低值 {rec[name]['still_below_thr']} | NaN {rec[name]['nan']}")
        stats[key] = rec
        print()

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    _plot(cases)
    print(f"已写出: {OUT_JSON} / {OUT_PNG}")


def _plot(cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ["griddata", "harmonic", "crosschannel"]
    n = len(cases)
    fig, axes = plt.subplots(n, 2 + len(methods), figsize=(3.0 * (2 + len(methods)), 3.0 * n))
    if n == 1:
        axes = axes[None, :]
    for r, cs in enumerate(cases):
        ci = cs["ci"]
        vmin, vmax = 180, 310
        ims = [("raw", cs["arr"][ci]), ("mask", cs["mask"].astype(float))]
        for mt in methods:
            ims.append((mt, cs["filled"][mt][ci]))
        for c, (title, img) in enumerate(ims):
            ax = axes[r, c]
            cmap = "gray" if title == "mask" else "turbo"
            ax.imshow(img, cmap=cmap, vmin=None if title == "mask" else vmin,
                      vmax=None if title == "mask" else vmax)
            rmse_txt = ""
            if title in methods:
                rmse_txt = ""  # 真实无 GT，不标误差
            ax.set_title(f"{title}\n{cs['ch']} idx={cs['idx']} {rmse_txt}", fontsize=8)
            ax.axis("off")
    fig.suptitle("Real missing frames — griddata vs harmonic vs crosschannel "
                 "(IR1 strips / WV blocks)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=105)
    plt.close(fig)


if __name__ == "__main__":
    main()
