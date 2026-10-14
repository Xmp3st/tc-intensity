# -*- coding: utf-8 -*-
"""benchmark_inpaint.py —— 定量比较「局部缺失补全」方法（有真值的合成评测）。

为什么需要合成评测
    真实缺失区域（被 nan_to_num 置 0 的像素）**没有 ground truth**，无法直接算误差。
    于是：
      1) 从真实数据中提取「真实形态」的缺失掩码
         （IR1 = 细横条带为主；WV = 罕见大块）；
      2) 把这些掩码施加到**干净帧**上（掩码区置 0），得到带真值的伪缺失样本；
      3) 用不同方法补全，与真值比较 RMSE / MAE。
    这样既保留真实缺失的几何形态，又提供了客观、可复现的评价标准。

方法
    mean          通道均值（现有代码在「大面积缺失」时的兜底）
    nearest       最近有效像素（scipy griddata nearest）
    griddata      现有默认：散点线性插值（Delaunay 三角剖分）
    row_interp    逐行 1D 线性插值
    col_interp    逐列 1D 线性插值（对细横条带是天然最优）
    harmonic      迭代求解拉普拉斯方程（调和插值，平滑、无三角剖分伪影）

输出
    outputs/inpaint_benchmark.json   各方法 × 通道 × 缺失面积桶 的 RMSE/MAE
    outputs/inpaint_benchmark.png    方法对比柱状图 + 案例对照
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

H5 = "data/wpac_96.h5"
CACHE = "data/inpaint_bench_cache.npz"
OUT_JSON = "outputs/inpaint_benchmark.json"
OUT_PNG = "outputs/inpaint_benchmark.png"
CH = ["IR1", "WV", "PMW"]
LOW_THR = [150.0, 150.0, -999.0]
MIN_BLOB = 20
DILATE = 1

BUCKETS = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.20), (0.20, 0.60), (0.60, 1.01)]


# --------------------------------------------------------------------------- #
# 提取真实掩码 + 干净帧
# --------------------------------------------------------------------------- #
def extract(n_clean=400, n_mask_per_ch=250, seed=0):
    import h5py
    from tcintens.data.band_health import (
        detect_missing_channels, detect_missing_region)

    rng = np.random.default_rng(seed)
    clean = []
    pool = {"IR1": [], "WV": []}          # (area, mask) 列表

    with h5py.File(H5, "r") as hf:
        mat = hf["matrix"]
        N = mat.shape[0]
        for s in range(0, N, 512):
            e = min(s + 512, N)
            x = mat[s:e][..., :3].astype(np.float64)     # (B,H,W,3)
            for k in range(x.shape[0]):
                img = x[k]                                # (H,W,3)
                whole = detect_missing_channels(
                    np.ascontiguousarray(img.transpose(2, 0, 1)))
                ms = {}
                for ci in range(2):
                    name = CH[ci]
                    if whole[ci]:
                        ms[name] = None
                        continue
                    m = detect_missing_region(
                        img[:, :, ci][None], sentinel=0.0, low_thr=LOW_THR[ci],
                        min_blob=MIN_BLOB, dilate=DILATE)[0]
                    ms[name] = m
                is_clean = all(v is not None and not v.any() for v in ms.values())
                if is_clean and len(clean) < n_clean:
                    clean.append(img.astype(np.float32))
                for name in ("IR1", "WV"):
                    m = ms[name]
                    if m is not None and m.any():
                        pool[name].append(m)
            if s % 5120 == 0:
                print(f"  扫描 {s}/{N} | clean={len(clean)} "
                      f"IR1 masks={len(pool['IR1'])} WV masks={len(pool['WV'])}",
                      flush=True)

    # 按面积分位抽样，保证各面积档都有代表
    out = {"clean": np.stack(clean, 0)}
    for name in ("IR1", "WV"):
        lst = pool[name]
        areas = np.array([float(m.mean()) for m in lst])
        order = np.argsort(areas)
        n = min(n_mask_per_ch, len(lst))
        if n == 0:
            out[f"masks_{name}"] = np.zeros((0, 96, 96), bool)
            continue
        # 均匀取分位点 -> 覆盖小洞到大洞
        idx = order[np.linspace(0, len(lst) - 1, n).astype(int)]
        out[f"masks_{name}"] = np.stack([lst[i] for i in idx], 0)
        print(f"  {name}: 采用 {n} 个掩码, 面积 {areas[idx].min()*100:.2f}%"
              f"~{areas[idx].max()*100:.1f}%")

    os.makedirs(os.path.dirname(os.path.abspath(CACHE)), exist_ok=True)
    np.savez_compressed(CACHE, **out)
    print(f"  缓存已写出: {CACHE}")
    return out


# --------------------------------------------------------------------------- #
# 补全方法
# --------------------------------------------------------------------------- #
def m_mean(img, m, **_):
    out = img.copy()
    out[m] = float(img[~m].mean()) if (~m).any() else 0.0
    return out


def _scatter(img, m, how):
    from scipy.interpolate import griddata
    H, W = img.shape
    ys, xs = np.indices((H, W))
    coord = np.column_stack([xs.ravel(), ys.ravel()])
    v = (~m).ravel()
    filled = griddata(coord[v], img.ravel()[v], coord[m.ravel()], method=how)
    if how == "linear":
        nan = np.isnan(filled)
        if nan.any():
            filled[nan] = griddata(coord[v], img.ravel()[v],
                                   coord[m.ravel()][nan], method="nearest")
    out = img.copy()
    out.ravel()[m.ravel()] = filled
    return out


def m_nearest(img, m, **_):
    return _scatter(img, m, "nearest")


def m_griddata(img, m, **_):
    return _scatter(img, m, "linear")


def _axis_interp(img, m, axis):
    """沿 axis 做逐线 1D 线性插值（axis=0 -> 逐列；axis=1 -> 逐行）。"""
    out = img.copy()
    a = img if axis == 0 else img.T
    am = m if axis == 0 else m.T
    gmean = float(img[~m].mean()) if (~m).any() else 0.0
    for j in range(a.shape[1]):
        col_m = am[:, j]
        if not col_m.any():
            continue
        col_v = ~col_m
        nv = int(col_v.sum())
        if nv >= 2:
            a[col_m, j] = np.interp(np.nonzero(col_m)[0],
                                    np.nonzero(col_v)[0], a[col_v, j])
        elif nv == 1:
            a[col_m, j] = a[col_v, j][0]
        else:
            a[col_m, j] = gmean
    if axis == 1:
        return a.T
    return a


def m_col_interp(img, m, **_):
    return _axis_interp(img, m, axis=0)


def m_row_interp(img, m, **_):
    return _axis_interp(img, m, axis=1)


def m_harmonic(img, m, iters=100, init="col", **_):
    """调和插值：迭代解拉普拉斯方程（Dirichlet 边界）。

    init="col" 用列向 1D 插值作初值 —— 对 IR1 的细横条带几乎已是正解，
    迭代只需再平滑少数几次即可收敛（比均值初值快一个量级）。
    """
    from scipy import ndimage
    known = ~m
    if not known.any():
        return np.zeros_like(img, dtype=np.float64)
    if init == "col":
        out = _axis_interp(img, m, axis=0).astype(np.float64)
    else:
        out = img.astype(np.float64).copy()
        out[m] = float(img[known].mean())
    k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float64) / 4.0
    for _ in range(int(iters)):
        out[m] = ndimage.convolve(out, k, mode="nearest")[m]
    return out


def m_crosschannel(img, m, ctx=None, **_):
    """跨通道先验：用同帧另一通道做线性回归预测缺失区（WV<-IR1 等）。"""
    if not ctx or ctx.get("other") is None:
        return m_mean(img, m)
    other = np.asarray(ctx["other"], dtype=np.float64)
    v = ~m
    if int(v.sum()) < 8:
        return m_mean(img, m)
    A = np.column_stack([other[v], np.ones(int(v.sum()))])
    coef, *_ = np.linalg.lstsq(A, img[v], rcond=None)
    out = img.copy()
    out[m] = coef[0] * other[m] + coef[1]
    return out


METHODS = {
    "mean": m_mean,
    "nearest": m_nearest,
    "griddata": m_griddata,
    "row_interp": m_row_interp,
    "col_interp": m_col_interp,
    "harmonic": m_harmonic,
    "crosschannel": m_crosschannel,
}
METHOD_ORDER = ["mean", "nearest", "griddata", "row_interp", "col_interp",
                "harmonic", "crosschannel"]


# --------------------------------------------------------------------------- #
# 评测
# --------------------------------------------------------------------------- #
def _bucket(a):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= a < hi:
            return i
    return len(BUCKETS) - 1


def run_bench(edge_blur=False, seed=0, only=None):
    data = np.load(CACHE)
    clean = data["clean"]
    rng = np.random.default_rng(seed)
    methods = METHOD_ORDER if only is None else only

    errs = {ch: {mt: {} for mt in methods} for ch in ("IR1", "WV")}
    # errs[ch][mt][bucket] = (sum sq, sum abs, count)
    for ch, ci in (("IR1", 0), ("WV", 1)):
        masks = data[f"masks_{ch}"]
        for mi in range(masks.shape[0]):
            m = masks[mi].copy()
            k = int(rng.integers(clean.shape[0]))
            truth = clean[k, :, :, ci].astype(np.float64)
            img = truth.copy()
            img[m] = 0.0
            if edge_blur:
                from scipy import ndimage
                ring = ndimage.binary_dilation(m, iterations=1) & (~m)
                img[ring] = img[ring] * 0.5      # 粗略模拟 bilinear 涂抹
            b = _bucket(float(m.mean()))
            ctx = {"other": clean[k, :, :, 1 - ci].astype(np.float64)}
            for mt in methods:
                f = METHODS[mt](img, m, ctx=ctx)
                e = f[m] - truth[m]
                d = errs[ch][mt].setdefault(b, [0.0, 0.0, 0])
                d[0] += float((e ** 2).sum())
                d[1] += float(np.abs(e).sum())
                d[2] += int(e.size)

    res = {}
    for ch in ("IR1", "WV"):
        res[ch] = {}
        for mt in methods:
            tot_sq = tot_ab = tot_n = 0.0
            per = {}
            for b, (sq, ab, n) in sorted(errs[ch][mt].items()):
                per[str(b)] = {"rmse": round(float(np.sqrt(sq / n)), 3),
                               "mae": round(float(ab / n), 3), "n": n}
                tot_sq += sq
                tot_ab += ab
                tot_n += n
            res[ch][mt] = {
                "rmse": round(float(np.sqrt(tot_sq / tot_n)), 3) if tot_n else None,
                "mae": round(float(tot_ab / tot_n), 3) if tot_n else None,
                "n_pixels": int(tot_n),
                "by_area": per,
            }
    return res


def _print_table(res, methods, title):
    print(f"\n{'='*84}\n{title}\n{'='*84}")
    for ch in ("IR1", "WV"):
        print(f"\n[{ch}]  各方法补全误差（越低越好）")
        print(f"{'method':<12}{'RMSE(K)':>10}{'MAE(K)':>10}   "
              f"{'<2%':>8}{'2-5%':>8}{'5-20%':>8}{'20-60%':>9}{'>60%':>8}")
        base = res[ch].get("griddata", {}).get("rmse") or 1e9
        for mt in methods:
            d = res[ch][mt]
            row = f"{mt:<12}{d['rmse']:>10.3f}{d['mae']:>10.3f}   "
            for b in range(len(BUCKETS)):
                v = d["by_area"].get(str(b), {}).get("rmse")
                row += f"{v:>8.2f}" if v is not None else f"{'-':>8}"
            star = "  <-- 现状" if mt == "griddata" else ""
            print(row + star)
        # 相对现状的改进
        print(f"  相对 griddata 的 RMSE 变化：", end="")
        for mt in methods:
            if mt == "griddata":
                continue
            d = res[ch][mt]["rmse"]
            print(f"{mt} {(d-base)/base*100:+.1f}%  ", end="")
        print()


def _plot(res, methods, edge_blur, seed, sfx=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for j, ch in enumerate(("IR1", "WV")):
        ax = axes[j]
        vals = [res[ch][mt]["rmse"] for mt in methods]
        cols = ["#9aa0a6" if mt == "griddata" else "#1a73e8" for mt in methods]
        bars = ax.bar(range(len(methods)), vals, color=cols)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("inpaint RMSE (K)")
        ax.set_title(f"{ch} — inpaint error (blur={edge_blur})")
        best = int(np.argmin(vals))
        bars[best].set_color("#188038")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=.3)

    # 案例对照：一帧 + 各方法（单独一图）
    data = np.load(CACHE)
    masks = data["masks_IR1"]
    # 挑一个中等条带
    areas = masks.reshape(masks.shape[0], -1).mean(1)
    mi = int(np.argmin(np.abs(areas - 0.015)))
    m = masks[mi]
    truth = data["clean"][0, :, :, 0].astype(np.float64)
    img = truth.copy(); img[m] = 0
    panel = [img]
    ctx_case = {"other": data["clean"][0, :, :, 1].astype(np.float64)}
    for mt in methods:
        panel.append(METHODS[mt](img, m, ctx=ctx_case))
    fig2, axs = plt.subplots(1, len(panel) + 1, figsize=(2.1 * (len(panel) + 1), 2.6))
    axs[0].imshow(truth, cmap="turbo", vmin=180, vmax=300)
    axs[0].set_title("truth", fontsize=8)
    axs[0].axis("off")
    for i, (mt, p) in enumerate(zip(methods, panel)):
        # 只显示掩码区域附近的细节
        axs[i + 1].imshow(p, cmap="turbo", vmin=180, vmax=300)
        rmse = np.sqrt(np.mean((p[m] - truth[m]) ** 2))
        axs[i + 1].set_title(f"{mt}\n{rmse:.2f}K", fontsize=8)
        axs[i + 1].axis("off")
    fig2.suptitle(f"IR1 case (area={m.mean()*100:.1f}%) — inpainted detail", fontsize=10)
    fig2.tight_layout()

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    png = OUT_PNG.replace(".png", sfx + ".png")
    fig.savefig(png, dpi=110)
    fig2.savefig(png.replace(".png", "_case.png"), dpi=110)
    plt.close("all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true", help="强制重新提取掩码缓存")
    ap.add_argument("--edge-blur", action="store_true", help="模拟边缘涂抹污染")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None, help="逗号分隔的方法名")
    args = ap.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if args.extract or not os.path.exists(CACHE):
        extract()
    methods = [m.strip() for m in args.only.split(",")] if args.only else METHOD_ORDER
    res = run_bench(edge_blur=args.edge_blur, seed=args.seed, only=methods)
    _print_table(res, methods,
                 f"补全方法对比（edge_blur={args.edge_blur}）")
    out = {"edge_blur": args.edge_blur, "seed": args.seed, "methods": methods,
           "buckets": BUCKETS, "results": res}
    os.makedirs("outputs", exist_ok=True)
    sfx = "_blur" if args.edge_blur else ""
    jpath = OUT_JSON.replace(".json", sfx + ".json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    if not args.only:
        _plot(res, methods, args.edge_blur, args.seed, sfx)
    print(f"\n已写出: {jpath}")


if __name__ == "__main__":
    main()
