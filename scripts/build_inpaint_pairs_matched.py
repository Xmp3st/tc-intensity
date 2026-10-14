#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按「真实损坏帧的缺测面积分布」重建 U-Net 训练的增广掩码池。

背景
----
原 `build_inpaint_pairs.py` 从真实掩码库 `logs/ir1_strip_masks.npz`（3355 个，
来自训练年真实损坏帧，面积中位 3.30%）里**只取了面积 5%~40% 的子集**
（1155 个，中位 8.05%）⇒ 训练时的损坏面积系统性偏大，与部署时真实见到的
损伤（中位 ~3.3%、多数 <5%）不匹配。下游实验里 U-Net 的优势恰好只出现在
「大块损坏」档，正是这一分布错配的直接后果。

做法
----
1. 扫描**全集** IR1（与检测同口径：sentinel=0 + low_thr=150K + min_blob=20 +
   dilate=1），统计**真实损坏帧**的缺测面积占比分布 → 目标分布；
   同时给出训练年(2003-2013)/测试年(2014-2016)两个子分布作对照。
2. 对掩码库做**按面积直方图的加权重采样**：把面积轴分箱后，
   使重采样池的每箱计数 ∝ 真实损坏帧的每箱占比（即逐箱频率匹配）。
3. 复用原 pairs 的训练/测试干净帧索引（同一 by_year 划分），只替换掩码池，
   落盘新 pairs + 统计 json + 分布对比图。

用法
----
  python scripts/build_inpaint_pairs_matched.py
  python scripts/build_inpaint_pairs_matched.py --pool-size 4000 --out logs/inpaint_pairs_areamatch.npz
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 面积占比分箱边界（覆盖掩码库有效区间 0.4%~50%，低端加密以分辨真实分布的主峰）
DEFAULT_EDGES = [0.0, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025,
                 0.03, 0.04, 0.05, 0.065, 0.08, 0.10, 0.13, 0.18, 0.25,
                 0.35, 0.50, 1.0]


def parse_years(spec):
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(str(y) for y in range(int(a), int(b) + 1))
        else:
            out.add(str(part))
    return out


def scan_real_area(h5_path, idx, col, batch=512, verbose=True):
    """逐帧算 IR1 缺测面积占比，返回 (area_frac_all, area_frac_damaged)。

    与训练/评估完全同一口径：detect_missing_region(sentinel=0, low_thr=150K,
    min_blob=20, dilate=1)。
    """
    import h5py
    from tcintens.data.band_health import detect_missing_region

    area = np.zeros(len(idx), dtype=np.float32)
    with h5py.File(h5_path, "r") as hf:
        mat = hf["matrix"]
        for s in range(0, len(idx), batch):
            rows = mat[idx[s:s + batch]]
            for k in range(rows.shape[0]):
                ch = rows[k][..., col].astype(np.float64)
                m = detect_missing_region(ch[None], sentinel=0.0,
                                          low_thr=[150.0], min_blob=20, dilate=1)[0]
                if m.any():
                    area[s + k] = float(m.mean())
            if verbose:
                print(f"  ... {min(s + batch, len(idx))}/{len(idx)}", flush=True)
    dam = area > 0
    return area, area[dam]


def hist_freq(x, edges):
    h = np.histogram(x, bins=edges)[0].astype(np.float64)
    s = h.sum()
    return (h / s) if s > 0 else h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/wpac_96.h5")
    ap.add_argument("--csv", default="data/wpac_compact.csv")
    ap.add_argument("--masks", default="logs/ir1_strip_masks.npz")
    ap.add_argument("--base-pairs", default="logs/inpaint_pairs.npz",
                    help="复用其 train/test 干净帧索引")
    ap.add_argument("--out", default="logs/inpaint_pairs_areamatch.npz")
    ap.add_argument("--pool-size", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-years", default="2003-2013")
    ap.add_argument("--test-years", default="2014-2016")
    ap.add_argument("--ir1-col", type=int, default=0)
    args = ap.parse_args()

    os.chdir(ROOT)

    edges = np.asarray(DEFAULT_EDGES, dtype=np.float64)
    nbin = len(edges) - 1

    df = pd.read_csv(args.csv)
    df["year"] = df["ID"].astype(str).str[:4]
    all_idx = np.sort(df["matrix_index"].to_numpy().astype(int))
    tr_years, te_years = parse_years(args.train_years), parse_years(args.test_years)

    # ---------- 1) 真实损坏帧面积分布（全集） ----------
    print(f"[scan] 全集 {len(all_idx)} 帧，逐帧检测 IR1 缺测面积 ...", flush=True)
    area_all, area_dam = scan_real_area(args.h5, all_idx, args.ir1_col)
    year_of = df.set_index("matrix_index")["year"].reindex(all_idx).to_numpy()
    dam = area_all > 0
    dam_tr = dam & np.isin(year_of, list(tr_years))
    dam_te = dam & np.isin(year_of, list(te_years))
    real_tr = area_all[dam_tr]
    real_te = area_all[dam_te]

    def q(a):
        if len(a) == 0:
            return {}
        ps = [10, 25, 50, 75, 90, 95, 99, 100]
        v = np.percentile(a, ps)
        return {f"p{p}": round(float(x), 5) for p, x in zip(ps, v)}

    print(f"[real] 全集损坏帧 {int(dam.sum())}/{len(all_idx)}"
          f"（{100*dam.mean():.1f}%）")
    print(f"[real] 训练年损坏 {int(dam_tr.sum())}｜测试年损坏 {int(dam_te.sum())}")
    print(f"[real] 缺测面积% 分位（全集损坏帧）: {q(area_dam*100)}")

    # ---------- 2) 掩码库面积分布 ----------
    z = np.load(args.masks)
    lib = z["masks"].astype(np.uint8)
    lib_area = lib.reshape(len(lib), -1).mean(1).astype(np.float64)
    print(f"[lib] {len(lib)} 个掩码｜面积% min {lib_area.min()*100:.3f} "
          f"中位 {np.median(lib_area)*100:.3f} p90 {np.percentile(lib_area,90)*100:.3f} "
          f"max {lib_area.max()*100:.3f}")

    # ---------- 3) 按面积直方图重采样到匹配真实分布 ----------
    # 目标分布 = 全集损坏帧（训练域与部署域一致的最合理描述）
    p_real = hist_freq(area_dam, edges)
    lib_bin = np.clip(np.digitize(lib_area, edges) - 1, 0, nbin - 1)
    cand = [np.where(lib_bin == b)[0] for b in range(nbin)]

    rng = np.random.default_rng(args.seed)
    picks, bin_targets, bin_avail, bin_taken = [], [], [], []
    for b in range(nbin):
        n_t = int(round(p_real[b] * args.pool_size))
        n_a = int(len(cand[b]))
        bin_targets.append(n_t)
        bin_avail.append(n_a)
        if n_t <= 0 or n_a == 0:
            bin_taken.append(0)
            continue
        take = rng.choice(cand[b], size=n_t, replace=True)   # 有放回：精确凑频率
        picks.append(take)
        bin_taken.append(n_t)
    pool_idx = (np.concatenate(picks) if picks else np.zeros(0, int))
    rng.shuffle(pool_idx)
    pool = lib[pool_idx]
    pool_area = lib_area[pool_idx]

    # 缺失箱（真实有占比但库里无候选）的占比损失
    lost = float(sum(p_real[b] for b in range(nbin)
                     if bin_targets[b] > 0 and bin_avail[b] == 0))
    p_new = hist_freq(pool_area, edges)
    tv = float(0.5 * np.abs(p_new - p_real).sum())         # 总变差距离

    print(f"[pool] 采样 {len(pool)} 个掩码｜面积% 中位 {np.median(pool_area)*100:.3f} "
          f"p90 {np.percentile(pool_area,90)*100:.3f}")
    print(f"[pool] 直方图匹配：被丢弃占比 {lost*100:.3f}%，总变差距离 {tv:.4f}")

    # ---------- 4) 复用干净帧索引，落盘 ----------
    bp = np.load(args.base_pairs)
    tr_clean = bp["train_clean_idx"].astype(int)
    te_clean = bp["test_clean_idx"].astype(int)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, train_clean_idx=tr_clean,
                        test_clean_idx=te_clean, masks=pool.astype(np.uint8))

    stats = {
        "h5": args.h5, "csv": args.csv, "masks_src": args.masks,
        "base_pairs": args.base_pairs, "out": args.out,
        "n_train_clean": int(len(tr_clean)), "n_test_clean": int(len(te_clean)),
        "n_masks_pool": int(len(pool)), "n_masks_lib": int(len(lib)),
        "real_damaged_frames": {
            "all": int(dam.sum()), "train_years": int(dam_tr.sum()),
            "test_years": int(dam_te.sum()),
            "area_frac_quantiles_all_pct": q(area_dam * 100),
            "area_frac_quantiles_train_years_pct": q(real_tr * 100),
            "area_frac_quantiles_test_years_pct": q(real_te * 100),
        },
        "lib_area_frac_pct": {
            "min": round(float(lib_area.min() * 100), 4),
            "median": round(float(np.median(lib_area) * 100), 4),
            "p90": round(float(np.percentile(lib_area, 90) * 100), 4),
            "max": round(float(lib_area.max() * 100), 4),
        },
        "pool_area_frac_pct": {
            "min": round(float(pool_area.min() * 100), 4),
            "median": round(float(np.median(pool_area) * 100), 4),
            "p90": round(float(np.percentile(pool_area, 90) * 100), 4),
            "max": round(float(pool_area.max() * 100), 4),
        },
        "histogram_matching": {
            "edges": [round(float(e), 5) for e in edges],
            "real_freq": [round(float(x), 6) for x in p_real],
            "pool_freq": [round(float(x), 6) for x in p_new],
            "lib_candidates_per_bin": bin_avail,
            "pool_count_per_bin": bin_taken,
            "lost_mass": round(lost, 6),
            "total_variation_distance": round(tv, 6),
        },
        "params": {"pool_size": args.pool_size, "seed": args.seed,
                   "ir1_col": args.ir1_col,
                   "detect": {"sentinel": 0.0, "low_thr": 150.0,
                              "min_blob": 20, "dilate": 1}},
    }
    sp = os.path.splitext(args.out)[0] + "_stats.json"
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("[save]", args.out)
    print("[save]", sp)

    # ---------- 5) 分布对比图 ----------
    try:
        _plot(area_dam, lib_area, pool_area, edges, stats,
              os.path.splitext(args.out)[0] + "_dist.png")
    except Exception as e:                                  # 可视化失败不影响主流程
        print("[warn] plot failed:", e)


def _plot(real, lib_area, pool_area, edges, stats, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tcintens.utils.plotting import use_cjk_font
    use_cjk_font()

    e = np.asarray(edges) * 100
    centers = 0.5 * (e[:-1] + e[1:])
    w = np.diff(e)

    def dens(a):
        h = np.histogram(a * 100, bins=e)[0].astype(float)
        return h / max(h.sum(), 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    ax = axes[0]
    ax.bar(centers, dens(real), width=w * 0.9, color="#888780",
           alpha=0.85, label=f"真实损坏帧 (n={stats['real_damaged_frames']['all']})")
    ax.bar(centers, dens(lib_area), width=w * 0.55, color="#378ADD",
           alpha=0.9, label="掩码库(全集, 3355)")
    ax.bar(centers, dens(pool_area), width=w * 0.3, color="#E24B4A",
           alpha=0.95, label=f"新采样池 (n={stats['n_masks_pool']})")
    ax.set_xlabel("缺测面积占比 (%)"); ax.set_ylabel("频率")
    ax.set_title("面积分布：真实损坏帧 vs 掩码库 vs 新池")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    lab = ["真实损坏帧", "掩码库全集", "旧训练池(5~40%)", "新采样池"]
    med = [np.median(real) * 100, np.median(lib_area) * 100, 8.05,
           np.median(pool_area) * 100]
    p90 = [np.percentile(real, 90) * 100, np.percentile(lib_area, 90) * 100,
           12.8, np.percentile(pool_area, 90) * 100]
    x = np.arange(len(lab)); wd = 0.38
    b1 = ax.bar(x - wd / 2, med, wd, color="#2C7FB8", label="中位面积")
    b2 = ax.bar(x + wd / 2, p90, wd, color="#F0A202", label="p90 面积")
    for b in (b1, b2):
        for r in b:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.15,
                    f"{r.get_height():.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=9)
    ax.set_ylabel("缺测面积占比 (%)")
    ax.set_title("中位 / p90 面积对照")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("[save]", path)


if __name__ == "__main__":
    main()
