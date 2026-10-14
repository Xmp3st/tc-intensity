#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在「真实损坏帧的面积分布」下，对比 旧U-Net / 新U-Net / griddata 的像素级修复质量。

为什么单独写：原 `eval_inpaint_unet.py` 用「面积 5%~40%」的掩码池评估（中位 8%），
与真实损坏帧（中位 ~3.2%）不符。本脚本统一改用**按真实面积分布重采样的掩码池**，
并让**同一张合成损坏图**同时经过三个修复器，从而得到严格可比的配对结果，
另给按缺测面积分档的细分（真实分布主峰在 1%~8%）。

用法：
  python scripts/eval_unet_areamatch_compare.py \
      --pairs logs/inpaint_pairs_areamatch.npz \
      --ckpt-old outputs/tcir_inpaint_unet/checkpoints/best.ckpt \
      --ckpt-new outputs/tcir_inpaint_unet_areamatch/checkpoints/best.ckpt \
      --out outputs/tcir_inpaint_unet_areamatch
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

NORM_MEAN, NORM_STD = 260.864, 32.849
SENTINEL_NORM = (0.0 - NORM_MEAN) / NORM_STD

BINS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 1.0]
BLAB = ["<1%", "1-2%", "2-3%", "3-5%", "5-8%", "8-12%", ">12%"]


def gridata_inpaint(damaged_norm, mask):
    from tcintens.data.band_health import inpaint_regions
    raw = (damaged_norm * NORM_STD + NORM_MEAN).astype(np.float64)
    m = mask.astype(bool)
    filled = inpaint_regions(raw[None], m[None], method="griddata")
    return ((filled[0] - NORM_MEAN) / NORM_STD).astype(np.float32)


def region_metrics(gt_norm, pred_norm, mask):
    gt_k = gt_norm * NORM_STD + NORM_MEAN
    pr_k = pred_norm * NORM_STD + NORM_MEAN
    m = mask > 0
    out = {}
    for name, sel in [("full", np.ones_like(m, bool)), ("roi", m)]:
        e = (pr_k - gt_k)[sel]
        out[f"{name}_rmse"] = float(np.sqrt(np.mean(e ** 2)))
        out[f"{name}_mae"] = float(np.mean(np.abs(e)))
    return out


def rmse(a):
    a = np.asarray(a, np.float64)
    return float(np.sqrt(np.mean(a ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/wpac_96.h5")
    ap.add_argument("--pairs", default="logs/inpaint_pairs_areamatch.npz")
    ap.add_argument("--ckpt-old", default="outputs/tcir_inpaint_unet/checkpoints/best.ckpt")
    ap.add_argument("--ckpt-new", default="outputs/tcir_inpaint_unet_areamatch/checkpoints/best.ckpt")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="outputs/tcir_inpaint_unet_areamatch")
    args = ap.parse_args()

    os.chdir(ROOT)
    import h5py
    import torch
    from tcintens.models.unet import UNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(ck):
        net = UNet(in_channels=2, out_channels=1, base=48).to(device)
        z = torch.load(ck, map_location=device)
        net.load_state_dict(z["model"]); net.eval()
        return net, z.get("best_roi")

    net_old, roi_old = load(args.ckpt_old)
    net_new, roi_new = load(args.ckpt_new)
    print(f"[model] old={args.ckpt_old} (best_roi={roi_old})")
    print(f"[model] new={args.ckpt_new} (best_roi={roi_new})")

    z = np.load(args.pairs)
    test_idx = z["test_clean_idx"].astype(int)
    masks = z["masks"].astype(np.float32)
    rng = np.random.default_rng(args.seed)
    n = min(args.n, len(test_idx))
    sel = np.sort(rng.choice(len(test_idx), size=n, replace=False))
    print(f"[data] 掩码池 {len(masks)}（真实面积分布）｜评估 {n} 张测试干净帧")

    def infer(net, x):
        xt = torch.from_numpy(x[None]).to(device)
        with torch.no_grad():
            return net(xt).squeeze(0).squeeze(0).cpu().numpy()

    rows = []
    with h5py.File(args.h5, "r") as hf:
        mat = hf["matrix"]
        for si in sel:
            idx = int(test_idx[si])
            clean = np.asarray(mat[idx, ..., 0], np.float32)
            clean_n = (clean - NORM_MEAN) / NORM_STD

            m = masks[rng.integers(len(masks))].copy()
            if rng.random() < 0.5:
                m = m[:, ::-1].copy()
            shift = int(rng.integers(-24, 25))
            if shift:
                m = np.roll(m, shift, axis=0)

            damaged = clean_n.copy()
            damaged[m > 0.5] = SENTINEL_NORM
            x = np.stack([damaged, m], axis=0).astype(np.float32)

            gd = gridata_inpaint(damaged, m)
            uo = infer(net_old, x)
            un = infer(net_new, x)

            mg = region_metrics(clean_n, gd, m)
            mo = region_metrics(clean_n, uo, m)
            mn = region_metrics(clean_n, un, m)
            rows.append({
                "idx": idx, "mask_frac": float(m.mean()),
                "gd_roi_rmse": mg["roi_rmse"], "gd_roi_mae": mg["roi_mae"],
                "gd_full_rmse": mg["full_rmse"],
                "old_roi_rmse": mo["roi_rmse"], "old_roi_mae": mo["roi_mae"],
                "old_full_rmse": mo["full_rmse"],
                "new_roi_rmse": mn["roi_rmse"], "new_roi_mae": mn["roi_mae"],
                "new_full_rmse": mn["full_rmse"],
            })

    af = np.array([r["mask_frac"] for r in rows])
    agg = {k: float(np.mean([r[k] for r in rows])) for k in
           ["gd_roi_rmse", "gd_roi_mae", "old_roi_rmse", "old_roi_mae",
            "new_roi_rmse", "new_roi_mae", "gd_full_rmse", "old_full_rmse",
            "new_full_rmse"]}
    print("\n===== 像素级汇总（亮温 K，越低越好；掩码池=真实面积分布）=====")
    print(f"缺测区 RMSE : griddata {agg['gd_roi_rmse']:.3f} | 旧U-Net {agg['old_roi_rmse']:.3f} "
          f"| 新U-Net {agg['new_roi_rmse']:.3f}")
    print(f"缺测区 MAE  : griddata {agg['gd_roi_mae']:.3f} | 旧U-Net {agg['old_roi_mae']:.3f} "
          f"| 新U-Net {agg['new_roi_mae']:.3f}")
    print(f"全图 RMSE    : griddata {agg['gd_full_rmse']:.3f} | 旧U-Net {agg['old_full_rmse']:.3f} "
          f"| 新U-Net {agg['new_full_rmse']:.3f}")

    print("\n===== 按缺测面积分档（缺测区 RMSE）=====")
    bin_rows = []
    for lo, hi, lab in zip(BINS[:-1], BINS[1:], BLAB):
        s = (af >= lo) & (af < hi)
        if s.sum() == 0:
            continue
        e = {"bin": lab, "n": int(s.sum()),
             "gd": rmse([r["gd_roi_rmse"] for r, k in zip(rows, s) if k]),
             "old": rmse([r["old_roi_rmse"] for r, k in zip(rows, s) if k]),
             "new": rmse([r["new_roi_rmse"] for r, k in zip(rows, s) if k])}
        bin_rows.append(e)
        print(f"{lab:7s} n={e['n']:3d}  gd {e['gd']:7.3f} | old {e['old']:7.3f} "
              f"| new {e['new']:7.3f}   (new-old {e['new']-e['old']:+.3f})")

    # 配对胜率
    d = np.array([r["new_roi_rmse"] - r["old_roi_rmse"] for r in rows])
    from scipy import stats
    t, p = stats.ttest_rel([r["new_roi_rmse"] for r in rows],
                           [r["old_roi_rmse"] for r in rows])
    win = float(np.mean(d < 0))
    print(f"\n[配对] 新 vs 旧（缺测区 RMSE）：均值差 {d.mean():+.4f} K，t={t:.2f}，"
          f"p={p:.2e}，新胜率 {win*100:.1f}%")

    os.makedirs(args.out, exist_ok=True)
    keys = ["gd_roi_rmse", "gd_roi_mae", "gd_full_rmse", "old_roi_rmse",
            "old_roi_mae", "old_full_rmse", "new_roi_rmse", "new_roi_mae",
            "new_full_rmse"]
    with open(os.path.join(args.out, "eval_areamatch_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({"pairs": args.pairs, "ckpt_old": args.ckpt_old,
                   "ckpt_new": args.ckpt_new, "n": n,
                   "best_roi_old": roi_old, "best_roi_new": roi_new,
                   "aggregate": agg, "by_area_bin": bin_rows,
                   "paired_new_vs_old": {"mean_diff": float(d.mean()),
                                         "t": float(t), "p": float(p),
                                         "win_rate_new": win}},
                  f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "eval_areamatch_per_sample.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "mask_frac"] + keys)
        for r in rows:
            w.writerow([r["idx"], r["mask_frac"]] + [r[k] for k in keys])
    print("[save]", os.path.join(args.out, "eval_areamatch_summary.json"))

    try:
        _plot(rows, af, agg, bin_rows, os.path.join(args.out, "eval_areamatch.png"))
    except Exception as e:
        print("[warn] plot failed:", e)


def _plot(rows, af, agg, bin_rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tcintens.utils.plotting import use_cjk_font
    use_cjk_font()
    C = {"gd": "#378ADD", "old": "#888780", "new": "#E24B4A"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    ax = axes[0]
    lab = ["griddata", "旧U-Net(5~40%)", "新U-Net(真实分布)"]
    v = [agg["gd_roi_rmse"], agg["old_roi_rmse"], agg["new_roi_rmse"]]
    b = ax.bar(lab, v, color=[C["gd"], C["old"], C["new"]])
    for r, x in zip(b, v):
        ax.text(r.get_x() + r.get_width() / 2, x + 0.05, f"{x:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("缺测区 RMSE (K)")
    ax.set_title("缺测区修复精度（真实面积分布掩码池）")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    x = np.arange(len(bin_rows)); w = 0.26
    ax.bar(x - w, [e["gd"] for e in bin_rows], w, label="griddata", color=C["gd"])
    ax.bar(x, [e["old"] for e in bin_rows], w, label="旧U-Net", color=C["old"])
    ax.bar(x + w, [e["new"] for e in bin_rows], w, label="新U-Net", color=C["new"])
    ax.set_xticks(x); ax.set_xticklabels([f"{e['bin']}\nn={e['n']}" for e in bin_rows],
                                         fontsize=8)
    ax.set_ylabel("缺测区 RMSE (K)")
    ax.set_title("按缺测面积分档")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print("[save]", path)


if __name__ == "__main__":
    main()
