#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评估 U-Net 修复效果，并与 griddata 插值基线对比。

在「测试年(2014-2016)干净帧」上贴掩码合成损坏图，分别用：
  (1) U-Net 修复
  (2) griddata 空间插值（项目既有 inpaint_regions）
修复后与干净真值对比，计算：
  - 缺测区域内 RMSE / MAE（核心：补回来的像素准不准）
  - 全图 RMSE / MAE / PSNR / SSIM
并生成可视化对比图（干净 / 损坏 / griddata / U-Net）。

用法：
  python scripts/eval_inpaint_unet.py
  python scripts/eval_inpaint_unet.py --ckpt outputs/tcir_inpaint_unet/checkpoints/best.ckpt
"""
import argparse
import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NORM_MEAN = 260.864
NORM_STD = 32.849
SENTINEL_NORM = (0.0 - NORM_MEAN) / NORM_STD


def gridata_inpaint(damaged_norm, mask):
    """用项目 griddata 插值修复归一化后的损坏图（在原始亮温域做，再归一化）。"""
    from tcintens.data.band_health import inpaint_regions
    # 转回原始亮温域
    raw = damaged_norm * NORM_STD + NORM_MEAN
    raw = raw.astype(np.float64)
    m = mask.astype(bool)
    # inpaint_regions 接受 (C,H,W)
    arr = raw[None]                       # (1,H,W)
    filled = inpaint_regions(arr, m[None], method="griddata")
    out = (filled[0] - NORM_MEAN) / NORM_STD
    return out.astype(np.float32)


def psnr(gt, pred, mask=None):
    if mask is not None and mask.sum() > 0:
        gt = gt[mask > 0]; pred = pred[mask > 0]
    mse = float(np.mean((gt - pred) ** 2))
    if mse == 0:
        return float("inf")
    # 归一化空间里的 PSNR：范围约 [-8, 2]，用 20*log10(1/mse^0.5) 无意义，
    # 改用真实亮温域动态范围。这里统一转回亮温域算。
    return float(10 * np.log10(1.0 / mse))


def metrics_region(gt_norm, pred_norm, mask):
    """在归一化空间计算缺测区域与全图的 RMSE/MAE（亮温单位 kn 需还原）。"""
    # 转回亮温域（Kelvin）
    gt_k = gt_norm * NORM_STD + NORM_MEAN
    pred_k = pred_norm * NORM_STD + NORM_MEAN
    m = mask > 0
    out = {}
    for name, sel in [("full", np.ones_like(m, bool)), ("roi", m)]:
        e = (pred_k - gt_k)[sel]
        out[f"{name}_rmse"] = float(np.sqrt(np.mean(e ** 2)))
        out[f"{name}_mae"] = float(np.mean(np.abs(e)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/wpac_96.h5")
    ap.add_argument("--pairs", default="logs/inpaint_pairs.npz")
    ap.add_argument("--ckpt", default="outputs/tcir_inpaint_unet/checkpoints/best.ckpt")
    ap.add_argument("--n", type=int, default=500, help="评估样本数（测试年干净帧抽样）")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="outputs/tcir_inpaint_unet")
    args = ap.parse_args()

    import h5py
    import torch
    from tcintens.models.unet import UNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=2, out_channels=1, base=48).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[model] 加载 {args.ckpt}（best_roi={ck.get('best_roi')}）")

    z = np.load(args.pairs)
    test_idx = z["test_clean_idx"].astype(int)
    masks = z["masks"].astype(np.float32)
    rng = np.random.default_rng(args.seed)
    n = min(args.n, len(test_idx))
    sel_idx = np.sort(rng.choice(len(test_idx), size=n, replace=False))
    print(f"[data] 测试干净帧 {len(test_idx)}，抽样 {n}")

    rows = []
    with h5py.File(args.h5, "r") as hf:
        mat = hf["matrix"]
        for j, si in enumerate(sel_idx):
            idx = int(test_idx[si])
            clean = np.asarray(mat[idx, ..., 0], dtype=np.float32)
            clean_n = (clean - NORM_MEAN) / NORM_STD

            m = masks[rng.integers(len(masks))].copy()
            if rng.random() < 0.5:
                m = m[:, ::-1].copy()
            shift = int(rng.integers(-24, 25))
            if shift:
                m = np.roll(m, shift, axis=0)

            damaged = clean_n.copy()
            damaged[m > 0.5] = SENTINEL_NORM

            # griddata 基线
            gd = gridata_inpaint(damaged, m)

            # U-Net
            x = np.stack([damaged, m], axis=0).astype(np.float32)
            xt = torch.from_numpy(x[None]).to(device)
            with torch.no_grad():
                un = model(xt).squeeze(0).squeeze(0).cpu().numpy()

            mg = metrics_region(clean_n, gd, m)
            mu = metrics_region(clean_n, un, m)
            rows.append({
                "idx": idx, "mask_frac": float(m.mean()),
                "gd_roi_rmse": mg["roi_rmse"], "gd_roi_mae": mg["roi_mae"],
                "gd_full_rmse": mg["full_rmse"], "gd_full_mae": mg["full_mae"],
                "unet_roi_rmse": mu["roi_rmse"], "unet_roi_mae": mu["roi_mae"],
                "unet_full_rmse": mu["full_rmse"], "unet_full_mae": mu["full_mae"],
            })

            if j < 6:
                _save_case(os.path.join(args.out, "cases"), j, clean, damaged,
                           m, gd, un)

    # 汇总
    keys = ["gd_roi_rmse", "gd_roi_mae", "gd_full_rmse", "gd_full_mae",
            "unet_roi_rmse", "unet_roi_mae", "unet_full_rmse", "unet_full_mae"]
    agg = {}
    for k in keys:
        vals = np.array([r[k] for r in rows])
        agg[k] = float(np.mean(vals))
    print("\n===== 汇总（亮温 K，越低越好）=====")
    print(f"缺测区域 RMSE : griddata {agg['gd_roi_rmse']:.3f}  vs  U-Net {agg['unet_roi_rmse']:.3f}")
    print(f"缺测区域 MAE  : griddata {agg['gd_roi_mae']:.3f}  vs  U-Net {agg['unet_roi_mae']:.3f}")
    print(f"全图 RMSE      : griddata {agg['gd_full_rmse']:.3f}  vs  U-Net {agg['unet_full_rmse']:.3f}")
    print(f"全图 MAE       : griddata {agg['gd_full_mae']:.3f}  vs  U-Net {agg['unet_full_mae']:.3f}")

    jp = os.path.join(args.out, "eval_summary.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump({"n": n, "aggregate": agg, "per_sample": rows}, f,
                  ensure_ascii=False, indent=2)
    cp = os.path.join(args.out, "eval_summary.csv")
    with open(cp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "mask_frac"] + keys)
        for r in rows:
            w.writerow([r["idx"], r["mask_frac"]] + [r[k] for k in keys])
    print("[save]", jp)
    print("[save]", cp)


def _save_case(odir, j, clean, damaged, mask, gd, un):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tcintens.utils.plotting import use_cjk_font
    use_cjk_font()
    os.makedirs(odir, exist_ok=True)
    # 统一亮温域显示。
    # 注意：clean 传入时已是原始亮温(K)，damaged/gd/un 是归一化值 ⇒ 只对后者做反归一化，
    # 否则真值会被放大到数千 K，vmin/vmax 爆掉导致后三格全黑。
    def to_k(a):
        return a * NORM_STD + NORM_MEAN
    imgs = [clean, to_k(damaged), to_k(gd), to_k(un)]
    titles = ["干净真值", "损坏(贴掩码)", "griddata 插值", "U-Net 修复"]
    # 拉伸范围只在未损坏像素上取（排除哨兵 0K，避免拉垮对比度）
    valid = clean[mask < 0.5]
    vmin = float(np.nanmin(valid)) - 5
    vmax = float(np.nanmax(valid)) + 5
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    for ax, im, t in zip(axes, imgs, titles):
        ax.imshow(im, cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(t, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"IR1 条状缺测修复对比（case {j}）")
    fig.tight_layout()
    fp = os.path.join(odir, f"case_{j:02d}.png")
    fig.savefig(fp, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
