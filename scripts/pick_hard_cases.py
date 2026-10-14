#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pick_hard_cases.py —— 从 corrected_vs_raw.csv 逐样本明细里挑「两个极端」的样本，
画出 [原始 glitch 输入 | 插值修复后输入] 的成对对比图。

选取口径（按 abs_err_improve = |err_raw| - |err_corrected|）：
  * improved     : abs_err_improve 最大 —— 修复把大错纠正回来的样本
  * not improved : abs_err_improve 最小（最负）—— 本来对、修复后反而偏离更多的样本

图上：RAW 面板用红色半透明覆盖「被修复的缺测像素」（= raw 与 corrected 的差异位置），
      因为缺测通常只发生在 IR1 单通道的细条带上，不标出来在合成配色里几乎看不出来。

用法
  python scripts/pick_hard_cases.py -c configs/tcir_wpac_train.yaml \
      --csv outputs/tcir_wpac/tcir_wpac_bb/corrraw/resnet50__griddata/corrected_vs_raw.csv \
      --set experiment.name=bb_resnet50_s42 \
      --set experiment.output_dir=outputs/tcir_wpac/tcir_wpac_bb \
      --topk 6

说明：本脚本只取数据、不推理（预测值直接读 CSV），因此不必加载权重；
      但需保证数据侧配置与生成该 CSV 时一致（interp 修复 + 同一 split_seed）。
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tcintens.data.dataset import create_datasets
from tcintens.data.tcir import TCIRDataset
from tcintens.utils.config import load_config
from tcintens.utils.logger import get_logger
from tcintens.utils.plotting import use_cjk_font
from tcintens.utils.repro import align_preprocessing

C_RAW = "#C44E52"
C_CORR = "#55A868"
MISS_RGB = (1.0, 0.12, 0.12)
# 亮温通道的物理下限：低于它视为缺测，不计入显示拉伸（否则 0 会把对比度拉垮）
_BT_CHANNELS = ("IR1", "WV", "IR2", "VIS")


def denorm_for_display(x, mean, std):
    """(C,H,W) 归一化张量 -> 反归一化物理量 numpy。"""
    x = x.cpu().numpy().astype(np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    return x * std[:, None, None] + mean[:, None, None]


def composite_shared(arrs, ch_names, miss_mask=None):
    """共享拉伸的伪 RGB。

    与 corrected_vs_raw.to_composite_shared 的区别：计算 vmin/vmax 时**排除缺测像素**
    （亮温通道 <=1K 的值），并把传给 miss_mask 的像素强制成红色，让“被修复的位置”
    在视觉上直接可见。
    """
    C = arrs[0].shape[0]
    lows = np.array([1.0 if str(n).upper() in _BT_CHANNELS else -np.inf
                     for n in ch_names], dtype=np.float64)

    vmin = np.full(C, np.inf)
    vmax = np.full(C, -np.inf)
    for a in arrs:
        for c in range(C):
            v = a[c].ravel()
            v = v[v > lows[c]]
            if v.size == 0:
                continue
            vmin[c] = min(vmin[c], float(v.min()))
            vmax[c] = max(vmax[c], float(v.max()))

    outs = []
    for a in arrs:
        rgb = np.zeros((a.shape[1], a.shape[2], 3), dtype=np.float32)
        for i in range(3):
            ci = min(i, C - 1)
            d = vmax[ci] - vmin[ci]
            if d > 1e-8:
                rgb[..., i] = np.clip((a[ci] - vmin[ci]) / d, 0.0, 1.0)
        rgb = np.nan_to_num(rgb)
        if miss_mask is not None:
            m = miss_mask.astype(np.float32)
            for i in range(3):
                rgb[..., i] = rgb[..., i] * (1 - 0.55 * m) + MISS_RGB[i] * 0.55 * m
        outs.append(np.clip(rgb, 0, 1))
    return outs


def _corner(ax, text, color, fs=8.5):
    ax.text(0.04, 0.96, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=fs, color="white", zorder=5,
            bbox=dict(facecolor=color, edgecolor="none", boxstyle="round,pad=0.22"))


def read_method_name(csv_path, fallback="griddata"):
    """从同目录的 corrected_vs_raw.json 读出本次 corrected 用的是哪种补全方法。"""
    jf = os.path.join(os.path.dirname(os.path.abspath(csv_path)),
                      "corrected_vs_raw.json")
    if not os.path.exists(jf):
        return fallback
    try:
        with open(jf, encoding="utf-8") as fh:
            cm = (json.load(fh) or {}).get("corrected_method", {}) or {}
        name = cm.get("missing_region_method") or fallback
        if cm.get("missing_region_feather"):
            name += f"+feather{cm['missing_region_feather']}"
        return name
    except Exception:
        return fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/tcir_wpac_train.yaml")
    ap.add_argument("--csv", required=True, help="corrected_vs_raw.csv 路径")
    ap.add_argument("--set", action="append", default=[], help="覆盖配置 KEY=VAL")
    ap.add_argument("--topk", type=int, default=6, help="每组挑几张")
    ap.add_argument("--out", default=None, help="输出图路径（默认与 CSV 同目录）")
    args = ap.parse_args()

    os.chdir(ROOT)
    use_cjk_font()
    logger = get_logger("tc-hardcases")

    df = pd.read_csv(args.csv)
    imp = df.nlargest(args.topk, "abs_err_improve").reset_index(drop=True)
    wor = df.nsmallest(args.topk, "abs_err_improve").reset_index(drop=True)
    logger.info(f"明细共 {len(df)} 个 glitch 样本")
    logger.info(f"improved 区间 {imp['abs_err_improve'].min():+.2f} ~ "
                f"{imp['abs_err_improve'].max():+.2f} kn")
    logger.info(f"not-improved 区间 {wor['abs_err_improve'].min():+.2f} ~ "
                f"{wor['abs_err_improve'].max():+.2f} kn")

    # ---- 数据（只取图，不推理）----
    cfg = load_config(args.config, args.set)
    cfg, _ = align_preprocessing(cfg, overrides=getattr(cfg, "overrides", []),
                                 logger=logger)
    d = cfg.to_dict()
    tcfg = d["data"]["tcir"]
    ch_names = tcfg.get("channels", ["IR1", "WV", "PMW"])
    norm_mean, norm_std = tcfg.get("norm_mean"), tcfg.get("norm_std")

    datasets, _ = create_datasets(cfg)
    test_ds = datasets["test"]
    test_ds.return_corrected_flag = True
    raw_ds = TCIRDataset(
        h5_path=test_ds.h5_path, indices=list(test_ds.indices),
        channels=test_ds.channels, target=test_ds.target,
        label_unit=test_ds.label_unit, nan_mode=test_ds.nan_mode,
        normalize=test_ds.normalize, norm_mean=test_ds.norm_mean,
        norm_std=test_ds.norm_std, resize=test_ds.resize,
        split="test", missing_band="none",
        frame_labels=list(test_ds.labels),
    )

    def grab(test_index):
        xr, _ = raw_ds[int(test_index)]
        xc, _, _ = test_ds[int(test_index)]
        return (denorm_for_display(xr, norm_mean, norm_std),
                denorm_for_display(xc, norm_mean, norm_std))

    # ---- 绘图：2 组 × (2 行 × 3 样本 × 2 面板) ----
    k = args.topk
    per_row = 3 if k % 3 == 0 else 2
    rows_per_group = int(np.ceil(k / per_row))
    nrow, ncol = rows_per_group * 2, per_row * 2
    top, bottom, hspace = 0.845, 0.015, 0.46
    fig = plt.figure(figsize=(ncol * 2.05, nrow * 2.6))
    gs = fig.add_gridspec(nrow, ncol, wspace=0.05, hspace=hspace,
                          left=0.01, right=0.99, top=top, bottom=bottom)

    row_h = (top - bottom) / (nrow + (nrow - 1) * hspace)

    groups = [
        ("improved", imp,
         f"修复收益最大 Top-{k}：原始输入大错 → 修复后纠回（improved）", "#2E7D32"),
        ("not_improved", wor,
         f"修复后反而偏离最多 Top-{k}：原始输入已接近真值 → 修复后跑偏（not improved）",
         "#B0413E"),
    ]
    for gi, (_tag, sub, title, tcol) in enumerate(groups):
        r_head = gi * rows_per_group
        fig.text(0.5, top - r_head * row_h * (1 + hspace) + 0.052, title,
                 ha="center", va="bottom", fontsize=12.5, color=tcol)
        for kk in range(len(sub)):
            r_local, c = divmod(kk, per_row)
            row = r_head + r_local
            rec = sub.iloc[kk]
            a, b = grab(rec["test_index"])
            miss = np.abs(a[0] - b[0]) > 0.5          # IR1 被修复的像素
            im_raw, im_cor = composite_shared([a, b], ch_names, miss_mask=miss)

            axr = fig.add_subplot(gs[row, c * 2])
            axr.imshow(im_raw)
            axr.axis("off")
            _corner(axr, "原始 RAW", C_RAW)
            axr.set_title(
                f"#{gi * k + kk + 1}  {rec['ID']}\n"
                f"真值 {rec['true_vmax']:.0f} kn  |  原始 Pred {rec['pred_raw']:.1f}"
                f"  (Δ{rec['err_raw']:+.1f})",
                fontsize=8, pad=3)

            axc = fig.add_subplot(gs[row, c * 2 + 1])
            axc.imshow(im_cor)
            axc.axis("off")
            _corner(axc, "修复后 CORRECTED", C_CORR)
            axc.set_title(
                f"修复 Pred {rec['pred_corrected']:.1f}"
                f"  (Δ{rec['err_corrected']:+.1f})\n"
                f"Δ|误差| {rec['abs_err_improve']:+.1f} kn"
                f"  ({'改善' if rec['abs_err_improve'] > 0 else '恶化'})",
                fontsize=8, pad=3)

    method = read_method_name(args.csv)
    fig.suptitle(
        f"resnet50 · 局部缺测修复的输入对照（{method} 修复；"
        f"RAW 面板红色 = 被修复的缺测像素）",
        fontsize=13.5, y=0.965)

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.csv)),
                                   f"hard_cases_top{k}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)

    sel = pd.concat([imp.assign(group="improved"), wor.assign(group="not_improved")])
    sel_csv = os.path.splitext(out)[0] + ".csv"
    sel.to_csv(sel_csv, index=False, encoding="utf-8-sig")

    logger.info(f"对比图: {out}")
    logger.info(f"选中样本: {sel_csv}")
    for tag, s in (("improved", imp), ("not improved", wor)):
        print(f"\n[{tag}]")
        print(s[["ID", "true_vmax", "pred_raw", "pred_corrected",
                 "err_raw", "err_corrected", "abs_err_improve"]].to_string(index=False))


if __name__ == "__main__":
    main()
