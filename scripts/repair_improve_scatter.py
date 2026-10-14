"""修复方法 improve 散点图（仅"被修复的图片"，即真实损坏帧）

improve = |err_raw| - |err_fixed|     正值 = 修复后更准（误差被减小）

对比两种修补：griddata（插值）vs U-Net（学习式）。
输出三面板：
  1) improve ~ 真值 Vmax        （看收益是否随强度变化）
  2) improve ~ 缺测面积 (%)      （看收益是否随损坏大小变化）
  3) 配对散点 griddata vs U-Net （谁更准，逐样本）

用法：
    python scripts/repair_improve_scatter.py
    python scripts/repair_improve_scatter.py \
        --csv outputs/tcir_wpac_ir1/a3_repair_compare_areamatch/per_sample.csv \
        --unet-col pred_unet2 --out outputs/tcir_wpac_ir1/a3_repair_compare_areamatch \
        --tag _areamatch
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from tcintens.utils.plotting import use_cjk_font  # noqa: E402

C_GD = "#378ADD"   # griddata（插值）
C_UN = "#1D9E75"   # U-Net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/tcir_wpac_ir1/a3_repair_compare/per_sample.csv")
    ap.add_argument("--unet-col", default="pred_unet", help="U-Net 预测列名")
    ap.add_argument("--out", default="outputs/tcir_wpac_ir1/a3_repair_compare")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    os.chdir(ROOT)
    use_cjk_font()

    df = pd.read_csv(args.csv)
    d = df[df.ir1_damaged == 1].copy()
    n = len(d)

    # improve（正 = 更准）
    d["imp_gd"] = np.abs(d.err_raw) - np.abs(d.err_griddata)
    err_un_col = args.unet_col.replace("pred_", "err_")   # pred_unet -> err_unet
    d["imp_un"] = np.abs(d.err_raw) - np.abs(d[err_un_col])
    d["area_pct"] = d.area_frac * 100.0

    imp_gd = d.imp_gd.to_numpy()
    imp_un = d.imp_un.to_numpy()
    vmax = d.true_vmax.to_numpy()
    area = d.area_pct.to_numpy()

    stats = {
        "n_damaged": int(n),
        "griddata": {
            "mean_improve": float(np.mean(imp_gd)),
            "median_improve": float(np.median(imp_gd)),
            "win_rate_vs_raw": float(np.mean(imp_gd > 0)),
            "total_mse_drop_frac": float(
                (np.mean(d.err_raw ** 2) - np.mean(d.err_griddata ** 2)) / np.mean(d.err_raw ** 2)),
        },
        "unet": {
            "mean_improve": float(np.mean(imp_un)),
            "median_improve": float(np.median(imp_un)),
            "win_rate_vs_raw": float(np.mean(imp_un > 0)),
            "total_mse_drop_frac": float(
                (np.mean(d.err_raw ** 2) - np.mean(d[err_un_col] ** 2)) / np.mean(d.err_raw ** 2)),
        },
    }
    # 配对：U-Net vs griddata
    diff = imp_un - imp_gd
    stats["unet_minus_griddata_improve"] = {
        "mean": float(np.mean(diff)),
        "median": float(np.median(diff)),
        "unet_better_frac": float(np.mean(diff > 1e-9)),
        "gd_better_frac": float(np.mean(diff < -1e-9)),
        "tie_frac": float(np.mean(np.abs(diff) <= 1e-9)),
    }

    # ---------- 作图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))

    def band(ax):
        ax.axhline(0, color="#444", lw=1.1, ls="--", alpha=0.8)
        ax.grid(alpha=0.25)

    # 面板1：improve ~ 真值
    ax = axes[0]
    ax.scatter(vmax, imp_gd, s=20, alpha=0.6, c=C_GD, edgecolors="none", label="griddata 插值")
    ax.scatter(vmax, imp_un, s=20, alpha=0.6, c=C_UN, edgecolors="none", label="U-Net")
    band(ax)
    ax.set_xlabel("真值 Vmax (kn)")
    ax.set_ylabel("improve = |raw 误差| - |修复后误差|  (kn)")
    ax.set_title("修复收益 ~ 台风强度")
    ax.legend(fontsize=9)

    # 面板2：improve ~ 缺测面积
    ax = axes[1]
    ax.scatter(area, imp_gd, s=20, alpha=0.6, c=C_GD, edgecolors="none", label="griddata 插值")
    ax.scatter(area, imp_un, s=20, alpha=0.6, c=C_UN, edgecolors="none", label="U-Net")
    band(ax)
    ax.set_xlabel("缺测面积占全图 (%)")
    ax.set_ylabel("improve (kn)")
    ax.set_title("修复收益 ~ 缺测面积")
    ax.legend(fontsize=9)

    # 面板3：配对
    ax = axes[2]
    lo = float(min(imp_gd.min(), imp_un.min())) - 2
    hi = float(max(imp_gd.max(), imp_un.max())) + 2
    ax.scatter(imp_gd, imp_un, s=20, alpha=0.6, c="#888780", edgecolors="none",
               label="每个损坏帧")
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.1, alpha=0.7, label="y = x（两种方法等收益）")
    band(ax)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("griddata 插值 · improve (kn)")
    ax.set_ylabel("U-Net · improve (kn)")
    st = stats["unet_minus_griddata_improve"]
    ax.set_title(f"逐样本配对（n={n}）\n虚线以上=U-Net 更好 {st['unet_better_frac']*100:.1f}% | 以下=插值更好 {st['gd_better_frac']*100:.1f}%")
    ax.legend(fontsize=9, loc="upper left")

    fig.suptitle("两种修补方法的 improve 散点图 · 仅真实损坏帧（a3 臂 · 测试 2014-2016）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs(args.out, exist_ok=True)
    png = os.path.join(args.out, f"improve_scatter{args.tag}.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)

    out_csv = os.path.join(args.out, f"improve_scatter{args.tag}.csv")
    d[["matrix_index", "ID", "true_vmax", "ir1_damaged", "area_pct",
       "err_raw", "err_griddata", err_un_col, "imp_gd", "imp_un"]].to_csv(out_csv, index=False)

    out_json = os.path.join(args.out, f"improve_scatter{args.tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 终端摘要
    print(f"损坏帧 n = {n}")
    print(f"  griddata : 平均 improve {stats['griddata']['mean_improve']:+.3f} kn | "
          f"中位 {stats['griddata']['median_improve']:+.3f} | 胜率(>0) {stats['griddata']['win_rate_vs_raw']*100:.1f}% | "
          f"总 MSE 降幅 {stats['griddata']['total_mse_drop_frac']*100:.1f}%")
    print(f"  U-Net    : 平均 improve {stats['unet']['mean_improve']:+.3f} kn | "
          f"中位 {stats['unet']['median_improve']:+.3f} | 胜率(>0) {stats['unet']['win_rate_vs_raw']*100:.1f}% | "
          f"总 MSE 降幅 {stats['unet']['total_mse_drop_frac']*100:.1f}%")
    print(f"  U-Net - griddata （逐样本 improve 差）: 均值 {st['mean']:+.4f} kn | "
          f"U-Net 更好 {st['unet_better_frac']*100:.1f}% | 插值更好 {st['gd_better_frac']*100:.1f}% | "
          f"并列 {st['tie_frac']*100:.1f}%")
    print(f"[saved] {png}\n        {out_csv}\n        {out_json}")


if __name__ == "__main__":
    main()
