# -*- coding: utf-8 -*-
"""
backbone_compare_plot.py —— 多 backbone 对比可视化

读取 logs/backbone_compare.csv（由 scripts/train_backbones.py 产出），
并附加 vgg16 参考基线（来自 outputs/tcir_wpac/tcir_wpac_v2_3x，80 轮口径），
绘制：
  (A) 各 backbone 的验证集 / 测试集 RMSE 分组柱状图
  (B) 参数量 vs 测试集 RMSE 散点（效率视角）
  (C) 指标明细表

用法：
  python scripts/backbone_compare_plot.py \
      --csv logs/backbone_compare.csv \
      --out outputs/tcir_wpac/tcir_wpac_bb/backbone_compare.png
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# vgg16 参考基线（来自 v2 口径：outputs/tcir_wpac/tcir_wpac_v2_3x/，80 轮 + patience20）
VGG16_REF = {
    "backbone": "vgg16*", "params_M": 138.36, "best_epoch": 27,
    "val_rmse": 13.065, "val_mae": 9.794, "val_r2": 0.8416,
    "test_rmse": 13.539, "test_mae": 10.293, "test_r2": 0.833,
    "minutes": 66.0, "ref": True,
}


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("val_rmse"):
                continue
            rec = {k: (float(v) if k in (
                "params_M", "val_rmse", "val_mae", "val_r2",
                "test_rmse", "test_mae", "test_r2", "minutes") and v not in ("", None)
                else v) for k, v in r.items()}
            rec["ref"] = False
            rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="logs/backbone_compare.csv")
    ap.add_argument("--out", default="outputs/tcir_wpac/tcir_wpac_bb/backbone_compare.png")
    ap.add_argument("--no-ref", action="store_true", help="不附加 vgg16 参考基线")
    args = ap.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rows = load_rows(args.csv)
    if not args.no_ref:
        rows.append(dict(VGG16_REF))
    rows.sort(key=lambda r: r["test_rmse"])

    names = [r["backbone"] for r in rows]
    val = [r["val_rmse"] for r in rows]
    test = [r["test_rmse"] for r in rows]
    params = [r["params_M"] for r in rows]
    colors = ["#9aa0a6" if r.get("ref") else "#1a73e8" for r in rows]

    fig = plt.figure(figsize=(13.5, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.26)

    # ---- (A) 分组柱状图 ----
    ax = fig.add_subplot(gs[0, 0])
    x = range(len(names))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], val, w, label="Val RMSE",
                color="#8ab4f8", edgecolor="#1a73e8")
    b2 = ax.bar([i + w / 2 for i in x], test, w, label="Test RMSE",
                color=colors, edgecolor="#0b57d0")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}",
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8.5)
    ax.axhline(13.539, ls="--", lw=1.1, color="#d93025", alpha=.85)
    ax.annotate("vgg16* test = 13.54 (80 ep)", xy=(0.0, 13.539),
                xytext=(-0.42, 15.7), fontsize=8.5, color="#d93025",
                ha="left",
                arrowprops=dict(arrowstyle="->", color="#d93025", lw=1))
    ax.set_xlim(-0.62, len(names) - 0.38)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9.5)
    ax.set_ylabel("RMSE (knots)")
    ax.set_title("Backbone comparison — RMSE (lower is better)\n"
                 "96x96x3, 40 epochs + early stop, fixed storm split",
                 fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=.25, ls=":")
    ax.set_ylim(0, max(test + val) * 1.16)

    # ---- (B) 参数量 vs 测试 RMSE ----
    ax2 = fig.add_subplot(gs[0, 1])
    for r in rows:
        mk = "*" if r.get("ref") else "o"
        sz = 260 if r.get("ref") else 120
        ax2.scatter(r["params_M"], r["test_rmse"], s=sz, marker=mk,
                    color="#9aa0a6" if r.get("ref") else "#1a73e8",
                    edgecolor="#0b57d0", zorder=3)
        ax2.annotate(f"{r['backbone']}  ({r['params_M']:.1f}M)",
                     (r["params_M"], r["test_rmse"]),
                     textcoords="offset points", xytext=(7, 5), fontsize=8.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("Trainable params (M, log scale)")
    ax2.set_ylabel("Test RMSE (knots)")
    ax2.set_title("Accuracy vs. model size", fontsize=11)
    ax2.grid(alpha=.25, ls=":")
    ax2.margins(x=.28, y=.22)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    print(f"saved -> {args.out}")

    # ---- (C) 文本表 ----
    hdr = (f"{'backbone':<17}{'params(M)':>10}{'best_ep':>8}{'val_RMSE':>10}"
           f"{'val_MAE':>9}{'val_R2':>8}{'test_RMSE':>11}{'test_MAE':>10}"
           f"{'test_R2':>9}{'min':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['backbone']:<17}{r['params_M']:>10.2f}{str(r.get('best_epoch', '')):>8}"
              f"{r['val_rmse']:>10.4f}{r['val_mae']:>9.4f}{r['val_r2']:>8.4f}"
              f"{r['test_rmse']:>11.4f}{r['test_mae']:>10.4f}{r['test_r2']:>9.4f}"
              f"{r.get('minutes', 0):>7.1f}")
    best = min([r for r in rows if not r.get("ref")], key=lambda r: r["test_rmse"])
    print(f"\n最优（不含参考基线）: {best['backbone']}  test RMSE={best['test_rmse']:.4f}")


if __name__ == "__main__":
    main()
