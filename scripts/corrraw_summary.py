# -*- coding: utf-8 -*-
"""
corrraw_summary.py —— 汇总「4 骨干 × 2 补全方法」= 8 组 raw vs corrected 消融。

读取每个 (model, method) 子目录下的 `corrected_vs_raw.json` / `.csv`，
产出：
  outputs/tcir_wpac/tcir_wpac_bb/corrraw_summary.csv    逐组明细
  outputs/tcir_wpac/tcir_wpac_bb/corrraw_summary.json   汇总
  outputs/tcir_wpac/tcir_wpac_bb/corrraw_summary.png    四联图

四联图：
  (1) glitch 子集 RMSE：raw vs corrected（按骨干分组，实心=griddata / 斜纹=crosschannel）
  (2) 修复收益 Δ = raw − corrected（正=修复有效）
  (3) 全测试集 RMSE：raw vs corrected
  (4) 逐样本改善比例（|err| 变小 / 变大 / 不变）

用法：
  python scripts/corrraw_summary.py
  python scripts/corrraw_summary.py --root outputs/tcir_wpac/tcir_wpac_bb/corrraw
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from tcintens.utils.plotting import use_cjk_font  # noqa: E402

use_cjk_font()

C_RAW = "#C44E52"
C_COR = "#55A868"
C_GAP = "#1a73e8"

MODEL_ORDER = ["resnet18", "resnet34", "resnet50", "efficientnet_b0"]
MODEL_LABEL = {"resnet18": "resnet18\n(11.2M)", "resnet34": "resnet34\n(21.3M)",
               "resnet50": "resnet50\n(23.6M)", "efficientnet_b0": "efficientnet_b0\n(4.1M)"}
METHOD_ORDER = ["griddata", "crosschannel"]


def collect(root):
    """扫描 <root>/<model>__<method>/corrected_vs_raw.json，返回行列表。"""
    rows = []
    for jf in sorted(glob.glob(os.path.join(root, "*", "corrected_vs_raw.json"))):
        d = os.path.basename(os.path.dirname(jf))
        if "__" not in d:
            continue
        model, method = d.split("__", 1)
        j = json.load(open(jf, encoding="utf-8"))
        g, a = j["glitch_subset"], j["all_test"]
        row = {
            "model": model, "method": method, "dir": os.path.dirname(jf),
            "n_glitch": j["n_glitch"], "n_test": j["n_test"],
            # glitch 子集
            "g_raw_rmse": g["raw"]["rmse"], "g_cor_rmse": g["corrected"]["rmse"],
            "g_raw_mae": g["raw"]["mae"], "g_cor_mae": g["corrected"]["mae"],
            "g_raw_r2": g["raw"]["r2"], "g_cor_r2": g["corrected"]["r2"],
            # 全测试集
            "a_raw_rmse": a["raw"]["rmse"], "a_cor_rmse": a["corrected"]["rmse"],
            "a_raw_mae": a["raw"]["mae"], "a_cor_mae": a["corrected"]["mae"],
            "a_raw_r2": a["raw"]["r2"], "a_cor_r2": a["corrected"]["r2"],
        }
        # 逐样本改善比例（从 csv 读，最好与最差都有记录）
        cf = os.path.join(os.path.dirname(jf), "corrected_vs_raw.csv")
        if os.path.exists(cf):
            c = pd.read_csv(cf)
            if "abs_err_improve" in c.columns:
                imp = c["abs_err_improve"].values
                row["n_improved"] = int((imp > 0).sum())
                row["n_worsened"] = int((imp < 0).sum())
                row["n_tied"] = int((imp == 0).sum())
                row["median_improve"] = float(np.median(imp))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/tcir_wpac/tcir_wpac_bb/corrraw")
    args = ap.parse_args()
    root = args.root
    rows = collect(root)
    if not rows:
        raise SystemExit(f"{root} 下没找到 corrected_vs_raw.json，先跑 corrected_vs_raw.py")

    df = pd.DataFrame(rows)
    models = [m for m in MODEL_ORDER if m in set(df["model"])] + \
             sorted(set(df["model"]) - set(MODEL_ORDER))
    methods = [m for m in METHOD_ORDER if m in set(df["method"])] + \
              sorted(set(df["method"]) - set(METHOD_ORDER))
    df["model"] = pd.Categorical(df["model"], categories=models, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=methods, ordered=True)
    df = df.sort_values(["model", "method"]).reset_index(drop=True)

    # 修复收益
    df["g_delta"] = df["g_raw_rmse"] - df["g_cor_rmse"]
    df["a_delta"] = df["a_raw_rmse"] - df["a_cor_rmse"]
    df["g_rel"] = df["g_delta"] / df["g_raw_rmse"] * 100.0

    csv_out = os.path.join(os.path.dirname(root), "corrraw_summary.csv")
    json_out = os.path.join(os.path.dirname(root), "corrraw_summary.json")
    png_out = os.path.join(os.path.dirname(root), "corrraw_summary.png")
    df.drop(columns=["dir"]).to_csv(csv_out, index=False)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({"root": root, "n_groups": len(df),
                   "rows": json.loads(df.drop(columns=["dir"]).to_json(orient="records"))},
                  f, ensure_ascii=False, indent=2)

    # ---------------- 图 ----------------
    nrow = max(1, len(models))
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.6))
    fig.suptitle("Raw(glitch) vs Corrected —— 4 骨干 × 2 补全方法（共 %d 组；glitch 子集）"
                 % len(df), fontsize=13.5, fontweight="bold", y=0.975)

    nm = len(models)
    w = 0.8 / (len(methods) * 2)     # 每根柱宽
    x = np.arange(nm)

    def pair_positions(i):
        """模型 i 内各方法的 (raw_x, cor_x) 列表"""
        out = []
        for k in range(len(methods)):
            off = (k - (len(methods) - 1) / 2) * (2 * w) * 1.12
            out.append((i + off - w / 2, i + off + w / 2))
        return out

    # (1) glitch 子集 RMSE
    ax = axes[0, 0]
    for k, met in enumerate(methods):
        for i in range(nm):
            r = df[(df["model"] == models[i]) & (df["method"] == met)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            xr, xc = pair_positions(i)[k]
            hatch = "" if met == "griddata" else "//"
            ax.bar(xr, r["g_raw_rmse"], w, color=C_RAW, edgecolor="white",
                   hatch=hatch, linewidth=0.6)
            ax.bar(xc, r["g_cor_rmse"], w, color=C_COR, edgecolor="white",
                   hatch=hatch, linewidth=0.6)
            ax.text(xc, r["g_cor_rmse"] + 0.25, f"{r['g_cor_rmse']:.2f}",
                    ha="center", fontsize=7.5)
            ax.text(xr, r["g_raw_rmse"] + 0.25, f"{r['g_raw_rmse']:.2f}",
                    ha="center", fontsize=7.5, color="#8a2b2e")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in models], fontsize=9)
    ax.set_ylabel("glitch 子集 RMSE (knots)")
    ax.set_ylim(0, df[["g_raw_rmse", "g_cor_rmse"]].values.max() * 1.18)
    ax.set_title("(1) 有缺测样本上的 RMSE：红=raw glitch，绿=corrected", fontsize=11.5)
    ax.grid(axis="y", ls=":", alpha=0.4)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=C_RAW, label="raw (glitch)"),
                       Patch(facecolor=C_COR, label="corrected"),
                       Patch(facecolor="#bbb", label="实心 = griddata"),
                       Patch(facecolor="#bbb", hatch="//", label="斜纹 = crosschannel")],
              fontsize=8, loc="upper left", ncol=2)

    # (2) 修复收益 Δ
    ax = axes[0, 1]
    w2 = 0.8 / len(methods)
    for k, met in enumerate(methods):
        vals, xs = [], []
        for i in range(nm):
            r = df[(df["model"] == models[i]) & (df["method"] == met)]
            vals.append(r.iloc[0]["g_delta"] if len(r) else np.nan)
            xs.append(i + (k - (len(methods) - 1) / 2) * w2)
        hatch = "" if met == "griddata" else "//"
        ax.bar(xs, vals, w2 * 0.9, color=C_GAP, alpha=0.55 + 0.35 * (hatch == ""),
               edgecolor="white", hatch=hatch, label=f"Δ {met}")
        for xi, v in zip(xs, vals):
            if np.isfinite(v):
                ax.text(xi, v + 0.06, f"{v:+.2f}", ha="center", fontsize=8)
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in models], fontsize=9)
    ax.set_ylabel("Δ RMSE = raw − corrected (knots)")
    vmax = float(np.nanmax([df["g_delta"].max(), 0.0]))
    ax.set_ylim(0, vmax * 1.30)
    ax.set_title("(2) 修复收益（越大=修复越有价值）", fontsize=11.5)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left")

    # (3) 全测试集
    ax = axes[1, 0]
    for k, met in enumerate(methods):
        for i in range(nm):
            r = df[(df["model"] == models[i]) & (df["method"] == met)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            xr, xc = pair_positions(i)[k]
            hatch = "" if met == "griddata" else "//"
            ax.bar(xr, r["a_raw_rmse"], w, color=C_RAW, edgecolor="white",
                   hatch=hatch, linewidth=0.6)
            ax.bar(xc, r["a_cor_rmse"], w, color=C_COR, edgecolor="white",
                   hatch=hatch, linewidth=0.6)
            ax.text(xc, r["a_cor_rmse"] + 0.06, f"{r['a_cor_rmse']:.2f}",
                    ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in models], fontsize=9)
    ax.set_ylabel("全测试集 RMSE (knots)")
    ax.set_ylim(13.0, df[["a_raw_rmse", "a_cor_rmse"]].values.max() * 1.06)
    ax.set_title("(3) 全测试集 RMSE（含 2256 张干净帧，故差距被稀释）", fontsize=11.5)
    ax.grid(axis="y", ls=":", alpha=0.4)

    # (4) 改善比例
    ax = axes[1, 1]
    if {"n_improved", "n_worsened"} <= set(df.columns):
        for k, met in enumerate(methods):
            imp, wor = [], []
            for i in range(nm):
                r = df[(df["model"] == models[i]) & (df["method"] == met)]
                n_tot = r.iloc[0]["n_glitch"] if len(r) else 1
                imp.append(r.iloc[0]["n_improved"] / n_tot * 100 if len(r) else np.nan)
                wor.append(r.iloc[0]["n_worsened"] / n_tot * 100 if len(r) else np.nan)
            xs = x + (k - (len(methods) - 1) / 2) * w2
            hatch = "" if met == "griddata" else "//"
            ax.bar(xs, imp, w2 * 0.9, color=C_COR, edgecolor="white", hatch=hatch,
                   label=f"改善 {met}")
            ax.bar(xs, wor, w2 * 0.9, bottom=imp, color=C_RAW, edgecolor="white",
                   hatch=hatch, label=f"变差 {met}")
            for xi, a_, b_ in zip(xs, imp, wor):
                if np.isfinite(a_):
                    ax.text(xi, a_ / 2, f"改善 {a_:.0f}%", ha="center", va="center",
                            fontsize=7.5, color="white", fontweight="bold")
                    ax.text(xi, a_ + b_ / 2, f"变差 {b_:.0f}%", ha="center", va="center",
                            fontsize=7.5, color="white", fontweight="bold")
        ax.set_ylim(0, 128)
        ax.set_ylabel("占 glitch 样本比例 (%)")
        ax.set_title("(4) 逐样本：修复后 |误差| 改善 vs 变差", fontsize=11.5)
        ax.grid(axis="y", ls=":", alpha=0.4)
        ax.legend(fontsize=7.5, loc="upper left", ncol=2, framealpha=0.95)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in models], fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(png_out, dpi=118)
    plt.close(fig)

    # ---------------- 控制台总表 ----------------
    print(f"已保存: {png_out}\n          {csv_out}\n          {json_out}\n")
    print(f"glitch 子集 n={int(df['n_glitch'].iloc[0])}, 全测试集 n={int(df['n_test'].iloc[0])}\n")
    hdr = (f"{'model':<16}{'method':<14}{'raw RMSE':>10}{'cor RMSE':>10}{'Δ':>8}"
           f"{'相对改善':>9}{'cor MAE':>9}{'cor R²':>8}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in df.iterrows():
        print(f"{r['model']:<16}{r['method']:<14}{r['g_raw_rmse']:>10.3f}"
              f"{r['g_cor_rmse']:>10.3f}{r['g_delta']:>+8.3f}{r['g_rel']:>8.1f}%"
              f"{r['g_cor_mae']:>9.3f}{r['g_cor_r2']:>8.3f}")
    print()
    piv = df.pivot_table(index="model", columns="method", values="g_delta")
    piv = piv.reindex(index=models, columns=methods)
    print("修复收益 Δ(glitch RMSE) 透视表：")
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
