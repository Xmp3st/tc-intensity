#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""improve 值（|err_raw| - |err_corrected|）对 true Vmax 的简单线性拟合。

读 corrraw/<model>__<method>/corrected_vs_raw.csv（每行一个 glitch 样本），
对 (true_vmax, abs_err_improve) 做最小二乘直线拟合，报告斜率/截距/R²/相关系数/p 值，
并给出「拟合线穿过 0 的强度」= 修复由收益转为损害的交叉点。

产物：outputs/tcir_wpac/tcir_wpac_bb/corrraw_summary_fit.{png,csv,json}
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from tcintens.utils.plotting import use_cjk_font

MODEL_ORDER = ["resnet18", "resnet34", "resnet50", "efficientnet_b0"]
MODEL_LABEL = {"resnet18": "ResNet18", "resnet34": "ResNet34",
               "resnet50": "ResNet50", "efficientnet_b0": "EfficientNet-B0"}
METHOD_COLOR = {"griddata": "#2b6cb0", "crosschannel": "#c05621"}


def fit_one(df):
    """返回该数据集的拟合统计。y=improve, x=true"""
    x = df["true_vmax"].to_numpy(dtype=float)
    y = df["abs_err_improve"].to_numpy(dtype=float)
    lr = stats.linregress(x, y)          # slope, intercept, r, p, se
    sp = stats.spearmanr(x, y)
    yhat = lr.slope * x + lr.intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # 拟合线过零点：修复中性的 Vmax
    cross = (-lr.intercept / lr.slope) if lr.slope != 0 else float("nan")
    return dict(
        n=len(df),
        slope=float(lr.slope), intercept=float(lr.intercept),
        r=float(lr.rvalue), r2=float(r2),
        p=float(lr.pvalue), se_slope=float(lr.stderr),
        spearman=float(sp.statistic), p_spearman=float(sp.pvalue),
        cross_zero=float(cross),
        mean_improve=float(y.mean()), med_improve=float(np.median(y)),
        frac_pos=float(np.mean(y > 0)),
        vmax_min=float(x.min()), vmax_max=float(x.max()),
    )


def binned(df, nb=6):
    x = df["true_vmax"].to_numpy(dtype=float)
    y = df["abs_err_improve"].to_numpy(dtype=float)
    edges = np.linspace(x.min(), x.max(), nb + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nb - 1)
    rows = []
    for k in range(nb):
        m = idx == k
        if m.sum() == 0:
            continue
        rows.append(dict(lo=edges[k], hi=edges[k + 1], c=0.5 * (edges[k] + edges[k + 1]),
                         n=int(m.sum()), mean=float(y[m].mean()),
                         sem=float(y[m].std(ddof=1) / np.sqrt(m.sum())) if m.sum() > 1 else 0.0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/tcir_wpac/tcir_wpac_bb/corrraw")
    ap.add_argument("--out", default="outputs/tcir_wpac/tcir_wpac_bb/corrraw_summary_fit")
    args = ap.parse_args()

    use_cjk_font()

    # 收集数据集
    data = {}
    for d in sorted(glob.glob(os.path.join(args.root, "*"))):
        f = os.path.join(d, "corrected_vs_raw.csv")
        if not os.path.exists(f):
            continue
        model, method = os.path.basename(d).split("__")
        data[(model, method)] = pd.read_csv(f)

    keys = [(m, met) for m in MODEL_ORDER for met in ("griddata", "crosschannel")
            if (m, met) in data]

    rows = []
    for (m, met) in keys:
        st = fit_one(data[(m, met)])
        st.update(model=m, method=met)
        rows.append(st)
    tab = pd.DataFrame(rows)

    pd.set_option("display.width", 220)
    print("\n=== improve ~ true_vmax 简单线性拟合（y = slope*x + intercept）===")
    print(tab[["model", "method", "n", "slope", "intercept", "r", "r2", "p",
               "cross_zero", "mean_improve", "frac_pos"]].round(4).to_string(index=False))
    print("\n--- 非参数相关（Spearman，抗离群）---")
    print(tab[["model", "method", "spearman", "p_spearman"]].round(4).to_string(index=False))
    print("\n--- 拟合线在参考强度上的预测 improve (knots) ---")
    ref = [40, 60, 80, 100, 120]
    hdr = "  ".join([f"{v:>6d}" for v in ref])
    print(f"{'model':<16}{'method':<14}{hdr}")
    for _, r in tab.iterrows():
        vals = "  ".join([f"{r['slope']*v + r['intercept']:>6.2f}" for v in ref])
        print(f"{r['model']:<16}{r['method']:<14}{vals}")
    print("\n--- 汇总（8 组统计）---")
    print(f"斜率: 均值 {tab['slope'].mean():.4f}  范围 [{tab['slope'].min():.4f}, "
          f"{tab['slope'].max():.4f}] (all>0: {bool((tab['slope']>0).all())})")
    print(f"R2  : 均值 {tab['r2'].mean():.4f}  范围 [{tab['r2'].min():.4f}, {tab['r2'].max():.4f}]")
    print(f"p<0.01 的组数: {(tab['p']<0.01).sum()}/8 ; Spearman<0 的组数: {(tab['spearman']<0).sum()}/8")

    # ---------- 图 ----------
    fig = plt.figure(figsize=(15.5, 9.6))
    gs = fig.add_gridspec(2, 4, hspace=0.34, wspace=0.26,
                          left=0.055, right=0.985, top=0.90, bottom=0.07)

    # (1)(2) resnet50 两个方法的主散点（含拟合线）
    for j, met in enumerate(("griddata", "crosschannel")):
        df = data.get(("resnet50", met))
        ax = fig.add_subplot(gs[0, j])
        if df is None:
            ax.axis("off")
            continue
        x = df["true_vmax"].to_numpy(float)
        y = df["abs_err_improve"].to_numpy(float)
        c = METHOD_COLOR[met]
        ax.axhline(0, color="#888", lw=1.0, ls="--")
        ax.scatter(x, y, s=16, alpha=0.55, color=c, edgecolor="none")
        st = fit_one(df)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, st["slope"] * xs + st["intercept"], color="#c00", lw=2.2,
                label=f"拟合 y={st['slope']:.3f}x{st['intercept']:+.1f}\n"
                      f"R2={st['r2']:.3f}  r={st['r']:.3f}  p={st['p']:.1e}")
        b = binned(df, 6)
        ax.plot([r["c"] for r in b], [r["mean"] for r in b], "o-", color="black",
                lw=1.6, ms=5, label="分箱均值(6 箱)")
        if np.isfinite(st["cross_zero"]) and x.min() <= st["cross_zero"] <= x.max():
            ax.axvline(st["cross_zero"], color="#c00", lw=1.2, ls=":")
            ax.annotate(f"过零 {st['cross_zero']:.0f} kn", (st["cross_zero"], ax.get_ylim()[1]),
                        textcoords="offset points", xytext=(4, -12), fontsize=8.5, color="#c00")
        ax.set_xlabel("true Vmax (knots)")
        ax.set_ylabel("improve = |err_raw| - |err_corrected| (knots)")
        ax.set_title(f"({j+1}) ResNet50 · {met}", fontsize=11.5)
        ax.grid(ls=":", alpha=0.4)
        ax.legend(fontsize=8, loc="upper left")

    # (3) 各模型斜率对比
    ax = fig.add_subplot(gs[0, 2])
    w = 0.38
    xs = np.arange(len(MODEL_ORDER))
    for k, met in enumerate(("griddata", "crosschannel")):
        ss, ee = [], []
        for m in MODEL_ORDER:
            r = tab[(tab.model == m) & (tab.method == met)]
            ss.append(r["slope"].iloc[0] if len(r) else np.nan)
            ee.append(r["se_slope"].iloc[0] if len(r) else 0.0)
        ax.bar(xs + (k - 0.5) * w, ss, w * 0.92, yerr=ee, capsize=3,
               color=METHOD_COLOR[met], label=met, edgecolor="white")
        for xi, v in zip(xs + (k - 0.5) * w, ss):
            ax.text(xi, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER], fontsize=8.5,
                       rotation=18, ha="right")
    ax.set_ylabel("拟合斜率 (knots improve / knots Vmax)")
    ax.set_title("(3) 斜率：修复收益随强度增长", fontsize=11.5)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.legend(fontsize=8.5)

    # (4) 各模型 R² 对比
    ax = fig.add_subplot(gs[0, 3])
    for k, met in enumerate(("griddata", "crosschannel")):
        rr = [tab[(tab.model == m) & (tab.method == met)]["r2"].iloc[0]
              if len(tab[(tab.model == m) & (tab.method == met)]) else np.nan
              for m in MODEL_ORDER]
        ax.bar(xs + (k - 0.5) * w, rr, w * 0.92, color=METHOD_COLOR[met],
               label=met, edgecolor="white")
        for xi, v in zip(xs + (k - 0.5) * w, rr):
            ax.text(xi, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER], fontsize=8.5,
                       rotation=18, ha="right")
    ax.set_ylabel("R2 (拟合优度)")
    ax.set_ylim(0, max(0.6, float(tab["r2"].max()) * 1.25))
    ax.set_title("(4) 拟合优度 R2", fontsize=11.5)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.legend(fontsize=8.5)

    # (5)-(8) 四个模型各自的分箱均值曲线（含 SEM 误差棒）
    for i, m in enumerate(MODEL_ORDER):
        ax = fig.add_subplot(gs[1, i])
        ax.axhline(0, color="#888", lw=1.0, ls="--")
        for met in ("griddata", "crosschannel"):
            df = data.get((m, met))
            if df is None:
                continue
            b = binned(df, 6)
            ax.errorbar([r["c"] for r in b], [r["mean"] for r in b],
                        yerr=[r["sem"] for r in b], marker="o", ms=5, lw=1.8,
                        capsize=3, color=METHOD_COLOR[met], label=met)
        ax.set_xlabel("true Vmax (knots)")
        if i == 0:
            ax.set_ylabel("分箱均值 improve (knots)")
        ax.set_title(f"({5+i}) {MODEL_LABEL[m]}", fontsize=11)
        ax.grid(ls=":", alpha=0.4)
        ax.legend(fontsize=8)

    fig.suptitle("局部缺测修复收益（improve）对真值强度（true Vmax）的线性拟合"
                 " · 8 组 = 4 骨干 × 2 补全方法（glitch 子集 n=637）",
                 fontsize=13.5, y=0.965)
    fig.savefig(args.out + ".png", dpi=150)
    tab.to_csv(args.out + ".csv", index=False)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump({f"{m}__{met}": fit_one(data[(m, met)]) for (m, met) in keys},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[图表] {args.out}.png")
    print(f"[数据] {args.out}.csv / .json")


if __name__ == "__main__":
    main()
