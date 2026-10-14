# -*- coding: utf-8 -*-
"""
inpaint_arms_compare.py —— 多种补全方法的 3v3（三 seed 配对）下游对比图。

臂（arm）= 仅 `data.tcir.missing_region_*` 不同，其余完全同参
（vgg16 / 96² / split_seed=42 / 80 轮 / patience 20 / batch 64 / AdamW 1e-3）。

  griddata               : v2_3x/run0_s42, run1_s43, run2_s44
  crosschannel           : cc_v1(s42), cc_s43, cc_s44
  crosschannel+feather=3 : ccf_s42, ccf_s43, ccf_s44
  harmonic               : harm_s42, harm_s43, harm_s44   （存在才画）

四联图：
  (1) 测试集 RMSE（按 seed 分组柱）
  (2) val / test 均值（点 + 横线）
  (3) 相对 griddata 的配对差 Δtest（每个 seed 一个点）
  (4) 修正组 / 干净组 均值（3 seed 均值）

用法：
  python scripts/inpaint_arms_compare.py
  python scripts/inpaint_arms_compare.py --out outputs/inpaint_arms_compare.png
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from tcintens.utils.plotting import use_cjk_font  # noqa: E402

use_cjk_font()
O = "outputs/tcir_wpac"

# (显示名, 颜色, [三个 seed 的目录]，顺序必须是 s42/s43/s44)
ARMS = [
    ("griddata", "#8a8f98",
     [f"{O}/tcir_wpac_v2_3x/run0_s42", f"{O}/tcir_wpac_v2_3x/run1_s43",
      f"{O}/tcir_wpac_v2_3x/run2_s44"]),
    ("crosschannel", "#1a73e8",
     [f"{O}/tcir_wpac_cc_v1", f"{O}/tcir_wpac_cc_s43", f"{O}/tcir_wpac_cc_s44"]),
    ("crosschannel+feather3", "#e37400",
     [f"{O}/tcir_wpac_ccf_s42", f"{O}/tcir_wpac_ccf_s43", f"{O}/tcir_wpac_ccf_s44"]),
    ("harmonic", "#7b1fa2",
     [f"{O}/tcir_wpac_harm_s42", f"{O}/tcir_wpac_harm_s43", f"{O}/tcir_wpac_harm_s44"]),
]

SEEDS = ["s42", "s43", "s44"]


def arm_available(dirs):
    need = ["metrics.json", "test_metrics.json", "grouped_eval.json"]
    return all(all(os.path.exists(f"{d}/{f}") for f in need) for d in dirs)


def load(dirs):
    out = []
    for d in dirs:
        m = json.load(open(f"{d}/metrics.json", encoding="utf-8"))
        t = json.load(open(f"{d}/test_metrics.json", encoding="utf-8"))["metrics"]
        g = json.load(open(f"{d}/grouped_eval.json", encoding="utf-8"))
        out.append({"val": -m["best_score"], "test": t["rmse"], "mae": t["mae"],
                    "r2": t["r2"], "corr": g["corrected"]["rmse"],
                    "clean": g["clean"]["rmse"]})
    return out


def paired_t(d):
    d = np.asarray(d, dtype=float)
    sd = d.std(ddof=1)
    if sd == 0:
        return float("inf")
    return float(d.mean() / (sd / np.sqrt(len(d))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/inpaint_arms_compare.png")
    args = ap.parse_args()

    arms = [(nm, c, load(d)) for nm, c, d in ARMS if arm_available(d)]
    if len(arms) < 2:
        raise SystemExit("可用臂不足 2 个，先跑训练/评估与 grouped_eval。")

    base = arms[0][2]
    n = len(arms)
    ncol = 2
    nrow = int(np.ceil(n / ncol))

    fig = plt.figure(figsize=(13.6, 4.9 * nrow))
    fig.suptitle("补全方法下游对比（每臂 3 个 seed 配对；仅补全方法不同，split_seed=42 固定）",
                 fontsize=13.5, fontweight="bold", y=0.98)

    # ---- (1)(2) 上排：test RMSE 分组柱 + val/test 均值 ----
    ax = fig.add_subplot(nrow, ncol, 1)
    x = np.arange(3)
    w = 0.8 / n
    for k, (nm, c, rows) in enumerate(arms):
        t = np.array([r["test"] for r in rows])
        xk = x + (k - (n - 1) / 2) * w
        ax.bar(xk, t, w * 0.9, color=c, label=f"{nm} （均值 {t.mean():.2f}）",
               edgecolor="white")
        for xi, v in zip(xk, t):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.6, rotation=90)
        ax.axhline(t.mean(), ls="--", lw=1.0, color=c, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s[1:]}" for s in SEEDS])
    ax.set_ylabel("测试集 RMSE (knots)")
    ax.set_ylim(13.2, 15.05)
    ax.set_title("(1) 测试集 RMSE（3 seed，虚线=各臂均值）", fontsize=11.5)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.legend(fontsize=8.5, loc="upper left", ncol=1)

    ax = fig.add_subplot(nrow, ncol, 2)
    for k, (nm, c, rows) in enumerate(arms):
        v = np.array([r["val"] for r in rows])
        t = np.array([r["test"] for r in rows])
        ax.scatter(np.full(3, k - 0.13) + np.array([-0.05, 0, 0.05]), v, s=52,
                   color=c, marker="o", alpha=0.85)
        ax.scatter(np.full(3, k + 0.13) + np.array([-0.05, 0, 0.05]), t, s=52,
                   color=c, marker="s", alpha=0.85)
        ax.hlines(v.mean(), k - 0.30, k - 0.02, color=c, lw=2.6)
        ax.hlines(t.mean(), k + 0.02, k + 0.30, color=c, lw=2.6)
        ax.text(k - 0.16, v.mean() - 0.20, f"{v.mean():.3f}", ha="center",
                fontsize=8.5, color=c, fontweight="bold")
        ax.text(k + 0.16, t.mean() + 0.09, f"{t.mean():.3f}", ha="center",
                fontsize=8.5, color=c, fontweight="bold")
    ax.set_xticks(range(n))
    ax.set_xticklabels([a[0] for a in arms], fontsize=8.5, rotation=12)
    ax.set_ylabel("RMSE (knots)")
    ax.set_title("(2) val / test 均值（●val ■test，横线=均值）", fontsize=11.5)
    ax.grid(axis="y", ls=":", alpha=0.4)

    # ---- (3)(4) 下排：配对差 + 分组均值 ----
    ax = fig.add_subplot(nrow, ncol, 3)
    gd_t = np.array([r["test"] for r in base])
    ax.axhline(0, color="#333", lw=1.2)
    for k, (nm, c, rows) in enumerate(arms):
        if k == 0:
            continue
        dt = np.array([r["test"] for r in rows]) - gd_t
        t = paired_t(dt)
        lab = f"{nm}  均值 {dt.mean():+.3f}  配对 t={t:.2f}"
        ax.plot(x, dt, "o-", color=c, lw=2, ms=8, label=lab)
        ax.hlines(dt.mean(), -0.35, 2.35, color=c, ls="--", lw=1.2)
        for i in range(3):
            ax.annotate(f"{dt[i]:+.2f}", (i, dt[i]), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s[1:]}" for s in SEEDS])
    ax.set_ylabel("Δ 测试集 RMSE = 方法 - griddata (knots)")
    ax.set_title("(3) 配对差（正=比 griddata 差；df=2）", fontsize=11.5)
    ax.set_ylim(-0.15, 1.25)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left")

    ax = fig.add_subplot(nrow, ncol, 4)
    xx = np.arange(2)
    w = 0.8 / n
    gd_c = np.array([r["corr"] for r in base])
    gd_cl = np.array([r["clean"] for r in base])
    for k, (nm, c, rows) in enumerate(arms):
        vals = np.array([np.mean([r["corr"] for r in rows]),
                         np.mean([r["clean"] for r in rows])])
        xk = xx + (k - (n - 1) / 2) * w
        ax.bar(xk, vals, w * 0.9, color=c, label=nm, edgecolor="white")
        for xi, v in zip(xk, vals):
            ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)
    for i, (a, b) in enumerate([(gd_c.mean(), 0), (gd_cl.mean(), 0)]):
        pass
    ax.set_xticks(xx)
    ax.set_xticklabels(["修正组 (n=637)", "干净组 (n=2256)"])
    ax.set_ylabel("测试集 RMSE (knots)（3 seed 均值）")
    ax.set_title("(4) 分组均值：修正组与干净组", fontsize=11.5)
    ax.set_ylim(12.0, 15.0)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.legend(fontsize=8.5, loc="upper left", ncol=2)
    # 在各柱上方标注相对 griddata 的差
    for k, (nm, c, rows) in enumerate(arms):
        if k == 0:
            continue
        vals = np.array([np.mean([r["corr"] for r in rows]),
                         np.mean([r["clean"] for r in rows])])
        ref = np.array([gd_c.mean(), gd_cl.mean()])
        for i in range(2):
            ax.annotate(f"{vals[i]-ref[i]:+.2f}", xy=(i + (k - (n - 1) / 2) * w, vals[i]),
                        xytext=(i + (k - (n - 1) / 2) * w, vals[i] + 0.26),
                        ha="center", fontsize=8, color=c, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=118)
    plt.close(fig)

    print(f"已保存: {args.out}\n")
    hdr = f"{'arm':<24}{'val 均值':>10}{'test 均值':>11}{'修正组':>9}{'干净组':>9}{'Δtest':>10}{'配对t':>8}"
    print(hdr)
    print("-" * len(hdr))
    for k, (nm, c, rows) in enumerate(arms):
        v = np.mean([r["val"] for r in rows])
        t = np.mean([r["test"] for r in rows])
        cc = np.mean([r["corr"] for r in rows])
        cl = np.mean([r["clean"] for r in rows])
        dt = np.array([r["test"] for r in rows]) - gd_t
        tt = 0.0 if k == 0 else paired_t(dt)
        dsm = 0.0 if k == 0 else dt.mean()
        print(f"{nm:<24}{v:>10.4f}{t:>11.4f}{cc:>9.4f}{cl:>9.4f}{dsm:>+10.4f}{tt:>8.2f}")
    print("\n逐 seed Δtest:")
    for k, (nm, c, rows) in enumerate(arms):
        if k == 0:
            continue
        dt = np.array([r["test"] for r in rows]) - gd_t
        print(f"  {nm:<24}{np.round(dt, 4)}")


if __name__ == "__main__":
    main()
