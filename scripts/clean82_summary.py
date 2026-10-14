# -*- coding: utf-8 -*-
"""汇总「无损坏子集 8:2」两组实验（泄漏 random / 对照 by_storm）并出图。

读 outputs/tcir_wpac/tcir_clean82_{leak,bystorm}_s{42,43,44}/test_metrics.json，
输出 outputs/tcir_wpac/clean82_compare.{png,csv,json}。
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tcintens.utils.plotting import use_cjk_font

ARMS = [("leak", "random（不防泄露）", "#c0392b"),
        ("bystorm", "by_storm（防泄露）", "#2b6cb0")]
SEEDS = [42, 43, 44]
OUT_DIR = "outputs/tcir_wpac"
REF = [("resnet50 · 含损坏/by_storm/70-15-15", 13.384),
       ("vgg16 v2 · 含损坏/by_storm/70-15-15", 13.539)]


def load():
    rows = []
    for arm, label, _ in ARMS:
        for s in SEEDS:
            p = os.path.join(OUT_DIR, "tcir_clean82_%s_s%d" % (arm, s),
                             "test_metrics.json")
            if not os.path.exists(p):
                print("[skip] 缺 %s" % p)
                continue
            with open(p, encoding="utf-8") as f:
                m = json.load(f)["metrics"]
            rows.append(dict(arm=arm, label=label, seed=s,
                             rmse=float(m["rmse"]), mae=float(m["mae"]),
                             r2=float(m["r2"])))
    return rows


def paired_t(d):
    d = np.asarray(d, float)
    n = len(d)
    if n < 2:
        return float("nan"), float("nan")
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return float("nan"), float("nan")
    t = d.mean() / se
    from scipy import stats
    return float(t), float(2 * stats.t.sf(abs(t), df=n - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "clean82_compare.png"))
    args = ap.parse_args()
    use_cjk_font()

    rows = load()
    if not rows:
        print("没有任何结果，退出")
        return
    by = {}
    for r in rows:
        by.setdefault(r["arm"], {})[r["seed"]] = r

    # ---- 表 ----
    import pandas as pd
    df = pd.DataFrame(rows)[["arm", "label", "seed", "rmse", "mae", "r2"]]
    print("\n=== 逐 run ===")
    print(df.round(3).to_string(index=False))

    stat = {}
    print("\n=== 各臂均值 ===")
    for arm, label, _ in ARMS:
        sub = by.get(arm, {})
        if not sub:
            continue
        rm = np.array([sub[s]["rmse"] for s in SEEDS if s in sub])
        ma = np.array([sub[s]["mae"] for s in SEEDS if s in sub])
        r2 = np.array([sub[s]["r2"] for s in SEEDS if s in sub])
        stat[arm] = dict(label=label, n=len(rm), rmse_mean=float(rm.mean()),
                         rmse_std=float(rm.std(ddof=1)) if len(rm) > 1 else 0.0,
                         mae_mean=float(ma.mean()), r2_mean=float(r2.mean()),
                         rmse=rm.tolist())
        print("  %-24s RMSE %.3f ± %.3f | MAE %.3f | R2 %.4f"
              % (label, rm.mean(), stat[arm]["rmse_std"], ma.mean(), r2.mean()))

    delta = None
    if "leak" in by and "bystorm" in by:
        common = [s for s in SEEDS if s in by["leak"] and s in by["bystorm"]]
        d = np.array([by["leak"][s]["rmse"] - by["bystorm"][s]["rmse"] for s in common])
        t, p = paired_t(d)
        delta = dict(seeds=common, per_seed=d.tolist(), mean=float(d.mean()),
                     t=t, p=p)
        print("\n=== 配对差（leak - by_storm，负=泄漏让结果虚好）===")
        for s, v in zip(common, d):
            print("  seed %d : %+.3f" % (s, v))
        print("  均值 %+.3f  配对 t=%.2f (df=%d, p=%.3f)" % (d.mean(), t, len(d) - 1, p))

    # ---- 图 ----
    fig = plt.figure(figsize=(14, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.30)

    # (1) 逐 seed RMSE
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(SEEDS))
    w = 0.36
    for k, (arm, label, c) in enumerate(ARMS):
        vals = [by.get(arm, {}).get(s, {}).get("rmse", np.nan) for s in SEEDS]
        ax.bar(x + (k - 0.5) * w, vals, w * 0.92, color=c, label=label, edgecolor="white")
        for xi, v in zip(x + (k - 0.5) * w, vals):
            if np.isfinite(v):
                ax.text(xi, v + 0.12, "%.2f" % v, ha="center", fontsize=8.5)
        if arm in stat:
            ax.axhline(stat[arm]["rmse_mean"], ls="--", lw=1.1, color=c, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(["seed %d" % s for s in SEEDS])
    ax.set_ylabel("测试集 RMSE (knots)")
    ax.set_ylim(0, 18.2)
    ax.set_title("(1) 测试集 RMSE：泄漏 vs 防泄露", fontsize=11.5)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.legend(fontsize=8.5, loc="upper left")

    # (2) 配对差
    ax = fig.add_subplot(gs[0, 1])
    if delta is not None:
        d = np.array(delta["per_seed"])
        ax.axhline(0, color="#333", lw=1.2)
        ax.plot(x, d, "o-", color="#7d3c98", lw=2, ms=8)
        ax.hlines(d.mean(), -0.35, 2.35, color="#7d3c98", ls="--", lw=1.2)
        for i, v in enumerate(d):
            ax.annotate("%+.2f" % v, (i, v), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=9, color="#7d3c98")
        ax.text(0.05, 0.92, "均值 %+.3f\n配对 t=%.2f (df=%d, p=%.3g)"
                % (d.mean(), delta["t"], len(d) - 1, delta["p"]),
                transform=ax.transAxes, fontsize=9, color="#7d3c98",
                va="top", ha="left")
        ax.set_xticks(x)
        ax.set_xticklabels(["seed %d" % s for s in delta["seeds"]])
        ax.set_ylabel("Δ RMSE = leak - by_storm (knots)")
        ax.set_title("(2) 配对差（负=泄漏使结果虚好）", fontsize=11.5)
        ax.grid(axis="y", ls=":", alpha=0.4)

    # (3) MAE / R2
    ax = fig.add_subplot(gs[0, 2])
    xs = np.arange(2)
    for k, (arm, label, c) in enumerate(ARMS):
        if arm not in stat:
            continue
        ax.bar(xs + (k - 0.5) * 0.36, [stat[arm]["mae_mean"], stat[arm]["r2_mean"] * 20],
               0.33, color=c, label=label, edgecolor="white")
        ax.text(xs[0] + (k - 0.5) * 0.36, stat[arm]["mae_mean"] + 0.15,
                "%.2f" % stat[arm]["mae_mean"], ha="center", fontsize=8.5)
        ax.text(xs[1] + (k - 0.5) * 0.36, stat[arm]["r2_mean"] * 20 + 0.15,
                "%.3f" % stat[arm]["r2_mean"], ha="center", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(["MAE (knots)", "R2 (×20)"])
    ax.set_ylim(0, 22)
    ax.set_title("(3) 均值：MAE 与 R2", fontsize=11.5)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.legend(fontsize=8.5, loc="upper left")

    fig.suptitle("无损坏子集 (project 口径, 15986 帧) · resnet50 · 8:2 划分 · 3 seed",
                 fontsize=13.5, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("\n[图] %s" % args.out)

    with open(os.path.join(OUT_DIR, "clean82_compare.csv"), "w", newline="",
              encoding="utf-8") as f:
        df.to_csv(f, index=False)
    with open(os.path.join(OUT_DIR, "clean82_compare.json"), "w",
              encoding="utf-8") as f:
        json.dump({"rows": rows, "stats": stat, "paired": delta}, f,
                  ensure_ascii=False, indent=2)
    print("[数据] %s/clean82_compare.{csv,json}" % OUT_DIR)


if __name__ == "__main__":
    main()
