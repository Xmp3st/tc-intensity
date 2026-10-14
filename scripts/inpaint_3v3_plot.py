# -*- coding: utf-8 -*-
"""
inpaint_3v3_plot.py —— griddata vs crosschannel 的三种子（3v3）配对对比图。

griddata    : v2_3x/run0_s42, run1_s43, run2_s44
crosschannel: cc_v1(s42), cc_s43, cc_s44
除 missing_region_method 外完全同参（split_seed=42 / 80轮 / patience20）。

四联图：
  (1) 测试集 RMSE（按 seed 配对）
  (2) val / test 分布（散点 + 均值）
  (3) 配对差 Δ = crosschannel − griddata（每个 seed 一个点）
  (4) 修正组 / 干净组 均值
"""
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

GD = [("s42", "outputs/tcir_wpac/tcir_wpac_v2_3x/run0_s42"),
      ("s43", "outputs/tcir_wpac/tcir_wpac_v2_3x/run1_s43"),
      ("s44", "outputs/tcir_wpac/tcir_wpac_v2_3x/run2_s44")]
CC = [("s42", "outputs/tcir_wpac/tcir_wpac_cc_v1"),
      ("s43", "outputs/tcir_wpac/tcir_wpac_cc_s43"),
      ("s44", "outputs/tcir_wpac/tcir_wpac_cc_s44")]
C = {"griddata": "#8a8f98", "crosschannel": "#1a73e8"}


def load(runs):
    out = []
    for tag, d in runs:
        m = json.load(open(f"{d}/metrics.json", encoding="utf-8"))
        t = json.load(open(f"{d}/test_metrics.json", encoding="utf-8"))["metrics"]
        g = json.load(open(f"{d}/grouped_eval.json", encoding="utf-8"))
        out.append({"tag": tag, "val": -m["best_score"], "test": t["rmse"],
                    "mae": t["mae"], "r2": t["r2"],
                    "corr": g["corrected"]["rmse"], "clean": g["clean"]["rmse"]})
    return out


gd, cc = load(GD), load(CC)
seeds = [x["tag"] for x in gd]
gd_v = np.array([x["val"] for x in gd]); gd_t = np.array([x["test"] for x in gd])
cc_v = np.array([x["val"] for x in cc]); cc_t = np.array([x["test"] for x in cc])
gd_c = np.array([x["corr"] for x in gd]); gd_cl = np.array([x["clean"] for x in gd])
cc_c = np.array([x["corr"] for x in cc]); cc_cl = np.array([x["clean"] for x in cc])

fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.4))
fig.suptitle("补全方法 3v3 配对对比（griddata vs crosschannel；仅补全方法不同，split_seed=42 固定）",
             fontsize=13.5, fontweight="bold", y=0.975)
x = np.arange(3); w = 0.36

# (1) test RMSE by seed
ax = axes[0, 0]
ax.bar(x - w / 2, gd_t, w, color=C["griddata"], label="griddata", edgecolor="white")
ax.bar(x + w / 2, cc_t, w, color=C["crosschannel"], label="crosschannel", edgecolor="white")
for i in range(3):
    ax.text(i - w / 2, gd_t[i] + 0.03, f"{gd_t[i]:.2f}", ha="center", fontsize=9.5)
    ax.text(i + w / 2, cc_t[i] + 0.03, f"{cc_t[i]:.2f}", ha="center", fontsize=9.5,
            color="#1a56c4", fontweight="bold")
    ax.text(i, min(gd_t[i], cc_t[i]) - 0.17, f"{cc_t[i]-gd_t[i]:+.2f}", ha="center",
            fontsize=9, color="#c5221f")
ax.axhline(gd_t.mean(), ls="--", lw=1.1, color=C["griddata"])
ax.axhline(cc_t.mean(), ls="--", lw=1.1, color=C["crosschannel"])
ax.text(2.42, gd_t.mean() - 0.03, f"均值 {gd_t.mean():.2f}", fontsize=9,
        color=C["griddata"], ha="right", va="top")
ax.text(2.42, cc_t.mean() + 0.03, f"均值 {cc_t.mean():.2f}", fontsize=9,
        color=C["crosschannel"], ha="right", va="bottom")
ax.set_xticks(x); ax.set_xticklabels([f"seed {s[1:]}" for s in seeds])
ax.set_ylabel("测试集 RMSE (knots)"); ax.set_ylim(13.2, 14.85)
ax.set_title("(1) 测试集 RMSE —— 3 个 seed 全部为负收益", fontsize=11.5)
ax.grid(axis="y", ls=":", alpha=0.4); ax.legend(fontsize=9, loc="upper left")

# (2) val / test 分布
ax = axes[0, 1]
for i, (nm, v, t) in enumerate([("griddata", gd_v, gd_t), ("crosschannel", cc_v, cc_t)]):
    ax.scatter(np.full(3, i - 0.14) + np.array([-0.04, 0, 0.04]), v, s=64,
               color=C[nm], marker="o", alpha=0.85, label=f"{nm} val")
    ax.scatter(np.full(3, i + 0.14) + np.array([-0.04, 0, 0.04]), t, s=64,
               color=C[nm], marker="s", alpha=0.85, label=f"{nm} test")
    ax.hlines(v.mean(), i - 0.30, i - 0.02, color=C[nm], lw=2.6)
    ax.hlines(t.mean(), i + 0.02, i + 0.30, color=C[nm], lw=2.6)
    ax.text(i - 0.16, v.mean() - 0.22, f"{v.mean():.3f}", ha="center", fontsize=9,
            color=C[nm], fontweight="bold")
    ax.text(i + 0.16, t.mean() + 0.10, f"{t.mean():.3f}", ha="center", fontsize=9,
            color=C[nm], fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["griddata", "crosschannel"])
ax.set_ylabel("RMSE (knots)"); ax.set_title("(2) val 打平、test 分化（●val ■test，横线=均值）",
                                            fontsize=11.5)
ax.grid(axis="y", ls=":", alpha=0.4); ax.legend(fontsize=8.5, ncol=2, loc="upper left")

# (3) 配对差
ax = axes[1, 0]
dv = cc_v - gd_v; dt = cc_t - gd_t
ax.axhline(0, color="#333", lw=1.2)
ax.plot(x, dv, "o-", color="#f29900", lw=2, ms=9, label="Δ val")
ax.plot(x, dt, "s-", color="#c5221f", lw=2, ms=9, label="Δ test")
ax.hlines(dv.mean(), -0.35, 2.35, color="#f29900", ls="--", lw=1.2)
ax.hlines(dt.mean(), -0.35, 2.35, color="#c5221f", ls="--", lw=1.2)
ax.text(2.4, dv.mean(), f"均值 {dv.mean():+.3f}", fontsize=9, color="#f29900",
        ha="right", va="bottom")
ax.text(2.4, dt.mean(), f"均值 {dt.mean():+.3f}", fontsize=9, color="#c5221f",
        ha="right", va="bottom")
for i in range(3):
    ax.annotate(f"{dt[i]:+.2f}", (i, dt[i]), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=9, color="#c5221f")
ax.set_xticks(x); ax.set_xticklabels([f"seed {s[1:]}" for s in seeds])
ax.set_ylabel("Δ RMSE = crosschannel - griddata (knots)")
ax.set_title("(3) 配对差 —— 正=更差。test 三点同号（配对 t≈3.9, p≈0.06）", fontsize=11.5)
ax.set_ylim(-0.42, 0.62); ax.grid(axis="y", ls=":", alpha=0.4)
ax.legend(fontsize=9, loc="upper left")

# (4) 分组均值
ax = axes[1, 1]
xx = np.array([0, 1])
b1 = ax.bar(xx - w / 2, [gd_c.mean(), gd_cl.mean()], w, color=C["griddata"],
            label="griddata", edgecolor="white")
b2 = ax.bar(xx + w / 2, [cc_c.mean(), cc_cl.mean()], w, color=C["crosschannel"],
            label="crosschannel", edgecolor="white")
for b, v in zip(b1, [gd_c.mean(), gd_cl.mean()]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.2f}", ha="center", fontsize=9.5)
for b, v in zip(b2, [cc_c.mean(), cc_cl.mean()]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.2f}", ha="center",
            fontsize=9.5, color="#1a56c4", fontweight="bold")
ax.set_xticks(xx); ax.set_xticklabels([f"修正组 (n=637)", f"干净组 (n=2256)"])
ax.set_ylabel("测试集 RMSE (knots)  (3 seed 均值)")
ax.set_title("(4) 两个组都一致变差 ~0.3 → 全局模型质量偏移，非补全值问题", fontsize=11.5)
ax.set_ylim(12.0, 14.9); ax.grid(axis="y", ls=":", alpha=0.4)
ax.legend(fontsize=9, loc="upper left")
for i, (a, b_) in enumerate([(gd_c.mean(), cc_c.mean()), (gd_cl.mean(), cc_cl.mean())]):
    ax.annotate(f"{b_-a:+.2f}", xy=(i + w / 2, b_), xytext=(i + w / 2, b_ + 0.30),
                ha="center", fontsize=10, color="#c5221f", fontweight="bold")

fig.tight_layout(rect=[0, 0, 1, 0.945])
os.makedirs("outputs", exist_ok=True)
fig.savefig("outputs/inpaint_3v3_compare.png", dpi=118)

print("已保存: outputs/inpaint_3v3_compare.png\n")
print(f"griddata     val {gd_v.mean():.4f} ± {gd_v.std(ddof=1):.4f} | "
      f"test {gd_t.mean():.4f} ± {gd_t.std(ddof=1):.4f}")
print(f"crosschannel val {cc_v.mean():.4f} ± {cc_v.std(ddof=1):.4f} | "
      f"test {cc_t.mean():.4f} ± {cc_t.std(ddof=1):.4f}")
print(f"Δval  均值 {dv.mean():+.4f} (逐 seed {np.round(dv,4)})")
print(f"Δtest 均值 {dt.mean():+.4f} (逐 seed {np.round(dt,4)})")
t = dt.mean() / (dt.std(ddof=1) / np.sqrt(len(dt)))
print(f"配对 t(test) = {t:.2f}  (df=2)")
