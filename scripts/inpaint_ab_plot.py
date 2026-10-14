# -*- coding: utf-8 -*-
"""
inpaint_ab_plot.py —— 缺失补全方法对下游 Vmax 回归影响的 A/B 对照图。

对照两组（除补全方法外完全同参：seed=42 / split_seed=42 / 80轮 / patience=20）：
  A = griddata    : outputs/tcir_wpac/tcir_wpac_v2_3x/run0_s42
  B = crosschannel: outputs/tcir_wpac/tcir_wpac_cc_v1

四联图：
  (1) 整体 val/test RMSE
  (2) 分组 RMSE（修正组 / 干净组）
  (3) 补全精度（合成 benchmark，有真值）
  (4) 交叉诊断：模型 x 评估补全方法 的 2x2 混淆矩阵（测模型是否依赖补全风格）
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

use_cjk_font(verbose=True)

PAIRS = [("griddata", "outputs/tcir_wpac/tcir_wpac_v2_3x/run0_s42"),
         ("crosschannel", "outputs/tcir_wpac/tcir_wpac_cc_v1")]
LABELS = {"griddata": "griddata\n(原实现)", "crosschannel": "crosschannel\n(新)"}
C = {"griddata": "#8a8f98", "crosschannel": "#1a73e8"}


def _load(method, d):
    g = json.load(open(f"{d}/grouped_eval.json", encoding="utf-8"))
    val = json.load(open(f"{d}/metrics.json", encoding="utf-8"))["best_score"]
    try:
        t = json.load(open(f"{d}/test_metrics.json", encoding="utf-8"))["metrics"]
    except Exception:
        t = {"rmse": g["overall"]["rmse"]}
    return {"method": method, "val_rmse": -val, "test_rmse": t["rmse"],
            "corr": g["corrected"], "clean": g["clean"], "gap": g["gap_rmse"],
            "n_corr": g["n_corrected"], "n_clean": g["n_clean"]}


res = [_load(m, d) for m, d in PAIRS]
names = [r["method"] for r in res]
colors = [C[m] for m in names]

fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.6))
fig.suptitle("缺失补全方法对下游 Vmax 回归的影响（仅补全方法不同，其余完全同参 seed=42）",
             fontsize=14, fontweight="bold", y=0.975)

x = np.arange(2)
w = 0.34

# ---- (1) val / test RMSE ----
ax = axes[0, 0]
v_vals = [r["val_rmse"] for r in res]
t_vals = [r["test_rmse"] for r in res]
b1 = ax.bar(x - w / 2, v_vals, w, label="验证集 RMSE", color=colors, alpha=0.5,
            edgecolor="white")
b2 = ax.bar(x + w / 2, t_vals, w, label="测试集 RMSE", color=colors, edgecolor="white")
for b, v in zip(b1, v_vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center",
            fontsize=9.5, color="#444")
for b, v in zip(b2, t_vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center",
            fontsize=9.5, fontweight="bold", color="#111")
ax.set_xticks(x)
ax.set_xticklabels([LABELS[n] for n in names])
ax.set_ylabel("RMSE (knots)")
ax.set_title("(1) 整体误差 —— 越低越好", fontsize=11.5)
ax.set_ylim(12.3, max(t_vals) + 0.85)
ax.grid(axis="y", ls=":", alpha=0.4)
ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
ax.annotate(f"Δval {v_vals[1]-v_vals[0]:+.2f}    Δtest {t_vals[1]-t_vals[0]:+.2f}",
            xy=(0.56, 0.965), xycoords="axes fraction", ha="center", va="top",
            fontsize=10.5, color="#c5221f",
            bbox=dict(boxstyle="round,pad=0.38", fc="#fdecea", ec="#c5221f", lw=0.9))

# ---- (2) 修正组 / 干净组 RMSE ----
ax = axes[0, 1]
corr = [r["corr"]["rmse"] for r in res]
clean = [r["clean"]["rmse"] for r in res]
b1 = ax.bar(x - w / 2, corr, w, label=f"修正组 (n={res[0]['n_corr']})", color=colors,
            alpha=0.5, edgecolor="white")
b2 = ax.bar(x + w / 2, clean, w, label=f"干净组 (n={res[0]['n_clean']})", color=colors,
            edgecolor="white")
for b, v in zip(b1, corr):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center", fontsize=9.5)
for b, v in zip(b2, clean):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center",
            fontsize=9.5, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([LABELS[n] for n in names])
ax.set_ylabel("RMSE (knots)")
ax.set_title("(2) 分组误差 —— 退步主要来自「干净组」", fontsize=11.5)
ax.set_ylim(12.3, max(clean) + 0.85)
ax.grid(axis="y", ls=":", alpha=0.4)
ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
ax.annotate(f"干净组 {clean[1]-clean[0]:+.2f}  |  修正组 {corr[1]-corr[0]:+.2f}",
            xy=(0.56, 0.965), xycoords="axes fraction", ha="center", va="top",
            fontsize=10.5, color="#c5221f",
            bbox=dict(boxstyle="round,pad=0.38", fc="#fdecea", ec="#c5221f", lw=0.9))

# ---- (3) 补全精度（合成 benchmark，有真值） ----
ax = axes[1, 0]
ch = ["IR1", "WV"]
gv = [11.106, 10.876]
cv = [8.708, 5.746]
xx = np.arange(2)
b1 = ax.bar(xx - w / 2, gv, w, color=C["griddata"], label="griddata (原实现)")
b2 = ax.bar(xx + w / 2, cv, w, color=C["crosschannel"], label="crosschannel (新)")
for b, v in zip(b1, gv):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.18, f"{v:.1f}", ha="center", fontsize=9.5)
for b, v in zip(b2, cv):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.18, f"{v:.1f}", ha="center",
            fontsize=9.5, fontweight="bold", color="#1a56c4")
for i, (a, b_) in enumerate(zip(gv, cv)):
    ax.annotate(f"-{(a-b_)/a*100:.0f}%", xy=(i + w / 2, b_), xytext=(i + w / 2, b_ + 1.35),
                ha="center", fontsize=11, color="#0b7a3b", fontweight="bold")
ax.set_xticks(xx)
ax.set_xticklabels(ch)
ax.set_ylabel("补全 RMSE (K)  ↓越低越好")
ax.set_title("(3) 补全精度（合成 benchmark，有真值）", fontsize=11.5)
ax.grid(axis="y", ls=":", alpha=0.4)
ax.legend(fontsize=9, framealpha=0.9)
ax.set_ylim(0, max(gv) * 1.28)

# ---- (4) 交叉诊断 2x2 ----
ax = axes[1, 1]
mat = np.array([[14.1581, 14.1356],     # 行=训练用; 列=评估用
                [14.5376, 14.4988]])    # [griddata, crosschannel]
im = ax.imshow(mat, cmap="RdYlGn_r", vmin=14.0, vmax=14.65, aspect="auto")
ax.set_xticks([0, 1]); ax.set_xticklabels(["评估: griddata", "评估: crosschannel"], fontsize=9.5)
ax.set_yticks([0, 1]); ax.set_yticklabels(["训练:\ngriddata", "训练:\ncrosschannel"], fontsize=9.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=13,
                fontweight="bold", color="#111")
for i in range(2):
    ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor="#111", lw=2.2))  # 对角线=同分布
ax.set_title("(4) 交叉诊断：模型 × 评估补全方法\n"
             "框内为同分布；换评估补全仅差 0.02~0.04 → 模型不依赖补全风格",
             fontsize=10.5)
ax.text(1.02, -0.30, "值=测试集 RMSE", transform=ax.transAxes, ha="right",
        fontsize=9, color="#555")

fig.tight_layout(rect=[0, 0, 1, 0.945])
os.makedirs("outputs", exist_ok=True)
out = "outputs/inpaint_ab_compare.png"
fig.savefig(out, dpi=118)
print("已保存:", out)
print("\n数值汇总:")
for r in res:
    print(f"  {r['method']:14s} val {r['val_rmse']:.4f} | test {r['test_rmse']:.4f} | "
          f"corr {r['corr']['rmse']:.4f} | clean {r['clean']['rmse']:.4f} | gap {r['gap']:+.4f}")
print("\n交叉诊断 (行=训练补全, 列=评估补全):")
print(f"  griddata x griddata       = {mat[0,0]:.4f}")
print(f"  griddata x crosschannel   = {mat[0,1]:.4f}")
print(f"  crosschannel x griddata   = {mat[1,0]:.4f}")
print(f"  crosschannel x crosschannel= {mat[1,1]:.4f}")
