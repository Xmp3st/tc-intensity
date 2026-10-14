#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/worst_grouped_viz.py —— 测试集「误差最大 top-N」与「Pred vs True」可视化，
                              **按 clean / corrected 分组对比**。

与 error_analysis.py / visualize_test.py 的区别：
  * 数据集开启 return_corrected_flag，逐样本拿到 was_corrected（原图是否有缺失、被插值修复）；
  * worst-N 图的每张缩略图左上角带分组徽标 [CLEAN] / [CORR]；
  * 额外产出独立的 Pred-vs-True 分面板对比图 + 指标表（clean / corrected / overall）。

用法（最终模型 best_of_3，预处理自动对齐 run1 的 config_used.yaml）：
  python scripts/worst_grouped_viz.py -c configs/tcir_wpac_train.yaml \
      --set experiment.name=run1_s43 \
      --set experiment.output_dir=outputs/tcir_wpac/tcir_wpac_v1_3x \
      --ckpt outputs/tcir_wpac/tcir_wpac_v1_3x/best_of_3.ckpt --worst 12

输出（写入 experiment 目录 = .../run1_s43/）：
  worst12_grouped.png      top-N 缩略图（带 clean/corr 徽标）
  worst12_grouped.csv      top-N 明细（含 was_corrected 列）
  pred_vs_true_grouped.png Pred-vs-True 分组面板 + 指标表
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import h5py
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tcintens.data.dataset import create_datasets
from tcintens.models.tc_model import build_model
from tcintens.utils.config import load_config
from tcintens.utils.logger import get_logger
from tcintens.utils.metrics import regression_metrics
from tcintens.utils.misc import resolve_device
from tcintens.utils.repro import align_preprocessing
from tcintens.engine.common import load_checkpoint

C_CLEAN = "#55A868"   # 绿：干净
C_CORR = "#C44E52"    # 红：修正


# --------------------------------------------------------------------------- #
def denorm_for_display(x, mean, std):
    """(C,H,W) 归一化张量 -> 反归一化物理量 numpy。"""
    x = x.cpu().numpy().astype(np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    return x * std[:, None, None] + mean[:, None, None]


def to_composite(x):
    """(C,H,W) -> 伪 RGB (H,W,3)：逐通道 min-max 拉伸后映射 R/G/B。"""
    x = np.ascontiguousarray(np.transpose(x, (1, 2, 0)))
    c = x.shape[-1]
    out = np.zeros((x.shape[0], x.shape[1], 3), dtype=np.float32)
    src = [0, 1, 2] if c >= 3 else (list(range(c)) + [0] * (3 - c))
    for i in range(3):
        v = x[..., src[i]]
        vmin, vmax = np.nanmin(v), np.nanmax(v)
        out[..., i] = (v - vmin) / (vmax - vmin) if (vmax - vmin) > 1e-8 else 0.0
    return np.nan_to_num(out)


def _badge(ax, corr):
    tag = "CORR" if corr else "CLEAN"
    col = C_CORR if corr else C_CLEAN
    ax.text(0.03, 0.97, tag, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, fontweight="bold", color="white",
            bbox=dict(facecolor=col, edgecolor="none", boxstyle="round,pad=0.25"))


# VIS 是原始 TCIR 的第 3 通道（CH_MAP: IR1=0, WV=1, VIS=2, PMW=3），
# 在 extract_wpac.py 抽取紧凑 h5 时被弃用，故需回原始 TCIR.h5 现取。
def load_vis(h5f, orig_idx, R):
    """从原始 TCIR.h5 读一帧 VIS 通道：NaN→0 后 bilinear 下采样到 R×R。"""
    fr = np.asarray(h5f["matrix"][int(orig_idx)])[..., 2].astype(np.float32)
    fr = np.nan_to_num(fr, nan=0.0)
    t = torch.from_numpy(fr)[None, None]
    t = F.interpolate(t, size=(R, R), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/tcir_wpac_train.yaml")
    ap.add_argument("--ckpt", default=None,
                    help="指定权重；默认 outputs/<name>/checkpoints/best.ckpt")
    ap.add_argument("--set", action="append", default=[],
                    help="覆盖配置 KEY=VAL")
    ap.add_argument("--worst", type=int, default=12, help="误差最大的前 N 张")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--tcir-h5", default="data/TCIR.h5",
                    help="原始 TCIR h5（用于取 VIS 通道；相对项目根）")
    ap.add_argument("--info-csv", default="data/wpac_info.csv",
                    help="原始 wpac_info.csv（紧凑行 -> 原始 matrix_index 映射）")
    ap.add_argument("--no-vis", action="store_true", help="不附加 VIS 面板")
    args = ap.parse_args()

    os.chdir(ROOT)
    cfg = load_config(args.config, args.set)
    logger = get_logger("tc-worst")

    # 一致性守卫：对齐缺失修正 / 划分等键到训练时保存的 config_used.yaml
    cfg, _ = align_preprocessing(cfg, overrides=getattr(cfg, "overrides", []),
                                 logger=logger)
    d = cfg.to_dict()
    exp = d["experiment"]
    tcfg = d["data"]["tcir"]
    device = resolve_device(d["device"].get("mode", "auto"))

    datasets, _ = create_datasets(cfg)
    test_ds = datasets["test"]
    test_ds.return_corrected_flag = True
    n_test = len(test_ds)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=exp.get("num_workers", 4))

    model = build_model(cfg).to(device)
    ckpt_path = args.ckpt or os.path.join(
        exp.get("output_dir", "outputs"), exp.get("name", "exp"),
        "checkpoints", "best.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到权重: {ckpt_path}")
    load_checkpoint(ckpt_path, model, map_location=device)
    model.eval()

    # ---------- 全测试集推理（带 flag） ----------
    yt, yp, fl = [], [], []
    with torch.no_grad():
        for x, y, flag in loader:
            out = model(x.to(device, non_blocking=True))
            yp.append(out.cpu().numpy().ravel())
            yt.append(y.numpy().ravel())
            fl.append(np.asarray(flag).ravel())
    yt = np.concatenate(yt).astype(np.float64)
    yp = np.concatenate(yp).astype(np.float64)
    fl = np.concatenate(fl).astype(bool)
    abs_err = np.abs(yp - yt)
    logger.info(f"测试集 N={n_test}；修正 {int(fl.sum())} / 干净 {int((~fl).sum())}")

    # ---------- 指标（分组） ----------
    def grp(mask):
        if int(mask.sum()) == 0:
            return {"rmse": None, "mae": None, "r2": None, "n": 0}
        m = regression_metrics(yt[mask], yp[mask])
        return {"rmse": m["rmse"], "mae": m["mae"], "r2": m["r2"], "n": int(mask.sum())}

    overall = grp(np.ones(n_test, bool))
    g_c = grp(~fl)
    g_r = grp(fl)

    # ---------- 元数据（storm ID） ----------
    csv_path = tcfg.get("csv_path")
    id_col = tcfg.get("storm_col", "ID")
    idx_col = tcfg.get("index_col", "matrix_index")
    real_idx = np.asarray(test_ds.indices, dtype=int)
    if csv_path and os.path.exists(csv_path):
        meta_df = pd.read_csv(csv_path).set_index(idx_col)
        sids = [str(meta_df.iloc[ri][id_col]) for ri in real_idx]
    else:
        sids = [f"r{ri}" for ri in real_idx]

    # ---------- top-N worst（按 |err| 降序） ----------
    n_top = min(args.worst, n_test)
    order = np.argsort(-abs_err)[:n_top]
    comp_rows = real_idx[order]            # 紧凑 h5 行号

    out_dir = os.path.join(exp.get("output_dir", "outputs"), exp.get("name", "exp"))
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 紧凑行 -> 原始 matrix_index 映射（供回原始 h5 取 VIS） ----------
    orig_map = None
    if os.path.exists(args.info_csv):
        _info = pd.read_csv(args.info_csv).sort_values("matrix_index").reset_index(drop=True)
        orig_map = _info["matrix_index"].values.astype(np.int64)
    show_vis = (not args.no_vis) and (orig_map is not None) and os.path.exists(args.tcir_h5)
    if not args.no_vis and not show_vis:
        logger.warning(f"未找到 {args.tcir_h5} 或 {args.info_csv}，本次不附加 VIS 面板")

    # ---------- 图1：worst-N 缩略图（伪RGB + VIS，带分组徽标） ----------
    norm_mean = tcfg.get("norm_mean")
    norm_std = tcfg.get("norm_std")
    R = int(tcfg.get("resize", 96))
    per_row = 3 if n_top > 6 else 2
    rows = int(np.ceil(n_top / per_row))
    ncol = per_row * (2 if show_vis else 1)

    fig = plt.figure(figsize=(ncol * 2.1, rows * 2.9))
    gs = fig.add_gridspec(rows, ncol, wspace=0.06, hspace=0.30)

    vis_p995 = np.full(n_top, np.nan)
    vis_state = np.array(["n/a"] * n_top, dtype=object)
    h5f = h5py.File(args.tcir_h5, "r") if show_vis else None
    try:
        for k, i in enumerate(order):
            r, c = divmod(k, per_row)
            x, _, _ = test_ds[int(i)]
            im = to_composite(denorm_for_display(x, norm_mean, norm_std))
            ax = fig.add_subplot(gs[r, c * (2 if show_vis else 1)])
            ax.imshow(im)
            ax.axis("off")
            _badge(ax, bool(fl[i]))
            ax.set_title(f"#{k+1} {sids[i]}\nPred {yp[i]:.1f} / True {yt[i]:.1f}\n"
                         f"\u0394={yp[i]-yt[i]:+.1f} kn", fontsize=8)
            if show_vis:
                axv = fig.add_subplot(gs[r, c * 2 + 1])
                vis = load_vis(h5f, orig_map[int(comp_rows[k])], R)
                p995 = float(np.percentile(vis, 99.5))
                sat_frac = float((vis >= 2.15).mean())
                vis_p995[k] = p995
                vis_state[k] = ("night" if p995 < 0.1
                                else "sat" if sat_frac >= 0.5 else "ok")
                axv.imshow(vis, cmap="gray", vmin=0.0, vmax=1.0)
                axv.axis("off")
                st = vis_state[k]
                lbl = {"night": "VIS (night / n.a.)",
                       "sat": "VIS (saturated)"}.get(st, "VIS")
                axv.set_title(lbl, fontsize=8,
                              color="#888888" if st != "ok" else "black")
    finally:
        if h5f is not None:
            h5f.close()

    ttl = f"Worst-{n_top} test samples (green=CLEAN, red=CORRECTED)"
    if show_vis:
        ttl += "  |  L: pseudo-RGB (IR1/WV/PMW)   R: VIS"
    fig.suptitle(ttl + f"  |  overall RMSE={overall['rmse']:.2f} kn", fontsize=12)
    worst_png = os.path.join(out_dir, "worst12_grouped.png")
    fig.savefig(worst_png, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---------- 明细 CSV（含 was_corrected / VIS 信息） ----------
    worst_df = pd.DataFrame({
        "rank": np.arange(1, n_top + 1),
        "test_index": order,
        "matrix_index": comp_rows,
        id_col: [sids[i] for i in order],
        "was_corrected": fl[order],
        "true_vmax": np.round(yt[order], 2),
        "pred_vmax": np.round(yp[order], 2),
        "abs_err": np.round(abs_err[order], 2),
        "signed_err": np.round((yp - yt)[order], 2),
        "vis_p99_5": np.round(vis_p995, 3),
        "vis_state": vis_state,
        "vis_daytime": (vis_p995 >= 0.1),
    })
    worst_csv = os.path.join(out_dir, "worst12_grouped.csv")
    worst_df.to_csv(worst_csv, index=False)

    # ---------- 图2：Pred vs True 分组面板 + 指标表 ----------
    fig = plt.figure(figsize=(15, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[4.6, 2.1], hspace=0.40)

    # 坐标轴：从 0 起、覆盖全部点、上端留 5 kn 裕量（共享同一范围，便于三面板对比）
    _hi = float(max(yt.max(), yp.max())) + 5.0
    lim = [0.0, _hi]

    def scatter_panel(ax, mask, color, title, m):
        ax.scatter(yt[mask], yp[mask], s=7, alpha=0.4, c=color)
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect("equal"); ax.set_box_aspect(1)
        ticks = np.arange(0, lim[1] + 1, 25)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xlabel("True Vmax (kn)"); ax.set_ylabel("Pred Vmax (kn)")
        ax.set_title(f"{title}\nRMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  "
                     f"R2={m['r2']:.3f}  (n={m['n']})", fontsize=10)
        return ax

    ax0 = fig.add_subplot(gs[0, 0])
    scatter_panel(ax0, np.ones(n_test, bool), "#4C72B0", "Overall", overall)

    ax1 = fig.add_subplot(gs[0, 1])
    scatter_panel(ax1, ~fl, C_CLEAN, "Clean", g_c)

    ax2 = fig.add_subplot(gs[0, 2])
    scatter_panel(ax2, fl, C_CORR, "Corrected", g_r)

    # 指标表
    axt = fig.add_subplot(gs[1, :])
    axt.axis("off")
    tab = [
        ["Group", "N", "RMSE (kn)", "MAE (kn)", "R2"],
        ["Overall", overall["n"], f"{overall['rmse']:.3f}", f"{overall['mae']:.3f}", f"{overall['r2']:.4f}"],
        ["Clean", g_c["n"], f"{g_c['rmse']:.3f}", f"{g_c['mae']:.3f}", f"{g_c['r2']:.4f}"],
        ["Corrected", g_r["n"], f"{g_r['rmse']:.3f}", f"{g_r['mae']:.3f}", f"{g_r['r2']:.4f}"],
        ["Gap (corr-clean)", "", f"{g_r['rmse']-g_c['rmse']:+.3f}", f"{g_r['mae']-g_c['mae']:+.3f}", ""],
    ]
    t = axt.table(cellText=tab[1:], colLabels=tab[0], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.5)
    axt.set_title("Test metrics by correction group", fontsize=11, pad=14)

    fig.suptitle("Pred vs True — clean vs corrected", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pv_png = os.path.join(out_dir, "pred_vs_true_grouped.png")
    fig.savefig(pv_png, dpi=120)
    plt.close(fig)

    logger.info(f"worst-N 明细: {worst_csv}")
    logger.info(f"worst-N 缩略图: {worst_png}")
    logger.info(f"Pred-vs-True 分组图: {pv_png}")
    logger.info(f"overall RMSE={overall['rmse']:.3f} | clean {g_c['rmse']:.3f} (n={g_c['n']}) "
                f"| corrected {g_r['rmse']:.3f} (n={g_r['n']}) | gap {g_r['rmse']-g_c['rmse']:+.3f}")


if __name__ == "__main__":
    main()
