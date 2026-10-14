#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/corrected_vs_raw.py —— 同一模型对「修正后(corrected)」与「原始 glitch(raw)」
两种输入的对比（只针对测试集中**做过插值修正**的那批样本）。

动机：模型是在「局部区域插值修复后」的数据上训练的。本脚本对同一批有缺测的样本，
      分别构造两种输入喂给同一权重：
        * corrected：missing_band=interp（缺测区域被 griddata 空间插值填补）
        * raw      ：missing_band=none  （保留原始缺测=0 的“花屏”glitch）
      比较两者预测误差，量化「插值修复」到底挽回多少；并给出可视化对照图。

用法（最终模型 best_of_3，预处理自动对齐 run2 的 config_used.yaml）：
  python scripts/corrected_vs_raw.py -c configs/tcir_wpac_train.yaml \
      --set experiment.name=run2_s44 \
      --set experiment.output_dir=outputs/tcir_wpac/tcir_wpac_v2_3x \
      --ckpt outputs/tcir_wpac/tcir_wpac_v2_3x/best_of_3.ckpt --show 12

输出（写入 experiment 目录 = .../run2_s44/）：
  corrected_vs_raw.png           top-N（预测差异最大）的 [RAW | CORRECTED] 对照缩略图
  corrected_vs_raw_metrics.png   指标对比（RMSE 柱状 + pred_raw-vs-pred_corr 散点 + 指标表）
  corrected_vs_raw.csv           全部修正样本的逐样本明细
  corrected_vs_raw.json          汇总指标（raw vs corrected，修正子集 + 全体）
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
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tcintens.data.dataset import create_datasets
from tcintens.data.tcir import TCIRDataset
from tcintens.models.tc_model import build_model
from tcintens.utils.config import load_config
from tcintens.utils.logger import get_logger
from tcintens.utils.metrics import regression_metrics
from tcintens.utils.misc import resolve_device
from tcintens.utils.repro import align_preprocessing
from tcintens.engine.common import load_checkpoint

C_RAW = "#C44E52"    # 红：原始 glitch
C_CORR = "#55A868"   # 绿：修正后


# --------------------------------------------------------------------------- #
def denorm_for_display(x, mean, std):
    """(C,H,W) 归一化张量 -> 反归一化物理量 numpy。"""
    x = x.cpu().numpy().astype(np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    return x * std[:, None, None] + mean[:, None, None]


def to_composite_shared(arrs):
    """arrs: [(C,H,W) 物理量, ...] -> [伪RGB, ...]，**共享**逐通道 min-max 拉伸。

    共享拉伸可保证 raw/corrected 两个面板用同一色彩尺度，缺测“花屏”一眼可见。
    """
    stats = []
    for a in arrs:
        xs = np.ascontiguousarray(np.transpose(a, (1, 2, 0))).reshape(-1, a.shape[0])
        stats.append((np.nanmin(xs, axis=0), np.nanmax(xs, axis=0)))
    vmin = np.min(np.stack([s[0] for s in stats]), axis=0)
    vmax = np.max(np.stack([s[1] for s in stats]), axis=0)

    outs = []
    for a in arrs:
        xs = np.ascontiguousarray(np.transpose(a, (1, 2, 0)))
        c = xs.shape[-1]
        src = [0, 1, 2] if c >= 3 else (list(range(c)) + [0] * (3 - c))
        rgb = np.zeros((xs.shape[0], xs.shape[1], 3), dtype=np.float32)
        for i in range(3):
            ci = src[i]
            v = xs[..., ci]
            d = vmax[ci] - vmin[ci]
            rgb[..., i] = (v - vmin[ci]) / d if d > 1e-8 else 0.0
        outs.append(np.nan_to_num(rgb))
    return outs


def _corner(ax, text, color):
    ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, fontweight="bold", color="white",
            bbox=dict(facecolor=color, edgecolor="none", boxstyle="round,pad=0.25"))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/tcir_wpac_train.yaml")
    ap.add_argument("--ckpt", default=None,
                    help="指定权重；默认 outputs/<name>/checkpoints/best.ckpt")
    ap.add_argument("--set", action="append", default=[], help="覆盖配置 KEY=VAL")
    ap.add_argument("--show", type=int, default=12, help="对照缩略图的样本数")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-dir", default=None,
                    help="输出目录（默认 outputs/<output_dir>/<name>/）。"
                         "多种补全方法对比时用它分目录，避免互相覆盖。")
    args = ap.parse_args()

    os.chdir(ROOT)
    cfg = load_config(args.config, args.set)
    logger = get_logger("tc-corrraw")

    # 一致性守卫：对齐缺失修正 / 划分等键到训练时保存的 config_used.yaml
    cfg, _ = align_preprocessing(cfg, overrides=getattr(cfg, "overrides", []),
                                 logger=logger)
    d = cfg.to_dict()
    exp = d["experiment"]
    tcfg = d["data"]["tcir"]
    device = resolve_device(d["device"].get("mode", "auto"))

    datasets, _ = create_datasets(cfg)
    test_ds = datasets["test"]          # 修正版（interp）
    test_ds.return_corrected_flag = True
    n_test = len(test_ds)
    nw = exp.get("num_workers", 4)

    # ---- 同一批 test 样本的「原始 glitch」版本（missing_band=none，别的全部一致）----
    raw_ds = TCIRDataset(
        h5_path=test_ds.h5_path,
        indices=list(test_ds.indices),
        channels=test_ds.channels,
        target=test_ds.target,
        label_unit=test_ds.label_unit,
        nan_mode=test_ds.nan_mode,
        normalize=test_ds.normalize,
        norm_mean=test_ds.norm_mean,
        norm_std=test_ds.norm_std,
        resize=test_ds.resize,
        split="test",
        missing_band="none",
        frame_labels=list(test_ds.labels),
    )

    # ---- 模型 ----
    model = build_model(cfg).to(device)
    ckpt_path = args.ckpt or os.path.join(
        exp.get("output_dir", "outputs"), exp.get("name", "exp"),
        "checkpoints", "best.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到权重: {ckpt_path}")
    load_checkpoint(ckpt_path, model, map_location=device)
    model.eval()
    logger.info(f"权重: {ckpt_path}")

    # ---- 推理：corrected / raw 两次前向（顺序一致，均为 test_ds.indices 顺序）----
    def infer(ds):
        ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=nw)
        preds, trues, flags = [], [], []
        with torch.no_grad():
            for batch in ld:
                if len(batch) == 3:
                    x, y, f = batch
                    flags.append(np.asarray(f).ravel())
                else:
                    x, y = batch
                out = model(x.to(device, non_blocking=True))
                preds.append(out.cpu().numpy().ravel())
                trues.append(y.numpy().ravel())
        p = np.concatenate(preds).astype(np.float64)
        t = np.concatenate(trues).astype(np.float64)
        f = (np.concatenate(flags).astype(bool) if flags else None)
        return p, t, f

    yp_corr, yt, fl = infer(test_ds)
    yp_raw, yt2, _ = infer(raw_ds)
    assert np.allclose(yt, yt2), "两次取样的标签顺序不一致！"
    glitch = fl.astype(bool)
    n_gl = int(glitch.sum())
    logger.info(f"测试集 N={n_test}；其中做过修正(glitch) {n_gl} 张 / 干净 {n_test - n_gl} 张")

    # ---- 指标 ----
    def m(y_true, y_pred):
        r = regression_metrics(np.asarray(y_true, float), np.asarray(y_pred, float))
        return {"rmse": r["rmse"], "mae": r["mae"], "r2": r["r2"], "n": int(len(y_true))}

    sub = glitch
    res = {
        "n_test": n_test, "n_glitch": n_gl,
        "ckpt": os.path.abspath(ckpt_path),
        "model": exp.get("name", "exp"),
        # 记录本次「corrected」一列到底用的是哪种补全方法（cross-method 对比时必须留痕）
        "corrected_method": {
            "missing_band": tcfg.get("missing_band"),
            "missing_region_method": tcfg.get("missing_region_method"),
            "missing_region_feather": tcfg.get("missing_region_feather"),
            "missing_region_deg": tcfg.get("missing_region_deg"),
            "missing_region_iters": tcfg.get("missing_region_iters"),
        },
        "glitch_subset": {
            "raw": m(yt[sub], yp_raw[sub]),
            "corrected": m(yt[sub], yp_corr[sub]),
        },
        "all_test": {
            "raw": m(yt, yp_raw),
            "corrected": m(yt, yp_corr),
        },
    }
    res["all_test"]["delta_rmse_raw_minus_corr"] = round(
        res["all_test"]["raw"]["rmse"] - res["all_test"]["corrected"]["rmse"], 4)
    res["glitch_subset"]["delta_rmse_raw_minus_corr"] = round(
        res["glitch_subset"]["raw"]["rmse"] - res["glitch_subset"]["corrected"]["rmse"], 4)

    abs_raw = np.abs(yp_raw - yt)
    abs_corr = np.abs(yp_corr - yt)

    # ---- 元数据（storm ID）----
    csv_path = tcfg.get("csv_path")
    id_col = tcfg.get("storm_col", "ID")
    idx_col = tcfg.get("index_col", "matrix_index")
    real_idx = np.asarray(test_ds.indices, dtype=int)
    if csv_path and os.path.exists(csv_path):
        meta_df = pd.read_csv(csv_path).set_index(idx_col)
        sids = [str(meta_df.iloc[ri][id_col]) for ri in real_idx]
    else:
        sids = [f"r{ri}" for ri in real_idx]

    out_dir = args.out_dir or os.path.join(exp.get("output_dir", "outputs"),
                                           exp.get("name", "exp"))
    os.makedirs(out_dir, exist_ok=True)

    # ---- 逐样本 CSV（全部 glitch 样本）----
    gidx = np.where(glitch)[0]
    df = pd.DataFrame({
        "test_index": gidx,
        "matrix_index": real_idx[gidx],
        id_col: [sids[i] for i in gidx],
        "true_vmax": np.round(yt[gidx], 2),
        "pred_raw": np.round(yp_raw[gidx], 2),
        "pred_corrected": np.round(yp_corr[gidx], 2),
        "err_raw": np.round((yp_raw - yt)[gidx], 2),
        "err_corrected": np.round((yp_corr - yt)[gidx], 2),
        "abs_err_raw": np.round(abs_raw[gidx], 2),
        "abs_err_corrected": np.round(abs_corr[gidx], 2),
        "delta_pred_corr_minus_raw": np.round((yp_corr - yp_raw)[gidx], 2),
        "abs_err_improve": np.round((abs_raw - abs_corr)[gidx], 2),
    }).sort_values("abs_err_improve", ascending=False).reset_index(drop=True)
    csv_out = os.path.join(out_dir, "corrected_vs_raw.csv")
    df.to_csv(csv_out, index=False)

    json_out = os.path.join(out_dir, "corrected_vs_raw.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # ================= 图1：RAW vs CORRECTED 缩略图对照 =================
    norm_mean = tcfg.get("norm_mean")
    norm_std = tcfg.get("norm_std")
    # 取「修正前后预测差异最大」的前 N 个 glitch 样本（最能量化修复影响）
    delta_pred = np.abs(yp_corr - yp_raw)
    cand = np.where(glitch)[0]
    order = cand[np.argsort(-delta_pred[cand])][:min(args.show, n_gl)]

    n_top = len(order)
    per_row = 3 if n_top > 6 else (2 if n_top > 2 else 1)
    rows = int(np.ceil(n_top / per_row))
    fig = plt.figure(figsize=(per_row * 4.6, rows * 2.9))
    gs = fig.add_gridspec(rows, per_row * 2, wspace=0.06, hspace=0.32)
    for k, i in enumerate(order):
        r, c = divmod(k, per_row)
        xr, _ = raw_ds[int(i)]          # raw_ds 未开 flag -> 2 元组
        xc, _, _ = test_ds[int(i)]      # test_ds 开了 flag -> 3 元组
        raw_phys = denorm_for_display(xr, norm_mean, norm_std)
        cor_phys = denorm_for_display(xc, norm_mean, norm_std)
        im_raw, im_cor = to_composite_shared([raw_phys, cor_phys])
        axr = fig.add_subplot(gs[r, c * 2])
        axr.imshow(im_raw); axr.axis("off")
        _corner(axr, "RAW (glitch)", C_RAW)
        axr.set_title(f"#{k+1} {sids[i]}\nPred_raw {yp_raw[i]:.1f} / True {yt[i]:.1f}\n"
                      f"\u0394={yp_raw[i]-yt[i]:+.1f} kn", fontsize=8)
        axc = fig.add_subplot(gs[r, c * 2 + 1])
        axc.imshow(im_cor); axc.axis("off")
        _corner(axc, "CORRECTED", C_CORR)
        axc.set_title(f"Pred_corr {yp_corr[i]:.1f} / True {yt[i]:.1f}\n"
                      f"\u0394={yp_corr[i]-yt[i]:+.1f} kn\n"
                      f"\u0394pred={yp_corr[i]-yp_raw[i]:+.1f} kn", fontsize=8)
    g = res["glitch_subset"]
    fig.suptitle(
        f"Corrected vs Raw (glitch) — top-{n_top} by |\u0394pred|  |  "
        f"glitch subset (n={n_gl}) RMSE: raw={g['raw']['rmse']:.2f} -> "
        f"corrected={g['corrected']['rmse']:.2f} kn", fontsize=12)
    fig.savefig(os.path.join(out_dir, "corrected_vs_raw.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ================= 图2：指标对比 =================
    fig = plt.figure(figsize=(14, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.25, 1.15], wspace=0.32)

    # (a) RMSE 柱状：glitch 子集 raw vs corrected（+ 全体对照）
    ax0 = fig.add_subplot(gs[0, 0])
    labels = ["glitch\nraw", "glitch\ncorrected"]
    vals = [g["raw"]["rmse"], g["corrected"]["rmse"]]
    bars = ax0.bar(labels, vals, color=[C_RAW, C_CORR])
    for b, v in zip(bars, vals):
        ax0.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=11)
    ax0.set_ylabel("RMSE (kn)")
    ax0.set_ylim(0, max(vals) * 1.2)
    ax0.set_title(f"Glitch subset RMSE (n={n_gl})\n"
                  f"\u0394={g['raw']['rmse']-g['corrected']['rmse']:+.2f} kn", fontsize=11)

    # (b) pred_raw vs pred_corr 散点（y=x），按「是否改善」着色
    ax1 = fig.add_subplot(gs[0, 1])
    improved = abs_corr[glitch] < abs_raw[glitch]
    ax1.scatter(yp_raw[glitch][improved], yp_corr[glitch][improved], s=9, alpha=0.5,
                c=C_CORR, label=f"improved ({int(improved.sum())})")
    ax1.scatter(yp_raw[glitch][~improved], yp_corr[glitch][~improved], s=9, alpha=0.5,
                c=C_RAW, label=f"not improved ({int((~improved).sum())})")
    lim = [0.0, float(max(yp_raw[glitch].max(), yp_corr[glitch].max(),
                          yt[glitch].max())) + 5.0]
    ax1.plot(lim, lim, "k--", lw=1)
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.set_aspect("equal"); ax1.set_box_aspect(1)
    ax1.set_xlabel("Pred Vmax — RAW (kn)")
    ax1.set_ylabel("Pred Vmax — CORRECTED (kn)")
    ax1.set_title("Prediction: raw vs corrected", fontsize=11)
    ax1.legend(loc="upper left", fontsize=8)

    # (c) 指标表
    axt = fig.add_subplot(gs[0, 2]); axt.axis("off")
    tab = [
        ["Input", "N", "RMSE", "MAE", "R2"],
        ["glitch RAW", g["raw"]["n"], f"{g['raw']['rmse']:.3f}",
         f"{g['raw']['mae']:.3f}", f"{g['raw']['r2']:.3f}"],
        ["glitch CORR", g["corrected"]["n"], f"{g['corrected']['rmse']:.3f}",
         f"{g['corrected']['mae']:.3f}", f"{g['corrected']['r2']:.3f}"],
        ["all RAW", res["all_test"]["raw"]["n"], f"{res['all_test']['raw']['rmse']:.3f}",
         f"{res['all_test']['raw']['mae']:.3f}", f"{res['all_test']['raw']['r2']:.3f}"],
        ["all CORR", res["all_test"]["corrected"]["n"],
         f"{res['all_test']['corrected']['rmse']:.3f}",
         f"{res['all_test']['corrected']['mae']:.3f}",
         f"{res['all_test']['corrected']['r2']:.3f}"],
    ]
    t = axt.table(cellText=tab[1:], colLabels=tab[0], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.6)
    axt.set_title("Metrics: raw vs corrected", fontsize=11, pad=12)

    fig.suptitle("Local-inpainting repair: corrected vs raw (glitch) input",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, "corrected_vs_raw_metrics.png"), dpi=120)
    plt.close(fig)

    logger.info(f"对照缩略图: {os.path.join(out_dir, 'corrected_vs_raw.png')}")
    logger.info(f"指标对比图: {os.path.join(out_dir, 'corrected_vs_raw_metrics.png')}")
    logger.info(f"逐样本明细: {csv_out}")
    logger.info(f"汇总 JSON : {json_out}")
    logger.info(f"[glitch n={n_gl}] RAW RMSE={g['raw']['rmse']:.3f} MAE={g['raw']['mae']:.3f} "
                f"R2={g['raw']['r2']:.3f} -> CORRECTED RMSE={g['corrected']['rmse']:.3f} "
                f"MAE={g['corrected']['mae']:.3f} R2={g['corrected']['r2']:.3f} "
                f"(Δ RMSE={g['raw']['rmse']-g['corrected']['rmse']:+.3f} kn)")
    logger.info(f"[all  n={n_test}] RAW RMSE={res['all_test']['raw']['rmse']:.3f} -> "
                f"CORRECTED RMSE={res['all_test']['corrected']['rmse']:.3f} "
                f"(Δ={res['all_test']['raw']['rmse']-res['all_test']['corrected']['rmse']:+.3f} kn)")


if __name__ == "__main__":
    main()
