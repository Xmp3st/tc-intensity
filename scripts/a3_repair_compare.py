#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 a3 臂（仅干净图训练）的模型下，对比测试集上 griddata vs U-Net 修复的效果。

背景：四个「插值类」补全方法都证明「补得更准 ≠ 下游更好」，而新训的 U-Net 修复网络
      在**像素级**明显更强（缺测区 RMSE −20%）。本脚本回答：把修复用在**下游强度识别**
      上，a3 这个「只见过干净图」的模型，用哪种修复（或不修复）时强度回归更准？

- 模型：outputs/tcir_wpac_ir1/ir1_year_a3_clean/checkpoints/best.ckpt（IR1 单通道 vgg16）
- 测试集：by_year 未见年份 2014-2016（全集 4107；其中 IR1 损坏 519）
- 三个输入变体（其余完全一致）：
    raw      : 原始输入，不做任何修复        —— a3 的原始口径（对照基线）
    griddata : 项目默认空间插值修复
    unet     : U-Net 学习式修复（logs/inpaint_unet best.ckpt）
- 输出：全集 / 干净 / 损坏 三组 RMSE/MAE/R²/偏置 + 逐样本配对（含 U-Net vs griddata 胜率、t 检验）

用法：
  python scripts/a3_repair_compare.py
  python scripts/a3_repair_compare.py --ckpt-arm outputs/tcir_wpac_ir1/ir1_year_a3_clean/checkpoints/best.ckpt
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CFG = "configs/tcir_wpac_ir1_year.yaml"
NAME = "ir1_year_a3_clean"
ARMS = {
    "a1": "ir1_year_a1_raw",       # 无修正
    "a2": "ir1_year_a2_interp",    # 带修正(griddata) 训练
    "a3": "ir1_year_a3_clean",     # 仅干净图训练
    "a4": "ir1_year_a4_strip",     # 无修正+条状增广
}
OUT_DIR = "outputs/tcir_wpac_ir1"
UNET_CKPT = "outputs/tcir_inpaint_unet/checkpoints/best.ckpt"
NORM_MEAN, NORM_STD = 260.864, 32.849
SENTINEL_NORM = (0.0 - NORM_MEAN) / NORM_STD          # ≈ -7.941，与 U-Net 训练一致


def metrics(y, p):
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    e = p - y
    ss_res = float(np.sum(e ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(np.abs(e))),
        "bias": float(np.mean(e)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def paired_t(a, b):
    """配对 t 检验（b-a 的均值是否显著非 0）。返回 (均值差, t, p, n)。"""
    from scipy import stats
    d = np.asarray(b, np.float64) - np.asarray(a, np.float64)
    if len(d) < 2:
        return float("nan"), float("nan"), float("nan"), len(d)
    t, p = stats.ttest_rel(np.asarray(b), np.asarray(a))
    return float(d.mean()), float(t), float(p), len(d)


def resolve_h5(cfg):
    d = cfg.to_dict()["data"]; t = d.get("tcir", {})
    h5 = t.get("h5_path", "data/wpac_96.h5")
    root = d.get("root_dir", "")
    if root and not os.path.isabs(h5):
        cand = os.path.join(root, h5)
        if os.path.exists(cand):
            h5 = cand
    if not os.path.isabs(h5):
        h5 = os.path.join(ROOT, h5)
    return h5


_ID_CACHE = {}


def ids_for(idx):
    if not _ID_CACHE:
        import pandas as pd
        df = pd.read_csv(os.path.join(ROOT, "data/wpac_compact.csv"))
        for k, v in zip(df["matrix_index"].to_numpy().astype(int), df["ID"].to_numpy()):
            _ID_CACHE[int(k)] = str(v)
    return [_ID_CACHE.get(int(i), "?") for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="a3", choices=list(ARMS.keys()),
                    help="用哪个臂的模型做对比（默认 a3 仅干净图训练）")
    ap.add_argument("--ckpt-arm", default=None, help="回归模型权重（默认 <arm> 的 best.ckpt）")
    ap.add_argument("--unet-ckpt", default=UNET_CKPT, help="第 1 个 U-Net 修复权重（变体 unet）")
    ap.add_argument("--unet-ckpt2", default=None,
                    help="可选：第 2 个 U-Net 修复权重（变体 unet2，如同一模型的另一种训练分布）")
    ap.add_argument("--tag", default="", help="输出目录后缀（避免覆盖既有结果）")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    name = ARMS[args.arm]

    os.chdir(ROOT)
    import h5py
    import torch
    from tcintens.utils.config import load_config
    from tcintens.utils.repro import align_preprocessing
    from tcintens.data.dataset import create_datasets
    from tcintens.models.tc_model import build_model
    from tcintens.models.unet import UNet
    from tcintens.engine.common import load_checkpoint
    from tcintens.data.band_health import detect_missing_region, inpaint_regions

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(CFG, [f"experiment.name={name}"])
    cfg, _ = align_preprocessing(cfg, overrides=getattr(cfg, "overrides", []), logger=None)
    cfg.set("experiment.output_dir", args.out)
    t = cfg.to_dict()["data"]["tcir"]

    datasets, meta = create_datasets(cfg)
    test_ds = datasets["test"]
    idx = np.asarray(test_ds.indices, dtype=int)
    y = np.asarray(test_ds.labels, dtype=float)
    N = len(idx)
    print(f"[data] 测试集 N={N}（by_year 2014-2016）", flush=True)

    # ---- 读 raw IR1（紧凑 h5 第 0 通道）----
    h5p = resolve_h5(cfg)
    raw = np.empty((N, 96, 96), np.float32)
    order = np.argsort(idx, kind="stable")
    sidx = idx[order]
    with h5py.File(h5p, "r") as hf:
        mat = hf["matrix"]
        for s in range(0, N, 512):
            e = min(s + 512, N)
            blk = np.asarray(mat[sidx[s:e]])[..., 0].astype(np.float32)  # (b,96,96)
            raw[order[s:e]] = blk
    raw = np.nan_to_num(raw, nan=0.0)

    # ---- 缺测检测（与训练同口径）----
    sentinel = t.get("missing_region_sentinel", 0.0)
    low_thr = t.get("missing_region_low_thr", [150.0])
    min_blob = t.get("missing_region_min_blob", 20)
    dilate = t.get("missing_region_dilate", 1)
    masks = np.zeros((N, 96, 96), bool)
    for i in range(N):
        m = detect_missing_region(raw[i][None], sentinel=sentinel, low_thr=low_thr,
                                  min_blob=min_blob, dilate=dilate)
        masks[i] = m[0]
    dam = masks.any(axis=(1, 2))
    area = masks.mean(axis=(1, 2))
    print(f"[mask] 损坏帧 {int(dam.sum())} / 干净 {int((~dam).sum())}"
          f"（损坏帧缺测面积 中位 {np.median(area[dam])*100:.2f}%）", flush=True)

    # ---- 模型 ----
    ck_arm = args.ckpt_arm or os.path.join(args.out, name, "checkpoints", "best.ckpt")
    model = build_model(cfg).to(device)
    load_checkpoint(ck_arm, model, map_location=device)
    model.eval()
    print(f"[model] 回归模型 {ck_arm}", flush=True)

    unet = UNet(in_channels=2, out_channels=1, base=48).to(device)
    ck = torch.load(args.unet_ckpt, map_location=device)
    unet.load_state_dict(ck["model"])
    unet.eval()
    print(f"[model] U-Net 修复 {args.unet_ckpt}（best_roi={ck.get('best_roi')}）", flush=True)

    unet2 = None
    if args.unet_ckpt2:
        unet2 = UNet(in_channels=2, out_channels=1, base=48).to(device)
        ck2 = torch.load(args.unet_ckpt2, map_location=device)
        unet2.load_state_dict(ck2["model"])
        unet2.eval()
        print(f"[model] U-Net#2 修复 {args.unet_ckpt2}（best_roi={ck2.get('best_roi')}）",
              flush=True)

    def infer_reg(xb):          # xb:(B,1,96,96) float32 -> (B,)
        outs = []
        with torch.no_grad():
            for s in range(0, len(xb), 64):
                xt = torch.from_numpy(xb[s:s + 64]).to(device)
                outs.append(model(xt).float().cpu().numpy().ravel())
        return np.concatenate(outs)

    def infer_unet(dn, m, net):
        # dn:(B,96,96) 归一化, m:(B,96,96) bool -> (B,96,96) 修复(归一化)
        x = np.stack([dn, m.astype(np.float32)], axis=1).astype(np.float32)
        outs = []
        with torch.no_grad():
            for s in range(0, len(x), 64):
                xt = torch.from_numpy(x[s:s + 64]).to(device)
                outs.append(net(xt).cpu().numpy()[:, 0])
        return np.concatenate(outs, 0)

    # ---- 变体推理（分批，避免一次性大数组）----
    keys = ["raw", "griddata", "unet"]
    if unet2 is not None:
        keys.append("unet2")
    pred = {k: np.empty(N, np.float64) for k in keys}
    nets = {"unet": unet, "unet2": unet2} if unet2 is not None else {"unet": unet}
    for s in range(0, N, 256):
        e = min(s + 256, N)
        r = raw[s:e]
        m = masks[s:e]
        dn = (r - NORM_MEAN) / NORM_STD

        pred["raw"][s:e] = infer_reg(dn[:, None].astype(np.float32))

        # griddata：在原始亮温域修复，再归一化
        gd = np.empty_like(dn)
        for j in range(e - s):
            if m[j].any():
                f = inpaint_regions(r[j][None].astype(np.float64), m[j][None],
                                    method="griddata", fill_values=[NORM_MEAN],
                                    max_frac=1.0)
                gd[j] = (f[0] - NORM_MEAN) / NORM_STD
            else:
                gd[j] = dn[j]
        pred["griddata"][s:e] = infer_reg(gd[:, None].astype(np.float32))

        # U-Net 系列：仅对**损坏帧**修复（干净帧原样通过），与真实部署一致；
        # 损坏像素置哨兵（与 U-Net 训练一致）。
        rows_dmg = np.where(m.reshape(e - s, -1).any(axis=1))[0]
        for k in keys:
            if k not in nets:
                continue
            un_full = dn.copy()
            if len(rows_dmg):
                dm_dn = dn[rows_dmg].copy()
                dm_dn[m[rows_dmg]] = SENTINEL_NORM
                un_full[rows_dmg] = infer_unet(dm_dn, m[rows_dmg], nets[k])
            pred[k][s:e] = infer_reg(un_full[:, None].astype(np.float32))
        print(f"  ... {e}/{N}", flush=True)

    # ---- 分组指标 ----
    groups = {"overall": np.ones(N, bool), "clean": ~dam, "damaged": dam}
    res = {}
    for k in keys:
        res[k] = {g: metrics(y[sel], pred[k][sel]) for g, sel in groups.items()}
    res["n_clean"] = int((~dam).sum())
    res["n_damaged"] = int(dam.sum())

    def _line(k):
        o, c, d = res[k]["overall"], res[k]["clean"], res[k]["damaged"]
        return (f"{k:9s} | 全集 RMSE {o['rmse']:.3f} MAE {o['mae']:.3f} R2 {o['r2']:.4f} "
                f"| 干净 {c['rmse']:.3f} (n={c['n']}) "
                f"| 损坏 {d['rmse']:.3f} MAE {d['mae']:.3f} bias {d['bias']:+.3f} (n={d['n']})")

    print(f"\n===== {args.arm} 模型下 各修复对比 =====", flush=True)
    for k in keys:
        print(_line(k), flush=True)

    # ---- 损坏帧配对分析（各修复变体 vs griddata / raw）----
    e_abs = {k: np.abs(pred[k] - y) for k in keys}
    rmse = {k: float(np.sqrt(np.mean((pred[k][dam] - y[dam]) ** 2))) for k in keys}
    dm = dam
    pair = {}
    if dm.sum() > 1:
        pair["gd_vs_raw_rmse"] = rmse["griddata"] - rmse["raw"]
        for k in keys:
            if k in ("raw", "griddata"):
                continue
            pair[f"{k}_vs_raw_rmse"] = rmse[k] - rmse["raw"]
            d_mean, t, p, n = paired_t(e_abs["griddata"][dm], e_abs[k][dm])
            pair[f"{k}_minus_gd_absmean"] = d_mean
            pair[f"{k}_vs_gd_abs_t"] = t
            pair[f"{k}_vs_gd_abs_p"] = p
            pair[f"{k}_win_rate_vs_gd"] = float(np.mean(e_abs[k][dm] < e_abs["griddata"][dm]))
            print(f"\n[损坏帧 n={int(dm.sum())}] {k} vs griddata（|误差|）："
                  f"均值差 {d_mean:+.4f} kn，t={t:.2f}，p={p:.2e}，"
                  f"{k} 胜率 {pair[f'{k}_win_rate_vs_gd']*100:.1f}%", flush=True)
        if "unet2" in keys:
            d_mean, t, p, n = paired_t(e_abs["unet"][dm], e_abs["unet2"][dm])
            pair["unet2_minus_unet_absmean"] = d_mean
            pair["unet2_vs_unet_abs_t"] = t
            pair["unet2_vs_unet_abs_p"] = p
            pair["unet2_win_rate_vs_unet"] = float(np.mean(e_abs["unet2"][dm] < e_abs["unet"][dm]))
            print(f"\n[损坏帧 n={int(dm.sum())}] unet2 vs unet（|误差|）："
                  f"均值差 {d_mean:+.4f} kn，t={t:.2f}，p={p:.2e}，"
                  f"unet2 胜率 {pair['unet2_win_rate_vs_unet']*100:.1f}%", flush=True)

    out_dir = os.path.join(args.out, f"{args.arm}_repair_compare" + (f"_{args.tag}" if args.tag else ""))
    os.makedirs(out_dir, exist_ok=True)
    # 逐样本
    ids = ids_for(idx)
    with open(os.path.join(out_dir, "per_sample.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["matrix_index", "ID", "true_vmax", "ir1_damaged", "area_frac"]
                   + [f"pred_{k}" for k in keys] + [f"err_{k}" for k in keys])
        for j in range(N):
            w.writerow([int(idx[j]), ids[j], float(y[j]), int(dam[j]), float(area[j])]
                       + [float(pred[k][j]) for k in keys]
                       + [float(pred[k][j] - y[j]) for k in keys])
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"arm": args.arm, "ckpt_arm": ck_arm, "unet_ckpt": args.unet_ckpt,
                   "unet_ckpt2": args.unet_ckpt2, "tag": args.tag,
                   "n_total": N, "n_clean": res["n_clean"], "n_damaged": res["n_damaged"],
                   "metrics": {k: res[k] for k in keys}, "paired_damaged": pair},
                  f, ensure_ascii=False, indent=2)

    _plot(res, keys, y, pred, dam, os.path.join(out_dir, "compare.png"), args.arm)
    print(f"\n[save] {out_dir}/summary.json / per_sample.csv / compare.png", flush=True)


def _plot(res, keys, y, pred, dam, path, arm="a3"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tcintens.utils.plotting import use_cjk_font
    use_cjk_font()

    label = {"raw": "raw(不修复)", "griddata": "griddata 插值",
             "unet": "U-Net 修复", "unet2": "U-Net#2 修复"}
    colors = {"raw": "#888780", "griddata": "#378ADD",
              "unet": "#1D9E75", "unet2": "#E24B4A"}
    groups = [("overall", "全集"), ("clean", "干净帧"), ("damaged", "损坏帧")]
    x = np.arange(len(groups)); width = 0.8 / len(keys)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for i, k in enumerate(keys):
        vals = [res[k][g]["rmse"] for g, _ in groups]
        pos = x + i * width - 0.4 + width / 2
        b = ax.bar(pos, vals, width * 0.9, label=label.get(k, k), color=colors.get(k, "#999"))
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.12, f"{v:.2f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([g[1] for g in groups])
    ax.set_ylabel("RMSE (kn)")
    ax.set_title(f"{arm} 模型 · 各修复的测试集 RMSE")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    for k in keys:
        if k == "raw":
            continue
        ax.scatter(y[dam], np.abs(pred[k][dam] - y[dam]), s=14, alpha=0.55,
                   c=colors.get(k, "#999"), label=f"{label.get(k, k)} |误差|")
    lim = [y[dam].min() - 5, y[dam].max() + 5]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.6)
    ax.set_xlabel("True Vmax (kn)"); ax.set_ylabel("|Error| (kn)")
    ax.set_title(f"损坏帧逐样本 |误差|（n={int(dam.sum())}）")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
