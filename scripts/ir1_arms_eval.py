#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IR1 · 按年份 8:2 · 四臂对照 —— 逐样本推理 + 分组指标 + 汇总图表。

对每个臂：
  1) 用训练时保存的 config_used.yaml 对齐预处理，重建**同一个**测试集；
  2) 加载 best.ckpt 推理，逐样本落盘 preds.csv；
  3) 统计「全体 / IR1 干净 / IR1 损坏」三档 RMSE、MAE、R²、偏置。
最后汇总成表 + 柱状图 + JSON。

用法：
  python scripts/ir1_arms_eval.py
  python scripts/ir1_arms_eval.py --only a1,a2
"""
import argparse
import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "configs/tcir_wpac_ir1_year.yaml")
OUT_DIR = os.path.join(ROOT, "outputs/tcir_wpac_ir1")

ARMS = [
    ("a1", "ir1_year_a1_raw",    "无修正"),
    ("a2", "ir1_year_a2_interp", "带修正(griddata)"),
    ("a3", "ir1_year_a3_clean",  "仅干净图训练"),
    ("a4", "ir1_year_a4_strip",  "无修正+条状增广"),
    ("a5", "ir1_year_a5_unet",   "U-Net 修复输入"),
]


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


def resolve_h5(cfg):
    d = cfg.to_dict()["data"]; t = d.get("tcir", {})
    h5 = t.get("h5_path", "data/TCIR.h5")
    root = d.get("root_dir", "")
    if root and not os.path.isabs(h5):
        cand = os.path.join(root, h5)
        if os.path.exists(cand):
            h5 = cand
    if not os.path.isabs(h5):
        h5 = os.path.join(ROOT, h5)
    return h5


def ir1_damage_flags(cfg, test_indices):
    from tcintens.data.band_health import scan_damage
    from tcintens.data.tcir import CHANNELS, _damage_low_thr_list
    t = cfg.to_dict()["data"]["tcir"]
    channels = t.get("channels", ["IR1", "WV", "PMW"])
    ch_idx = [CHANNELS.index(c) for c in channels]
    nd, nw, imf, mxf = scan_damage(
        resolve_h5(cfg), test_indices, ch_idx,
        sentinel=t.get("missing_region_sentinel", 0.0),
        low_thr_list=_damage_low_thr_list(channels, t),
        min_blob=t.get("missing_region_min_blob", 20),
        dilate=t.get("missing_region_dilate", 1),
        missing_detect=t.get("missing_detect", "constant"))
    return (nd > 0), imf


def run_arm(tag, name, label, device):
    import torch
    from torch.utils.data import DataLoader
    from tcintens.utils.config import load_config
    from tcintens.utils.repro import align_preprocessing
    from tcintens.data.dataset import create_datasets
    from tcintens.models.tc_model import build_model
    from tcintens.engine.common import load_checkpoint

    adir = os.path.join(OUT_DIR, name)
    ckpt = os.path.join(adir, "checkpoints", "best.ckpt")
    if not os.path.exists(ckpt):
        print(f"[skip] {tag}: 无 best.ckpt")
        return None

    cfg = load_config(CFG, [f"experiment.name={name}"])
    cfg, _ = align_preprocessing(cfg, overrides=getattr(cfg, "overrides", []), logger=None)
    cfg.set("experiment.output_dir", "outputs/tcir_wpac_ir1")

    datasets, meta = create_datasets(cfg)
    test_ds = datasets["test"]
    idx = np.asarray(test_ds.indices, dtype=int)
    bs = cfg.to_dict()["train"].get("batch_size", 32)
    loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4)

    model = build_model(cfg).to(device)
    load_checkpoint(ckpt, model, map_location=device)
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            yp.append(model(x).float().cpu().numpy())
            yt.append(np.asarray(y))
    yt = np.concatenate(yt); yp = np.concatenate(yp)

    # IR1 损坏标记（与输入是否已修正无关，标记的是「原始是否缺测」）
    dam, imf = ir1_damage_flags(cfg, idx.tolist())

    ids = ids_for(idx)
    with open(os.path.join(adir, "preds.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["matrix_index", "ID", "true_vmax", "pred", "err", "ir1_damaged", "ir1_area_frac"])
        for j in range(len(idx)):
            w.writerow([int(idx[j]), ids[j], float(yt[j]), float(yp[j]),
                        float(yp[j] - yt[j]), int(dam[j]), float(imf[j])])

    res = {"arm": tag, "name": name, "label": label,
           "overall": metrics(yt, yp),
           "clean": metrics(yt[~dam], yp[~dam]),
           "damaged": metrics(yt[dam], yp[dam]),
           "n_clean": int((~dam).sum()), "n_damaged": int(dam.sum())}
    print(f"[{tag}] {label}: 全集 RMSE={res['overall']['rmse']:.3f} "
          f"| 干净 {res['clean']['rmse']:.3f} (n={res['n_clean']}) "
          f"| 损坏 {res['damaged']['rmse']:.3f} (n={res['n_damaged']})", flush=True)
    return res


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
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    want = None if not args.only else set(x.strip() for x in args.only.split(","))

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for tag, name, label in ARMS:
        if want is not None and tag not in want:
            continue
        r = run_arm(tag, name, label, device)
        if r:
            results.append(r)

    jp = os.path.join(OUT_DIR, "arms_grouped.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 扁平 CSV
    cp = os.path.join(OUT_DIR, "arms_grouped.csv")
    with open(cp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "label", "group", "n", "rmse", "mae", "bias", "r2"])
        for r in results:
            for g in ("overall", "clean", "damaged"):
                m = r[g]
                w.writerow([r["arm"], r["label"], g, m["n"], round(m["rmse"], 4),
                            round(m["mae"], 4), round(m["bias"], 4), round(m["r2"], 4)])

    make_plot(results, os.path.join(OUT_DIR, "arms_compare.png"))
    print("[save]", jp)
    print("[save]", cp)


def make_plot(results, out_png):
    if not results:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tcintens.utils.plotting import use_cjk_font
    use_cjk_font()

    groups = [("overall", "全集"), ("clean", "IR1 干净帧"), ("damaged", "IR1 损坏帧")]
    labels = [f"{r['arm']} {r['label']}" for r in results]
    x = np.arange(len(groups))
    width = 0.8 / max(len(results), 1)
    colors = ["#888780", "#1D9E75", "#378ADD", "#D85A30", "#7B5EA7"]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for i, r in enumerate(results):
        vals = [r[g]["rmse"] for g, _ in groups]
        pos = x + i * width - 0.4 + width / 2
        b = ax.bar(pos, vals, width * 0.92, label=labels[i],
                   color=colors[i % len(colors)])
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.15,
                    f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([g[1] for g in groups])
    ax.set_ylabel("RMSE (kn)")
    ax.set_title("IR1 单通道 · 按台风年份 8:2（测试 2014-2016，全集 4107 帧）· 五臂对照（含 U-Net 修复输入）")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print("[save]", out_png)


if __name__ == "__main__":
    main()
