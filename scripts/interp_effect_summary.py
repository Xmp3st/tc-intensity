#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 improve_vs_position.csv 归纳「插值何时有效 / 模型何时失效」的量化结论。

输入：outputs/tcir_wpac/tcir_wpac_bb/improve_vs_position.csv（逐样本，含 raw/corrected 预测）
输出：out_dir/interp_effect_summary.{json,csv}（分强度箱、分位置组的偏置/误差/命中率）

用法：
  python scripts/interp_effect_summary.py
  python scripts/interp_effect_summary.py --method griddata --out-dir outputs/tcir_wpac/tcir_wpac_bb
"""
import argparse
import csv
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(ROOT, "outputs/tcir_wpac/tcir_wpac_bb/improve_vs_position.csv")

INT_BINS = [(0, 34, "TD  <34kn"),
            (34, 64, "TS  34-64"),
            (64, 96, "Cat1-2 64-96"),
            (96, 999, "Cat3+ >=96")]


def _rmse(a):
    a = np.asarray(a, dtype=np.float64)
    return float(np.sqrt(np.mean(a ** 2))) if a.size else float("nan")


def load(path, method):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if method and r.get("method") != method:
                continue
            rows.append(r)
    def col(k):
        return np.array([float(r[k]) for r in rows], dtype=np.float64)
    d = {k: col(k) for k in
         ["true_vmax", "pred_raw", "pred_corrected", "err_raw", "err_corrected",
          "abs_err_raw", "abs_err_corrected", "abs_err_improve",
          "central_frac", "c_min", "area_frac", "n_px"]}
    d["ID"] = [r["ID"] for r in rows]
    return rows, d


def group_stats(d, mask):
    """给一组样本算 raw/corrected 的偏置、绝对误差、命中率。"""
    if mask.sum() == 0:
        return None
    er, ec = d["err_raw"][mask], d["err_corrected"][mask]
    ar, ac = d["abs_err_raw"][mask], d["abs_err_corrected"][mask]
    imp = d["abs_err_improve"][mask]
    return {
        "n": int(mask.sum()),
        "true_mean": float(d["true_vmax"][mask].mean()),
        "bias_raw": float(er.mean()),          # 负=低估
        "bias_corrected": float(ec.mean()),
        "mae_raw": float(ar.mean()),
        "mae_corrected": float(ac.mean()),
        "rmse_raw": _rmse(er),
        "rmse_corrected": _rmse(ec),
        "improve_mean": float(imp.mean()),
        "improve_median": float(np.median(imp)),
        "hit_rate": float((imp > 0).mean()),    # 修复后 |误差| 变小的比例
        "hurt20_rate": float((imp < -20).mean()),  # 被明显推坏(>20kn)的比例
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--method", default="griddata")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs/tcir_wpac/tcir_wpac_bb"))
    ap.add_argument("--central-r", type=float, default=24.0, help="中心圆盘半径(px)")
    args = ap.parse_args()

    rows, d = load(args.csv, args.method)
    tv = d["true_vmax"]
    out = {"method": args.method, "n": len(rows)}

    # ---- 1. 按强度分箱 ----
    bins = {}
    for lo, hi, name in INT_BINS:
        m = (tv >= lo) & (tv < hi)
        bins[name] = group_stats(d, m)
    out["by_intensity"] = bins

    # ---- 2. 按缺测位置分组（中心圆盘 r=24 内是否有缺测）----
    is_central = d["central_frac"] > 0
    out["by_position"] = {
        "central": group_stats(d, is_central),
        "edge": group_stats(d, ~is_central),
    }

    # ---- 3. 强度 × 位置 交互 ----
    inter = {}
    for lo, hi, name in INT_BINS:
        m = (tv >= lo) & (tv < hi)
        inter[name] = {
            "central": group_stats(d, m & is_central),
            "edge": group_stats(d, m & ~is_central),
        }
    out["intensity_x_position"] = inter

    # ---- 4. 尾部集中度（|err_raw| 最大 10%）----
    order = np.argsort(-d["abs_err_raw"])
    k = max(1, int(round(0.10 * len(order))))
    tail = np.zeros(len(order), dtype=bool)
    tail[order[:k]] = True
    rest = ~tail
    ss_raw = float(np.sum(d["err_raw"] ** 2))
    drop_tail = float(np.sum(d["err_raw"][tail] ** 2) - np.sum(d["err_corrected"][tail] ** 2))
    drop_rest = float(np.sum(d["err_raw"][rest] ** 2) - np.sum(d["err_corrected"][rest] ** 2))
    out["tail"] = {
        "k_worst": int(k),
        "tail_rmse_raw": _rmse(d["err_raw"][tail]),
        "tail_rmse_corrected": _rmse(d["err_corrected"][tail]),
        "mse_drop_tail_share": drop_tail / (drop_tail + drop_rest) if (drop_tail + drop_rest) else float("nan"),
        "overall_hit_rate": float((d["abs_err_improve"] > 0).mean()),
        "overall_improve_mean": float(d["abs_err_improve"].mean()),
        "overall_improve_median": float(np.median(d["abs_err_improve"])),
    }

    # ---- 5. 模型整体偏置（原始输入）----
    out["model_bias_raw"] = {
        "overall_bias": float(d["err_raw"].mean()),
        "overall_mae": float(d["abs_err_raw"].mean()),
        "corr_bias_vs_true": float(np.corrcoef(tv, d["err_raw"])[0, 1]),
        "n_underestimate": int((d["err_raw"] < 0).sum()),
        "frac_underestimate": float((d["err_raw"] < 0).mean()),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    jp = os.path.join(args.out_dir, "interp_effect_summary.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 扁平 CSV，便于贴表
    cp = os.path.join(args.out_dir, "interp_effect_summary.csv")
    with open(cp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope", "group", "n", "true_mean", "bias_raw", "bias_corrected",
                    "mae_raw", "mae_corrected", "rmse_raw", "rmse_corrected",
                    "improve_mean", "improve_median", "hit_rate", "hurt20_rate"])
        for name, s in bins.items():
            if s:
                w.writerow(["intensity", name] + [s["n"], s["true_mean"], s["bias_raw"],
                           s["bias_corrected"], s["mae_raw"], s["mae_corrected"],
                           s["rmse_raw"], s["rmse_corrected"], s["improve_mean"],
                           s["improve_median"], s["hit_rate"], s["hurt20_rate"]])
        for g in ["central", "edge"]:
            s = out["by_position"][g]
            if s:
                w.writerow(["position", g] + [s["n"], s["true_mean"], s["bias_raw"],
                           s["bias_corrected"], s["mae_raw"], s["mae_corrected"],
                           s["rmse_raw"], s["rmse_corrected"], s["improve_mean"],
                           s["improve_median"], s["hit_rate"], s["hurt20_rate"]])

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("\n[save]", jp)
    print("[save]", cp)


if __name__ == "__main__":
    main()
