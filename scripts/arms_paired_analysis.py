#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""五臂对照的「逐样本配对分析」：修复方式（raw / griddata / U-Net）谁在下游更强。

读各臂 ir1_arms_eval.py 落盘的 preds.csv，按 matrix_index 对齐，做：
  * 全集 / 干净帧 / 损坏帧 的 RMSE/MAE/bias 汇总；
  * **损坏帧**上的配对 |误差| 检验：a5(U-Net) vs a2(griddata)、a5 vs a1(raw)、a2 vs a1；
  * 按缺测面积分档的 RMSE（看 U-Net 优势是否集中在大块损坏）。

用法：
  python scripts/arms_paired_analysis.py
  python scripts/arms_paired_analysis.py --arms a1,a2,a5
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs/tcir_wpac_ir1")
ARMS = {
    "a1": "ir1_year_a1_raw",
    "a2": "ir1_year_a2_interp",
    "a3": "ir1_year_a3_clean",
    "a4": "ir1_year_a4_strip",
    "a5": "ir1_year_a5_unet",
}
LABEL = {"a1": "raw(无修正)", "a2": "griddata", "a3": "仅干净图训练",
         "a4": "条状增广", "a5": "U-Net 修复"}


def rmse(e):
    return float(np.sqrt(np.mean(np.asarray(e, np.float64) ** 2)))


def paired_t(a, b):
    """配对 t 检验（b-a），返回 (均值差, t, p, n)。"""
    from scipy import stats
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if len(a) < 2:
        return float("nan"), float("nan"), float("nan"), len(a)
    t, p = stats.ttest_rel(b, a)
    return float((b - a).mean()), float(t), float(p), len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="a1,a2,a5")
    args = ap.parse_args()
    want = [x.strip() for x in args.arms.split(",") if x.strip()]

    dfs = {}
    for tag in want:
        p = os.path.join(OUT_DIR, ARMS[tag], "preds.csv")
        if not os.path.exists(p):
            print(f"[skip] {tag}: 缺 {p}")
            continue
        dfs[tag] = pd.read_csv(p).set_index("matrix_index")
    use = [t for t in want if t in dfs]
    if not use:
        print("无可分析臂"); return

    # 对齐公共样本
    common = None
    for t in use:
        s = set(dfs[t].index)
        common = s if common is None else (common & s)
    common = sorted(common)
    for t in use:
        dfs[t] = dfs[t].loc[common]

    ref = dfs[use[0]]
    y = ref["true_vmax"].to_numpy()
    dam = ref["ir1_damaged"].to_numpy().astype(bool)
    area = ref["ir1_area_frac"].to_numpy()

    print(f"公共样本 N={len(common)}  损坏帧 n={int(dam.sum())}  干净帧 n={int((~dam).sum())}\n")

    # ---- 分组汇总 ----
    print("=" * 88)
    print(f"{'臂':<16}{'全集RMSE':>10}{'干净RMSE':>11}{'损坏RMSE':>11}{'损坏bias':>11}{'损坏MAE':>10}")
    print("-" * 88)
    for t in use:
        e = dfs[t]["err"].to_numpy()
        print(f"{t + ' ' + LABEL.get(t, ''):<16}"
              f"{rmse(e):>10.3f}{rmse(e[~dam]):>11.3f}{rmse(e[dam]):>11.3f}"
              f"{e[dam].mean():>11.3f}{np.abs(e[dam]).mean():>10.3f}")
    print("=" * 88)

    # ---- 损坏帧配对检验 ----
    out = {"n_total": len(common), "n_damaged": int(dam.sum()), "pairs": {}}
    if dam.sum() > 1:
        print("\n--- 损坏帧配对 |误差| 检验（b 相对 a 更优则均差为负）---")
        pairs = [("a2", "a5"), ("a1", "a5"), ("a1", "a2")]
        for a, b in pairs:
            if a not in dfs or b not in dfs:
                continue
            ea = np.abs(dfs[a]["err"].to_numpy()[dam]); eb = np.abs(dfs[b]["err"].to_numpy()[dam])
            d, t, p, n = paired_t(ea, eb)
            win = float(np.mean(eb < ea))
            print(f"  {b}({LABEL.get(b)}) vs {a}({LABEL.get(a)}): "
                  f"|误差|均值差 {d:+.4f} kn, t={t:.2f}, p={p:.2e}, {b} 胜率 {win*100:.1f}%")
            out["pairs"][f"{b}_vs_{a}"] = {"d_absmean": d, "t": t, "p": p, "win_rate": win, "n": n}
        # RMSE 差
        for a, b in pairs:
            if a in dfs and b in dfs:
                ra = rmse(dfs[a]["err"].to_numpy()[dam]); rb = rmse(dfs[b]["err"].to_numpy()[dam])
                out["pairs"][f"{b}_vs_{a}"]["rmse_diff"] = rb - ra

        # ---- 按缺测面积分档 ----
        print("\n--- 损坏帧按缺测面积分档 RMSE ---")
        bins = [0, 0.01, 0.02, 0.04, 0.08, 1.0]
        labs = ["<1%", "1-2%", "2-4%", "4-8%", ">8%"]
        ab = pd.Series(pd.cut(area, bins=bins, labels=labs))
        hdr = "".join(f"{t:>12}" for t in use)
        print(f"{'面积档':<8}{'n':>5}{hdr}")
        area_tab = {}
        for g in labs:
            sel = dam & (ab == g).to_numpy()
            if sel.sum() == 0:
                continue
            row = {t: rmse(dfs[t]["err"].to_numpy()[sel]) for t in use}
            area_tab[g] = {"n": int(sel.sum()), **row}
            print(f"{g:<8}{int(sel.sum()):>5}" + "".join(f"{row[t]:>12.3f}" for t in use))
        out["area_bins"] = area_tab

    jp = os.path.join(OUT_DIR, "arms_paired_analysis.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[save] {jp}")


if __name__ == "__main__":
    main()
