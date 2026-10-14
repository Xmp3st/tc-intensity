# -*- coding: utf-8 -*-
"""扫描 WPAC 全量帧，挑出「完全无损坏」的子集，落盘成独立 CSV。

两个口径：
  project : n_damaged == 0，用项目默认缺测检测参数
            （sentinel=0, low_thr=[150,150,-999], min_blob=20, dilate=1）
            即：IR1/WV 无任何 >=20px 的缺测连通块，且三通道均无整通道缺失。
  strict  : IR1/WV 逐像素完全无缺测（min_blob=1, dilate=0）+ 无整通道缺失。
            真正的「零缺测」，连 1 个 0 像素都没有。

输出：
  data/wpac_clean.csv          （project 口径；列与 wpac_compact.csv 一致）
  data/wpac_clean_strict.csv   （strict 口径）
  logs/clean_subset_stats.json （统计明细）
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tcintens.data.band_health import frame_damage_stats, scan_damage  # noqa: E402
from tcintens.data.tcir import CHANNELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/wpac_96.h5")
    ap.add_argument("--csv", default="data/wpac_compact.csv")
    ap.add_argument("--channels", default="IR1,WV,PMW")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    channels = args.channels.split(",")
    ch_idx = [CHANNELS.index(c) for c in channels]
    low_thr = [150.0 if c in ("IR1", "WV") else -999.0 for c in channels]

    df = pd.read_csv(args.csv)
    idx = df["matrix_index"].to_numpy().astype(int)
    print(f"[clean] 全量 {len(idx)} 帧，channels={channels}, ch_idx={ch_idx}", flush=True)

    # ---- project 口径（默认参数）----
    nd, nw, imgf, maxf = scan_damage(
        args.h5, idx, ch_idx, sentinel=0.0, low_thr_list=low_thr,
        min_blob=20, dilate=1, missing_detect="constant", batch=args.batch)
    print(f"[clean] project 口径扫描完成：clean={int((nd == 0).sum())}", flush=True)

    # ---- strict 口径（逐像素）----
    nd_s, nw_s, imgf_s, maxf_s = scan_damage(
        args.h5, idx, ch_idx, sentinel=0.0, low_thr_list=low_thr,
        min_blob=1, dilate=0, missing_detect="constant", batch=args.batch)
    print(f"[clean] strict  口径扫描完成：clean={int((nd_s == 0).sum())}", flush=True)

    # ---- 落盘 ----
    for tag, keep in (("wpac_clean", nd == 0), ("wpac_clean_strict", nd_s == 0)):
        sub = df.loc[keep].reset_index(drop=True)
        out = os.path.join(os.path.dirname(args.csv), f"{tag}.csv")
        sub.to_csv(out, index=False)
        print(f"[clean] 写出 {out}: {len(sub)} 行, {sub['ID'].nunique()} 个风暴", flush=True)

    # ---- 统计 ----
    def _stat(keep, nd_arr, imgf_arr, maxf_arr):
        sub = df.loc[keep]
        return {
            "n": int(keep.sum()),
            "frac_of_all": round(float(keep.mean()), 4),
            "n_storms": int(sub["ID"].nunique()),
            "vmax_mean": round(float(sub["Vmax"].mean()), 2),
            "vmax_std": round(float(sub["Vmax"].std()), 2),
            "vmax_min": float(sub["Vmax"].min()),
            "vmax_max": float(sub["Vmax"].max()),
            "bad_channel_hist": {str(k): int(v) for k, v in
                                 zip(*np.unique(nd_arr, return_counts=True))},
            "image_frac_mean": round(float(imgf_arr.mean()), 5),
            "max_ch_frac_p95": round(float(np.percentile(maxf_arr, 95)), 5),
        }

    stats = {
        "all": {"n": int(len(df)), "n_storms": int(df["ID"].nunique()),
                "vmax_mean": round(float(df["Vmax"].mean()), 2)},
        "project_clean": _stat(nd == 0, nd, imgf, maxf),
        "strict_clean": _stat(nd_s == 0, nd_s, imgf_s, maxf_s),
        # 非 clean 的坏道数分布（project 口径）
        "project_all_bad_channel_hist": {
            str(k): int(v) for k, v in zip(*np.unique(nd, return_counts=True))},
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/clean_subset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    print("[clean] 完成", flush=True)


if __name__ == "__main__":
    main()
