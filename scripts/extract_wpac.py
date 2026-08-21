# -*- coding: utf-8 -*-
"""
extract_wpac.py —— 把 TCIR 中 WPAC(西北太平洋) 子集抽取成紧凑、可快速读取的本地 h5

为什么需要它：
    TCIR 原始 h5 约 30GB，且 WSL 磁盘顺序读 ~50MB/s。若训练时每轮从 30GB 文件
    随机读 1.4 万帧，I/O 会成为瓶颈（数分钟/epoch）。
    本脚本一次性（慢速读一次）把 WPAC 帧：
      * 只保留 IR1/WV/PMW 三通道（弃用白天不稳定的 VIS）
      * 下采样 201->resize（默认 96）
      * NaN 置 0
    写出为 (M, resize, resize, 3) 的紧凑 h5（约 2GB），并把标签/风暴ID 写配套 CSV。
    同时在该过程中直接计算逐通道 mean/std（基于 resize 后的分布，与训练输入一致）。

用法：
    python scripts/extract_wpac.py \
        --h5 data/TCIR.h5 \
        --csv data/wpac_info.csv \
        --out-h5 data/wpac_96.h5 \
        --out-csv data/wpac_compact.csv \
        --channels IR1 WV PMW --resize 96 --start 27000
"""
import argparse
import csv
import json
import sys

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

CH_MAP = {"IR1": 0, "WV": 1, "VIS": 2, "PMW": 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="原始 TCIR h5（含 matrix + info）")
    ap.add_argument("--csv", required=True, help="wpac_info.csv（matrix_index/Vmax/ID/data_set）")
    ap.add_argument("--out-h5", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--channels", nargs="+", default=["IR1", "WV", "PMW"])
    ap.add_argument("--resize", type=int, default=96)
    ap.add_argument("--start", type=int, default=27000,
                    help="WPAC 帧的 matrix_index 最小约 27322，从 27000 起读即可跳过前部无关帧")
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--stats-out", default="data_tcir_wpac_stats.json")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    rec = {int(r.matrix_index): (float(r.Vmax), str(r.ID), str(r.data_set))
           for r in df.itertuples()}
    wpac_idx = set(rec.keys())
    ch_idx = [CH_MAP[c] for c in args.channels]
    R = args.resize

    sums = np.zeros(len(ch_idx), dtype=np.float64)
    sqs = np.zeros(len(ch_idx), dtype=np.float64)
    counts = np.zeros(len(ch_idx), dtype=np.int64)
    rows_matrix, rows_meta = [], []

    with h5py.File(args.h5, "r") as hf:
        mat = hf["matrix"]
        N = mat.shape[0]
        start = min(args.start, N)
        for s in range(start, N, args.batch):
            e = min(s + args.batch, N)
            x = mat[s:e]                                      # (B, 201, 201, 4)
            sel, meta = [], []
            for j in range(x.shape[0]):
                gi = s + j
                if gi not in wpac_idx:
                    continue
                fr = x[j][..., ch_idx].astype(np.float32)      # (201, 201, C)
                fr = np.nan_to_num(fr, nan=0.0)
                sel.append(fr)
                meta.append((gi,) + rec[gi])
            if sel:
                t = torch.from_numpy(np.stack(sel, 0)).permute(0, 3, 1, 2)  # (K,C,201,201)
                t = F.interpolate(t, size=(R, R), mode="bilinear", align_corners=False)
                out = t.permute(0, 2, 3, 1).numpy().astype(np.float32)     # (K,R,R,C)
                rows_matrix.append(out)
                rows_meta.extend(meta)
                sums += out.sum((0, 1, 2))
                sqs += (out ** 2).sum((0, 1, 2))
                counts += out.shape[0] * R * R

    M = np.concatenate(rows_matrix, axis=0)                   # (M, R, R, C)
    with h5py.File(args.out_h5, "w") as hf:
        hf.create_dataset("matrix", data=M)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["matrix_index", "Vmax", "ID", "data_set"])
        for i, (gi, vmax, sid, ds) in enumerate(rows_meta):
            w.writerow([i, vmax, sid, ds])

    mean = sums / counts
    std = np.sqrt(np.clip(sqs / counts - mean ** 2, 1e-12, None))
    json.dump({"channels": list(args.channels), "mean": mean.tolist(), "std": std.tolist()},
              open(args.stats_out, "w"), indent=2)
    print(f"extracted matrix: {M.shape}  frames={M.shape[0]}")
    print(f"stats mean: {[round(v, 3) for v in mean.tolist()]}")
    print(f"stats std : {[round(v, 3) for v in std.tolist()]}")
    print(f"wrote {args.out_h5}  and  {args.out_csv}  and  {args.stats_out}")


if __name__ == "__main__":
    sys.exit(main())
