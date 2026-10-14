# -*- coding: utf-8 -*-
"""extract_wpac_hires.py —— 流式抽取「高分辨率」WPAC 紧凑 h5（低内存）

与 extract_wpac.py 的唯一本质区别：
    后者把所有帧先堆在内存 list 里，最后 np.concatenate 一次性写出。
    在 96² 时约 2.2GB 尚可；到 128²/160²/201² 需要 4/6/10GB（concat 时翻倍），
    本机 WSL 只有 7GB RAM，必然 OOM。
    本脚本改为「预分配 dataset + 分块写入」，峰值内存只与 --src-batch 有关，
    与目标分辨率无关。

其余流程完全对齐 extract_wpac.py（保证与 wpac_96.h5 的行序、通道、NaN 处理一致）：
    * 通道取 [IR1, WV, PMW]（用 CH_MAP 映射到原始 4 通道位置）
    * 行序 = wpac_info.csv 中 matrix_index 升序（与 wpac_96.h5 完全相同）
    * 先 nan_to_num(nan=0)，再 bilinear 下采样到 --resize
    * 顺带统计逐通道 mean/std（在目标分辨率下）

用法：
    python scripts/extract_wpac_hires.py --resize 160 \
        --out-h5 data/wpac_160.h5 --stats-out data/wpac_160_stats.json
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

CH_MAP = {"IR1": 0, "WV": 1, "VIS": 2, "PMW": 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/TCIR.h5", help="原始 TCIR h5")
    ap.add_argument("--csv", default="data/wpac_info.csv", help="WPAC 帧清单")
    ap.add_argument("--out-h5", required=True)
    ap.add_argument("--channels", nargs="+", default=["IR1", "WV", "PMW"])
    ap.add_argument("--resize", type=int, required=True)
    ap.add_argument("--src-batch", type=int, default=256,
                    help="每次从源 h5 读入的帧数（决定峰值内存）")
    ap.add_argument("--stats-out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    rec = {int(r.matrix_index): (float(r.Vmax), str(r.ID), str(r.data_set))
           for r in df.itertuples()}
    targets = sorted(rec.keys())                 # 与 wpac_96.h5 完全相同的行序
    M = len(targets)
    tgt_set = set(targets)
    pos = {gi: i for i, gi in enumerate(targets)}
    ch_idx = [CH_MAP[c] for c in args.channels]
    R = args.resize
    C = len(ch_idx)
    print(f"targets={M} frames | resize={R} | channels={args.channels}", flush=True)

    sums = np.zeros(C, dtype=np.float64)
    sqs = np.zeros(C, dtype=np.float64)
    counts = 0
    s0, s1 = targets[0], targets[-1]
    print(f"source range to read: matrix_index [{s0}, {s1}]", flush=True)

    with h5py.File(args.h5, "r") as hf, h5py.File(args.out_h5, "w") as out:
        mat = hf["matrix"]
        ds = out.create_dataset(
            "matrix", shape=(M, R, R, C), dtype="float32",
            chunks=(min(64, M), R, R, C), compression=None)
        written = 0
        for s in range(s0, s1 + 1, args.src_batch):
            e = min(s + args.src_batch, s1 + 1)
            idxs = [gi for gi in range(s, e) if gi in tgt_set]
            if not idxs:
                continue
            blk = mat[idxs]                            # (k,201,201,4) 只读所需帧
            fr = blk[..., ch_idx].astype(np.float32)   # (k,201,201,C)
            fr = np.nan_to_num(fr, nan=0.0)
            t = torch.from_numpy(fr).permute(0, 3, 1, 2)   # (k,C,201,201)
            if t.shape[-1] != R:
                t = F.interpolate(t, size=(R, R), mode="bilinear",
                                  align_corners=False)
            arr = t.permute(0, 2, 3, 1).contiguous().numpy().astype(np.float32)
            rows = [pos[gi] for gi in idxs]            # 升序，满足 h5py fancy 索引要求
            ds[rows] = arr
            # ⚠️ 必须用 float64 累加：单块运行和可达 1e9 量级，float32 的 ULP≈128，
            #    逐元素累加会累积 ~1% 的系统性漂移（实测 WV 均值偏 1.79%，
            #    并因 mean²>sqs/counts 被 clip 而让 std 假报 0）。数据本身不受影响。
            a64 = arr.astype(np.float64)
            sums += a64.sum(axis=(0, 1, 2))
            sqs += (a64 ** 2).sum(axis=(0, 1, 2))
            counts += arr.shape[0] * R * R
            prev = written
            written += len(idxs)
            if written // 4000 != prev // 4000:
                print(f"  ... {written}/{M}", flush=True)

    mean = sums / counts
    std = np.sqrt(np.clip(sqs / counts - mean ** 2, 1e-12, None))
    print("mean:", [round(float(v), 4) for v in mean])
    print("std :", [round(float(v), 4) for v in std])
    if args.stats_out:
        json.dump({"channels": list(args.channels), "resize": R,
                   "mean": mean.tolist(), "std": std.tolist()},
                  open(args.stats_out, "w"), indent=2)
        print("wrote", args.stats_out)
    print(f"wrote {args.out_h5}  ({os.path.getsize(args.out_h5) / 1e9:.2f} GB)")


if __name__ == "__main__":
    sys.exit(main())
