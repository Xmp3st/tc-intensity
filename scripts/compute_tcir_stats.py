# -*- coding: utf-8 -*-
"""
compute_tcir_stats.py —— 计算 TCIR 逐通道均值/标准差，用于归一化

输出 JSON，例如：
    {"mean": [..], "std": [..]}

把结果填到 configs/tcir_regression.yaml 的
data.tcir.norm_mean / norm_std，并把 data.tcir.normalize 设为 true。

用法：
    python scripts/compute_tcir_stats.py --h5 data/TCIR-ATLN_EPAC_WPAC.h5 \
        --channels IR1 WV PMW --out data/tcir_stats.json
"""
import argparse
import json
import sys

sys.path.insert(0, ".")
from tcintens.data.tcir import compute_tcir_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--channels", nargs="+", default=["IR1", "WV", "PMW"])
    ap.add_argument("--sample-every", type=int, default=1,
                    help="每隔多少帧采样一次（大数据集可设大一点加速）")
    ap.add_argument("--out", default="data/tcir_stats.json")
    args = ap.parse_args()

    mean, std = compute_tcir_stats(args.h5, tuple(args.channels), args.sample_every)
    out = {"channels": list(args.channels), "mean": mean, "std": std}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"已写入 {args.out}")
    print(f"  通道   : {args.channels}")
    print(f"  mean   : {[round(x, 4) for x in mean]}")
    print(f"  std    : {[round(x, 4) for x in std]}")


if __name__ == "__main__":
    sys.exit(main())
