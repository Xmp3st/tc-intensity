# -*- coding: utf-8 -*-
"""
scan_bands.py —— 训练前识别 WPAC 影像中各通道(频带)的缺失情况

TCIR 的 PMW(被动微波)等通道仅在卫星过顶时才有观测，缺测时通常被写成全 0
(抽取阶段 np.nan_to_num 已把 NaN 置 0)。本脚本调用 tcintens.data.band_health
分块扫描 h5，统计每个通道「整通道常数(即缺失)」的图像数量，并打印摘要；
同时缓存逐图缺失掩码 (N, C) 到 logs/band_missing_mask.npz，供 drop 策略复用。

用法:
  python scripts/scan_bands.py --h5 data/wpac_96.h5 --channels IR1 WV PMW
"""
import argparse
import json
import os
import sys

# 允许以脚本方式直接运行（项目根在 sys.path）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tcintens.data.band_health import scan_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/wpac_96.h5")
    ap.add_argument("--channels", nargs="+", default=["IR1", "WV", "PMW"])
    ap.add_argument("--mode", default="constant",
                    help="缺失判定: constant(整通道常数) | var(方差<eps)")
    ap.add_argument("--mask-out", default="logs/band_missing_mask.npz")
    ap.add_argument("--json-out", default="logs/band_scan.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.mask_out), exist_ok=True)
    mask, stats = scan_dataset(args.h5, args.channels, mode=args.mode)
    import numpy as np
    np.savez_compressed(args.mask_out, mask=mask.astype(np.int8),
                        channels=np.array(args.channels, dtype="S"))
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"扫描完成: {args.h5}  ({stats['n_images']} 张图)")
    print("  逐通道缺失(整通道常数=缺失):")
    for c in args.channels:
        print(f"    {c:5s}: 缺失 {stats['missing_constant'][c]:6d} / "
              f"全0 {stats['missing_allzero'][c]:6d}  "
              f"(均值 {stats['raw_mean'][c]}, 标准差 {stats['raw_std'][c]})")
    print(f"  含任意缺失通道的图像: {stats['images_with_any_missing']} 张")
    print(f"  每图缺失通道数分布: {stats['per_image_missing_count']}")
    print(f"  掩码已缓存: {args.mask_out}")
    print(f"  摘要已写:   {args.json_out}")


if __name__ == "__main__":
    main()
