# -*- coding: utf-8 -*-
"""
scan_bands.py —— 训练前识别 WPAC 影像的缺失情况（两类）

(A) 整通道缺失：整通道为常数(全0)的图像（PMW 等卫星未观测帧）。
    缓存逐图掩码 (N, C) 到 logs/band_missing_mask.npz，供 missing_band=drop 复用。
(B) 局部区域缺失：某通道图像内某些空间区域缺数据（PMW 条带缺口 / IR1/WV 涂抹带），
    用连通块+物理下界识别，统计每通道含缺测块的图数与缺测像素占比；
    对应 missing_band=interp 的处理对象（默认仅 IR1/WV，PMW 的 0 为合法值不参与）。

用法:
  python scripts/scan_bands.py --h5 data/wpac_96.h5 --channels IR1 WV PMW
  python scripts/scan_bands.py --region-channels IR1 WV --low-thr 150 150
"""
import argparse
import json
import os
import sys

# 允许以脚本方式直接运行（项目根在 sys.path）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tcintens.data.band_health import scan_dataset, scan_region_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/wpac_96.h5")
    ap.add_argument("--channels", nargs="+", default=["IR1", "WV", "PMW"])
    ap.add_argument("--mode", default="constant",
                    help="整通道缺失判定: constant | var")
    ap.add_argument("--mask-out", default="logs/band_missing_mask.npz")
    ap.add_argument("--json-out", default="logs/band_scan.json")
    # 区域缺失扫描参数
    ap.add_argument("--region-channels", nargs="+", default=["IR1", "WV"])
    ap.add_argument("--low-thr", nargs="+", type=float, default=[150.0, 150.0],
                    help="各区域通道的物理下界(对齐 --region-channels)；低于此判缺测")
    ap.add_argument("--min-blob", type=int, default=20)
    ap.add_argument("--sentinel", type=float, default=0.0)
    args = ap.parse_args()

    # ---------- (A) 整通道缺失 ----------
    os.makedirs(os.path.dirname(args.mask_out), exist_ok=True)
    mask, stats = scan_dataset(args.h5, args.channels, mode=args.mode)
    import numpy as np
    np.savez_compressed(args.mask_out, mask=mask.astype(np.int8),
                        channels=np.array(args.channels, dtype="S"))
    print(f"扫描完成: {args.h5}  ({stats['n_images']} 张图)")
    print("【整通道缺失】(整通道常数=缺失):")
    for c in args.channels:
        print(f"    {c:5s}: 缺失 {stats['missing_constant'][c]:6d} / "
              f"全0 {stats['missing_allzero'][c]:6d}  "
              f"(均值 {stats['raw_mean'][c]}, 标准差 {stats['raw_std'][c]})")
    print(f"  含任意整通道缺失的图像: {stats['images_with_any_missing']} 张")
    print(f"  每图缺失通道数分布: {stats['per_image_missing_count']}")
    print(f"  整通道掩码已缓存: {args.mask_out}")

    # ---------- (B) 局部区域缺失 ----------
    # 把 low_thr 对齐到 region-channels
    lt = list(args.low_thr)
    if len(lt) == 1 and len(args.region_channels) > 1:
        lt = lt * len(args.region_channels)
    low_thr_for = {c: (lt[i] if i < len(lt) else None)
                   for i, c in enumerate(args.region_channels)}
    _, _, rstats = scan_region_stats(
        args.h5, args.region_channels, sentinel=args.sentinel,
        low_thr=lt, min_blob=args.min_blob)
    print(f"\n【局部区域缺失】通道 {args.region_channels} "
          f"(sentinel={args.sentinel}, low_thr={lt}, min_blob={args.min_blob}):")
    for c in args.region_channels:
        print(f"    {c:5s}: 含缺测块的图 {rstats['images_with_region_blob'][c]:6d} / "
              f"{rstats['n_images']}  "
              f"缺测像素 {rstats['region_missing_pixels'][c]:8d} "
              f"(占比 {rstats['region_missing_ratio'][c]*100:.4f}%)")

    stats["region"] = rstats
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n摘要已写: {args.json_out}")


if __name__ == "__main__":
    main()
