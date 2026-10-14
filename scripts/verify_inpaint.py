# -*- coding: utf-8 -*-
"""验证「局部区域缺失插值」的正确性、真实数据效果与数据集集成。

1. 合成测试：在有平滑梯度的通道中央挖一个缺测洞(0 / 或涂抹低值)，
   用 griddata 插值修复，比较修复后与真值的误差（应很小）。
2. 真实测试：取 WPAC 中已知含 IR1/WV 局部缺测的图(idx=276)，
   修复前后统计对比，并保存可视化对比图 outputs/inpaint_check.png。
3. 数据集集成：以 missing_band="interp" 构造 TCIRDataset，逐样本取数，
   确认返回有限张量、无异常。
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tcintens.data.band_health import (
    detect_missing_region, inpaint_regions, correct_region_missing)
from tcintens.data.tcir import TCIRDataset

H5 = "data/wpac_96.h5"
OUT = "outputs/inpaint_check.png"


def synthetic_test():
    print("=== 合成正确性测试 ===")
    H = W = 64
    yy, xx = np.mgrid[0:H, 0:W]
    truth = (xx / W * 200 + yy / H * 60 + 180.0)  # 平滑梯度, 物理范围 ~180-260K
    arr = truth.copy()
    # 在中心挖 20x20 缺测洞
    r0, r1, c0, c1 = 22, 42, 22, 42
    hole = arr[r0:r1, c0:c1]
    arr[r0:r1, c0:c1] = 0.0

    # (a) 哨兵精确0 检测
    m0 = detect_missing_region(arr[None], sentinel=0.0, low_thr=None, min_blob=20)
    assert m0[0].sum() == (r1 - r0) * (c1 - c0), "哨兵0 检测应覆盖整个洞"
    filled0 = inpaint_regions(arr[None], m0, method="griddata")
    err0 = np.sqrt(np.mean((filled0[0][r0:r1, c0:c1] - truth[r0:r1, c0:c1]) ** 2))
    print(f"  哨兵0 检测+插值: 洞内 RMSE={err0:.3f} K (应很小)")

    # (b) 涂抹低值(如 50K) + low_thr=150 检测
    arr2 = truth.copy()
    arr2[r0:r1, c0:c1] = 50.0
    m1 = detect_missing_region(arr2[None], sentinel=0.0, low_thr=150.0, min_blob=20)
    assert m1[0].sum() == (r1 - r0) * (c1 - c0), "low_thr 检测应覆盖整个低值洞"
    filled1 = inpaint_regions(arr2[None], m1, method="griddata")
    err1 = np.sqrt(np.mean((filled1[0][r0:r1, c0:c1] - truth[r0:r1, c0:c1]) ** 2))
    print(f"  low_thr=150 检测+插值: 洞内 RMSE={err1:.3f} K (应很小)")
    print("  合成测试通过 ✓" if err0 < 5 and err1 < 5 else "  合成测试异常 ✗")
    return err0, err1


def real_test(idx=276):
    import h5py
    print(f"\n=== 真实数据测试 (h5 idx={idx}) ===")
    with h5py.File(H5, "r") as hf:
        raw = hf["matrix"][idx]  # (96,96,3) -> IR1,WV,PMW
    arr = np.ascontiguousarray(raw.transpose(2, 0, 1)).astype(np.float32)  # (3,96,96)
    ch_names = ["IR1", "WV", "PMW"]
    low_thr = [150.0, 150.0, -999.0]
    before = arr.copy()
    mask = np.zeros_like(arr, dtype=bool)
    for ci, c in enumerate(ch_names):
        if c not in ("IR1", "WV"):
            continue
        lt = low_thr[ci] if low_thr[ci] > -900 else None
        m = detect_missing_region(arr[ci:ci + 1], sentinel=0.0, low_thr=lt,
                                  min_blob=20, dilate=1)
        mask[ci] = m[0]
    filled, _ = correct_region_missing(
        arr, sentinel=0.0, low_thr=low_thr, min_blob=20, dilate=1, method="griddata")
    for ci, c in enumerate(ch_names):
        n_miss = int(mask[ci].sum())
        if n_miss == 0:
            print(f"  [{c}] 无区域缺测")
            continue
        bmin = before[ci][mask[ci]].min()
        bmax = before[ci][mask[ci]].max()
        amean = filled[ci][mask[ci]].mean()
        print(f"  [{c}] 缺测像素 {n_miss:5d} | 修复前范围[{bmin:.1f},{bmax:.1f}] "
              f"-> 修复后均值 {amean:.1f} | 仍<150 的像素数: {int((filled[ci] < 150).sum())}")
    _save_figure(before, filled, mask, ch_names, idx)
    print(f"  对比图已保存: {OUT}")
    return mask


def _save_figure(before, after, mask, ch_names, idx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    titles = [f"{c} 修复前" for c in ch_names] + \
             [f"{c} 缺测掩码" for c in ch_names] + \
             [f"{c} 修复后" for c in ch_names]
    for j in range(3):
        for k in range(3):
            ax = axes[j, k]
            ci = k
            if j == 0:
                im = before[ci]
            elif j == 1:
                im = mask[ci].astype(float)
            else:
                im = after[ci]
            ax.imshow(im, cmap="viridis")
            ax.set_title(titles[j * 3 + k], fontsize=9)
            ax.axis("off")
    fig.suptitle(f"WPAC h5 idx={idx} —— 局部区域缺失插值(griddata)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=110)
    plt.close(fig)


def dataset_test():
    print("\n=== 数据集集成测试 (missing_band='interp') ===")
    ds = TCIRDataset(
        h5_path=H5, indices=list(range(300)),
        channels=("IR1", "WV", "PMW"),
        nan_mode="zero", normalize=True,
        norm_mean=[260.864, 232.83, 0.638], norm_std=[32.849, 15.389, 1.71],
        missing_band="interp",
        missing_region_channels=["IR1", "WV"],
        missing_region_sentinel=0.0,
        missing_region_low_thr=[150.0, 150.0, -999.0],
        missing_region_min_blob=20, missing_region_dilate=1,
        missing_region_method="griddata",
        frame_labels=np.full(300, 50.0), split="train")
    bad = 0
    checked = 0
    for i in range(0, 300, 30):
        x, y = ds[i]
        checked += 1
        if not torch_is_finite(x):
            bad += 1
    print(f"  取数 {checked} 个样本, 非有限张量 {bad} 个 (应为0)")
    print("  数据集集成测试通过 ✓" if bad == 0 else "  数据集集成异常 ✗")


def torch_is_finite(t):
    import torch
    return bool(torch.isfinite(t).all())


def config_test():
    """用真实配置文件 + missing_band=interp 覆盖，构建数据集并取数。"""
    print("\n=== 真实配置端到端测试 (load_config + create_tcir_datasets) ===")
    from tcintens.utils.config import load_config
    from tcintens.data.tcir import create_tcir_datasets
    cfg = load_config("configs/tcir_wpac_train.yaml",
                      overrides=["data.tcir.missing_band=interp"])
    datasets, meta = create_tcir_datasets(cfg)
    print(f"  meta: n_train={meta['n_train']} n_val={meta['n_val']} "
          f"n_test={meta['n_test']} missing_band={meta['missing_band']}")
    ds = datasets["test"]
    bad = 0
    for i in range(min(60, len(ds))):
        x, y = ds[i]
        if not torch_is_finite(x) or not torch_is_finite(y):
            bad += 1
    print(f"  测试集取数 {min(60, len(ds))} 个, 非有限 {bad} 个 (应为0)")
    print("  配置端到端测试通过 ✓" if bad == 0 else "  配置端到端异常 ✗")


if __name__ == "__main__":
    synthetic_test()
    real_test(276)
    dataset_test()
    config_test()
    print("\n全部验证完成。")
