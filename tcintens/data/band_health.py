# -*- coding: utf-8 -*-
"""
band_health.py —— 频带(通道)缺失检测与修正

背景：
    TCIR 的 PMW(被动微波)等通道仅在卫星过顶时有观测，缺测时通常被写成全 0
    （抽取阶段 np.nan_to_num 已把 NaN 置 0）。缺测通道若直接送进网络，会引入
    一个恒定的异常值（归一化后仍是常数偏移），既浪费通道又可能干扰学习。

本模块提供：
    1. detect_missing_channels : 判定 (C,H,W) 中哪些通道整通道缺失
    2. correct_missing         : 按策略修正（none / zero / mean）
    3. scan_dataset            : 向量化扫描整个 h5，统计每通道缺失数
    4. ensure_band_mask        : 取/生成缺失掩码缓存（供 drop 策略复用）

缺失判定（mode）：
    constant : 整通道 min==max（即常数）。缺测通道恒为常数(0)，真实卫星通道必有空间变化，
              因此该判据稳健、默认采用。
    var      : 通道方差 < var_eps。可作为补充。

修正策略（strategy）：
    none : 不处理（保持现状，即 0；作为对照基线）
    zero : 显式把缺失通道置 0（与 none 数值等价，但语义明确）
    mean : 用该通道原始均值填充缺失通道 -> 归一化后变为 0（中性，不引入偏移）
"""
import os

import numpy as np


# --------------------------------------------------------------------------- #
# 单样本检测 / 修正
# --------------------------------------------------------------------------- #
def detect_missing_channels(arr: np.ndarray, mode: str = "constant",
                             var_eps: float = 1e-6) -> np.ndarray:
    """判定 (C, H, W) 数组中哪些通道整通道缺失，返回 (C,) bool。

    arr : (C, H, W) 原始空间（归一化前的浮点数组）
    mode: "constant" -> min==max；"var" -> 方差 < var_eps
    """
    arr = np.asarray(arr, dtype=np.float64)
    if mode == "constant":
        mn = arr.min(axis=(-2, -1))
        mx = arr.max(axis=(-2, -1))
        return (mn == mx)
    elif mode == "var":
        return (arr.var(axis=(-2, -1)) < var_eps)
    else:
        raise ValueError(f"未知 missing_detect mode: {mode}")


def correct_missing(arr: np.ndarray, missing_mask: np.ndarray,
                    strategy: str = "none", fill_values=None) -> np.ndarray:
    """按策略修正缺失通道。

    arr         : (C, H, W) 原始空间浮点
    missing_mask : (C,) bool（detect_missing_channels 的输出）
    strategy    : none | zero | mean
    fill_values : (C,) 各通道原始均值（mean 策略用，归一化后变 0）
    """
    arr = np.asarray(arr, dtype=np.float64)
    out = arr.copy()
    missing = np.where(np.asarray(missing_mask))[0]
    if missing.size == 0 or strategy in ("none",):
        return out
    for c in missing:
        if strategy == "zero":
            out[c] = 0.0
        elif strategy == "mean":
            v = fill_values[c] if fill_values is not None else out[c].mean()
            out[c] = float(v)
        else:
            raise ValueError(f"未知 missing_band strategy: {strategy}")
    return out.astype(arr.dtype) if out.dtype != arr.dtype else out


# --------------------------------------------------------------------------- #
# 数据集级扫描 / 掩码缓存
# --------------------------------------------------------------------------- #
def scan_dataset(h5_path: str, channels, batch: int = 2000,
                 mode: str = "constant") -> tuple:
    """向量化扫描 h5，返回 (mask, stats)。

    mask : (N, C) int8，mask[i,c]==1 表示第 i 帧第 c 通道缺失
    stats: 人类可读摘要（每通道缺失数、含任意缺失的图数、每图缺失数分布等）
    """
    import h5py
    ch_idx = list(range(len(channels)))
    n_ch = len(channels)
    miss_const = np.zeros(n_ch, dtype=np.int64)
    miss_zero = np.zeros(n_ch, dtype=np.int64)
    raw_sum = np.zeros(n_ch, dtype=np.float64)
    raw_sq = np.zeros(n_ch, dtype=np.float64)
    raw_cnt = np.zeros(n_ch, dtype=np.int64)
    per_img = []
    N = 0
    with h5py.File(h5_path, "r") as hf:
        mat = hf["matrix"]
        total = mat.shape[0]
        for s in range(0, total, batch):
            e = min(s + batch, total)
            x = mat[s:e][..., ch_idx].astype(np.float64)   # (B,H,W,C)
            B = x.shape[0]
            N += B
            mn = x.min(axis=(1, 2))
            mx = x.max(axis=(1, 2))
            const = (mn == mx)
            zero = (mn == 0) & (mx == 0)
            miss_const += const.sum(axis=0)
            miss_zero += zero.sum(axis=0)
            raw_sum += x.sum(axis=(0, 1, 2))
            raw_sq += (x ** 2).sum(axis=(0, 1, 2))
            raw_cnt += B * x.shape[1] * x.shape[2]
            per_img.append(const.astype(np.int8))
    mask = np.concatenate(per_img, axis=0)
    mean = raw_sum / raw_cnt
    std = np.sqrt(np.clip(raw_sq / raw_cnt - mean ** 2, 1e-12, None))
    stats = {
        "h5_path": h5_path,
        "n_images": int(N),
        "channels": list(channels),
        "missing_constant": {channels[c]: int(miss_const[c]) for c in ch_idx},
        "missing_allzero": {channels[c]: int(miss_zero[c]) for c in ch_idx},
        "raw_mean": {channels[c]: round(float(mean[c]), 4) for c in ch_idx},
        "raw_std": {channels[c]: round(float(std[c]), 4) for c in ch_idx},
        "images_with_any_missing": int(mask.any(axis=1).sum()),
        "per_image_missing_count": {
            str(k): int((mask.sum(axis=1) == k).sum()) for k in range(n_ch + 1)
        },
    }
    return mask.astype(np.int8), stats


def ensure_band_mask(h5_path: str, channels, mask_path: str,
                     mode: str = "constant", batch: int = 2000) -> np.ndarray:
    """读取缺失掩码缓存；若不存在则用 scan_dataset 生成并保存。

    返回 (N, C) int8 掩码，行号与 h5 帧顺序(紧凑行号)一一对应。
    """
    if os.path.exists(mask_path):
        with np.load(mask_path) as z:
            return z["mask"].astype(np.int8)
    mask, _ = scan_dataset(h5_path, channels, batch=batch, mode=mode)
    os.makedirs(os.path.dirname(os.path.abspath(mask_path)), exist_ok=True)
    np.savez_compressed(mask_path, mask=mask.astype(np.int8),
                        channels=np.array(list(channels), dtype="S"))
    return mask
