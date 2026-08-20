# -*- coding: utf-8 -*-
"""
tcir.py —— TCIR (Tropical Cyclone Image-to-intensity Regression) 数据集加载器

数据来源（Hsuan-Tien Lin 团队，KDD 2018）：
    https://www.csie.ntu.edu.tw/~htlin/program/TCIR/
    https://github.com/BoyoChen/TCIR

HDF5 文件结构（两个 key）：
    matrix : N x 201 x 201 x 4  (float32, channel-last, 含 NaN)
             通道顺序固定为 [IR1, WV, VIS, PMW]
    info   : pandas DataFrame，含标签列
             强度 intensity(主目标, 单位 knot)、size(nmi)、pres(hPa)、
             region、year、id(风暴ID)、lat、lon 等

设计要点：
    1. 懒加载：不一次性把 matrix 读进内存（70k 帧约数十 GB），
       而是按样本索引从 h5 文件中切片读取（DataLoader 多进程安全）。
    2. 多通道：默认取 3 通道 [IR1, WV, PMW]（弃用白天极不稳定的 VIS），
       也可配置取全部 4 通道。
    3. NaN 处理：zero（置 0）或 interp（按行线性插值）。
    4. 切分防泄漏：默认按「风暴 ID」分组切分 train/val/test，
       保证同一台风的连续帧不会同时出现在训练与测试集。
"""
import os
import random

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# TCIR matrix 的 4 个通道固定顺序
CHANNELS = ("IR1", "WV", "VIS", "PMW")

# knot -> m/s 换算
KNOT_TO_MS = 0.514444


# --------------------------------------------------------------------------- #
# 单样本数据集
# --------------------------------------------------------------------------- #
class TCIRDataset(Dataset):
    """读取单个 TCIR HDF5 文件的子集（按索引列表）。"""

    def __init__(
        self,
        h5_path: str,
        indices=None,
        channels=("IR1", "WV", "PMW"),
        target: str = "intensity",
        label_unit: str = "knots",
        nan_mode: str = "zero",
        normalize: bool = False,
        norm_mean=None,
        norm_std=None,
        resize: int = None,
        split: str = "train",
        aug: dict = None,
    ):
        self.h5_path = h5_path
        self.channels = list(channels)
        self.ch_idx = [CHANNELS.index(c) for c in self.channels]
        self.target = target
        self.label_unit = label_unit
        self.nan_mode = nan_mode
        self.normalize = normalize
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.resize = resize
        self.split = split
        self.aug = aug or {}

        # 标签：直接从 info 表读取（数据集已自带标注，无需人工标注）
        info = load_info(h5_path)
        labels = info[target].astype(float).to_numpy()
        if label_unit == "ms":
            labels = labels * KNOT_TO_MS  # 原始为 knot
        if indices is None:
            indices = list(range(len(labels)))
        self.indices = list(indices)
        self.labels = labels[self.indices]
        self._hf = None  # 延迟打开，兼容多进程

    # ---- 多进程安全：每个 worker 重新打开 h5 ----
    def _ensure_open(self):
        if self._hf is None:
            self._hf = h5py.File(self.h5_path, "r")
            self._matrix = self._hf["matrix"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._ensure_open()
        real_idx = self.indices[idx]
        # matrix[real_idx] -> (201, 201, 4)
        arr = self._matrix[real_idx][..., self.ch_idx]          # (H, W, C)
        arr = np.ascontiguousarray(arr.transpose(2, 0, 1)).astype(np.float32)  # (C, H, W)

        # 1) NaN 处理
        if self.nan_mode == "interp":
            arr = _fill_nan_interp(arr)
        else:  # zero
            arr = np.nan_to_num(arr, nan=0.0, copy=False)

        x = torch.from_numpy(arr)

        # 2) 归一化（可选，需提供逐通道均值/标准差）
        if self.normalize and self.norm_mean is not None:
            m = torch.tensor(self.norm_mean, dtype=x.dtype).view(-1, 1, 1)
            s = torch.tensor(self.norm_std, dtype=x.dtype).view(-1, 1, 1)
            x = (x - m) / s

        # 3) 尺寸调整（可选）
        if self.resize is not None and self.resize != x.shape[-1]:
            x = torch.nn.functional.interpolate(
                x.unsqueeze(0), size=(self.resize, self.resize),
                mode="bilinear", align_corners=False,
            ).squeeze(0)

        # 4) 训练集增强（台风无 canonical 朝向，旋转/翻转安全）
        if self.split == "train" and self.aug:
            if self.aug.get("rotate", 0) > 0:
                ang = random.uniform(-self.aug["rotate"], self.aug["rotate"])
                x = torchvision_rotate(x, ang)
            if self.aug.get("horizontal_flip", False) and random.random() < 0.5:
                x = torch.flip(x, dims=[-1])

        return x, torch.tensor(float(self.labels[idx]), dtype=torch.float32)

    def close(self):
        if self._hf is not None:
            self._hf.close()
            self._hf = None


def torchvision_rotate(x, angle):
    from torchvision.transforms.functional import rotate
    return rotate(x, angle)


# --------------------------------------------------------------------------- #
# info 表读取：优先用 pandas 官方读法（需 pytables），失败回退到纯 h5py 解析
# （兼容 pandas 写入 HDF5 的 fixed 与 table 两种布局，无需安装 pytables）
# --------------------------------------------------------------------------- #
def _decode(v):
    return v.decode("utf-8") if isinstance(v, bytes) else v


def _read_info_h5py(h5_path: str):
    with h5py.File(h5_path, "r") as hf:
        if "info" not in hf:
            raise KeyError(f"{h5_path} 中找不到 info 组")
        g = hf["info"]
        # pandas fixed 布局：block0_values(2D) + block0_items(列名)
        if "block0_values" in g:
            cols = [_decode(c) for c in g["block0_items"][()]]
            vals = g["block0_values"][()]
            return pd.DataFrame(vals, columns=cols)
        # pandas table 布局：每个列是一个 dataset
        cols = {}
        for k in g.keys():
            if k.startswith("_"):   # 跳过 pytables 索引组
                continue
            arr = g[k][()]
            if isinstance(arr, np.ndarray) and arr.dtype.kind in ("O", "S", "U"):
                arr = np.array([_decode(v) for v in arr])
            cols[_decode(k)] = arr
        if cols:
            return pd.DataFrame(cols)
        raise ValueError(
            "无法用 h5py 解析 info，请安装 pytables：pip install tables"
        )


def load_info(h5_path: str) -> pd.DataFrame:
    """读取 TCIR 的 info 表（pandas DataFrame）。"""
    try:
        return pd.read_hdf(h5_path, key="info", mode="r")
    except Exception:
        return _read_info_h5py(h5_path)


def _fill_nan_interp(arr: np.ndarray) -> np.ndarray:
    """对 (C, H, W) 每个通道沿行方向做线性插值填补 NaN。"""
    out = arr.copy()
    c, h, w = out.shape
    for ci in range(c):
        for r in range(h):
            row = out[ci, r]
            if not np.isnan(row).any():
                continue
            xx = np.arange(w)
            valid = ~np.isnan(row)
            if valid.sum() == 0:
                out[ci, r] = 0.0
            else:
                out[ci, r] = np.interp(xx, xx[valid], row[valid])
    return out


# --------------------------------------------------------------------------- #
# 切分 + 数据集构造（接入框架的 create_datasets）
# --------------------------------------------------------------------------- #
def create_tcir_datasets(cfg):
    """按配置构建 train/val/test，返回 (dict, meta)。"""
    d = cfg.to_dict()["data"]
    tcfg = d.get("tcir", {})
    h5_path = tcfg.get("h5_path", "data/TCIR.h5")
    # 若提供了 root_dir 且文件相对它存在，则拼接解析
    root = d.get("root_dir", "")
    if root and not os.path.isabs(h5_path):
        cand = os.path.join(root, h5_path)
        if os.path.exists(cand):
            h5_path = cand

    channels = tcfg.get("channels", ["IR1", "WV", "PMW"])
    target = tcfg.get("target", "intensity")
    label_unit = tcfg.get("label_unit", "knots")
    nan_mode = tcfg.get("nan_mode", "zero")
    normalize = tcfg.get("normalize", False)
    norm_mean = tcfg.get("norm_mean", None)
    norm_std = tcfg.get("norm_std", None)
    resize = tcfg.get("resize", None)
    ratios = d.get("split", [0.7, 0.15, 0.15])
    seed = cfg.get("experiment.seed", 42)
    regions = tcfg.get("regions", None)          # 可选：只取某些区域
    storm_col = tcfg.get("storm_col", "id")      # 按风暴 ID 切分防泄漏
    split_strategy = tcfg.get("split_strategy", "by_storm")

    info = load_info(h5_path)
    n = len(info)
    if regions:
        mask = info["region"].isin(regions).to_numpy()
        keep = np.where(mask)[0]
    else:
        keep = np.arange(n)

    # 训练集增强配置
    aug = cfg.to_dict().get("augmentation", {}).get("train", {})

    if split_strategy == "by_storm" and storm_col in info.columns:
        storm_ids = info[storm_col].to_numpy()[keep]
        rng = random.Random(seed)
        groups = {}
        for i, sid in zip(keep, storm_ids):
            groups.setdefault(sid, []).append(i)
        keys = list(groups.keys())
        rng.shuffle(keys)
        train, val, test = [], [], []
        r_tr, r_va = ratios[0], ratios[1]
        used = 0
        total = len(keys)
        for k in keys:
            used += 1
            frac = used / total
            if frac <= r_tr:
                train += groups[k]
            elif frac <= r_tr + r_va:
                val += groups[k]
            else:
                test += groups[k]
    else:
        # 随机切分（fallback）
        rng = random.Random(seed)
        idx_all = list(keep)
        rng.shuffle(idx_all)
        n_tr = int(round(len(idx_all) * ratios[0]))
        n_va = int(round(len(idx_all) * ratios[1]))
        train = idx_all[:n_tr]
        val = idx_all[n_tr:n_tr + n_va]
        test = idx_all[n_tr + n_va:]

    common = dict(
        h5_path=h5_path, channels=channels, target=target, label_unit=label_unit,
        nan_mode=nan_mode, normalize=normalize, norm_mean=norm_mean,
        norm_std=norm_std, resize=resize,
    )
    datasets = {
        "train": TCIRDataset(indices=train, split="train", aug=aug, **common),
        "val": TCIRDataset(indices=val, split="val", **common),
        "test": TCIRDataset(indices=test, split="test", **common),
    }
    meta = {
        "task": "regression",
        "class_names": [],
        "num_classes": 1,
        "in_channels": len(channels),
        "target": target,
        "label_unit": label_unit,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
    }
    return datasets, meta


# --------------------------------------------------------------------------- #
# 逐通道统计（可选：用于归一化的 mean/std 计算）
# --------------------------------------------------------------------------- #
def compute_tcir_stats(h5_path, channels=("IR1", "WV", "PMW"), sample_every=1):
    """遍历 matrix 计算逐通道均值/标准差，返回 (mean, std) 列表。"""
    ch_idx = [CHANNELS.index(c) for c in channels]
    sums = np.zeros(len(ch_idx), dtype=np.float64)
    sqs = np.zeros(len(ch_idx), dtype=np.float64)
    counts = np.zeros(len(ch_idx), dtype=np.int64)
    with h5py.File(h5_path, "r") as hf:
        mat = hf["matrix"]
        n = mat.shape[0]
        for i in range(0, n, sample_every):
            x = mat[i][..., ch_idx].astype(np.float64)   # (H, W, C)
            x = np.nan_to_num(x, nan=0.0)
            sums += x.sum(axis=(0, 1))
            sqs += (x ** 2).sum(axis=(0, 1))
            counts += x.shape[0] * x.shape[1]
    mean = sums / counts
    var = sqs / counts - mean ** 2
    std = np.sqrt(np.clip(var, 1e-12, None))
    return mean.tolist(), std.tolist()


# --------------------------------------------------------------------------- #
# 合成 TCIR（便于无真实数据时验证流程；含 NaN 以测试缺值处理）
# --------------------------------------------------------------------------- #
def make_synthetic_tcir(path, n=20, size=24, seed=0):
    """生成一个迷你 TCIR 风格 HDF5：matrix + info，含 NaN。"""
    rng = np.random.default_rng(seed)
    C = len(CHANNELS)
    matrix = np.zeros((n, size, size, C), dtype=np.float32)
    rows = []
    regions = ["WPAC", "EPAC", "ATL"]
    for i in range(n):
        # 中心暖核 + 噪声，强度与中心亮温相关（仅用于可学习性演示）
        yy, xx = np.mgrid[0:size, 0:size]
        cy = cx = size / 2
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        intensity = rng.uniform(20, 140)  # knot
        for c in range(C):
            base = rng.normal(0, 1, (size, size)).astype(np.float32)
            if c in (0, 3):  # IR1 / PMW：中心暖异常
                base -= (intensity / 140.0) * np.exp(-(r ** 2) / (2 * (size / 5) ** 2))
            matrix[i, ..., c] = base
        # 随机打一些 NaN（模拟缺测）
        mask = rng.random((size, size, C)) < 0.02
        matrix[i][mask] = np.nan
        sid = f"2016{regions[i % 3]}{i // 3 + 1:02d}"
        rows.append({
            "id": sid, "region": regions[i % 3], "year": 2016,
            "intensity": intensity, "size": rng.uniform(50, 300),
            "pres": rng.uniform(900, 1000), "lat": rng.uniform(0, 30),
            "lon": rng.uniform(120, 180),
        })
    info = pd.DataFrame(rows)
    with h5py.File(path, "w") as hf:
        hf.create_dataset("matrix", data=matrix)
        # 用 h5py 直接写 info 组（每列一个 dataset），无需 pytables
        g = hf.create_group("info")
        for col in info.columns:
            vals = info[col].to_numpy()
            if vals.dtype.kind in ("O", "U", "S"):
                g.create_dataset(
                    col,
                    data=np.array([str(v).encode("utf-8") for v in vals]),
                )
            else:
                g.create_dataset(col, data=vals)
    return path
