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
        missing_band: str = "none",
        missing_detect: str = "constant",
        fill_values=None,
        # ---- 局部区域缺失插值（missing_band=="interp" 时启用）----
        missing_region_channels=("IR1", "WV"),
        missing_region_sentinel: float = 0.0,
        missing_region_low_thr=None,
        missing_region_min_blob: int = 20,
        missing_region_dilate: int = 1,
        missing_region_method: str = "griddata",
        missing_region_max_frac: float = 1.0,
        missing_region_deg: int = 2,
        missing_region_iters: int = 100,
        missing_region_feather: int = 0,
        # ---- 训练期「条状缺失」干扰增广（模拟真实 IR1 条带缺测）----
        strip_dropout_prob: float = 0.0,
        strip_masks=None,
        return_corrected_flag: bool = False,
        frame_labels=None,
        # ---- 离线 U-Net 预修复的 IR1 缓存（见 scripts/build_unet_repaired_ir1.py）----
        # 给出 .npy 路径即启用：训练/评估时，凡在该缓存里的帧（真实缺测帧），
        # 其 IR1 通道整帧替换为修复后亮温。干净帧不在此缓存中，输入与原图一致。
        ir1_repaired_path: str = None,
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
        self.missing_band = missing_band
        self.missing_detect = missing_detect
        self.fill_values = fill_values
        self.missing_region_channels = list(missing_region_channels)
        self.missing_region_sentinel = missing_region_sentinel
        self.missing_region_low_thr = missing_region_low_thr
        self.missing_region_min_blob = missing_region_min_blob
        self.missing_region_dilate = missing_region_dilate
        self.missing_region_method = missing_region_method
        self.missing_region_max_frac = missing_region_max_frac
        self.missing_region_deg = missing_region_deg
        self.missing_region_iters = missing_region_iters
        self.missing_region_feather = missing_region_feather
        # 条状缺失干扰增广：strip_masks 为 (K,H,W) bool 的真实缺测掩码库
        self.strip_dropout_prob = float(strip_dropout_prob or 0.0)
        self.strip_masks = strip_masks
        # 可作为「跨通道参考」的通道索引白名单（PMW 的 0 是合法值，须排除）
        self.missing_region_ref_idx = [
            i for i, nm in enumerate(self.channels)
            if nm in self.missing_region_channels]
        self.return_corrected_flag = return_corrected_flag

        # 标签：CSV 模式直接由 frame_labels 传入（与 indices 顺序对齐）；
        #       否则从 h5 的 info 表读取（数据集自带标注，无需人工标注）。
        if frame_labels is not None:
            self.labels = np.asarray(frame_labels, dtype=np.float64)
            if indices is None:
                indices = list(range(len(self.labels)))
        else:
            info = load_info(h5_path)
            labels = info[target].astype(float).to_numpy()
            if label_unit == "ms":
                labels = labels * KNOT_TO_MS  # 原始为 knot
            if indices is None:
                indices = list(range(len(labels)))
            self.labels = labels[self.indices]
        self.indices = list(indices)
        self._hf = None  # 延迟打开，兼容多进程

        # ---- 离线 U-Net 预修复的 IR1：载入 (K,H,W) 修复值 + 行号映射 ----
        self.ir1_repaired_path = ir1_repaired_path
        self._ir1_rep = None
        self._ir1_rep_map = None
        if ir1_repaired_path and os.path.exists(ir1_repaired_path):
            self._ir1_rep = np.load(ir1_repaired_path)          # (K,H,W) float32
            idx_p = ir1_repaired_path.replace(".npy", "_idx.npy")
            if os.path.exists(idx_p):
                ridx = np.load(idx_p).astype(np.int64)
            else:                                               # 无索引则按顺序对齐
                ridx = np.arange(len(self._ir1_rep), dtype=np.int64)
            self._ir1_rep_map = {int(v): j for j, v in enumerate(ridx)}

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
        # matrix[real_idx] -> (H, W, C_h5)
        raw = self._matrix[real_idx]
        if raw.shape[-1] == len(self.channels):
            # 预处理过的紧凑 h5 已是目标通道（如 3 通道），直接取
            arr = raw[..., :len(self.channels)]
        else:
            arr = raw[..., self.ch_idx]                         # (H, W, C)
        arr = np.ascontiguousarray(arr.transpose(2, 0, 1)).astype(np.float32)  # (C, H, W)

        # 0) 离线 U-Net 预修复：命中缓存则用修复后的 IR1 整帧替换第 0 通道。
        #    只在「真实缺测帧」上生效（干净帧不在缓存），故对干净帧零扰动。
        if self._ir1_rep is not None:
            j = self._ir1_rep_map.get(int(real_idx))
            if j is not None and arr.shape[0] >= 1:
                arr[0] = np.asarray(self._ir1_rep[j], dtype=np.float32)

        # 1) NaN 处理
        if self.nan_mode == "interp":
            arr = _fill_nan_interp(arr)
        else:  # zero
            arr = np.nan_to_num(arr, nan=0.0, copy=False)

        # 1.5) 频带/区域缺失检测与修正（训练途中逐样本修正）
        was_corrected = False
        if self.missing_band == "interp":
            # ---- 局部「区域」缺失：用空间插值(inpainting)就地填补 ----
            # 仅对配置通道(默认 IR1/WV)生效；PMW 的 0 是合法值，不参与。
            # 缺测判定：精确等于 sentinel(0) 或低于物理下界 low_thr(捕获涂抹过渡带)。
            from .band_health import detect_missing_region, inpaint_regions
            n_ch = len(self.channels)
            lt_list = []
            for ci, ch_name in enumerate(self.channels):
                if ch_name not in self.missing_region_channels:
                    lt_list.append(-999.0)          # 该通道不参与
                else:
                    v = None
                    if self.missing_region_low_thr is not None:
                        try:
                            v = float(self.missing_region_low_thr[ci])
                        except (TypeError, IndexError):
                            v = None
                    lt_list.append(v if (v is not None and v > -900) else -999.0)
            mask = detect_missing_region(
                arr, sentinel=self.missing_region_sentinel,
                low_thr=lt_list, min_blob=self.missing_region_min_blob,
                dilate=self.missing_region_dilate)
            if mask.any():
                was_corrected = True
                arr = inpaint_regions(
                    arr, mask, method=self.missing_region_method,
                    fill_values=self.fill_values,
                    max_frac=self.missing_region_max_frac,
                    deg=self.missing_region_deg,
                    iters=self.missing_region_iters,
                    ref_idx=self.missing_region_ref_idx,
                    feather=self.missing_region_feather)
        elif self.missing_band != "none":
            # ---- 整通道缺失(老逻辑)：none 之外的 zero/mean 策略 ----
            from .band_health import detect_missing_channels, correct_missing
            mm = detect_missing_channels(arr, mode=self.missing_detect)
            if mm.any():
                was_corrected = True
                arr = correct_missing(arr, mm, strategy=self.missing_band,
                                      fill_values=self.fill_values)

        # 1.6) 训练期「条状缺失」干扰增广
        # 从「真实检测到的 IR1 条状缺测掩码库」随机取一个贴到本帧上（置哨兵 0），
        # 模拟卫星条带缺测，迫使模型对该形态鲁棒 —— 用于对照「无修正」基线。
        # 仅训练集生效；随机水平镜像 + 纵向平移以获得位置多样性。
        was_corrupted = False
        if (self.split == "train" and self.strip_dropout_prob > 0
                and self.strip_masks is not None and len(self.strip_masks)
                and random.random() < self.strip_dropout_prob):
            m = np.asarray(self.strip_masks[random.randrange(len(self.strip_masks))])
            m = m.astype(bool)
            if m.shape == arr.shape[1:]:
                if random.random() < 0.5:
                    m = m[:, ::-1]
                shift = random.randint(-arr.shape[1] // 4, arr.shape[1] // 4)
                if shift:
                    m = np.roll(m, shift, axis=0)
                arr[0][m] = 0.0
                was_corrupted = True

        x = torch.from_numpy(arr.astype(np.float32))   # 防止插值/修正路径 upcast 成 float64

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

        y = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        if self.return_corrected_flag:
            return x, y, bool(was_corrected or was_corrupted)
        return x, y

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
        # pandas fixed 布局：block0(数值列) + block1(字符串列) + axis1(行索引)
        if "block0_values" in g:
            cols = [_decode(c) for c in np.atleast_1d(g["block0_items"][()])]
            vals = np.atleast_2d(g["block0_values"][()])
            df = pd.DataFrame(vals, columns=cols)
            if "block1_items" in g:
                scols = [_decode(c) for c in np.atleast_1d(g["block1_items"][()])]
                svals = g["block1_values"][()]
                try:
                    if getattr(svals, "ndim", 0) == 2:
                        for j, sc in enumerate(scols):
                            df[sc] = svals[:, j]
                    else:  # 单列字符串常存为 1D object 数组
                        df[scols[0]] = np.array([_decode(v) for v in svals])
                except Exception:
                    pass
            if "axis1" in g and len(g["axis1"]) == len(df):
                df.index = g["axis1"][()]
            return df
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
def _split_by_storm(indices, labels, storm_ids, ratios, seed):
    """按风暴 ID 分组切分 train/val/test（防泄漏）。

    返回 (tr_i, tr_l, va_i, va_l, te_i, te_l)，其中 *i 为 frame 序号、
    *l 为对应标签，三者顺序一致。
    """
    rng = random.Random(seed)
    groups = {}
    for j, sid in enumerate(storm_ids):
        groups.setdefault(sid, []).append(j)
    keys = list(groups.keys())
    rng.shuffle(keys)
    r_tr, r_va = ratios[0], ratios[1]
    tr_i, tr_l, va_i, va_l, te_i, te_l = [], [], [], [], [], []
    total = len(keys)
    used = 0
    for k in keys:
        used += 1
        frac = used / total
        js = groups[k]
        if frac <= r_tr:
            tr_i += [indices[j] for j in js]; tr_l += [labels[j] for j in js]
        elif frac <= r_tr + r_va:
            va_i += [indices[j] for j in js]; va_l += [labels[j] for j in js]
        else:
            te_i += [indices[j] for j in js]; te_l += [labels[j] for j in js]
    return tr_i, tr_l, va_i, va_l, te_i, te_l


def _split_by_year(indices, labels, years, ratios, year_test=None):
    """按「台风年份」切分 train/val/test（时序/年份泛化，防泄露）。

    与 by_storm 的区别：by_storm 只保证「同一台风暴不跨集」，但训练/测试仍可能
    共享年份；by_year 进一步要求测试集是**完全没见过的年份**，等价于「向后预测未来
    年份」，既杜绝同类风暴的年份相关性泄漏，也更贴近业务外推场景。

    year_test : 显式指定测试年份（如 [2014, 2015, 2016]）；None 时按 ratios[2]
                从**最近的年份往前**累加，直到测试帧占比 >= ratios[2]（"约 8:2"）。
    ratios[1] > 0 时再从剩余（较早）年份的末端切出验证集，占比 >= ratios[1]。
    返回 (tr_i, tr_l, va_i, va_l, te_i, te_l)。
    """
    years = np.asarray([str(y) for y in years])
    uniq = sorted(set(years.tolist()))
    total = max(len(indices), 1)
    if year_test:
        test_years = set(str(y) for y in year_test)
    else:
        test_years, acc = set(), 0
        for y in reversed(uniq):
            if acc / total >= float(ratios[2]):
                break
            test_years.add(y)
            acc += int((years == y).sum())
    remaining = [y for y in uniq if y not in test_years]
    val_years = set()
    if len(ratios) > 1 and ratios[1] and float(ratios[1]) > 0:
        acc = 0
        for y in reversed(remaining):
            if acc / total >= float(ratios[1]):
                break
            val_years.add(y)
            acc += int((years == y).sum())
    tr_i, tr_l, va_i, va_l, te_i, te_l = [], [], [], [], [], []
    for j, y in enumerate(years):
        if y in test_years:
            te_i.append(indices[j]); te_l.append(labels[j])
        elif y in val_years:
            va_i.append(indices[j]); va_l.append(labels[j])
        else:
            tr_i.append(indices[j]); tr_l.append(labels[j])
    return tr_i, tr_l, va_i, va_l, te_i, te_l


def _damage_low_thr_list(channels, tcfg):
    """构造与 channels 顺序对齐的 low_thr 列表（<=-900 表示该通道不参与区域检测）。"""
    mr_ch = tcfg.get("missing_region_channels", ["IR1", "WV"])
    mr_lt = tcfg.get("missing_region_low_thr", None)
    out = []
    for ci, name in enumerate(channels):
        if name not in mr_ch:
            out.append(-999.0)
        else:
            v = None
            if mr_lt is not None:
                try:
                    v = float(mr_lt[ci])
                except (TypeError, IndexError):
                    v = None
            out.append(v if (v is not None and v > -900) else -999.0)
    return out


def _apply_damage_filter(split_name, frame_indices, labels, h5_path, channels,
                         ch_idx, tcfg):
    """剔除「过度损坏、难以可靠修复」的样本（默认仅测试集）。

    判据（三者取 OR，任一满足即剔除）：
      * 「坏道」数 >= damage_filter_bad_channels（默认 2）
        坏道定义由 damage_filter_bad_definition 决定：
            any   = 有缺测的通道数（整通道缺失 或 区域掩码非空）【默认】
            whole = 整通道缺失的通道数
      * 全图缺测面积占比   >= damage_filter_image_frac（默认 0.5）
      * 单通道缺测面积占比 >= damage_filter_channel_frac（默认 2/3）
    返回 (new_indices, new_labels, info)。
    """
    from .band_health import scan_damage
    n_in = len(frame_indices)
    if n_in == 0:
        return list(frame_indices), labels, {"split": split_name, "n_in": 0,
                                             "n_dropped": 0}
    n_damaged, n_whole, img_frac, max_frac = scan_damage(
        h5_path, frame_indices, ch_idx,
        sentinel=tcfg.get("missing_region_sentinel", 0.0),
        low_thr_list=_damage_low_thr_list(channels, tcfg),
        min_blob=tcfg.get("missing_region_min_blob", 20),
        dilate=tcfg.get("missing_region_dilate", 1),
        missing_detect=tcfg.get("missing_detect", "constant"))
    bad_def = tcfg.get("damage_filter_bad_definition", "any")
    bad_cnt = n_whole if bad_def == "whole" else n_damaged
    bad_thr = tcfg.get("damage_filter_bad_channels", 2)
    img_thr = tcfg.get("damage_filter_image_frac", 0.5)
    ch_thr = tcfg.get("damage_filter_channel_frac", 2.0 / 3.0)
    BIG = 1e9  # 阈值置 None 时禁用该判据
    c1 = bad_cnt >= (bad_thr if bad_thr is not None else BIG)
    c2 = img_frac >= (img_thr if img_thr is not None else BIG)
    c3 = max_frac >= (ch_thr if ch_thr is not None else BIG)
    drop = c1 | c2 | c3
    keep = ~drop
    new_i = [int(frame_indices[j]) for j in range(n_in) if keep[j]]
    new_l = ([float(labels[j]) for j in range(n_in) if keep[j]]
             if labels is not None else None)
    info = {
        "split": split_name, "n_in": n_in, "n_dropped": int(drop.sum()),
        "by_bad_channels": int(c1.sum()), "by_image_frac": int(c2.sum()),
        "by_channel_frac": int(c3.sum()), "bad_definition": bad_def,
    }
    return new_i, new_l, info


def create_tcir_datasets(cfg):
    """按配置构建 train/val/test，返回 (dict, meta)。

    两种标签来源：
      label_source=h5  : 从 h5 的 info 表读标签（需 pytables 或 h5py 回退）
      label_source=csv : 从 csv_path 读标签（推荐，配合用户整理的 wpac_info.csv）
                         —— 用 index_col 定位 h5 帧，target_col 取强度，
                            storm_col 做按风暴防泄漏切分
    """
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
    # 划分种子：默认跟随 experiment.seed；若显式给出 data.tcir.split_seed，则固定划分。
    # 目的：多次训练取最优（train_3x）时，各 run 用同一验证集，val 指标才可比、择优选模才无偏。
    seed = tcfg.get("split_seed", None)
    if seed is None:
        seed = cfg.get("experiment.seed", 42)
    regions = tcfg.get("regions", None)          # 可选：只取某些区域
    storm_col = tcfg.get("storm_col", "ID")      # 按风暴 ID 切分防泄漏
    split_strategy = tcfg.get("split_strategy", "by_storm")

    # 频带缺失修正配置
    missing_band = tcfg.get("missing_band", "none")        # none|zero|mean|drop|interp
    missing_detect = tcfg.get("missing_detect", "constant")  # constant|var
    missing_mask_path = tcfg.get("missing_mask_path", "logs/band_missing_mask.npz")
    # 局部区域缺失插值配置（missing_band=="interp" 时生效）
    missing_region_channels = tcfg.get("missing_region_channels", ["IR1", "WV"])
    missing_region_sentinel = tcfg.get("missing_region_sentinel", 0.0)
    missing_region_low_thr = tcfg.get("missing_region_low_thr", [150, 150, -999])
    missing_region_min_blob = tcfg.get("missing_region_min_blob", 20)
    missing_region_dilate = tcfg.get("missing_region_dilate", 1)
    missing_region_method = tcfg.get("missing_region_method", "griddata")
    missing_region_max_frac = tcfg.get("missing_region_max_frac", 1.0)
    missing_region_deg = tcfg.get("missing_region_deg", 2)
    missing_region_iters = tcfg.get("missing_region_iters", 100)
    missing_region_feather = tcfg.get("missing_region_feather", 0)

    # 离线 U-Net 预修复的 IR1 缓存（可选；见 scripts/build_unet_repaired_ir1.py）
    ir1_repaired_path = tcfg.get("ir1_repaired_path", None)
    if ir1_repaired_path and root and not os.path.isabs(ir1_repaired_path):
        cand = os.path.join(root, ir1_repaired_path)
        if os.path.exists(cand):
            ir1_repaired_path = cand
    if ir1_repaired_path:
        if os.path.exists(ir1_repaired_path):
            print(f"[ir1-repaired] 启用离线 U-Net 修复输入：{ir1_repaired_path}")
        else:
            print(f"[ir1-repaired] 警告：找不到 {ir1_repaired_path}，忽略（按未修复处理）")
            ir1_repaired_path = None

    # 过度损坏样本剔除（不可靠修复 -> 直接剔除，而非硬修）：默认仅作用于测试集
    damage_filter = bool(tcfg.get("damage_filter", False))
    damage_splits = tcfg.get("damage_filter_splits", ["test"]) or []
    ch_idx = [CHANNELS.index(c) for c in channels]
    damage_log = []

    aug = cfg.to_dict().get("augmentation", {}).get("train", {})

    # 训练期「条状缺失」干扰增广：载入真实检测到的缺测掩码库（见 scripts/build_strip_masks.py）
    strip_dropout_prob = float(tcfg.get("strip_dropout_prob", 0.0) or 0.0)
    strip_masks = None
    strip_masks_path = tcfg.get("strip_masks_path", None)
    if strip_dropout_prob > 0 and strip_masks_path:
        smp = strip_masks_path
        if root and not os.path.isabs(smp):
            cand = os.path.join(root, smp)
            if os.path.exists(cand):
                smp = cand
        if os.path.exists(smp):
            with np.load(smp) as z:
                strip_masks = z["masks"].astype(bool)
            print(f"[strip-dropout] 载入 {strip_masks.shape[0]} 个条状缺测掩码: {smp}"
                  f"（应用概率 {strip_dropout_prob}）")
        else:
            raise FileNotFoundError(
                f"strip_dropout_prob>0 但找不到掩码库 {smp}；"
                f"请先运行 scripts/build_strip_masks.py 生成。")

    common = dict(
        h5_path=h5_path, channels=channels, target=target, label_unit=label_unit,
        nan_mode=nan_mode, normalize=normalize, norm_mean=norm_mean,
        norm_std=norm_std, resize=resize,
        missing_band=missing_band, missing_detect=missing_detect,
        fill_values=norm_mean,
        missing_region_channels=missing_region_channels,
        missing_region_sentinel=missing_region_sentinel,
        missing_region_low_thr=missing_region_low_thr,
        missing_region_min_blob=missing_region_min_blob,
        missing_region_dilate=missing_region_dilate,
        missing_region_method=missing_region_method,
        missing_region_max_frac=missing_region_max_frac,
        missing_region_deg=missing_region_deg,
        missing_region_iters=missing_region_iters,
        missing_region_feather=missing_region_feather,
        strip_dropout_prob=strip_dropout_prob,
        strip_masks=strip_masks,
        ir1_repaired_path=ir1_repaired_path,
    )

    # drop 策略：剔除含缺失通道的样本（掩码行号与 h5 帧顺序一致）
    bad_rows = set()
    n_dropped_missing = 0
    if missing_band == "drop":
        from .band_health import ensure_band_mask
        mask = ensure_band_mask(h5_path, channels, missing_mask_path,
                                mode=missing_detect)
        bad_rows = set(int(i) for i in np.where(mask.any(axis=1))[0])
        n_dropped_missing = len(bad_rows)

    # --------------------------- CSV 驱动模式（推荐） ---------------------------
    csv_path = tcfg.get("csv_path", None)
    if csv_path and tcfg.get("label_source", "h5") == "csv":
        df = pd.read_csv(csv_path)
        idx_col = tcfg.get("index_col", "matrix_index")
        id_col = storm_col
        target_col = tcfg.get("target_col", "Vmax")
        if regions:
            rcol = tcfg.get("region_col", "data_set")
            df = df[df[rcol].isin(regions)].reset_index(drop=True)
        frame_indices = df[idx_col].to_numpy().astype(int)
        labels = df[target_col].astype(float).to_numpy()
        if label_unit == "ms":
            labels = labels * KNOT_TO_MS
        storm_ids = df[id_col].to_numpy()

        if bad_rows:
            keep_b = np.array([int(i) not in bad_rows for i in frame_indices])
            frame_indices = frame_indices[keep_b]
            labels = labels[keep_b]
            storm_ids = storm_ids[keep_b]

        if split_strategy == "by_storm":
            tr_i, tr_l, va_i, va_l, te_i, te_l = _split_by_storm(
                frame_indices, labels, storm_ids, ratios, seed)
        elif split_strategy == "by_year":
            # 年份来源：优先 CSV 的 year_col，否则从风暴 ID 前 4 位取（如 200911W -> 2009）
            year_col = tcfg.get("year_col", None)
            if year_col and year_col in df.columns:
                years = df[year_col].astype(str).to_numpy()
                if bad_rows:
                    years = years[keep_b]
            else:
                years = np.array([str(s)[:4] for s in storm_ids])
            tr_i, tr_l, va_i, va_l, te_i, te_l = _split_by_year(
                frame_indices, labels, years, ratios,
                year_test=tcfg.get("split_year_test", None))
        else:
            rng = random.Random(seed)
            perm = list(range(len(frame_indices)))
            rng.shuffle(perm)
            n = len(perm)
            n_tr = int(round(n * ratios[0]))
            n_va = int(round(n * ratios[1]))
            tr_i = [frame_indices[j] for j in perm[:n_tr]]; tr_l = [labels[j] for j in perm[:n_tr]]
            va_i = [frame_indices[j] for j in perm[n_tr:n_tr + n_va]]; va_l = [labels[j] for j in perm[n_tr:n_tr + n_va]]
            te_i = [frame_indices[j] for j in perm[n_tr + n_va:]]; te_l = [labels[j] for j in perm[n_tr + n_va:]]

        # 仅用「干净图」训练（train 侧过滤掉含缺测的帧；val/test 不变）
        clean_log = None
        if bool(tcfg.get("train_clean_only", False)) and len(tr_i) > 0:
            from .band_health import scan_damage
            nd, nw, imf, mxf = scan_damage(
                h5_path, tr_i, ch_idx,
                sentinel=missing_region_sentinel,
                low_thr_list=_damage_low_thr_list(channels, tcfg),
                min_blob=missing_region_min_blob,
                dilate=missing_region_dilate,
                missing_detect=missing_detect)
            keep_c = (nd == 0)
            n_before = len(tr_i)
            tr_i = [int(tr_i[j]) for j in range(n_before) if keep_c[j]]
            tr_l = [float(tr_l[j]) for j in range(n_before) if keep_c[j]]
            clean_log = {"n_before": n_before, "n_after": len(tr_i),
                         "n_dropped": n_before - len(tr_i)}
            print(f"[train-clean-only] 训练集 {n_before} -> {len(tr_i)}"
                  f"（剔除 {clean_log['n_dropped']} 张含缺测帧）")

        # 过度损坏样本剔除（默认仅 test）
        if damage_filter:
            if "train" in damage_splits:
                tr_i, tr_l, di = _apply_damage_filter(
                    "train", tr_i, tr_l, h5_path, channels, ch_idx, tcfg)
                damage_log.append(di)
            if "val" in damage_splits:
                va_i, va_l, di = _apply_damage_filter(
                    "val", va_i, va_l, h5_path, channels, ch_idx, tcfg)
                damage_log.append(di)
            if "test" in damage_splits:
                te_i, te_l, di = _apply_damage_filter(
                    "test", te_i, te_l, h5_path, channels, ch_idx, tcfg)
                damage_log.append(di)

        datasets = {
            "train": TCIRDataset(indices=tr_i, frame_labels=tr_l, split="train", aug=aug, **common),
            "val":   TCIRDataset(indices=va_i, frame_labels=va_l, split="val", **common),
            "test":  TCIRDataset(indices=te_i, frame_labels=te_l, split="test", **common),
        }
        meta_t = target_col
        n_tr, n_va, n_te = len(tr_i), len(va_i), len(te_i)
    else:
        # ---------------------- 原始 h5-info 模式（需 pytables 或 h5py 回退） ----------------------
        info = load_info(h5_path)
        n = len(info)
        if regions:
            mask = info["region"].isin(regions).to_numpy()
            keep = np.where(mask)[0]
        else:
            keep = np.arange(n)

        if bad_rows:
            keep = np.array([int(i) not in bad_rows for i in keep])

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
            rng = random.Random(seed)
            idx_all = list(keep)
            rng.shuffle(idx_all)
            n_tr2 = int(round(len(idx_all) * ratios[0]))
            n_va2 = int(round(len(idx_all) * ratios[1]))
            train = idx_all[:n_tr2]
            val = idx_all[n_tr2:n_tr2 + n_va2]
            test = idx_all[n_tr2 + n_va2:]

        # 过度损坏样本剔除（默认仅 test）；h5 模式标签由 info 按 index 读取，故 here 传 None
        if damage_filter:
            if "train" in damage_splits:
                train, _, di = _apply_damage_filter(
                    "train", train, None, h5_path, channels, ch_idx, tcfg)
                damage_log.append(di)
            if "val" in damage_splits:
                val, _, di = _apply_damage_filter(
                    "val", val, None, h5_path, channels, ch_idx, tcfg)
                damage_log.append(di)
            if "test" in damage_splits:
                test, _, di = _apply_damage_filter(
                    "test", test, None, h5_path, channels, ch_idx, tcfg)
                damage_log.append(di)

        datasets = {
            "train": TCIRDataset(indices=train, split="train", aug=aug, **common),
            "val": TCIRDataset(indices=val, split="val", **common),
            "test": TCIRDataset(indices=test, split="test", **common),
        }
        meta_t = target
        n_tr, n_va, n_te = len(train), len(val), len(test)

    meta = {
        "task": "regression",
        "class_names": [],
        "num_classes": 1,
        "in_channels": len(channels),
        "target": meta_t,
        "label_unit": label_unit,
        "n_train": n_tr,
        "n_val": n_va,
        "n_test": n_te,
        "n_dropped_missing": n_dropped_missing,
        "missing_band": missing_band,
        "train_clean_only": clean_log,
        "strip_dropout_prob": strip_dropout_prob,
        "ir1_repaired_path": ir1_repaired_path,
        "damage_filter": {"enabled": damage_filter, "splits": list(damage_splits),
                          "log": damage_log},
        "label_source": "csv" if (csv_path and tcfg.get("label_source", "h5") == "csv") else "h5",
    }
    if damage_filter and damage_log:
        for di in damage_log:
            if di.get("n_dropped"):
                print(f"[damage-filter] {di['split']}: 剔除 {di['n_dropped']}/{di['n_in']} 张"
                      f"（坏道数 {di['by_bad_channels']} / 全图面积 {di['by_image_frac']} / "
                      f"单道面积 {di['by_channel_frac']}；口径={di['bad_definition']}）")
    return datasets, meta


# --------------------------------------------------------------------------- #
# 逐通道统计（可选：用于归一化的 mean/std 计算）
# --------------------------------------------------------------------------- #
def compute_tcir_stats(h5_path, channels=("IR1", "WV", "PMW"), sample_every=1,
                       frame_indices=None, batch=2000):
    """遍历 matrix 计算逐通道均值/标准差，返回 (mean, std) 列表。

    frame_indices : 只统计这些帧（如 WPAC 子集，可由 CSV 的 matrix_index 给出）。
                    注意：非连续索引会触发逐帧读取，在挂载盘（如 WSL DrvFS）上极慢，
                    仅对子集统计时使用。
    batch         : 批量大小。frame_indices 为 None 时按【连续分块】(单次 hyperslab)
                    读取，远快于逐帧/乱序读取，适合在挂载盘上做全量统计。
    """
    ch_idx = [CHANNELS.index(c) for c in channels]
    sums = np.zeros(len(ch_idx), dtype=np.float64)
    sqs = np.zeros(len(ch_idx), dtype=np.float64)
    counts = np.zeros(len(ch_idx), dtype=np.int64)
    with h5py.File(h5_path, "r") as hf:
        mat = hf["matrix"]
        N = mat.shape[0]
        if frame_indices is None:
            # 全量统计：连续分块读取（单次 hyperslab，避免海量随机读）
            step = max(1, batch) * sample_every
            for s in range(0, N, step):
                e = min(s + step, N)
                x = mat[s:e][..., ch_idx].astype(np.float64)   # (B, H, W, C)
                x = np.nan_to_num(x, nan=0.0)
                sums += x.sum(axis=(0, 1, 2))
                sqs += (x ** 2).sum(axis=(0, 1, 2))
                counts += x.shape[0] * x.shape[1] * x.shape[2]
        else:
            for s in range(0, len(frame_indices), batch):
                idx = frame_indices[s:s + batch]
                x = mat[idx][..., ch_idx].astype(np.float64)
                x = np.nan_to_num(x, nan=0.0)
                sums += x.sum(axis=(0, 1, 2))
                sqs += (x ** 2).sum(axis=(0, 1, 2))
                counts += x.shape[0] * x.shape[1] * x.shape[2]
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
