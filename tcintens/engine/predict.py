# -*- coding: utf-8 -*-
"""engine/predict.py —— TCIR 适配的推理入口

支持三种输入：
  1) 单张 .npy:      python predict.py --image frame.npy -c configs/tcir_wpac_train.yaml
  2) 文件夹 .npy:    python predict.py --folder ./frames/ -c ... --out preds.csv
  3) 从 TCIR h5 取帧: python predict.py --h5 data/wpac_96.h5 --indices 10260,9603 -c ... --out preds.csv

输入张量约定（与训练完全一致）：
  - .npy 应为 (H, W, C)，C 与配置 channels 数一致，且通道顺序与 data.tcir.channels 相同；
    也兼容 (C, H, W) 自动转置。若是 4 通道则按 [IR1,WV,VIS,PMW] 截取配置通道。
  - h5 的 matrix 为 (..., 4) [IR1,WV,VIS,PMW] 或紧凑 (..., 3) [IR1,WV,PMW]，按配置通道截取。
归一化：先 (x-mean)/std（用 data.tcir.norm_mean/std），再按需下采样到 data.tcir.resize。
输出 Vmax，单位 knots；同时给出 m/s 换算。
"""
import csv
import os
from typing import List, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from ..data.tcir import CHANNELS
from ..models.tc_model import build_model
from ..utils.config import Config
from ..utils.misc import resolve_device
from .common import load_checkpoint

KNOT_TO_MS = 0.514444


def _load_array(path: str) -> np.ndarray:
    a = np.load(path)
    if a.ndim == 3:
        # 兼容 (C,H,W) 与 (H,W,C)
        if a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
            a = np.transpose(a, (1, 2, 0))
    return a.astype(np.float32)


def _preprocess(arr: np.ndarray, channels, norm_mean, norm_std, resize):
    """arr: (H, W, C_in) -> 归一化+下采样后的 (C,H,W) 张量。"""
    C_in = arr.shape[-1]
    if C_in != len(channels):
        if C_in == 4:
            idx = [CHANNELS.index(c) for c in channels]
            arr = arr[..., idx]
        else:
            raise ValueError(
                f"输入通道数 {C_in} 与配置通道数 {len(channels)} 不匹配，且非 4 通道")
    arr = np.nan_to_num(arr, nan=0.0).astype(np.float32)
    mean = np.array(norm_mean, dtype=np.float32).reshape(1, 1, -1)
    std = np.array(norm_std, dtype=np.float32).reshape(1, 1, -1)
    arr = (arr - mean) / std
    t = torch.from_numpy(arr.transpose(2, 0, 1))  # (C, H, W)
    if resize is not None and t.shape[-1] != resize:
        t = F.interpolate(t.unsqueeze(0), size=(resize, resize),
                          mode="bilinear", align_corners=False).squeeze(0)
    return t


def predict(cfg: Config, image: str = None, folder: str = None, h5: str = None,
            index: int = None, indices: str = None, out: str = None):
    d = cfg.to_dict()
    exp = d["experiment"]
    tcfg = d["data"]["tcir"]
    device = resolve_device(d["device"].get("mode", "auto"))
    task = d["data"].get("task", "regression")

    channels = tcfg.get("channels", ["IR1", "WV", "PMW"])
    norm_mean = tcfg.get("norm_mean")
    norm_std = tcfg.get("norm_std")
    resize = tcfg.get("resize")
    if norm_mean is None or norm_std is None:
        raise ValueError("配置缺少 data.tcir.norm_mean / norm_std，无法做归一化")

    # ---------- 收集 (source_name, array) ----------
    items: List[tuple] = []
    if image:
        items.append((os.path.basename(image), _load_array(image)))
    elif folder:
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(".npy"):
                items.append((f, _load_array(os.path.join(folder, f))))
    elif h5:
        with h5py.File(h5, "r") as hf:
            mat = hf["matrix"]
            idxs: List[int] = []
            if index is not None:
                idxs.append(int(index))
            if indices:
                idxs += [int(x) for x in indices.split(",") if x.strip() != ""]
            if not idxs:
                raise ValueError("必须提供 --index 或 --indices")
            for i in idxs:
                items.append((f"matrix[{i}]", np.asarray(mat[i])))
    else:
        raise ValueError("必须指定 --image / --folder / (--h5 + --index|--indices)")

    # ---------- 预处理成 batch ----------
    tensors = [_preprocess(a, channels, norm_mean, norm_std, resize) for _, a in items]
    x = torch.stack(tensors, 0).to(device)

    # ---------- 模型 + 权重 ----------
    model = build_model(cfg).to(device)
    ckpt_path = os.path.join(exp.get("output_dir", "outputs"), exp.get("name", "exp"),
                             "checkpoints", "best.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到最佳权重: {ckpt_path}，请先训练。")
    load_checkpoint(ckpt_path, model, map_location=device)
    model.eval()

    with torch.no_grad():
        out_t = model(x)
        if task == "regression":
            vals = out_t.cpu().numpy().reshape(-1).astype(float)
        else:
            probs = torch.softmax(out_t, 1).cpu().numpy()
            cls = np.argmax(probs, 1)
            vals = cls.astype(float)

    # ---------- 输出 ----------
    rows = []
    print("=" * 52)
    if task == "regression":
        print(f"{'source':<22} {'Vmax(kn)':>10} {'Vmax(m/s)':>10}")
        print("-" * 52)
        for (name, _), v in zip(items, vals):
            kn = float(v)
            ms = kn * KNOT_TO_MS
            print(f"{name:<22} {kn:>10.2f} {ms:>10.2f}")
            rows.append((name, round(kn, 3), round(ms, 3)))
        print("=" * 52)
    else:
        names = d["data"].get("class_names", [])
        print(f"{'source':<22} {'class':<14} {'prob':>8}")
        print("-" * 52)
        for (name, _), c, p in zip(items, cls, probs.max(1)):
            print(f"{name:<22} {names[int(c)]:<14} {float(p):>8.3f}")
            rows.append((name, names[int(c)], round(float(p), 4)))

    if out:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if task == "regression":
                w.writerow(["source", "Vmax_kn", "Vmax_ms"])
                for name, kn, ms in rows:
                    w.writerow([name, kn, ms])
            else:
                w.writerow(["source", "class", "prob"])
                for name, c, p in rows:
                    w.writerow([name, c, p])
        print(f"\n预测结果已保存: {out}")
    return rows
