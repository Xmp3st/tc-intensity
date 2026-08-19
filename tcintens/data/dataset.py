# -*- coding: utf-8 -*-
"""
dataset.py —— 热带气旋图像数据集

双模式数据组织：
  csv    —— 单个 CSV（label_file），含 image_path 与标签列，按 split 比例分层切分
  folder —— folder_root 下以「类别名」命名的子文件夹，自动分层切分

双任务：
  classification —— 6 级 CMA 强度（TD/TS/STS/TY/STY/SuperTY）
  regression    —— 连续风速（m/s）

对外主入口：create_datasets(cfg) -> dict{'train','val','test': Dataset, 'meta': ...}
"""
import csv
import os
import random
from typing import Dict, List, Tuple

from PIL import Image
from torch.utils.data import Dataset


class TCImageDataset(Dataset):
    """(path, target) 列表构成的通用数据集。

    target: 分类时为 int 类别索引；回归时为 float 风速。
    """

    def __init__(self, samples, task, transform=None):
        self.samples = samples
        self.task = task
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.task == "regression":
            import torch
            return img, torch.tensor(float(target), dtype=torch.float32)
        return img, int(target)


# --------------------------------------------------------------------------- #
# 路径解析：优先原路径，其次拼 root_dir
# --------------------------------------------------------------------------- #
def _resolve(path: str, root_dir: str) -> str:
    if os.path.exists(path):
        return path
    cand = os.path.join(root_dir, path)
    if os.path.exists(cand):
        return cand
    return path  # 实在找不到就返回原值，加载时再报错


# --------------------------------------------------------------------------- #
# 分层切分（保持各子集类别比例一致）
# --------------------------------------------------------------------------- #
def _stratified_split(samples, ratios, seed):
    """samples: List[(path, target)]。按 target 分层，按比例切分。
    返回 [(train, val, test)] 三个子集。
    """
    rng = random.Random(seed)
    by_class: Dict = {}
    for s in samples:
        by_class.setdefault(s[1], []).append(s)
    train, val, test = [], [], []
    r_train, r_val = ratios[0], ratios[1]
    for key, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * r_train))
        n_val = int(round(n * r_val))
        train += items[:n_train]
        val += items[n_train:n_train + n_val]
        test += items[n_train + n_val:]
    return train, val, test


# --------------------------------------------------------------------------- #
# CSV 模式
# --------------------------------------------------------------------------- #
def _load_csv(cfg) -> Tuple[List, List[str], str]:
    label_file = _resolve(cfg.get("data.label_file"), cfg.get("data.root_dir", ""))
    task = cfg.get("data.task", "classification")
    class_names = cfg.get("data.class_names", [])
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    rows = []
    with open(label_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = _resolve(row["image_path"].strip(), cfg.get("data.root_dir", ""))
            if task == "classification":
                lab = row.get("label", row.get("intensity", "")).strip()
                target = class_to_idx[lab] if lab in class_to_idx else int(lab)
            else:
                col = cfg.get("data.regression_target", "wind_speed_ms")
                target = float(row[col])
            rows.append((path, target))

    return rows, class_names, task


# --------------------------------------------------------------------------- #
# 文件夹模式
# --------------------------------------------------------------------------- #
def _load_folder(cfg) -> Tuple[List, List[str], str]:
    folder_root = cfg.get("data.folder_root", "data/processed")
    class_names = cfg.get("data.class_names", [])
    task = cfg.get("data.task", "classification")
    rows = []
    for idx, name in enumerate(class_names):
        d = os.path.join(folder_root, name)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
                rows.append((os.path.join(d, fn), idx))
    return rows, class_names, task


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def create_datasets(cfg):
    """根据配置构建 train/val/test 数据集，返回字典。"""
    from .transforms import build_transforms

    data_cfg = cfg.to_dict()["data"]
    aug_cfg = cfg.to_dict()["augmentation"]
    image_size = data_cfg["image_size"]
    mean = data_cfg["mean"]
    std = data_cfg["std"]
    task = data_cfg["task"]
    seed = cfg.get("experiment.seed", 42)

    mode = data_cfg.get("mode", "csv")
    if mode == "csv":
        rows, class_names, task = _load_csv(cfg)
    else:
        rows, class_names, task = _load_folder(cfg)

    if not rows:
        raise ValueError("未读到任何样本，请检查 data 配置（路径/列名/文件夹）。")

    ratios = data_cfg.get("split", [0.7, 0.15, 0.15])
    train_s, val_s, test_s = _stratified_split(rows, ratios, seed)

    t_train = build_transforms("train", image_size, aug_cfg, mean, std)
    t_eval = build_transforms("val", image_size, aug_cfg, mean, std)

    datasets = {
        "train": TCImageDataset(train_s, task, t_train),
        "val": TCImageDataset(val_s, task, t_eval),
        "test": TCImageDataset(test_s, task, t_eval),
    }
    meta = {
        "task": task,
        "class_names": class_names,
        "num_classes": len(class_names),
        "n_train": len(train_s),
        "n_val": len(val_s),
        "n_test": len(test_s),
    }
    return datasets, meta
