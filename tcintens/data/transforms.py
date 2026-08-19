# -*- coding: utf-8 -*-
"""
transforms.py —— 根据配置构造 torchvision v2 数据增强流水线

配置入口见 configs/default.yaml 的 `augmentation.train / augmentation.val`。
把某项设为 false 即关闭该增强；normalize 控制是否做 ImageNet 归一化。
"""
from typing import List

import torch
import torchvision.transforms.v2 as T


def build_transforms(
    split: str,
    image_size: int,
    aug_cfg: dict,
    mean: List[float],
    std: List[float],
):
    """根据 split('train'|'val'|'test') 与配置返回变换组合。"""
    cfg = aug_cfg.get(split, {})
    ops = []

    if split == "train":
        if cfg.get("random_resized_crop", False):
            ops.append(T.RandomResizedCrop(image_size, antialias=True))
        else:
            ops.append(T.Resize(image_size, antialias=True))
        if cfg.get("horizontal_flip", False):
            ops.append(T.RandomHorizontalFlip())
        rot = cfg.get("rotate", 0)
        if rot:
            ops.append(T.RandomRotation(degrees=rot))
        cj = cfg.get("color_jitter", None)
        if cj and any(cj):
            ops.append(
                T.ColorJitter(
                    brightness=cj[0], contrast=cj[1], saturation=cj[2], hue=cj[3]
                )
            )
    else:
        # val / test：固定预处理，便于可复现评估
        resize = cfg.get("resize", 256)
        ops.append(T.Resize(resize, antialias=True))
        crop = cfg.get("center_crop", image_size)
        ops.append(T.CenterCrop(crop))

    ops.append(T.ToImage())          # PIL/numpy -> torch.Tensor (uint8)
    ops.append(T.ToDtype(torch.float32, scale=True))  # 归一化到 [0,1]
    if cfg.get("normalize", True):
        ops.append(T.Normalize(mean=mean, std=std))
    return T.Compose(ops)
