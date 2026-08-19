# -*- coding: utf-8 -*-
"""
engine/common.py —— 训练引擎的通用构件

集中构造：优化器 / 学习率调度器 / 损失函数 / checkpoint 读写。
所有选择都由配置驱动，方便调参。
"""
import os
import torch


# --------------------------------------------------------------------------- #
# 优化器
# --------------------------------------------------------------------------- #
def build_optimizer(model, cfg) -> torch.optim.Optimizer:
    t = cfg.to_dict()["train"]
    lr = t.get("lr", 3e-4)
    wd = t.get("weight_decay", 1e-2)
    params = [p for p in model.parameters() if p.requires_grad]
    name = t.get("optimizer", "adamw").lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    # default adamw
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)


# --------------------------------------------------------------------------- #
# 学习率调度器
# --------------------------------------------------------------------------- #
def build_scheduler(optimizer, cfg):
    t = cfg.to_dict()["train"]
    name = t.get("scheduler", "cosine").lower()
    epochs = t.get("epochs", 50)
    if name == "none" or not name:
        return None
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=t.get("lr_step_size", 10), gamma=t.get("lr_gamma", 0.1)
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=t.get("lr_gamma", 0.1), patience=5
        )
    # default cosine
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


# --------------------------------------------------------------------------- #
# 损失函数
# --------------------------------------------------------------------------- #
def build_loss(cfg):
    t = cfg.to_dict()["train"]
    data_cfg = cfg.to_dict()["data"]
    task = data_cfg.get("task", "classification")
    choice = t.get("loss", "auto")
    if task == "regression":
        if choice == "huber":
            return torch.nn.HuberLoss()
        return torch.nn.MSELoss()
    # classification
    if choice == "cross_entropy":
        return torch.nn.CrossEntropyLoss(label_smoothing=t.get("label_smoothing", 0.0))
    # auto
    return torch.nn.CrossEntropyLoss(label_smoothing=t.get("label_smoothing", 0.0))


# --------------------------------------------------------------------------- #
# Checkpoint
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str, model, optimizer, epoch, metrics, cfg, meta, is_best=False):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": cfg.to_dict(),
            "meta": meta,
            "is_best": is_best,
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer=None, map_location="cpu"):
    # 我们自己的 checkpoint（含 config / metrics 等 python 对象），信任来源，关闭 weights_only
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt
