# -*- coding: utf-8 -*-
"""
engine/train.py —— 训练主循环

特性（全部由配置控制）：
  - 自动设备（CUDA/CPU）、固定种子
  - 混合精度（AMP）、梯度裁剪
  - 多种优化器 / 调度器 / 损失
  - 早停（early stopping）、Top-K checkpoint 保留
  - 同时输出控制台日志与 outputs/<exp>/train.log
  - 保存最终配置 config_used.yaml，保证可复现
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import create_datasets
from ..models.tc_model import build_model
from ..utils.config import Config
from ..utils.logger import get_logger
from ..utils.metrics import classification_metrics, regression_metrics
from ..utils.misc import count_parameters, humanize_count, resolve_device, set_seed
from .common import (
    build_loss,
    build_optimizer,
    build_scheduler,
    save_checkpoint,
)


def _run_one_epoch(model, loader, cfg, device, loss_fn, optimizer=None,
                   scaler=None, is_train=True):
    """跑一个 epoch，返回 (losses, y_true, y_pred)。"""
    task = cfg.to_dict()["data"].get("task", "classification")
    model.train(is_train)
    losses, yt, yp = [], [], []
    grad_clip = cfg.to_dict()["train"].get("grad_clip", 0.0)
    amp = scaler is not None

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            out = model(x)
            loss = loss_fn(out, y)
        if is_train:
            if amp:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        losses.append(loss.detach().float().item())
        if task == "regression":
            yt.append(y.detach().cpu().numpy())
            yp.append(out.detach().cpu().numpy())
        else:
            yt.append(y.detach().cpu().numpy())
            yp.append(out.detach().argmax(1).cpu().numpy())

    yt = np.concatenate(yt) if yt else np.array([])
    yp = np.concatenate(yp) if yp else np.array([])
    return float(np.mean(losses)), yt, yp


def train(cfg: Config):
    exp = cfg.to_dict()["experiment"]
    tr = cfg.to_dict()["train"]
    seed = exp.get("seed", 42)
    set_seed(seed)

    device = resolve_device(cfg.to_dict()["device"].get("mode", "auto"))
    logger = get_logger("tc", os.path.join(exp.get("output_dir", "outputs"),
                                           exp.get("name", "exp"), "train.log"))
    logger.info(f"device={device}  seed={seed}")
    cfg.save(os.path.join(exp.get("output_dir", "outputs"), exp.get("name", "exp"),
                          "config_used.yaml"))

    # 数据
    datasets, meta = create_datasets(cfg)
    logger.info(f"数据集: {meta}")
    nw = exp.get("num_workers", 4)
    train_dl = DataLoader(datasets["train"], batch_size=tr.get("batch_size", 32),
                          shuffle=True, num_workers=nw, pin_memory=(device.type == "cuda"))
    val_dl = DataLoader(datasets["val"], batch_size=tr.get("batch_size", 32),
                        shuffle=False, num_workers=nw, pin_memory=(device.type == "cuda"))

    # 模型
    model = build_model(cfg).to(device)
    logger.info(f"模型: {cfg.to_dict()['model'].get('backbone')}  可训练参数: "
                f"{humanize_count(count_parameters(model))}")

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    loss_fn = build_loss(cfg)
    use_amp = tr.get("mixed_precision", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda") if use_amp else None

    ckpt_dir = os.path.join(exp.get("output_dir", "outputs"), exp.get("name", "exp"),
                            "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 早停与 top-k 状态
    patience = tr.get("early_stopping_patience", 10)
    save_top_k = tr.get("save_top_k", 3)
    task = meta["task"]
    # monitor：分类用 macro_f1（越大越好），回归用 -rmse（越大越好）
    best_score = -1e9
    epochs_no_improve = 0
    topk = []  # list of (score, epoch)

    start_epoch = 0
    # 可选断点续训
    if exp.get("resume", False):
        last = os.path.join(ckpt_dir, "last.ckpt")
        if os.path.exists(last):
            ckpt = torch.load(last, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            logger.info(f"从 epoch {start_epoch} 续训")

    for epoch in range(start_epoch, tr.get("epochs", 50)):
        tr_loss, _, _ = _run_one_epoch(model, train_dl, cfg, device, loss_fn,
                                       optimizer, scaler, is_train=True)
        vl_loss, yt, yp = _run_one_epoch(model, val_dl, cfg, device, loss_fn,
                                         None, None, is_train=False)

        if task == "regression":
            m = regression_metrics(yt, yp)
            score = -m["rmse"]
            mon_text = f"val_rmse={m['rmse']:.4f} val_mae={m['mae']:.4f} r2={m['r2']:.4f}"
        else:
            m = classification_metrics(yt, yp, meta["class_names"])
            score = m["macro_f1"]
            mon_text = f"val_acc={m['accuracy']:.4f} val_macro_f1={m['macro_f1']:.4f}"

        logger.info(f"[epoch {epoch+1}/{tr.get('epochs',50)}] "
                    f"train_loss={tr_loss:.4f} val_loss={vl_loss:.4f} {mon_text}")

        # 调度器
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(vl_loss)
            else:
                scheduler.step()

        # 保存 last
        save_checkpoint(os.path.join(ckpt_dir, "last.ckpt"), model, optimizer,
                        epoch, m, cfg, meta)

        # 更新 best + top-k
        improved = score > best_score
        if improved:
            best_score = score
            epochs_no_improve = 0
            save_checkpoint(os.path.join(ckpt_dir, "best.ckpt"), model, optimizer,
                            epoch, m, cfg, meta, is_best=True)
        else:
            epochs_no_improve += 1

        topk.append((score, epoch))
        topk.sort(reverse=True)
        topk = topk[:save_top_k]
        # 持久化 top-k 文件
        for rank, (sc, ep) in enumerate(topk, 1):
            src = os.path.join(ckpt_dir, f"epoch_{ep}.ckpt")
            if os.path.exists(src):
                dst = os.path.join(ckpt_dir, f"top{rank}_epoch{ep}.ckpt")
                if not os.path.exists(dst):
                    import shutil
                    shutil.copy(src, dst)
        # 每个 epoch 也存一份 epoch_N.ckpt 供 top-k 引用
        save_checkpoint(os.path.join(ckpt_dir, f"epoch_{epoch}.ckpt"), model, optimizer,
                        epoch, m, cfg, meta)

        if epochs_no_improve >= patience:
            logger.info(f"早停：连续 {patience} 个 epoch 无提升。")
            break

    logger.info(f"训练结束。最佳 monitor 分数={best_score:.4f}")
    # 写出最终指标
    with open(os.path.join(os.path.dirname(ckpt_dir), "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"best_score": best_score, "task": task, "meta": meta}, f,
                  ensure_ascii=False, indent=2)
    return best_score
