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
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import create_datasets
from ..models.tc_model import build_model
from ..utils.config import Config
from ..utils.logger import get_logger
from ..utils.metrics import classification_metrics, regression_metrics
from ..utils.misc import count_parameters, humanize_count, resolve_device, set_seed
from ..utils.runlog import log_run
from .common import (
    build_loss,
    build_optimizer,
    build_scheduler,
    save_checkpoint,
)


def _fmt_secs(sec: float) -> str:
    """把秒数格式化成易读时长（心跳 / ETA 用）。"""
    sec = int(max(0, round(sec)))
    if sec < 60:
        return f"{sec}s"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _run_one_epoch(model, loader, cfg, device, loss_fn, optimizer=None,
                   scaler=None, is_train=True, log_every=0, epoch_tag="",
                   logger=None):
    """跑一个 epoch，返回 (losses, y_true, y_pred)。

    log_every > 0 时每 log_every 个 batch 打印一条进度（含已用时长）：
    若该行长时间不推进即说明训练卡住（GPU/磁盘 I/O stall），是最直接的防停滞信号。
    """
    task = cfg.to_dict()["data"].get("task", "classification")
    model.train(is_train)
    losses, yt, yp = [], [], []
    grad_clip = cfg.to_dict()["train"].get("grad_clip", 0.0)
    amp = scaler is not None
    n_batches = len(loader) if hasattr(loader, "__len__") else 0
    t_ep0 = time.time()

    for bi, (x, y) in enumerate(loader):
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
            yp.append(out.detach().cpu().argmax(1).numpy())

        # 批内进度心跳（只出现在训练期；若长时间不刷新 ⇒ 疑似卡住）
        if (is_train and log_every > 0 and n_batches and logger is not None
                and (bi + 1) % log_every == 0 and (bi + 1) < n_batches):
            logger.info(f"  [{epoch_tag}] batch {bi+1}/{n_batches} "
                        f"({(bi+1)/n_batches*100:.0f}%) loss={losses[-1]:.4f} "
                        f"已用 {_fmt_secs(time.time()-t_ep0)}")

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
    # 验证集可能为空（如 data.split=[0.8, 0.0, 0.2] 的纯 8:2 划分）：
    # 此时跳过每轮验证/早停，best.ckpt 取最后一轮。
    has_val = len(datasets["val"]) > 0
    val_dl = (DataLoader(datasets["val"], batch_size=tr.get("batch_size", 32),
                         shuffle=False, num_workers=nw,
                         pin_memory=(device.type == "cuda")) if has_val else None)
    if not has_val:
        logger.info("验证集为空(n_val=0)：跳过验证与早停，best.ckpt 取最后一轮。")

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
    # 检查点写入节流：默认每轮存 last（save_last_every 控制频率）；
    # save_top_k<=0 时不存 epoch_N.ckpt/top-k。长时训练经 9P 挂载写海量 195MB
    # 大文件会把 9P 服务器压挂，故提供这些开关降低写入压力。
    save_last = tr.get("save_last", True)
    save_last_every = max(1, int(tr.get("save_last_every", 1)))
    task = meta["task"]
    # monitor：分类用 macro_f1（越大越好），回归用 -rmse（越大越好）
    best_score = -1e9
    epochs_no_improve = 0
    topk = []  # list of (score, epoch)
    best_metrics = None
    best_epoch = -1

    start_epoch = 0
    # 可选断点续训
    if exp.get("resume", False):
        last = os.path.join(ckpt_dir, "last.ckpt")
        if os.path.exists(last):
            ckpt = torch.load(last, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            # 让学习率调度器追上已训练进度，否则续训首轮 LR 会跳回初始值
            if scheduler is not None and not isinstance(
                    scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                for _ in range(start_epoch):
                    scheduler.step()
            # 恢复最优记录：从既有 best.ckpt 读回 monitor 分数，
            # 否则续训首个 epoch（通常更差）会直接覆盖掉已有的 best.ckpt
            best_ckpt = os.path.join(ckpt_dir, "best.ckpt")
            if os.path.exists(best_ckpt):
                bc = torch.load(best_ckpt, map_location="cpu", weights_only=False)
                bm = bc.get("metrics", {}) or {}
                if task == "regression" and bm.get("rmse") is not None:
                    best_score = -float(bm["rmse"])
                elif task != "regression" and bm.get("macro_f1") is not None:
                    best_score = float(bm["macro_f1"])
                if bm:
                    best_metrics = bm
                    best_epoch = int(bc.get("epoch", -1))
            logger.info(f"从 epoch {start_epoch} 续训（已恢复最优 monitor 分数="
                        f"{best_score:.4f}，best_epoch={best_epoch + 1 if best_epoch >= 0 else 'NA'}）")

    total_epochs = tr.get("epochs", 50)
    # ---- 实时进度 / 心跳 / 停滞监测（防卡死）----
    #   log_every       : 每 N 个 batch 打一条进度（0=关闭）。长时间不推进 ⇒ 卡住。
    #   log_every_epochs: 每 N 轮打一条汇总心跳（含 ETA）。默认 5。
    #   stall_factor    : 单轮耗时 > 已完成轮耗时中位数 × 该系数 ⇒ 打「疑似停滞」告警。
    log_every = max(0, int(tr.get("log_every", 0)))
    hb_every = max(1, int(tr.get("log_every_epochs", 5)))
    stall_factor = float(tr.get("stall_factor", 3.0))
    epoch_times = []
    t_loop0 = time.time()
    logger.info(f"进度显示：批内每 {log_every or '关'} 个 batch 一条；"
                f"每 {hb_every} 轮一条心跳；停滞阈值 {stall_factor:.1f}× 中位耗时。")

    for epoch in range(start_epoch, total_epochs):
        t_ep = time.time()
        tr_loss, _, _ = _run_one_epoch(model, train_dl, cfg, device, loss_fn,
                                       optimizer, scaler, is_train=True,
                                       log_every=log_every,
                                       epoch_tag=f"epoch {epoch+1}/{total_epochs}",
                                       logger=logger)

        if val_dl is not None:
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
        else:
            # 无验证集：不做验证，monitor 分数不存在
            vl_loss, m, score = float("nan"), {"train_loss": round(tr_loss, 4)}, None
            mon_text = "no-val"

        logger.info(f"[epoch {epoch+1}/{total_epochs}] "
                    f"train_loss={tr_loss:.4f} val_loss={vl_loss:.4f} {mon_text}")

        # 调度器
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if val_dl is not None:
                    scheduler.step(vl_loss)
            else:
                scheduler.step()

        # 保存 last（按 save_last_every 节流，降低写入频率）
        if save_last and (epoch + 1) % save_last_every == 0:
            save_checkpoint(os.path.join(ckpt_dir, "last.ckpt"), model, optimizer,
                            epoch, m, cfg, meta)

        # 更新 best + top-k
        if val_dl is None:
            improved = (epoch + 1 >= total_epochs)     # 无验证集：best 取最后一轮
        else:
            improved = score > best_score
        if improved:
            if score is not None:
                best_score = score
            epochs_no_improve = 0
            best_metrics = m
            best_epoch = epoch
            save_checkpoint(os.path.join(ckpt_dir, "best.ckpt"), model, optimizer,
                            epoch, m, cfg, meta, is_best=True)
        else:
            epochs_no_improve += 1

        topk.append((0.0 if score is None else score, epoch))
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
        # 每个 epoch 也存一份 epoch_N.ckpt 供 top-k 引用（save_top_k<=0 时跳过）
        if save_top_k > 0:
            save_checkpoint(os.path.join(ckpt_dir, f"epoch_{epoch}.ckpt"), model, optimizer,
                            epoch, m, cfg, meta)

        # ---- 计时 · 停滞监测 · 定期心跳汇总（防停滞）----
        ep_time = time.time() - t_ep
        epoch_times.append(ep_time)
        if len(epoch_times) >= 3:
            med = float(np.median(epoch_times[:-1]))
            if med > 0 and ep_time > stall_factor * med:
                logger.warning(
                    f"[疑似停滞] 第 {epoch+1} 轮耗时 {_fmt_secs(ep_time)}，为历史中位 "
                    f"{_fmt_secs(med)} 的 {ep_time/med:.1f}×（阈值 {stall_factor:.1f}×）。"
                    f"训练可能卡在数据 I/O 或 GPU，请检查 nvidia-smi 与磁盘剩余空间。")
        # 每 hb_every 轮（以及最后一轮）打印一条汇总心跳
        if (epoch + 1) % hb_every == 0 or (epoch + 1) == total_epochs:
            avg_t = float(np.mean(epoch_times))
            elapsed = time.time() - t_loop0
            left = max(0, total_epochs - (epoch + 1)) * avg_t
            lr_now = optimizer.param_groups[0]["lr"]
            if has_val and best_score > -1e8:
                best_txt = f"best={best_score:.4f}(第{best_epoch+1}轮)"
            elif has_val:
                best_txt = "best=NA"
            else:
                best_txt = "best=最后一轮(无验证集)"
            logger.info(
                f"[心跳 {epoch+1}/{total_epochs}] train_loss={tr_loss:.4f} {mon_text} | "
                f"{best_txt} | lr={lr_now:.2e} | 本轮 {_fmt_secs(ep_time)} · "
                f"均 {_fmt_secs(avg_t)}/轮 | 已用 {_fmt_secs(elapsed)} · "
                f"剩余约 {_fmt_secs(left)}")

        if val_dl is not None and epochs_no_improve >= patience:
            logger.info(f"早停：连续 {patience} 个 epoch 无提升。")
            break

    _elapsed = _fmt_secs(time.time() - t_loop0)
    if has_val:
        logger.info(f"训练结束（总耗时 {_elapsed}）。最佳 monitor 分数={best_score:.4f}")
    else:
        logger.info(f"训练结束（总耗时 {_elapsed}；无验证集，best=最后一轮 "
                    f"epoch {best_epoch + 1}）。")
    # 写出最终指标
    with open(os.path.join(os.path.dirname(ckpt_dir), "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"best_score": (best_score if has_val and best_score > -1e8 else None),
                   "best_epoch": best_epoch, "has_val": has_val,
                   "task": task, "meta": meta}, f,
                  ensure_ascii=False, indent=2)

    # 记录本次训练的方式与结果到 logs/training_runs.csv（横向对比不同缺失修正策略等）
    if best_metrics is not None:
        try:
            log_run(cfg, best_metrics, best_epoch, meta=meta,
                    config_file=exp.get("_config_file", ""),
                    output_dir=os.path.join(exp.get("output_dir", "outputs"),
                                            exp.get("name", "exp")))
            logger.info(f"已写入训练日志: logs/training_runs.csv")
        except Exception as e:  # 日志失败不应影响训练产物
            logger.warning(f"写入训练日志失败(可忽略): {e}")
    return best_score
