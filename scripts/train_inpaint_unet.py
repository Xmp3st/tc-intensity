#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练「损坏 → 修正」的 U-Net（IR1 条状缺测图像修复）。

数据（在线合成，每个 epoch 随机配对）：
  干净帧 = 训练年(2003-2013) scan_damage(nd==0) 的 IR1 帧（12502 张）
  掩码   = 面积 5%~40% 的真实条状缺测掩码（1155 个）
  合成   = 干净帧 + 随机掩码（置哨兵 0）→ 损坏图；目标 = 干净图
  输入   = [损坏图(归一化), 掩码 0/1]  (2 通道)
  输出   = 修复后的 IR1（与输入同归一化空间）

训练：
  L1 损失（对缺测区域加权可选）、AdamW、梯度裁剪、AMP、断点续训。
  每 epoch 后记录「全图 L1 / 缺测区域 L1」，best 按缺测区域 L1 保存。

用法：
  python scripts/train_inpaint_unet.py            # 默认超参
  python scripts/train_inpaint_unet.py --epochs 50 --bs 64
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs", "tcir_inpaint_unet")

# IR1 归一化参数（与 configs/tcir_wpac_ir1_year.yaml 一致）
NORM_MEAN = 260.864
NORM_STD = 32.849
SENTINEL_NORM = (0.0 - NORM_MEAN) / NORM_STD   # ≈ -7.94


def fmt_secs(sec):
    """把秒数格式化成易读时长（心跳 / ETA 用）。"""
    sec = int(max(0, round(sec)))
    if sec < 60:
        return f"{sec}s"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


class InpaintDataset(Dataset):
    """在线合成配对：每次 __getitem__ 从干净帧 + 掩码库随机配对。"""

    def __init__(self, h5_path, clean_idx, masks, norm_mean, norm_std,
                 seed=0):
        import h5py
        self.h5_path = h5_path
        self.clean_idx = np.asarray(clean_idx, dtype=int)
        self.masks = np.asarray(masks, dtype=np.float32)  # (K,H,W) 0/1
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.rng = np.random.default_rng(seed)
        self._h5 = None

    def _open(self):
        import h5py
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self):
        return len(self.clean_idx)

    def __getitem__(self, i):
        hf = self._open()
        idx = int(self.clean_idx[i])
        # IR1 是紧凑 h5 的第 0 通道
        clean = np.asarray(hf["matrix"][idx, ..., 0], dtype=np.float32)  # (96,96)

        # 随机取一个掩码 + 随机水平镜像 + 随机纵向平移
        m = self.masks[self.rng.integers(len(self.masks))].copy()
        if self.rng.random() < 0.5:
            m = m[:, ::-1].copy()   # .copy() 消除负 stride（torch.from_numpy 不接受）
        shift = int(self.rng.integers(-24, 25))
        if shift:
            m = np.roll(m, shift, axis=0)

        # 归一化干净图
        clean_n = (clean - self.norm_mean) / self.norm_std
        # 损坏图：掩码处置哨兵（归一化后的 0 → SENTINEL_NORM）
        damaged = clean_n.copy()
        damaged[m > 0.5] = SENTINEL_NORM

        x = np.stack([damaged, m], axis=0).astype(np.float32)  # (2,96,96)
        y = clean_n[None].astype(np.float32)                    # (1,96,96)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m[None])


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    ms = torch.stack([b[2] for b in batch])
    return xs, ys, ms


def build_model(device):
    from tcintens.models.unet import UNet
    return UNet(in_channels=2, out_channels=1, base=48).to(device)


def masked_l1(pred, target, mask):
    """对缺测区域加权（掩码内像素权重更高），返回标量 + 缺测区域 L1。"""
    diff = torch.abs(pred - target)
    full = diff.mean()
    if mask.sum() > 0:
        roi = (diff * mask).sum() / mask.sum()
    else:
        roi = full
    # 缺测区域权重 4 倍，突出修复质量
    loss = full + 3.0 * roi
    return loss, full.detach(), roi.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/wpac_96.h5")
    ap.add_argument("--pairs", default="logs/inpaint_pairs.npz")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--save-every", type=int, default=10)
    # ---- 进度显示 / 防停滞 ----
    ap.add_argument("--log-every-batch", type=int, default=50,
                    help="每 N 个 batch 打印一条进度（0=关闭）；长时间不刷新即卡住")
    ap.add_argument("--log-every-epochs", type=int, default=5,
                    help="每 N 轮打印一条汇总心跳（含 ETA）")
    ap.add_argument("--stall-factor", type=float, default=3.0,
                    help="单轮耗时 > 历史中位 × 该系数 ⇒ 打「疑似停滞」告警")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z = np.load(args.pairs)
    train_idx = z["train_clean_idx"].astype(int)
    masks = z["masks"].astype(np.float32)
    print(f"[data] 训练干净帧 {len(train_idx)} / 掩码 {len(masks)}")

    ds = InpaintDataset(args.h5, train_idx, masks, NORM_MEAN, NORM_STD, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=4,
                    collate_fn=collate, drop_last=True, persistent_workers=True)

    model = build_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] U-Net 参数 {n_params/1e6:.2f} M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "checkpoints")
    os.makedirs(ckpt_path, exist_ok=True)

    start_epoch = 0
    best_roi = float("inf")
    if args.resume and os.path.exists(os.path.join(ckpt_path, "last.ckpt")):
        ck = torch.load(os.path.join(ckpt_path, "last.ckpt"), map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch = ck["epoch"] + 1
        best_roi = ck.get("best_roi", float("inf"))
        print(f"[resume] 从 epoch {start_epoch} 续训（best_roi={best_roi:.4f}）")

    log_path = os.path.join(args.out, "train.log")
    logf = open(log_path, "a", encoding="utf-8")

    n_batches = len(dl)
    epoch_times = []
    t_loop0 = time.time()
    hb = f"[进度] 批内每 {args.log_every_batch or '关'} 个 batch 一条；" \
         f"每 {args.log_every_epochs} 轮一条心跳；停滞阈值 {args.stall_factor:.1f}× 中位耗时。"
    print(hb, flush=True)
    logf.write(hb + "\n")
    logf.flush()

    for ep in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        tot_loss = 0.0
        tot_full = 0.0
        tot_roi = 0.0
        nb = 0
        for bi, (x, y, m) in enumerate(dl):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                pred = model(x)
                loss, full, roi = masked_l1(pred, y, m)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            tot_loss += loss.item()
            tot_full += full.item()
            tot_roi += roi.item()
            nb += 1
            # 批内进度心跳（长时间不刷新 ⇒ 疑似卡住）
            if (args.log_every_batch > 0 and (bi + 1) % args.log_every_batch == 0
                    and (bi + 1) < n_batches):
                pmsg = (f"  [ep {ep+1}/{args.epochs}] batch {bi+1}/{n_batches} "
                        f"({(bi+1)/n_batches*100:.0f}%) loss={loss.item():.4f} "
                        f"已用 {fmt_secs(time.time()-t0)}")
                print(pmsg, flush=True)
                logf.write(pmsg + "\n")
                logf.flush()
        sched.step()

        avg_loss = tot_loss / nb
        avg_full = tot_full / nb
        avg_roi = tot_roi / nb
        msg = (f"[epoch {ep+1}/{args.epochs}] loss={avg_loss:.4f} "
               f"full_l1={avg_full:.4f} roi_l1={avg_roi:.4f} "
               f"lr={sched.get_last_lr()[0]:.2e} "
               f"({time.time()-t0:.1f}s)")
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

        # 保存 last（断点续训）
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep,
                    "best_roi": best_roi},
                   os.path.join(ckpt_path, "last.ckpt"))
        # 保存 best（按缺测区域 L1）
        if avg_roi < best_roi:
            best_roi = avg_roi
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "best_roi": best_roi},
                       os.path.join(ckpt_path, "best.ckpt"))
            print(f"  [best] 缺测区域 L1 {best_roi:.4f} @ epoch {ep+1}", flush=True)

        # ---- 计时 · 停滞监测 · 定期心跳汇总（防停滞）----
        ep_time = time.time() - t0
        epoch_times.append(ep_time)
        if len(epoch_times) >= 3:
            med = float(np.median(epoch_times[:-1]))
            if med > 0 and ep_time > args.stall_factor * med:
                wmsg = (f"[疑似停滞] 第 {ep+1} 轮耗时 {fmt_secs(ep_time)}，为历史中位 "
                        f"{fmt_secs(med)} 的 {ep_time/med:.1f}×（阈值 {args.stall_factor:.1f}×）。"
                        f"训练可能卡在数据 I/O 或 GPU，请检查 nvidia-smi 与磁盘剩余空间。")
                print(wmsg, flush=True)
                logf.write(wmsg + "\n")
                logf.flush()
        if (ep + 1) % args.log_every_epochs == 0 or (ep + 1) == args.epochs:
            avg_t = float(np.mean(epoch_times))
            elapsed = time.time() - t_loop0
            left = max(0, args.epochs - (ep + 1)) * avg_t
            hmsg = (f"[心跳 {ep+1}/{args.epochs}] loss={avg_loss:.4f} roi_l1={avg_roi:.4f} "
                    f"best_roi={best_roi:.4f} | lr={sched.get_last_lr()[0]:.2e} | "
                    f"本轮 {fmt_secs(ep_time)} · 均 {fmt_secs(avg_t)}/轮 | "
                    f"已用 {fmt_secs(elapsed)} · 剩余约 {fmt_secs(left)}")
            print(hmsg, flush=True)
            logf.write(hmsg + "\n")
            logf.flush()

    logf.close()
    # 保存配置
    with open(os.path.join(args.out, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print(f"[done] best_roi={best_roi:.4f} 总耗时 {fmt_secs(time.time()-t_loop0)}")


if __name__ == "__main__":
    main()
