#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线预修复：用 U-Net 把 wpac_96.h5 中 IR1 的「真实缺测帧」修好并落盘。

动机
----
要把 U-Net 修复作为**训练/评估的输入侧预处理**（而不只是事后对比），最干净的做法是
在数据管线里替换 IR1 通道。但 DataLoader 是多进程（num_workers=4），在 worker 里跑
GPU 推理既不安全也不高效。故改为**一次性离线修复 + 缓存 npy**：

    data/ir1_unet_repaired.npy      (K, 96, 96) float32   修复后整帧 IR1（亮温 K）
    data/ir1_unet_repaired_idx.npy  (K,)        int64     对应 h5 行号（matrix_index）

之后由 tcir.py 的 `data.tcir.ir1_repaired_path` 在 `__getitem__` 中把损坏帧的 IR1
通道整帧替换为缓存值（干净帧不受影响）。这样训练与评估共用同一份修复结果，天然一致。

口径（与项目既有缺测检测完全对齐，勿随意改）
----
sentinel=0.0, low_thr=[150.0], min_blob=20, dilate=1（detect_missing_region）
U-Net 输入 = 归一化域 [(x-260.864)/32.849]，缺测像素置 SENTINEL_NORM；
输出反归一化回亮温域后**整帧**替换（与 a3_repair_compare.py 的 unet 变体一致）。

用法
----
  python scripts/build_unet_repaired_ir1.py \
      --unet-ckpt outputs/tcir_inpaint_unet_areamatch/checkpoints/best.ckpt \
      --out data/ir1_unet_repaired.npy
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tcintens.data.band_health import detect_missing_region  # noqa: E402
from tcintens.models.unet import UNet                        # noqa: E402

# IR1 逐通道归一化参数（与 configs/tcir_wpac_ir1_year.yaml 一致）
NORM_MEAN, NORM_STD = 260.864, 32.849
SENTINEL_NORM = (0.0 - NORM_MEAN) / NORM_STD          # ≈ -7.941


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/wpac_96.h5")
    ap.add_argument("--unet-ckpt",
                    default="outputs/tcir_inpaint_unet_areamatch/checkpoints/best.ckpt")
    ap.add_argument("--out", default="data/ir1_unet_repaired.npy")
    ap.add_argument("--ch", type=int, default=0, help="IR1 通道索引（默认 0）")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--sentinel", type=float, default=0.0)
    ap.add_argument("--low-thr", type=float, default=150.0)
    ap.add_argument("--min-blob", type=int, default=20)
    ap.add_argument("--dilate", type=int, default=1)
    args = ap.parse_args()
    os.chdir(ROOT)

    import h5py
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet = UNet(in_channels=2, out_channels=1, base=48).to(device)
    ck = torch.load(args.unet_ckpt, map_location=device)
    unet.load_state_dict(ck["model"])
    unet.eval()
    print(f"[unet] {args.unet_ckpt}（best_roi={ck.get('best_roi')}, epoch={ck.get('epoch')}）-> {device}",
          flush=True)

    t0 = time.time()
    idx_parts, rep_parts = [], []
    n_dmg = 0
    with h5py.File(args.h5, "r") as hf:
        mat = hf["matrix"]
        N = mat.shape[0]
        print(f"[h5] {args.h5}  N={N}  ch={args.ch}", flush=True)
        for s in range(0, N, args.batch):
            e = min(s + args.batch, N)
            x = mat[s:e, :, :, args.ch].astype(np.float32)      # (B,96,96) 亮温域
            B = x.shape[0]
            dmg_rows, masks = [], []
            for k in range(B):
                m = detect_missing_region(
                    x[k][None], sentinel=args.sentinel,
                    low_thr=[args.low_thr], min_blob=args.min_blob,
                    dilate=args.dilate)[0]
                if m.any():
                    dmg_rows.append(k)
                    masks.append(m)
            if dmg_rows:
                n_dmg += len(dmg_rows)
                dn = (x[dmg_rows] - NORM_MEAN) / NORM_STD
                mm = np.stack(masks, 0)                          # (b,96,96) bool
                dn[mm] = SENTINEL_NORM
                inp = np.stack([dn, mm.astype(np.float32)], 1)   # (b,2,96,96)
                outs = []
                with torch.no_grad():
                    for t in range(0, len(inp), 64):
                        xt = torch.from_numpy(inp[t:t + 64]).to(device)
                        outs.append(unet(xt).float().cpu().numpy()[:, 0])
                rep = np.concatenate(outs, 0) * NORM_STD + NORM_MEAN   # 回亮温域
                idx_parts.append(np.arange(s, e, dtype=np.int64)[dmg_rows])
                rep_parts.append(rep.astype(np.float32))
            if (s // args.batch) % 8 == 0:
                print(f"  {e}/{N}  累计损坏帧 {n_dmg}  {time.time()-t0:.0f}s", flush=True)

    idx = np.concatenate(idx_parts) if idx_parts else np.zeros(0, np.int64)
    rep = np.concatenate(rep_parts, 0) if rep_parts else np.zeros((0, 96, 96), np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, rep)
    idx_out = args.out.replace(".npy", "_idx.npy")
    np.save(idx_out, idx)

    stat = {
        "h5": args.h5, "unet_ckpt": args.unet_ckpt, "ch": args.ch,
        "n_frames": int(N), "n_damaged_repaired": int(len(idx)),
        "damaged_frac": round(float(len(idx) / max(N, 1)), 4),
        "detect": {"sentinel": args.sentinel, "low_thr": args.low_thr,
                   "min_blob": args.min_blob, "dilate": args.dilate},
        "norm_mean": NORM_MEAN, "norm_std": NORM_STD,
        "out": args.out, "idx": idx_out,
        "rep_min": float(rep.min()) if len(rep) else None,
        "rep_max": float(rep.max()) if len(rep) else None,
        "seconds": round(time.time() - t0, 1),
    }
    stat_out = args.out.replace(".npy", "_stats.json")
    with open(stat_out, "w", encoding="utf-8") as f:
        json.dump(stat, f, ensure_ascii=False, indent=2)
    print("\n[完成] " + json.dumps(stat, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
