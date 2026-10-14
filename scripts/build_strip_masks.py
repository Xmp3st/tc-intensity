#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建「真实条状缺测掩码库」，供训练期条状缺失干扰增广使用。

动机
----
IR1 的局部缺测以**横向细条带**为主（贴近图像上下边缘），是 TCIR 最典型的损伤形态。
要让模型对这类损伤鲁棒，最忠实的做法是「拿真实检测出来的缺测掩码」贴到干净图上，
而不是自己造矩形噪声。

做法
----
1. 只扫**训练年份**的帧（默认 2003–2013），避免把测试年份的损伤形态泄漏进训练；
2. 对 IR1 通道跑项目自身的缺测检测 `detect_missing_region`
   （sentinel=0 + low_thr=150K + min_blob=20 + dilate=1）；
3. 按形态筛选「条状」掩码：面积占比在 [min_af, max_af]、包围盒短边/长边 <= ratio、
   或贴边（真实条带 81% 贴边）；
4. 按掩码字节去重（条带高度重复，去重后多样性更高、文件更小）；
5. 落盘 npz：`masks` (K,96,96) uint8，附带统计 json 与预览图。

用法
----
  python scripts/build_strip_masks.py \
      --h5 data/wpac_96.h5 --csv data/wpac_compact.csv \
      --train-years 2003-2013 --out logs/ir1_strip_masks.npz
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def parse_years(spec):
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(str(y) for y in range(int(a), int(b) + 1))
        else:
            out.add(str(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/wpac_96.h5")
    ap.add_argument("--csv", default="data/wpac_compact.csv")
    ap.add_argument("--ir1-col", type=int, default=0, help="IR1 在紧凑 h5 中的通道下标")
    ap.add_argument("--train-years", default="2003-2013")
    ap.add_argument("--out", default="logs/ir1_strip_masks.npz")
    ap.add_argument("--min-area-frac", type=float, default=0.002)
    ap.add_argument("--max-area-frac", type=float, default=0.50)
    ap.add_argument("--max-ratio", type=float, default=0.75,
                    help="包围盒 短边/长边 上限（越小越细长；1.0=不限）")
    ap.add_argument("--min-blob", type=int, default=20)
    ap.add_argument("--dilate", type=int, default=1)
    ap.add_argument("--max-masks", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--preview", default=None, help="可选：预览 PNG 路径")
    args = ap.parse_args()

    import h5py
    from tcintens.data.band_health import detect_missing_region

    years = parse_years(args.train_years)
    df = pd.read_csv(args.csv)
    df["year"] = df["ID"].astype(str).str[:4]
    sel = df[df["year"].isin(years)]
    idx = np.sort(sel["matrix_index"].to_numpy().astype(int))
    print(f"[strip] 训练年份 {sorted(years)}，候选帧 {len(idx)}/{len(df)}")

    seen = set()
    masks = []
    n_detected = 0
    n_kept = 0
    area_list = []
    ratio_list = []
    with h5py.File(args.h5, "r") as hf:
        mat = hf["matrix"]
        for s in range(0, len(idx), args.batch):
            rows = mat[idx[s:s + args.batch]]            # (b,H,W,C) 升序，OK
            for k in range(rows.shape[0]):
                ch = rows[k][..., args.ir1_col].astype(np.float64)
                m = detect_missing_region(ch[None], sentinel=0.0, low_thr=[150.0],
                                          min_blob=args.min_blob,
                                          dilate=args.dilate)[0]
                if not m.any():
                    continue
                n_detected += 1
                af = float(m.mean())
                if af < args.min_area_frac or af > args.max_area_frac:
                    continue
                ys, xs = np.nonzero(m)
                hh = ys.max() - ys.min() + 1
                ww = xs.max() - xs.min() + 1
                ratio = min(hh, ww) / max(hh, ww, 1)
                touches = bool(ys.min() == 0 or ys.max() == ch.shape[0] - 1
                               or xs.min() == 0 or xs.max() == ch.shape[1] - 1)
                if ratio > args.max_ratio and not touches:
                    continue
                key = np.packbits(m).tobytes()
                if key in seen:
                    continue
                seen.add(key)
                masks.append(m.astype(np.uint8))
                area_list.append(af)
                ratio_list.append(float(ratio))
                n_kept += 1
            print(f"  ... {min(s + args.batch, len(idx))}/{len(idx)}  "
                  f"命中 {n_detected}  去重后入库 {len(masks)}", flush=True)

    if len(masks) > args.max_masks:
        rng = np.random.default_rng(0)
        pick = np.sort(rng.choice(len(masks), size=args.max_masks, replace=False))
        masks = [masks[i] for i in pick]
        area_list = [area_list[i] for i in pick]
        ratio_list = [ratio_list[i] for i in pick]

    arr = np.stack(masks, axis=0) if masks else np.zeros((0, 96, 96), np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, masks=arr)
    stats = {
        "h5": args.h5, "csv": args.csv, "train_years": sorted(years),
        "n_candidate_frames": int(len(idx)),
        "n_detected_frames": int(n_detected),
        "n_unique_masks": int(arr.shape[0]),
        "mask_shape": list(arr.shape[1:]),
        "area_frac": {
            "min": float(np.min(area_list)) if area_list else None,
            "median": float(np.median(area_list)) if area_list else None,
            "p90": float(np.percentile(area_list, 90)) if area_list else None,
            "max": float(np.max(area_list)) if area_list else None,
        },
        "bbox_short_long_ratio": {
            "median": float(np.median(ratio_list)) if ratio_list else None,
            "p90": float(np.percentile(ratio_list, 90)) if ratio_list else None,
        },
        "params": {"min_area_frac": args.min_area_frac,
                   "max_area_frac": args.max_area_frac,
                   "max_ratio": args.max_ratio,
                   "min_blob": args.min_blob, "dilate": args.dilate},
    }
    sp = os.path.splitext(args.out)[0] + "_stats.json"
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("[save]", args.out)
    print("[save]", sp)

    if args.preview and arr.shape[0] > 0:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from tcintens.utils.plotting import use_cjk_font
        use_cjk_font()
        n = min(12, arr.shape[0])
        rng = np.random.default_rng(1)
        pick = rng.choice(arr.shape[0], size=n, replace=False)
        fig, axes = plt.subplots(3, 4, figsize=(9, 7))
        for ax, i in zip(axes.ravel(), pick):
            ax.imshow(arr[i], cmap="gray_r", vmin=0, vmax=1)
            ax.set_title(f"#{i} 面积 {arr[i].mean()*100:.1f}%", fontsize=9)
            ax.axis("off")
        fig.suptitle("真实 IR1 条状缺测掩码（去重后抽样）")
        fig.tight_layout()
        fig.savefig(args.preview, dpi=130)
        print("[save]", args.preview)


if __name__ == "__main__":
    main()
