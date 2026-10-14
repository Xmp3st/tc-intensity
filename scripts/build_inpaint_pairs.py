#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建「损坏-修正」配对数据（IR1 图像修复）所需的干净帧索引 + 掩码筛选。

动机
----
用干净的 IR1 图 + 真实条状缺测掩码（面积 5%~40%）合成「人工损坏图」，
训练一个 U-Net 学「损坏 → 修正」映射（替代 griddata 插值）。

本脚本做两件事：
1. 扫描训练年(2003-2013)与测试年(2014-2016)的「干净」IR1 帧（scan_damage nd==0），
   保存其 matrix_index 列表（训练脚本据此加载干净图，在线随机贴掩码）。
2. 从 ir1_strip_masks.npz 筛出「面积占比在 [min_af, max_af]」的掩码子集，
   保存成 inpaint_masks.npz（保证贴上去的损坏面积符合用户要求 5%~40%）。

训练集不落盘整张图——在线随机配对（干净帧 × 掩码），每个 epoch 组合不同，泛化更好。
评估集同样用测试年干净帧 + 掩码在线合成（在 eval 脚本里做），保证测试域干净。

用法：
  python scripts/build_inpaint_pairs.py \
      --h5 data/wpac_96.h5 --csv data/wpac_compact.csv \
      --masks logs/ir1_strip_masks.npz \
      --out logs/inpaint_pairs.npz
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/wpac_96.h5")
    ap.add_argument("--csv", default="data/wpac_compact.csv")
    ap.add_argument("--masks", default="logs/ir1_strip_masks.npz")
    ap.add_argument("--out", default="logs/inpaint_pairs.npz")
    ap.add_argument("--min-af", type=float, default=0.05, help="掩码面积占比下限（默认 5%）")
    ap.add_argument("--max-af", type=float, default=0.40, help="掩码面积占比上限（默认 40%）")
    ap.add_argument("--train-years", default="2003-2013")
    ap.add_argument("--test-years", default="2014-2016")
    ap.add_argument("--ir1-col", type=int, default=0)
    args = ap.parse_args()

    from tcintens.data.band_health import scan_damage

    def years_set(spec):
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

    tr_years = years_set(args.train_years)
    te_years = years_set(args.test_years)

    df = pd.read_csv(args.csv)
    df["year"] = df["ID"].astype(str).str[:4]
    tr_df = df[df["year"].isin(tr_years)]
    te_df = df[df["year"].isin(te_years)]
    tr_idx = np.sort(tr_df["matrix_index"].to_numpy().astype(int))
    te_idx = np.sort(te_df["matrix_index"].to_numpy().astype(int))

    # 1) 扫描干净帧
    clean = {}
    for name, idx in [("train", tr_idx), ("test", te_idx)]:
        nd, nw, imf, mxf = scan_damage(
            args.h5, idx, [args.ir1_col],
            sentinel=0.0, low_thr_list=[150.0], min_blob=20, dilate=1,
            missing_detect="constant")
        keep = idx[nd == 0]
        clean[name] = keep.astype(int)
        print(f"[clean] {name}: {len(keep)}/{len(idx)} 干净帧 "
              f"({100 * (nd == 0).mean():.1f}%)")

    # 2) 筛掩码（面积 5%~40%）
    z = np.load(args.masks)
    masks = z["masks"].astype(bool)
    frac = masks.reshape(masks.shape[0], -1).mean(1)
    sel = (frac >= args.min_af) & (frac <= args.max_af)
    sub = masks[sel]
    print(f"[mask] {sel.sum()}/{len(masks)} 个掩码面积在 "
          f"{args.min_af*100:.0f}%~{args.max_af*100:.0f}%")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        train_clean_idx=clean["train"],
        test_clean_idx=clean["test"],
        masks=sub.astype(np.uint8),
    )
    stats = {
        "h5": args.h5, "csv": args.csv, "masks_src": args.masks,
        "n_train_clean": int(len(clean["train"])),
        "n_test_clean": int(len(clean["test"])),
        "n_masks": int(sub.shape[0]),
        "mask_area_frac_range": [args.min_af, args.max_af],
        "mask_area_frac": {
            "min": float(frac[sel].min()) if sel.any() else None,
            "median": float(np.median(frac[sel])) if sel.any() else None,
            "max": float(frac[sel].max()) if sel.any() else None,
        },
    }
    sp = os.path.splitext(args.out)[0] + "_stats.json"
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("[save]", args.out)
    print("[save]", sp)


if __name__ == "__main__":
    main()
