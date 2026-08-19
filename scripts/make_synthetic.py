# -*- coding: utf-8 -*-
"""
make_synthetic.py —— 生成「可学习」的合成热带气旋图像 + labels.csv

用途：在没有真实卫星数据的情况下，快速验证整套框架能否跑通。
生成规律（与强度正相关，模型应当能学到）：
  - 强度越高 → 中心「台风眼」越小越亮
  - 强度越高 → 螺旋云带越多、整体越亮
  - 每张图加入随机相位 / 抖动，制造类内差异

用法:
  python scripts/make_synthetic.py --out-dir data/synthetic --per-class 60 --size 224
"""
import argparse
import csv
import os
import random

import numpy as np
from PIL import Image


# 6 级 CMA 强度对应的代表风速（m/s），仅用于合成数据的标签
WIND_BY_CLASS = {"TD": 12, "TS": 21, "STS": 28, "TY": 37, "STY": 46, "SuperTY": 58}
CLASS_NAMES = list(WIND_BY_CLASS.keys())


def generate_one(class_idx: int, size: int, rng: random.Random) -> np.ndarray:
    cx = size / 2 + rng.uniform(-size * 0.08, size * 0.08)
    cy = size / 2 + rng.uniform(-size * 0.08, size * 0.08)
    yy, xx = np.mgrid[0:size, 0:size]
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)

    k = class_idx
    freq = 0.06 + k * 0.012
    phase = rng.uniform(0, 2 * np.pi)
    arms = 2 + k  # 云带旋臂数随强度增加
    bands = np.sin(arms * theta + r * freq + phase) * 0.5 + 0.5

    eye_r = max(6, 45 - k * 6)
    base = 0.35 + k * 0.07
    img = base * bands
    img = np.where(r < eye_r, 1.0, img)              # 明亮台风眼
    falloff = np.clip(1 - r / (size * 0.7), 0, 1)    # 边缘渐暗（海洋）
    img = img * falloff
    img = img + rng.normalvariate(0, 0.05)          # 噪声
    img = np.clip(img, 0, 1)

    # 轻微冷色调，强度越高越偏亮白
    rgb = np.stack([img, img * 0.92, img * 0.82], axis=-1) * 255
    return rgb.astype("uint8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/synthetic")
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    rows = []
    for ci, name in enumerate(CLASS_NAMES):
        sub = os.path.join(img_dir, name)
        os.makedirs(sub, exist_ok=True)
        for i in range(args.per_class):
            arr = generate_one(ci, args.size, rng)
            fn = f"{name}_{i:04d}.png"
            Image.fromarray(arr, "RGB").save(os.path.join(sub, fn))
            rel = os.path.join("images", name, fn)
            rows.append((rel, name, WIND_BY_CLASS[name]))

    label_file = os.path.join(args.out_dir, "labels.csv")
    with open(label_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "label", "wind_speed_ms"])
        for r in rows:
            w.writerow(r)
    print(f"已生成 {len(rows)} 张合成图 -> {img_dir}")
    print(f"标签文件 -> {label_file}")


if __name__ == "__main__":
    main()
