# -*- coding: utf-8 -*-
"""
engine/predict.py —— 对单张图片或整个文件夹做推理

用法:
  python predict.py --image path/to/img.png --set experiment.name=xxx
  python predict.py --folder path/to/dir --set experiment.name=xxx --out preds.csv

输出：分类时给出类别 + 概率；回归时给出预测风速(m/s)。
"""
import csv
import os
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..data.transforms import build_transforms
from ..models.tc_model import build_model
from ..utils.config import Config
from ..utils.misc import resolve_device


class _ImageList(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.paths[i]


def predict(cfg: Config, image: str = None, folder: str = None, out: str = None):
    exp = cfg.to_dict()["experiment"]
    data_cfg = cfg.to_dict()["data"]
    aug_cfg = cfg.to_dict()["augmentation"]
    device = resolve_device(cfg.to_dict()["device"].get("mode", "auto"))
    task = data_cfg.get("task", "classification")
    class_names = data_cfg.get("class_names", [])

    model = build_model(cfg).to(device)
    ckpt_path = os.path.join(exp.get("output_dir", "outputs"), exp.get("name", "exp"),
                             "checkpoints", "best.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到最佳权重: {ckpt_path}，请先训练。")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = build_transforms("val", data_cfg["image_size"], aug_cfg,
                                 data_cfg["mean"], data_cfg["std"])

    # 收集待预测路径
    if image:
        paths: List[str] = [image]
    elif folder:
        paths = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp"))]
    else:
        raise ValueError("必须指定 --image 或 --folder")

    loader = DataLoader(_ImageList(paths, transform), batch_size=1, shuffle=False)

    results = []
    with torch.no_grad():
        for x, p in loader:
            x = x.to(device)
            out_t = model(x)
            if task == "regression":
                val = float(out_t.cpu().numpy().reshape(-1)[0])
                results.append((p[0], round(val, 2)))
            else:
                probs = torch.softmax(out_t, 1).cpu().numpy()[0]
                idx = int(np.argmax(probs))
                results.append((p[0], class_names[idx], round(float(probs[idx]), 4)))

    # 打印 + 可选保存
    for r in results:
        print(r)
    if out:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if task == "regression":
                w.writerow(["image", "wind_speed_ms"])
                for p, v in results:
                    w.writerow([p, v])
            else:
                w.writerow(["image", "class", "prob"])
                for p, c, pr in results:
                    w.writerow([p, c, pr])
        print(f"\n预测结果已保存: {out}")
    return results
