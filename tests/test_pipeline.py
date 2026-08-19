# -*- coding: utf-8 -*-
"""
test_pipeline.py —— 端到端冒烟测试（无需真实数据）

做法：
  1) 生成小规模合成数据到临时目录
  2) 用轻量配置（mobilenet_v3_small / 64px / CPU / 2 epoch）跑 train
  3) 跑 evaluate，断言指标明显优于随机猜测

运行: python tests/test_pipeline.py   （或 pytest tests/）
"""
import importlib.util
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tcintens.utils.config import Config  # noqa: E402
from tcintens.engine import train, evaluate  # noqa: E402


def _load_make_synthetic():
    spec = importlib.util.spec_from_file_location(
        "make_synthetic", os.path.join(ROOT, "scripts", "make_synthetic.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pipeline_runs():
    tmp = tempfile.mkdtemp(prefix="tc_smoke_")
    ms = _load_make_synthetic()
    # 直接调用内部生成函数，避免 argparse
    import csv
    import random
    import numpy as np
    from PIL import Image

    rng = random.Random(0)
    img_dir = os.path.join(tmp, "images")
    class_names = ms.CLASS_NAMES
    rows = []
    for ci, name in enumerate(class_names):
        sub = os.path.join(img_dir, name)
        os.makedirs(sub, exist_ok=True)
        for i in range(30):
            arr = ms.generate_one(ci, 64, rng)
            fn = f"{name}_{i:04d}.png"
            Image.fromarray(arr, "RGB").save(os.path.join(sub, fn))
            rel = os.path.join("images", name, fn)
            rows.append((rel, name, ms.WIND_BY_CLASS[name]))
    label_file = os.path.join(tmp, "labels.csv")
    with open(label_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "label", "wind_speed_ms"])
        for r in rows:
            w.writerow(r)

    cfg = Config({
        "experiment": {"name": "smoke", "seed": 0, "output_dir": tmp, "num_workers": 0},
        "data": {
            "mode": "csv", "task": "classification",
            "root_dir": tmp, "label_file": label_file,
            "split": [0.6, 0.2, 0.2], "image_size": 64,
            "class_names": class_names,
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        },
        "augmentation": {
            "train": {"random_resized_crop": True, "horizontal_flip": True,
                      "rotate": 10, "color_jitter": [0.2, 0.2, 0.2, 0.0], "normalize": True},
            "val": {"resize": 72, "center_crop": 64, "normalize": True},
        },
        "model": {"backbone": "mobilenet_v3_small", "pretrained": True,
                  "freeze_backbone": False, "dropout": 0.3, "reg_hidden": 64},
        "train": {"epochs": 2, "batch_size": 8, "optimizer": "adamw", "lr": 1e-3,
                  "weight_decay": 1e-2, "scheduler": "none", "mixed_precision": False,
                  "early_stopping_patience": 99, "save_top_k": 1, "log_every": 5},
        "device": {"mode": "cpu"},
    })

    train(cfg)
    metrics = evaluate(cfg)

    best = json.load(open(os.path.join(tmp, "smoke", "metrics.json"), encoding="utf-8"))
    assert best["best_score"] > 0.2, f"训练未学到信号, best={best['best_score']}"
    assert metrics["accuracy"] > 0.2, f"测试精度过低: {metrics['accuracy']}"
    print(f"\n[smoke] best_macro_f1={best['best_score']:.3f} "
          f"test_acc={metrics['accuracy']:.3f} test_macro_f1={metrics['macro_f1']:.3f}")


if __name__ == "__main__":
    test_pipeline_runs()
    print("冒烟测试通过 ✅")
