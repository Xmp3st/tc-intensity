# -*- coding: utf-8 -*-
"""
engine/evaluate.py —— 在测试集上评估已训练模型

用法: python evaluate.py --set experiment.name=xxx
会自动加载 outputs/<name>/checkpoints/best.ckpt。
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
from ..utils.misc import resolve_device
from .common import load_checkpoint


def evaluate(cfg: Config):
    exp = cfg.to_dict()["experiment"]
    device = resolve_device(cfg.to_dict()["device"].get("mode", "auto"))
    logger = get_logger("tc-eval")

    datasets, meta = create_datasets(cfg)
    test_ds = datasets["test"]
    loader = DataLoader(test_ds, batch_size=cfg.to_dict()["train"].get("batch_size", 32),
                        shuffle=False, num_workers=exp.get("num_workers", 4))

    model = build_model(cfg).to(device)
    ckpt_path = os.path.join(exp.get("output_dir", "outputs"), exp.get("name", "exp"),
                             "checkpoints", "best.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到最佳权重: {ckpt_path}，请先训练。")
    load_checkpoint(ckpt_path, model, map_location=device)
    model.eval()

    yt, yp = [], []
    task = meta["task"]
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            out = model(x)
            if task == "regression":
                yp.append(out.cpu().numpy())
            else:
                yp.append(out.argmax(1).cpu().numpy())
            yt.append(y.numpy() if hasattr(y, "numpy") else np.array(y))
    yt = np.concatenate(yt)
    yp = np.concatenate(yp)

    if task == "regression":
        m = regression_metrics(yt, yp)
        logger.info(f"[TEST] RMSE={m['rmse']}  MAE={m['mae']}  R²={m['r2']}")
    else:
        m = classification_metrics(yt, yp, meta["class_names"])
        logger.info(f"[TEST] Accuracy={m['accuracy']}  Macro-F1={m['macro_f1']}")
        logger.info(f"[TEST] 混淆矩阵(行=真值,列=预测): {meta['class_names']}")
        for i, name in enumerate(meta["class_names"]):
            logger.info(f"  {name}: {m['confusion_matrix'][i]}")

    out_path = os.path.join(exp.get("output_dir", "outputs"), exp.get("name", "exp"),
                            "test_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"task": task, "metrics": m, "class_names": meta["class_names"]},
                  f, ensure_ascii=False, indent=2)
    logger.info(f"测试指标已保存: {out_path}")
    return m
