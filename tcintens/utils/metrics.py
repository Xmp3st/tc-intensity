# -*- coding: utf-8 -*-
"""
metrics.py —— 评估指标（不依赖 sklearn，纯 numpy 实现）

分类任务：accuracy / macro-F1 / 各类别 recall / 混淆矩阵
回归任务：MAE / RMSE / R²
"""
from typing import Dict, List

import numpy as np


# --------------------------------------------------------------------------- #
# 分类指标
# --------------------------------------------------------------------------- #
def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]
) -> Dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(class_names)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0

    per_class = {}
    f1_list = []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_list.append(f1)
        per_class[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(cm[i, :].sum()),
        }
    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


# --------------------------------------------------------------------------- #
# 回归指标
# --------------------------------------------------------------------------- #
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) == 0:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(float(r2), 4)}
