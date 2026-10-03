# -*- coding: utf-8 -*-
"""
runlog.py —— 训练运行日志（记录「方式」与「结果」，便于横向对比）

每次训练结束后追加一行到 logs/training_runs.csv，包含：
  * 方式：backbone / 输入通道数 / 缺失通道修正策略(missing_band) / 缺失判定 /
          损失函数 / epochs / batch_size / lr / optimizer
  * 数据规模：n_train / n_val / n_test / 被 drop 的缺失样本数
  * 结果：RMSE / MAE / R² / 最佳 epoch
  * 溯源：config 文件路径 / output_dir / 备注(notes)

用法（在 engine/train.py 训练结束时调用）：
  from ..utils.runlog import log_run
  log_run(cfg, best_metrics, best_epoch, meta=meta,
          config_file=exp.get("_config_file",""),
          output_dir=os.path.join(exp.get("output_dir"), exp.get("name")))

查看对比：直接用表格软件打开 logs/training_runs.csv，或
  python -c "import pandas as pd; print(pd.read_csv('logs/training_runs.csv'))"
"""
import csv
import os
from datetime import datetime

FIELDS = [
    "timestamp", "exp_name", "backbone", "in_channels",
    "missing_band", "missing_detect", "loss",
    "epochs", "batch_size", "lr", "optimizer",
    "n_train", "n_val", "n_test", "n_dropped_missing",
    "rmse", "mae", "r2", "best_epoch",
    "config_file", "output_dir", "notes",
]


def _to_dict(cfg):
    return cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)


def log_run(cfg, metrics, best_epoch, meta=None, config_file=None,
            output_dir=None, notes="", log_path="logs/training_runs.csv"):
    """追加一次训练运行记录。幂等：文件不存在时先写表头。"""
    d = _to_dict(cfg)
    exp = d.get("experiment", {})
    tr = d.get("train", {})
    mcfg = d.get("model", {})
    tcfg = d.get("data", {}).get("tcir", {})
    meta = meta or {}

    def num(x, nd=None):
        try:
            v = float(x)
            return round(v, nd) if nd is not None else v
        except (TypeError, ValueError):
            return x

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "exp_name": exp.get("name", ""),
        "backbone": mcfg.get("backbone", ""),
        "in_channels": mcfg.get("in_channels", ""),
        "missing_band": tcfg.get("missing_band", "none"),
        "missing_detect": tcfg.get("missing_detect", "constant"),
        "loss": tr.get("loss", ""),
        "epochs": tr.get("epochs", ""),
        "batch_size": tr.get("batch_size", ""),
        "lr": tr.get("lr", ""),
        "optimizer": tr.get("optimizer", ""),
        "n_train": meta.get("n_train", ""),
        "n_val": meta.get("n_val", ""),
        "n_test": meta.get("n_test", ""),
        "n_dropped_missing": meta.get("n_dropped_missing", 0),
        "rmse": num(metrics.get("rmse"), 3),
        "mae": num(metrics.get("mae"), 3),
        "r2": num(metrics.get("r2"), 4),
        "best_epoch": best_epoch,
        "config_file": config_file or exp.get("_config_file", ""),
        "output_dir": output_dir or os.path.join(exp.get("output_dir", "outputs"),
                                                 exp.get("name", "exp")),
        "notes": notes,
    }

    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})
    return row
