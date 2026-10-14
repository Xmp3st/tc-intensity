# -*- coding: utf-8 -*-
"""
grouped_eval.py —— 把测试集按「该样本是否做过缺失修正」分组，对比误差差距。

背景：WPAC 影像中有一部分样本含局部区域缺失（IR1/WV），训练/测试均已用空间插值
      修复；另一部分样本本身无缺失。本脚本把测试集分成两组：
        * 做过修正（was_corrected=True）：原图有缺失、被插值修复
        * 无修正（was_corrected=False）：原图本就干净
      分别计算整体 / 修正组 / 干净组的 RMSE/MAE/R²，量化「修正还能挽回多少」。

用法：
  # 默认对 outputs/<name>/checkpoints/best.ckpt 评估（预处理自动对齐训练配置）
  python scripts/grouped_eval.py -c configs/tcir_wpac_train.yaml

  # 对 3x 训练选出的最终模型评估（用其所属 run 的 config_used.yaml 对齐）
  python scripts/grouped_eval.py -c configs/tcir_wpac_train.yaml \
      --set experiment.name=tcir_wpac_v1_run0_s42 \
      --set experiment.output_dir=outputs/tcir_wpac/tcir_wpac_v1_3x \
      --ckpt outputs/tcir_wpac/tcir_wpac_v1_3x/best_of_3.ckpt
"""
import argparse
import json
import os
import sys

# 把项目根加入 sys.path（直接 python scripts/xxx.py 时也能 import tcintens）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tcintens.data.dataset import create_datasets
from tcintens.models.tc_model import build_model
from tcintens.utils.config import load_config
from tcintens.utils.logger import get_logger
from tcintens.utils.metrics import regression_metrics
from tcintens.utils.misc import resolve_device
from tcintens.utils.repro import align_preprocessing
from tcintens.engine.common import load_checkpoint


def _r(g):
    return {"rmse": g["rmse"], "mae": g["mae"], "r2": g["r2"], "n": g["n"]}


def _sub(a, b):
    if a is None or b is None:
        return None
    return round(a - b, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/tcir_wpac_train.yaml")
    ap.add_argument("--ckpt", default=None,
                    help="指定权重；默认 outputs/<name>/checkpoints/best.ckpt")
    ap.add_argument("--set", action="append", default=[],
                    help="覆盖配置 KEY=VAL（如 experiment.name=...）")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    os.chdir(ROOT)  # 相对 h5 路径依赖项目根 cwd
    cfg = load_config(args.config, args.set)
    logger = get_logger("tc-grouped")

    # 一致性守卫：把缺失修正/划分等键对齐到训练时保存的 config_used.yaml
    cfg, _ = align_preprocessing(cfg, overrides=getattr(cfg, "overrides", []),
                                 logger=logger)
    exp = cfg.to_dict()["experiment"]

    datasets, meta = create_datasets(cfg)
    test_ds = datasets["test"]
    test_ds.return_corrected_flag = True  # __getitem__ 额外返回是否被修正
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=exp.get("num_workers", 4))

    device = resolve_device(cfg.to_dict()["device"].get("mode", "auto"))
    model = build_model(cfg).to(device)
    ckpt_path = args.ckpt or os.path.join(
        exp.get("output_dir", "outputs"), exp.get("name", "exp"),
        "checkpoints", "best.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到权重: {ckpt_path}")
    load_checkpoint(ckpt_path, model, map_location=device)
    model.eval()

    yt, yp, fl = [], [], []
    with torch.no_grad():
        for x, y, flag in loader:
            x = x.to(device, non_blocking=True)
            out = model(x)
            yp.append(out.cpu().numpy().ravel())
            yt.append(y.numpy().ravel())
            fl.append(np.asarray(flag).ravel())
    yt = np.concatenate(yt)
    yp = np.concatenate(yp)
    fl = np.concatenate(fl).astype(bool)
    n_total = len(yt)
    n_cor = int(fl.sum())
    n_clean = n_total - n_cor

    def grp(mask):
        if int(mask.sum()) == 0:
            return {"rmse": None, "mae": None, "r2": None, "n": 0}
        m = regression_metrics(yt[mask], yp[mask])
        return {"rmse": m["rmse"], "mae": m["mae"], "r2": m["r2"], "n": int(mask.sum())}

    overall = grp(np.ones(n_total, dtype=bool))
    g_cor = grp(fl)
    g_clean = grp(~fl)

    result = {
        "ckpt": os.path.abspath(ckpt_path),
        "n_total": n_total, "n_corrected": n_cor, "n_clean": n_clean,
        "overall": _r(overall), "corrected": _r(g_cor), "clean": _r(g_clean),
        "gap_rmse": _sub(g_cor.get("rmse"), g_clean.get("rmse")),
        "gap_mae": _sub(g_cor.get("mae"), g_clean.get("mae")),
    }

    out_dir = os.path.join(exp.get("output_dir", "outputs"), exp.get("name", "exp"))
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "grouped_eval.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    def _f(v):
        return "N/A" if v is None else f"{v:.3f}"

    logger.info(f"分组评估完成：修正 {n_cor} / 干净 {n_clean} / 共 {n_total}")
    logger.info(f"  整体   RMSE={_f(overall['rmse'])}  MAE={_f(overall['mae'])}  R²={_f(overall['r2'])}")
    logger.info(f"  修正组 RMSE={_f(g_cor['rmse'])}  MAE={_f(g_cor['mae'])}  R²={_f(g_cor['r2'])}  (n={n_cor})")
    logger.info(f"  干净组 RMSE={_f(g_clean['rmse'])}  MAE={_f(g_clean['mae'])}  R²={_f(g_clean['r2'])}  (n={n_clean})")
    logger.info(f"  RMSE 差距(修正-干净) = {result['gap_rmse']}")

    _plot(result, yt, yp, fl, os.path.join(out_dir, "grouped_eval.png"))
    logger.info(f"已保存: {json_path} 与 grouped_eval.png")


def _plot(result, yt, yp, fl, path):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    # (1) 柱状：各组 RMSE
    labels = ["Overall", "Corrected", "Clean"]
    rmse = [result["overall"]["rmse"], result["corrected"]["rmse"],
            result["clean"]["rmse"]]
    cols = ["#4C72B0", "#C44E52", "#55A868"]
    bars = ax[0].bar(labels, [v if v is not None else float("nan") for v in rmse],
                     color=cols)
    for b, v in zip(bars, rmse):
        if v is None:
            continue
        ax[0].text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                   ha="center", va="bottom")
    ax[0].set_title("Test RMSE by group")
    ax[0].set_ylabel("RMSE (kn)")
    # (2) 散点：Pred vs True，按组着色
    ax[1].scatter(yt[~fl], yp[~fl], s=8, alpha=0.4, c="#55A868",
                  label=f"Clean (n={result['n_clean']})")
    ax[1].scatter(yt[fl], yp[fl], s=8, alpha=0.4, c="#C44E52",
                  label=f"Corrected (n={result['n_corrected']})")
    lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax[1].plot(lim, lim, "k--", lw=1)
    ax[1].set_xlabel("True Vmax (kn)")
    ax[1].set_ylabel("Pred Vmax (kn)")
    ax[1].set_title("Pred vs True (by correction)")
    ax[1].legend()
    # (3) 直方图：|误差| 分布
    ax[2].hist(np.abs(yt[~fl] - yp[~fl]), bins=40, alpha=0.6,
               color="#55A868", label="Clean")
    ax[2].hist(np.abs(yt[fl] - yp[fl]), bins=40, alpha=0.6,
               color="#C44E52", label="Corrected")
    ax[2].set_xlabel("|Error| (kn)")
    ax[2].set_ylabel("Samples")
    ax[2].set_title("|Error| distribution")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
