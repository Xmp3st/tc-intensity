# -*- coding: utf-8 -*-
"""
classification_metrics.py —— 把回归模型（Vmax）的分档结果当作分类来评估 accuracy。

回归模型本身不输出类别，故按台风强度分级阈值把 y_true / y_pred 离散化，
再计算 accuracy / 混淆矩阵 / per-class precision-recall-F1 / ±1 档准确率。

一次给出多套常用分档口径（阈值见 SCHEMES，可自行改）：
  * SS7     : Saffir–Simpson（1-min，与 JTWC/TCIR 标签一致）TD/TS/Cat1..Cat5
  * JMA4    : JMA 台风强度 4 级（Higa et al. 2021 用的分档；其原生为 10-min 口径，
              此处直接套数值仅作参考，跨口径比较需注明）
  * COARSE3 : 弱 / 中 / 强 三档
  * BINARY  : 是否达台风级（>=64 kn）
另附回归口径的 tolerance accuracy（|err| <= 5/10/15/20 kn 的占比）。

用法:
  python scripts/classification_metrics.py -c configs/tcir_wpac_train.yaml \
      --set experiment.name=run2_s44 \
      --set experiment.output_dir=outputs/tcir_wpac/tcir_wpac_v2_3x \
      --ckpt outputs/tcir_wpac/tcir_wpac_v2_3x/best_of_3.ckpt
"""
import argparse
import json
import os
import sys

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

INF = float("inf")

SCHEMES = {
    "SS7": {
        "edges": [-INF, 34, 64, 83, 96, 113, 137, INF],
        "names": ["TD(<34)", "TS(34-63)", "Cat1(64-82)", "Cat2(83-95)",
                  "Cat3(96-112)", "Cat4(113-136)", "Cat5(>=137)"],
    },
    "JMA4": {
        "edges": [-INF, 34, 64, 85, 105, INF],
        "names": ["TD(<34)", "Normal(34-63)", "Strong(64-84)",
                  "VeryStrong(85-104)", "Violent(>=105)"],
    },
    "COARSE3": {
        "edges": [-INF, 64, 96, INF],
        "names": ["Weak(<64)", "Medium(64-95)", "Strong(>=96)"],
    },
    "BINARY": {
        "edges": [-INF, 64, INF],
        "names": ["<64", ">=64"],
    },
}


def evaluate_scheme(yt, yp, edges, names):
    """按给定分档边界把回归值离散化后，计算分类指标。"""
    tc = np.digitize(yt, edges[1:-1], right=False)
    pc = np.digitize(yp, edges[1:-1], right=False)
    K = len(names)
    acc = float((tc == pc).mean())
    adj = float((np.abs(tc - pc) <= 1).mean())
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(tc, pc):
        cm[t, p] += 1
    per = []
    for k in range(K):
        tp = int(cm[k, k])
        fp = int(cm[:, k].sum() - tp)
        fn = int(cm[k, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per.append({"class": names[k], "support": int(cm[k, :].sum()),
                    "precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(f1, 4)})
    macro_f1 = float(np.mean([p["f1"] for p in per]))
    return {"names": names, "accuracy": round(acc, 4),
            "adjacent_accuracy": round(adj, 4), "macro_f1": round(macro_f1, 4),
            "confusion": cm.tolist(), "per_class": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/tcir_wpac_train.yaml")
    ap.add_argument("--ckpt", default=None,
                    help="指定权重；默认 outputs/<name>/checkpoints/best.ckpt")
    ap.add_argument("--set", action="append", default=[],
                    help="覆盖配置 KEY=VAL（如 experiment.name=...）")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    os.chdir(ROOT)
    cfg = load_config(args.config, args.set)
    logger = get_logger("tc-cls")
    cfg, _ = align_preprocessing(cfg, overrides=getattr(cfg, "overrides", []),
                                 logger=logger)
    exp = cfg.to_dict()["experiment"]

    datasets, meta = create_datasets(cfg)
    test_ds = datasets["test"]
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

    yt, yp = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            x = x.to(device, non_blocking=True)
            out = model(x)
            yp.append(out.cpu().numpy().ravel())
            yt.append(y.numpy().ravel())
    yt = np.concatenate(yt)
    yp = np.concatenate(yp)
    err = yp - yt
    n = len(yt)

    reg = regression_metrics(yt, yp)
    tol = {f"within_{t}kn": round(float((np.abs(err) <= t).mean()), 4)
           for t in (5, 10, 15, 20)}

    results = {"ckpt": os.path.abspath(ckpt_path), "n_test": int(n),
               "y_true_range": [float(yt.min()), float(yt.max())],
               "regression": {k: reg[k] for k in ("rmse", "mae", "r2")},
               "tolerance_accuracy": tol, "schemes": {}}

    log_lines = []
    for name, sc in SCHEMES.items():
        r = evaluate_scheme(yt, yp, sc["edges"], sc["names"])
        results["schemes"][name] = r
        log_lines.append(
            f"[{name}] accuracy={r['accuracy']:.4f}  "
            f"±1档={r['adjacent_accuracy']:.4f}  macroF1={r['macro_f1']:.4f}")
        for p in r["per_class"]:
            log_lines.append(
                f"    {p['class']:<20s} n={p['support']:<5d} "
                f"P={p['precision']:.3f} R={p['recall']:.3f} F1={p['f1']:.3f}")

    out_dir = os.path.join(exp.get("output_dir", "outputs"), exp.get("name", "exp"))
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "classification_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    for ln in log_lines:
        logger.info(ln)
    logger.info(f"回归: RMSE={reg['rmse']:.3f} MAE={reg['mae']:.3f} R²={reg['r2']:.3f}  (n={n})")
    logger.info("tolerance accuracy: " + "  ".join(f"<= {k.split('_')[1]}: {v:.3f}"
                                                   for k, v in tol.items()))
    logger.info(f"已保存: {json_path}")

    _plot(results, yt, yp, os.path.join(out_dir, "classification_metrics.png"))
    logger.info("已保存: classification_metrics.png")


def _plot(results, yt, yp, path):
    names = list(results["schemes"].keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    for ax, nm in zip(axes, names):
        r = results["schemes"][nm]
        cm = np.asarray(r["confusion"], dtype=float)
        row = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        labels = [c.split("(")[0] for c in r["names"]]
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] > 0:
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                            fontsize=7, color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(f"{nm}  acc={r['accuracy']:.3f}  "
                     f"±1={r['adjacent_accuracy']:.3f}  F1={r['macro_f1']:.3f}",
                     fontsize=10)
    fig.suptitle("Classification metrics of the regression model "
                 "(thresholded by intensity class)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
