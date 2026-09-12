# -*- coding: utf-8 -*-
"""
error_analysis.py —— 在完整测试集上评估 + 列出预测误差最大的样本

用法（与本框架一致，必须带上训练配置与正确的 backbone）：
    python scripts/error_analysis.py -c configs/tcir_wpac_train.yaml

输出（写入 experiment 目录，如 outputs/tcir_wpac/tcir_wpac_v1/）：
    test_predictions.csv : 逐样本 真实/预测/误差
    worst12.csv          : 误差最大的 top-k 样本（默认 12）

说明：
    - 测试集样本可精确映射回 wpac_compact.csv（紧凑 h5 与 CSV 行序一致），
      因此每个样本都能列出 风暴ID / matrix_index / 真实Vmax / 预测Vmax / 误差。
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# 把项目根加入路径（脚本在 scripts/ 下运行）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tcintens.data.dataset import create_datasets
from tcintens.models.tc_model import build_model
from tcintens.utils.config import parse_config
from tcintens.utils.metrics import regression_metrics
from tcintens.utils.misc import resolve_device
from tcintens.engine.common import load_checkpoint


def main():
    cfg = parse_config()
    d = cfg.to_dict()
    exp = d["experiment"]
    device = resolve_device(d["device"].get("mode", "auto"))
    tcfg = d["data"]["tcir"]
    csv_path = tcfg.get("csv_path")
    out_dir = os.path.join(exp.get("output_dir", "outputs"), exp.get("name", "exp"))

    # ---------- 数据 ----------
    datasets, meta = create_datasets(cfg)
    test_ds = datasets["test"]
    loader = DataLoader(
        test_ds,
        batch_size=d["train"].get("batch_size", 32),
        shuffle=False,
        num_workers=exp.get("num_workers", 4),
    )

    # ---------- 模型 + 权重 ----------
    model = build_model(cfg).to(device)
    ckpt_path = os.path.join(out_dir, "checkpoints", "best.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到最佳权重: {ckpt_path}，请先训练。")
    load_checkpoint(ckpt_path, model, map_location=device)
    model.eval()

    # ---------- 逐样本推理 ----------
    yt, yp = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            out = model(x)
            yp.append(out.cpu().numpy())
            yt.append(y.numpy() if hasattr(y, "numpy") else np.array(y))
    yt = np.concatenate(yt).astype(np.float64)
    yp = np.concatenate(yp).astype(np.float64)
    n = len(yt)

    # ---------- 总指标 ----------
    m = regression_metrics(yt, yp)
    print("=" * 60)
    print(f"[TEST 总统计] 样本数 N = {n}")
    print(f"  RMSE = {m['rmse']:.3f} kn")
    print(f"  MAE  = {m['mae']:.3f} kn")
    print(f"  R²   = {m['r2']:.4f}")
    print("=" * 60)

    # ---------- 映射回 CSV 元数据 ----------
    abs_err = np.abs(yp - yt)
    signed = yp - yt
    # 测试集样本顺序与 test_ds.indices 对齐
    real_idx = np.asarray(test_ds.indices, dtype=int)  # 紧凑 h5 行号 == CSV 行号

    id_col = tcfg.get("storm_col", "ID")
    idx_col = tcfg.get("index_col", "matrix_index")
    target_col = tcfg.get("target_col", "Vmax")

    if csv_path and os.path.exists(csv_path):
        meta_df = pd.read_csv(csv_path).set_index(idx_col)
        # 取与 test 顺序一致的元数据
        rows = []
        for ri in real_idx:
            row = meta_df.iloc[ri]
            rows.append({"matrix_index": ri, id_col: row[id_col], "data_set": row.get("data_set")})
        meta_rows = pd.DataFrame(rows)
        sid = meta_rows[id_col].to_numpy()
        dset = meta_rows["data_set"].to_numpy()
    else:
        sid = [f"r{ri}" for ri in real_idx]
        dset = [""] * n

    # ---------- 保存逐样本预测 ----------
    os.makedirs(out_dir, exist_ok=True)
    full = pd.DataFrame({
        "test_index": np.arange(n),
        "matrix_index": real_idx,
        id_col: sid,
        "data_set": dset,
        "true_vmax": np.round(yt, 3),
        "pred_vmax": np.round(yp, 3),
        "abs_err": np.round(abs_err, 3),
        "signed_err": np.round(signed, 3),
    })
    full_path = os.path.join(out_dir, "test_predictions.csv")
    full.to_csv(full_path, index=False)
    print(f"逐样本预测已保存: {full_path}  ({n} 行)")

    # ---------- 误差最大的 top-k ----------
    top_k = int(exp.get("top_k", 12)) if False else 12
    order = np.argsort(-abs_err)[:top_k]
    worst = pd.DataFrame({
        "rank": np.arange(1, len(order) + 1),
        "test_index": order,
        "matrix_index": real_idx[order],
        id_col: [sid[i] for i in order],
        "data_set": [dset[i] for i in order],
        "true_vmax": np.round(yt[order], 2),
        "pred_vmax": np.round(yp[order], 2),
        "abs_err": np.round(abs_err[order], 2),
        "signed_err": np.round(signed[order], 2),
    })
    worst_path = os.path.join(out_dir, "worst12.csv")
    worst.to_csv(worst_path, index=False)
    print(f"\n误差最大 top-{len(order)} 已保存: {worst_path}")
    print("\n" + "=" * 70)
    print(f"{'#':>2}  {'storm_ID':<14} {'m_idx':>6} {'true':>7} {'pred':>7} {'|err|':>6} {'err':>6}")
    print("-" * 70)
    for _, r in worst.iterrows():
        print(f"{int(r['rank']):>2}  {str(r[id_col]):<14} {int(r['matrix_index']):>6} "
              f"{r['true_vmax']:>7.1f} {r['pred_vmax']:>7.1f} {r['abs_err']:>6.1f} {r['signed_err']:>+6.1f}")
    print("=" * 70)
    print("（signed_err = pred - true；正值=高估，负值=低估）")


if __name__ == "__main__":
    main()
