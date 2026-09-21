#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/visualize_test.py —— 测试集预测可视化报告

生成一张图，包含：
  (1) 抽样若干台风卫星图（伪 RGB 合成：IR1->R, WV->G, PMW->B，逐通道 min-max 拉伸）
      每幅标题标出 预测 / 真实 Vmax(knot) 与误差 Δ
  (2) 测试集整体指标：RMSE / MAE / R²（基于散点子样本）
  (3) Pred vs True 散点图（含 y=x 参考线）

三种取样模式（互斥，优先级 --worst > --csv > --samples）：
  # 默认随机抽样 12 张（默认 config 即 vgg16，对应 outputs/tcir_wpac/tcir_wpac_v1 权重）
  python scripts/visualize_test.py -c configs/tcir_wpac_train.yaml

  # 误差最大的前 N 张：在完整测试集上推理并按 |pred-true| 取 top-N 画图
  python scripts/visualize_test.py -c configs/tcir_wpac_train.yaml --worst 12

  # 给定 CSV（如 error_analysis 产出的 worst12.csv / test_predictions.csv），
  # 按其中的 test_index（或 matrix_index）列画出这些样本
  python scripts/visualize_test.py -c configs/tcir_wpac_train.yaml \
      --csv outputs/tcir_wpac/tcir_wpac_v1/worst12.csv

输出: outputs/<experiment.name>/visualization.png
"""
import argparse
import os
import random
import sys

# 脚本位于 scripts/ 子目录，运行时会把 scripts/ 而非项目根加入 sys.path，
# 这里手动把项目根（脚本上级目录）插到搜索路径最前，保证能 import tcintens。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")  # 无显示环境（WSL/服务器）也能存图
import matplotlib.pyplot as plt

from tcintens.data.tcir import create_tcir_datasets
from tcintens.models.tc_model import build_model
from tcintens.utils.config import build_arg_parser, load_config
from tcintens.utils.metrics import regression_metrics
from tcintens.utils.misc import resolve_device


# --------------------------------------------------------------------------- #
# 显示用：反归一化 + 伪 RGB 合成
# --------------------------------------------------------------------------- #
def denorm_for_display(x, mean, std):
    """x: (C,H,W) 归一化张量 -> 反归一化回物理量 -> (C,H,W) numpy。"""
    x = x.cpu().numpy().astype(np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    return x * std[:, None, None] + mean[:, None, None]


def to_composite(x):
    """(C,H,W) -> 伪 RGB (H,W,3)：逐通道 min-max 拉伸后映射到 R/G/B。"""
    x = np.ascontiguousarray(np.transpose(x, (1, 2, 0)))
    c = x.shape[-1]
    out = np.zeros((x.shape[0], x.shape[1], 3), dtype=np.float32)
    src = [0, 1, 2] if c >= 3 else (list(range(c)) + [0] * (3 - c))
    for i in range(3):
        ci = src[i]
        v = x[..., ci]
        vmin, vmax = np.nanmin(v), np.nanmax(v)
        out[..., i] = (v - vmin) / (vmax - vmin) if (vmax - vmin) > 1e-8 else 0.0
    return np.nan_to_num(out)


# --------------------------------------------------------------------------- #
def main():
    parser = build_arg_parser()
    parser.add_argument("--samples", type=int, default=12, help="随机抽样的样本图数量")
    parser.add_argument("--scatter-n", type=int, default=800,
                        help="散点图最多使用的测试样本数（提速）")
    parser.add_argument("--out", default=None, help="输出 PNG 路径")
    parser.add_argument("--seed", type=int, default=1234, help="随机抽样随机种子")
    parser.add_argument("--worst", type=int, default=0,
                        help="若为 >0，则在完整测试集上计算并可视化误差最大的前 N 个样本"
                             "（覆盖 --samples / --csv）")
    parser.add_argument("--csv", default=None,
                        help="给定 CSV（如 worst12.csv / test_predictions.csv），"
                             "按其中 test_index（或 matrix_index）列画出这些样本（覆盖 --samples）")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    exp = cfg.to_dict()["experiment"]
    name = exp["name"]
    out_dir = exp.get("output_dir", "outputs")
    device = resolve_device(cfg.to_dict()["device"].get("mode", "auto"))

    # ---- 模型 + 权重 ----
    model = build_model(cfg).to(device)
    ckpt_path = os.path.join(out_dir, name, "checkpoints", "best.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到最佳权重: {ckpt_path}，请先训练。")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- 测试集 ----
    datasets, _ = create_tcir_datasets(cfg)
    test_ds = datasets["test"]
    n_test = len(test_ds)
    norm_mean = cfg.get("data.tcir.norm_mean")
    norm_std = cfg.get("data.tcir.norm_std")

    rng = random.Random(args.seed)

    # ---------------- 选定要画的样本（test-order 索引） ----------------
    sample_ids = []  # 可选：风暴 ID，仅 --csv 模式有
    if args.worst and args.worst > 0:
        # 全测试集推理，按 |pred-true| 取 top-N
        print(f"[worst] 在全部 {n_test} 个测试样本上推理，取误差最大前 {args.worst} ...")
        all_true, all_pred = [], []
        with torch.no_grad():
            for i in range(n_test):
                x, y = test_ds[i]
                p = model(x.unsqueeze(0).to(device)).item()
                all_true.append(float(y))
                all_pred.append(p)
        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        order = np.argsort(-np.abs(all_pred - all_true))[:args.worst]
        sample_idx = [int(i) for i in order]
    elif args.csv:
        # 按给定 CSV 的 test_index / matrix_index 列取样本
        df = pd.read_csv(args.csv)
        if "test_index" in df.columns:
            sample_idx = [int(v) for v in df["test_index"].tolist()]
            key_col = "test_index"
        elif "matrix_index" in df.columns:
            # matrix_index 是紧凑 h5 行号，需反查回 test-order 索引
            inv = {int(r): ti for ti, r in enumerate(test_ds.indices)}
            sample_idx = [inv[int(v)] for v in df["matrix_index"].tolist()]
            key_col = "matrix_index"
        else:
            raise ValueError("CSV 需含 test_index 或 matrix_index 列")
        if "ID" in df.columns:
            sample_ids = [str(v) for v in df["ID"].tolist()]
        print(f"[csv] 从 {args.csv} 读取 {len(sample_idx)} 个样本（列={key_col}）")
    else:
        sample_idx = rng.sample(range(n_test), min(args.samples, n_test))

    # ---------------- 画图用的样本图像 + 预测 ----------------
    imgs, preds, trues = [], [], []
    with torch.no_grad():
        for i in sample_idx:
            x, y = test_ds[i]
            p = model(x.unsqueeze(0).to(device)).item()
            imgs.append(to_composite(denorm_for_display(x, norm_mean, norm_std)))
            preds.append(p)
            trues.append(float(y))

    # 散点 + 指标（更大子样本，保持整体评估不变）
    scatter_idx = rng.sample(range(n_test), min(args.scatter_n, n_test))
    sp, st = [], []
    with torch.no_grad():
        for i in scatter_idx:
            x, y = test_ds[i]
            sp.append(model(x.unsqueeze(0).to(device)).item())
            st.append(float(y))
    sp = np.array(sp)
    st = np.array(st)
    m = regression_metrics(st, sp)

    # ------------------------------------------------------------------ #
    # 绘图
    # ------------------------------------------------------------------ #
    cols = 3
    rows = int(np.ceil(len(imgs) / cols))
    fig = plt.figure(figsize=(cols * 3.0, rows * 3.0 + 5.0))
    gs = fig.add_gridspec(rows + 2, cols, height_ratios=[3.0] * rows + [0.5, 4.5])

    for k, im in enumerate(imgs):
        r, c = divmod(k, cols)
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(im)
        ax.axis("off")
        err = preds[k] - trues[k]
        title = f"Pred {preds[k]:.1f} / True {trues[k]:.1f} kn\nΔ={err:+.1f}"
        if k < len(sample_ids) and sample_ids[k]:
            title = f"{sample_ids[k]}\n" + title
        ax.set_title(title, fontsize=9)

    # 指标条
    axc = fig.add_subplot(gs[rows, :])
    axc.axis("off")
    axc.text(0.5, 0.5,
             f"Test (n={len(st)})   RMSE={m['rmse']:.2f} kn   "
             f"MAE={m['mae']:.2f} kn   R²={m['r2']:.3f}",
             ha="center", va="center", fontsize=12, transform=axc.transAxes)

    # 散点图（方形、x/y 轴刻度间距一致，y=x 为真正 45° 对角线）
    axs = fig.add_subplot(gs[rows + 1, :])
    axs.scatter(st, sp, s=6, alpha=0.4, color="#2b7bba")
    lo, hi = float(min(st.min(), sp.min())), float(max(st.max(), sp.max()))
    axs.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
    axs.set_xlim(lo, hi)
    axs.set_ylim(lo, hi)
    axs.set_aspect("equal")        # 数据单位 1:1，刻度间距一致
    axs.set_box_aspect(1)          # 坐标轴框为正方形
    axs.set_xlabel("True Vmax (kn)")
    axs.set_ylabel("Pred Vmax (kn)")
    axs.set_title("Predicted vs True")
    axs.legend(loc="upper left")

    fig.tight_layout()
    out_path = args.out or os.path.join(out_dir, name, "visualization.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"可视化已保存: {out_path}")
    print(f"测试集指标(散点样本 n={len(st)}): {m}")


if __name__ == "__main__":
    main()
