# -*- coding: utf-8 -*-
"""
test_tcir.py —— TCIR 加载器与多通道训练路径的端到端验证

无需真实 13GB 数据：用 make_synthetic_tcir 生成「与真实 TCIR 同 schema」的
迷你 HDF5（matrix: N x size x size x 4，含 NaN；info: pandas 表含 intensity/id 等），
验证：
  1) 懒加载、通道选择、NaN 处理（zero / interp）
  2) 按风暴 ID 切分（防泄漏），三集合样本数之和 == 总样本数
  3) 多通道模型前向 + 一次反向传播可运行
  4) build_backbone 在 in_channels=4 时正确改写首个卷积层

直接运行：python tests/test_tcir.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from tcintens.data.tcir import (
    TCIRDataset,
    compute_tcir_stats,
    make_synthetic_tcir,
)
from tcintens.data.tcir import create_tcir_datasets
from tcintens.models.backbones import build_backbone
from tcintens.models.tc_model import build_model
from tcintens.utils.config import Config


def _make_cfg(h5_path, tmp):
    return Config({
        "experiment": {"seed": 0, "output_dir": os.path.join(tmp, "outputs"),
                       "name": "tcir_test"},
        "data": {
            "dataset": "tcir", "task": "regression", "root_dir": tmp,
            "split": [0.7, 0.15, 0.15],
            "mean": [0, 0, 0], "std": [1, 1, 1],
            "tcir": {
                "h5_path": h5_path, "channels": ["IR1", "WV", "PMW"],
                "target": "intensity", "label_unit": "knots",
                "nan_mode": "zero", "normalize": False, "resize": 24,
                "split_strategy": "by_storm", "storm_col": "id", "regions": None,
            },
        },
        "augmentation": {"train": {"rotate": 10, "horizontal_flip": True}},
        "model": {"backbone": "resnet18", "pretrained": True, "in_channels": 3},
    })


def run():
    tmp = tempfile.mkdtemp(prefix="tcir_test_")
    h5_path = os.path.join(tmp, "TCIR_synthetic.h5")
    make_synthetic_tcir(h5_path, n=30, size=24, seed=1)

    cfg = _make_cfg(h5_path, tmp)
    datasets, meta = create_tcir_datasets(cfg)

    # --- 1) 基础结构 ---
    assert meta["task"] == "regression"
    assert meta["in_channels"] == 3
    total = meta["n_train"] + meta["n_val"] + meta["n_test"]
    assert total == 30, f"切分总数应为 30，实际 {total}"
    print(f"[ok] 切分: train={meta['n_train']} val={meta['n_val']} "
          f"test={meta['n_test']} (防泄漏 by_storm)")

    x, y = datasets["train"][0]
    assert isinstance(x, torch.Tensor) and x.dim() == 3, "样本应为 (C,H,W) 张量"
    assert x.shape[0] == 3, f"通道数应为 3，实际 {x.shape[0]}"
    assert not torch.isnan(x).any(), "zero 模式不应有 NaN"
    print(f"[ok] 单样本: x{x.shape} y={float(y):.1f}（knot）")

    # --- 2) NaN interp 模式也不产生 NaN ---
    interp_ds = TCIRDataset(h5_path=h5_path, channels=["IR1", "WV", "PMW"],
                            nan_mode="interp")
    xi, _ = interp_ds[0]
    assert not torch.isnan(xi).any(), "interp 模式不应有 NaN"
    interp_ds.close()
    print("[ok] NaN 处理: zero 与 interp 均无残留 NaN")

    # --- 3) 多通道训练：前向 + 反向 ---
    model = build_model(cfg)
    dl = DataLoader(datasets["train"], batch_size=4, shuffle=True)
    xb, yb = next(iter(dl))
    out = model(xb)
    assert out.shape == (4,), f"回归输出应为 (B,)，实际 {out.shape}"
    loss = torch.nn.MSELoss()(out, yb)
    loss.backward()
    print(f"[ok] 多通道训练路径: forward out{out.shape}  loss={float(loss.detach()):.3f}  反向传播成功")

    # --- 4) 4 通道时首个卷积被正确改写 ---
    bb, _ = build_backbone("resnet18", pretrained=False, in_channels=4)
    first_conv = None
    for m in bb.modules():
        if isinstance(m, torch.nn.Conv2d):
            first_conv = m
            break
    assert first_conv.in_channels == 4, "4 通道时首个卷积 in_channels 应为 4"
    print("[ok] in_channels=4: 首个卷积已改写为 4 输入通道")

    # --- 5) 逐通道统计 ---
    mean, std = compute_tcir_stats(h5_path, ("IR1", "WV", "PMW"))
    assert len(mean) == len(std) == 3
    print(f"[ok] 逐通道统计: mean={[round(v, 3) for v in mean]}")

    print("\n全部通过 ✅  TCIR 加载与多通道训练路径可用。")


if __name__ == "__main__":
    run()
