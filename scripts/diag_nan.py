# -*- coding: utf-8 -*-
"""诊断训练 nan（v2）：数据 + 模型输出尺度 + 首个 batch 的 loss。

用法：
    PYTHONPATH=. python3.11 scripts/diag_nan.py
"""
import numpy as np
import torch

from tcintens.utils.config import load_config
from tcintens.data.dataset import create_datasets
from tcintens.models.tc_model import build_model

CFG = "configs/tcir_wpac_ir1_year.yaml"


def _pct(a, qs=(0, 1, 50, 99, 100)):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return "empty"
    return np.round(np.percentile(a, qs), 3)


def main():
    cfg = load_config(CFG, [])
    ds, meta = create_datasets(cfg)
    print("meta:", {k: meta.get(k) for k in
                    ("n_train", "n_val", "n_test", "in_channels", "missing_band")})

    for name in ("train", "val", "test"):
        d = ds[name]
        lab = np.asarray(d.labels, dtype=np.float64)
        if lab.size == 0:
            print(f"[{name}] empty")
            continue
        print(f"[{name}] n={lab.size} label pct={_pct(lab)} "
              f"nan={int(np.isnan(lab).sum())} inf={int(np.isinf(lab).sum())}")

    # ---- 输入抽样统计 ----
    d = ds["train"]
    n = len(d)
    idxs = np.linspace(0, n - 1, 600).astype(int)
    rows = []
    for i in idxs:
        x, y = d[i]
        v = x.numpy()
        rows.append((float(v.min()), float(v.max()), float(y),
                     int(np.isnan(v).sum()), int(np.isinf(v).sum())))
    w = np.array(rows, dtype=np.float64)
    print(f"\n[输入抽样 {len(idxs)}] x.min pct={_pct(w[:, 0])}")
    print(f"                  x.max pct={_pct(w[:, 1])}")
    print(f"                  nan={int(w[:, 3].sum())} inf={int(w[:, 4].sum())}")
    k = np.argsort(w[:, 0])[:3]
    print("  最小 x.min 3 帧:", w[k][:, :3])

    # ---- 模型首层权重 ----
    model = build_model(cfg).to("cuda").eval()
    for name, p in model.named_parameters():
        if p.ndim == 4:
            print(f"\n[first conv] {name} shape={tuple(p.shape)} "
                  f"|w|mean={float(p.abs().mean()):.5f} "
                  f"max={float(p.max()):.4f} min={float(p.min()):.4f} "
                  f"nan={int(torch.isnan(p).sum())}")

    # ---- 一个真实 batch 的前向 ----
    loader = torch.utils.data.DataLoader(ds["train"], batch_size=64, shuffle=True,
                                         num_workers=2)
    x, y = next(iter(loader))
    x, y = x.to("cuda"), y.to("cuda")
    print(f"\n[batch] x shape={tuple(x.shape)} x.min={float(x.min()):.3f} "
          f"x.max={float(x.max()):.3f} x.mean={float(x.mean()):.3f}")
    print(f"        y.min={float(y.min()):.1f} y.max={float(y.max()):.1f}")

    with torch.no_grad():
        feat = model.backbone(x) if hasattr(model, "backbone") else None
        out = model(x)
        if feat is not None:
            print(f"[backbone] feat shape={tuple(feat.shape)} "
                  f"min={float(feat.min()):.3f} max={float(feat.max()):.3f} "
                  f"mean={float(feat.mean()):.3f} nan={int(torch.isnan(feat).sum())}")
        print(f"[head out] min={float(out.min()):.3f} max={float(out.max()):.3f} "
              f"mean={float(out.mean()):.3f} nan={int(torch.isnan(out).sum())}")
        loss = torch.nn.functional.mse_loss(out, y)
        print(f"[loss] {float(loss):.4f}")

    # ---- fp16 溢出边界检查 ----
    print(f"\n[fp16] max=65504  sqrt(max)={65504 ** 0.5:.1f} kn  "
          f"→ MSE 在误差 |e|>{65504 ** 0.5:.0f} kn 时上溢")


if __name__ == "__main__":
    main()
