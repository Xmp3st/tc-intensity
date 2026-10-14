# -*- coding: utf-8 -*-
"""冒烟：确认 crosschannel + feather 接入数据管线后划分不变、数值安全。"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from tcintens.utils.config import load_config          # noqa: E402
from tcintens.data.dataset import create_datasets       # noqa: E402

cfg = load_config("configs/tcir_wpac_train.yaml", [
    "data.tcir.missing_band=interp",
    "data.tcir.missing_region_method=crosschannel",
    "data.tcir.missing_region_feather=3",
])
ds, meta = create_datasets(cfg)
print("划分:", meta.get("n_train"), meta.get("n_val"), meta.get("n_test"))

te = ds["test"]
te.return_corrected_flag = True
xs, n_corr = [], 0
for i in range(200):
    x, y, c = te[i]
    xs.append(x.numpy())
    n_corr += int(c)
X = np.stack(xs)
print("前 200 样本中修正过:", n_corr)
print("finite:", bool(np.isfinite(X).all()), "| min", round(float(X.min()), 3),
      "| max", round(float(X.max()), 3))
for ci, nm in enumerate(["IR1", "WV", "PMW"]):
    print(f"  {nm}: min {float(X[:, ci].min()):.3f} max {float(X[:, ci].max()):.3f}")
print("形状:", X.shape, "dtype:", X.dtype)

# 对照：同 200 样本在 griddata 下的极值（掩码一致 => 残留哨兵应完全相同）
cfg2 = load_config("configs/tcir_wpac_train.yaml", [
    "data.tcir.missing_band=interp",
    "data.tcir.missing_region_method=griddata",
])
ds2, _ = create_datasets(cfg2)
te2 = ds2["test"]
Y = np.stack([te2[i][0].numpy() for i in range(200)])
print("\n对照 griddata: IR1 min", round(float(Y[:, 0].min()), 3),
      "| WV min", round(float(Y[:, 1].min()), 3))
IR1_MEAN, IR1_STD = 260.864, 32.849
print("IR1 min 换算亮温: feather %.2f K  vs  griddata %.2f K"
      % (IR1_MEAN + X[:, 0].min() * IR1_STD, IR1_MEAN + Y[:, 0].min() * IR1_STD))
print("OK")
