# -*- coding: utf-8 -*-
"""校验「无损坏子集 + 8:2 划分」两种策略的样本数与泄漏程度。"""
import pandas as pd

from tcintens.data.dataset import create_datasets
from tcintens.utils.config import load_config

BASE = [
    "data.tcir.csv_path=data/wpac_clean.csv",
    "data.split=[0.8,0.0,0.2]",
    "model.backbone=resnet50",
    "train.epochs=40",
]

df = pd.read_csv("data/wpac_clean.csv")
idmap = dict(zip(df.matrix_index, df.ID))

for strat in ("random", "by_storm"):
    cfg = load_config("configs/tcir_wpac_train.yaml",
                      BASE + ["data.tcir.split_strategy=" + strat])
    ds, meta = create_datasets(cfg)
    tr = set(ds["train"].indices)
    te = set(ds["test"].indices)
    s_tr = {idmap[i] for i in tr}
    s_te = {idmap[i] for i in te}
    shared = s_tr & s_te
    leak_frac = sum(1 for i in te if idmap[i] in s_tr) / max(1, len(te))
    print("[%-8s] n_train=%d n_val=%d n_test=%d  frame_overlap=%d" % (
        strat, meta["n_train"], meta["n_val"], meta["n_test"], len(tr & te)))
    print("            storms train=%d test=%d shared=%d  leak_frac(test frames from seen storms)=%.3f"
          % (len(s_tr), len(s_te), len(shared), leak_frac))
