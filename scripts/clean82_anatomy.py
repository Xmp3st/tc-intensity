# -*- coding: utf-8 -*-
"""泄漏解剖：量化两种划分下「测试帧所属风暴在训练集中的曝光量」。

输出：每臂 —— 测试帧数、涉及风暴数、测试风暴在训练集的平均/中位曝光帧数、
      以及标签（Vmax）分布对比（排除因难度差异造成的伪结论）。
"""
import json

import numpy as np
import pandas as pd

from tcintens.data.dataset import create_datasets
from tcintens.utils.config import load_config

BASE = ["data.tcir.csv_path=data/wpac_clean.csv",
        "data.split=[0.8,0.0,0.2]",
        "model.backbone=resnet50"]

df = pd.read_csv("data/wpac_clean.csv")
idmap = dict(zip(df.matrix_index, df.ID))
vmap = dict(zip(df.matrix_index, df.Vmax))

out = {}
for strat in ("random", "by_storm"):
    cfg = load_config("configs/tcir_wpac_train.yaml",
                      BASE + ["data.tcir.split_strategy=" + strat])
    ds, meta = create_datasets(cfg)
    tr, te = list(ds["train"].indices), list(ds["test"].indices)

    tr_cnt = {}
    for i in tr:
        tr_cnt[idmap[i]] = tr_cnt.get(idmap[i], 0) + 1

    te_ids = [idmap[i] for i in te]
    te_storms = sorted(set(te_ids))
    expo = np.array([tr_cnt.get(s, 0) for s in te_storms], float)
    seen = expo > 0

    v_tr = np.array([vmap[i] for i in tr], float)
    v_te = np.array([vmap[i] for i in te], float)

    rec = dict(
        strategy=strat,
        n_train=len(tr), n_test=len(te),
        n_storms_train=len(tr_cnt), n_storms_test=len(te_storms),
        storms_seen_in_train=int(seen.sum()),
        test_frame_seen_frac=float(np.mean([tr_cnt.get(i, 0) > 0 for i in te_ids])),
        frames_per_test_storm_mean=float(expo.mean()),
        frames_per_test_storm_median=float(np.median(expo)),
        frames_per_test_storm_max=int(expo.max()),
        vmax_train_mean=round(float(v_tr.mean()), 2),
        vmax_test_mean=round(float(v_te.mean()), 2),
        vmax_train_std=round(float(v_tr.std()), 2),
        vmax_test_std=round(float(v_te.std()), 2),
    )
    out[strat] = rec
    print("=== %s ===" % strat)
    for k, v in rec.items():
        print("  %-28s %s" % (k, v))
    print()

with open("logs/clean82_anatomy.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("[数据] logs/clean82_anatomy.json")
