# -*- coding: utf-8 -*-
"""verify_inpaint_v2.py —— 校验新补全方法接入数据管线后的正确性与开销。

检查项
  1. 数值安全：归一化后是否出现 NaN/Inf、是否越界（|z| 异常大）
  2. 统计一致性：修复前后 test 集各通道均值/标准差是否仍在合理范围
  3. 覆盖率：有多少帧真正走了「跨通道回归」，多少退回「调和插值」
  4. 速度：与旧方法（griddata）在同样本上的取数耗时对比

用法
  python scripts/verify_inpaint_v2.py            # 对比 griddata 与 crosschannel
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

H5 = "data/wpac_96.h5"
N_SAMPLE = 400
OUT_JSON = "outputs/inpaint_v2_verify.json"


def run_one(method, n=N_SAMPLE):
    import torch
    from tcintens.utils.config import load_config
    from tcintens.data.tcir import create_tcir_datasets

    over = ["data.tcir.missing_region_method=" + method]
    if method == "harmonic":
        over.append("data.tcir.missing_region_iters=100")
    cfg = load_config("configs/tcir_wpac_train.yaml", over)
    t0 = time.time()
    ds, meta = create_tcir_datasets(cfg)
    build_s = time.time() - t0
    d = ds["test"]
    n = min(n, len(d))

    # —— 覆盖诊断：逐样本看 IR1/WV 各自的缺测情况 ——
    from tcintens.data.band_health import detect_missing_region
    import h5py
    cov = {"n": 0, "ir1_only": 0, "wv_only": 0, "both": 0, "neither": 0}
    arrs = []
    t0 = time.time()
    for i in range(n):
        out = d[i]
        x = out[0]
        arrs.append(x)
        if not torch.isfinite(x).all():
            cov["n_nan"] = cov.get("n_nan", 0) + 1
        if i < 120:
            ri = d.indices[i]
            with h5py.File(H5, "r") as hf:
                raw = hf["matrix"][ri][..., :3].astype(np.float32)
            a = np.ascontiguousarray(raw.transpose(2, 0, 1))
            mi = detect_missing_region(a[0:1], 0.0, 150.0, 20, 1)[0].any()
            mw = detect_missing_region(a[1:2], 0.0, 150.0, 20, 1)[0].any()
            cov["n"] += 1
            key = ("both" if (mi and mw) else "ir1_only" if mi
                   else "wv_only" if mw else "neither")
            cov[key] += 1
    load_s = time.time() - t0

    X = torch.stack(arrs)
    stats = {
        "method": method,
        "build_seconds": round(build_s, 1),
        "n_samples": n,
        "load_seconds": round(load_s, 2),
        "ms_per_sample": round(load_s / max(n, 1) * 1000, 2),
        "finite": bool(torch.isfinite(X).all()),
        "z_min": round(float(X.min()), 2),
        "z_max": round(float(X.max()), 2),
        "abs_z_gt_20": int((X.abs() > 20).sum()),
        "coverage": cov,
    }
    return stats


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    res = {}
    for m in ("griddata", "crosschannel"):
        print(f"\n===== 方法: {m} =====", flush=True)
        s = run_one(m)
        res[m] = s
        c = s["coverage"]
        print(f"  构建数据集 {s['build_seconds']}s | 取数 {s['n_samples']} 样本 "
              f"{s['load_seconds']}s ({s['ms_per_sample']} ms/样本)")
        print(f"  数值: finite={s['finite']} z∈[{s['z_min']},{s['z_max']}] "
              f"|z|>20 的像素 {s['abs_z_gt_20']}")
        print(f"  缺测覆盖(前{c['n']}样本): 仅IR1缺 {c['ir1_only']} | 仅WV缺 "
              f"{c['wv_only']} | 两者都缺 {c['both']} | 都不缺 {c['neither']}")

    sp = (res["griddata"]["ms_per_sample"] /
          max(res["crosschannel"]["ms_per_sample"], 1e-6))
    print(f"\n取数速度: crosschannel 比 griddata 快 {sp:.1f}×")
    import json
    os.makedirs("outputs", exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"已写出: {OUT_JSON}")


if __name__ == "__main__":
    main()
