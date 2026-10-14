# -*- coding: utf-8 -*-
"""
train_3x.py —— 自动训练 N 次（不同随机种子），取验证集 RMSE 最小的模型作为最终模型并记录。

流程：
  1. 每次训练用 experiment.seed + i 作为新种子（避免 3 次同初始化）；
  2. 每次训练写入独立子目录
       outputs/<base_name>_3x/run{i}_s{seed}/checkpoints/best.ckpt
     互不覆盖；
  3. 读取各 run 的 best.ckpt 内保存的验证集指标（regression_metrics），
     取 val RMSE 最小的那一个，复制为
       outputs/<base_name>_3x/best_of_3.ckpt
  4. 每次 train() 已在 logs/training_runs.csv 追加一行；本脚本额外写
       logs/train_3x_record.json 汇总 3 次结果并标注最终选用。

为什么用「验证集 RMSE」挑最终模型：best.ckpt 本身按 val RMSE 早停挑选，
用 val 选模型不涉及测试集，避免测试泄漏；最终模型的测试指标请用 grouped_eval /
evaluate 另行评估。

用法：
  python scripts/train_3x.py -c configs/tcir_wpac_train.yaml --runs 3
  python scripts/train_3x.py -c configs/tcir_wpac_train.yaml --runs 3 --missing-band interp
"""
import argparse
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tcintens.utils.config import load_config
from tcintens.models.tc_model import build_model
from tcintens.engine.train import train
from tcintens.engine.common import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/tcir_wpac_train.yaml")
    ap.add_argument("--runs", type=int, default=3, help="自动训练次数")
    ap.add_argument("--missing-band", default="interp",
                    help="缺失修正策略（默认 interp 局部区域插值）")
    ap.add_argument("--set", action="append", default=[],
                    help="额外覆盖配置 KEY=VAL")
    args = ap.parse_args()

    os.chdir(ROOT)
    cfg = load_config(args.config, args.set)
    d = cfg.to_dict()
    base_name = d["experiment"]["name"]
    base_out = d["experiment"].get("output_dir", "outputs")
    base_seed = int(d["experiment"].get("seed", 42))
    umbrella = os.path.join(base_out, f"{base_name}_3x")
    os.makedirs(umbrella, exist_ok=True)

    runs = []
    for i in range(args.runs):
        seed = base_seed + i
        # 固定修正策略与本次 run 的身份（独立目录，互不覆盖）
        cfg.set("data.tcir.missing_band", args.missing_band)
        cfg.set("experiment.seed", seed)
        run_name = f"run{i}_s{seed}"
        cfg.set("experiment.name", run_name)
        cfg.set("experiment.output_dir", umbrella)

        run_dir = os.path.join(umbrella, run_name)
        ckpt = os.path.join(run_dir, "checkpoints", "best.ckpt")
        done_marker = os.path.join(run_dir, "DONE")  # 完成标记，用于断点续跑
        print(f"\n===== 训练 run {i} (seed={seed}) -> {run_dir} =====", flush=True)

        success = False
        if os.path.exists(done_marker):
            # 已完成的 run 直接复用，不重复训练
            print("  已存在完成标记，跳过训练（复用已有 best.ckpt）。", flush=True)
            success = os.path.exists(ckpt)
        else:
            max_attempts = 2
            for attempt in range(1, max_attempts + 1):
                try:
                    train(cfg)
                    success = True
                    break
                except Exception as e:  # 捕获训练中途崩溃（如瞬时 CUDA 错误）后重试
                    print(f"  [WARN] run{i} 第 {attempt} 次尝试失败: {e!r}", flush=True)
                    if attempt < max_attempts:
                        print(f"  等待 15s 后重试 run{i} ...", flush=True)
                        time.sleep(15)
                    else:
                        print(f"  [ERROR] run{i} 多次重试仍失败，跳过。", flush=True)
            if success:
                # 仅在真正训练成功后才写完成标记，避免把半截模型当成完成
                open(done_marker, "w").close()

        if success and os.path.exists(ckpt):
            model = build_model(cfg)
            c = load_checkpoint(ckpt, model, map_location="cpu")
            m = c.get("metrics", {}) or {}
            runs.append({
                "run": i, "seed": seed, "name": run_name,
                "ckpt": os.path.abspath(ckpt),
                "val_rmse": m.get("rmse"), "val_mae": m.get("mae"),
                "val_r2": m.get("r2"),
            })
        else:
            runs.append({"run": i, "seed": seed, "name": run_name,
                         "ckpt": None})

    # 取验证集 RMSE 最小者
    valid = [r for r in runs if r.get("val_rmse") is not None]
    best = min(valid, key=lambda r: r["val_rmse"]) if valid else None
    if best is not None:
        dst = os.path.join(umbrella, "best_of_3.ckpt")
        shutil.copy(best["ckpt"], dst)
        best["final_ckpt"] = os.path.abspath(dst)

    record = {
        "base_name": base_name,
        "missing_band": args.missing_band,
        "umbrella": os.path.abspath(umbrella),
        "runs": runs,
        "best": best,
    }
    os.makedirs("logs", exist_ok=True)
    rec_path = os.path.join("logs", "train_3x_record.json")
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"\n===== 3x 训练完成 =====", flush=True)
    for r in runs:
        vr = f"{r.get('val_rmse'):.3f}" if r.get("val_rmse") is not None else "N/A"
        print(f"  run{r['run']} seed={r['seed']}  val_RMSE={vr}  -> {r['name']}",
              flush=True)
    if best is not None:
        print(f"  最佳: run{best['run']} seed={best['seed']}  val_RMSE={best['val_rmse']:.3f}",
              flush=True)
        print(f"  最终模型已复制为 {dst}", flush=True)
    print(f"  汇总已保存: {rec_path}", flush=True)


if __name__ == "__main__":
    main()
