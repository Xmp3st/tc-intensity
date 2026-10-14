#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IR1 单通道 · 按台风年份 8:2 · 四臂缺测修正对照 —— 训练 + 评估驱动脚本。

四臂（全部：IR1 单通道 / vgg16 / by_year 时序 holdout / 无验证集 / 测全集）：
  a1 无修正                        missing_band=none
  a2 带修正（griddata 空间插值）    missing_band=interp
  a3 仅用干净图训练                train_clean_only=true
  a4 无修正 + 训练期条状干扰增广    strip_dropout_prob=0.5

断点续跑：每个臂训练完写 DONE，评估产出 test_metrics.json 后写 DONE_EVAL；
重跑本脚本会自动跳过已完成的臂。

用法：
  python scripts/run_ir1_arms.py                # 全部
  python scripts/run_ir1_arms.py --only a2,a4   # 只跑指定臂
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
CFG = "configs/tcir_wpac_ir1_year.yaml"
OUT_DIR = os.path.join(ROOT, "outputs/tcir_wpac_ir1")

ARMS = [
    ("a1", "ir1_year_a1_raw",    []),
    ("a2", "ir1_year_a2_interp", ["data.tcir.missing_band=interp"]),
    ("a3", "ir1_year_a3_clean",  ["data.tcir.train_clean_only=true"]),
    ("a4", "ir1_year_a4_strip",  ["data.tcir.strip_dropout_prob=0.5"]),
    # a5：训练与测试输入都换成「离线 U-Net 修复后的 IR1」（真缺测帧被修复）
    ("a5", "ir1_year_a5_unet",
     ["data.tcir.ir1_repaired_path=data/ir1_unet_repaired.npy"]),
]


def run(cmd, log_path):
    print(f"> {' '.join(cmd)}", flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 78 + "\n> " + " ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只跑指定臂，如 a1,a3")
    ap.add_argument("--epochs", default=None, help="覆盖训练轮数")
    args = ap.parse_args()

    want = None if not args.only else set(x.strip() for x in args.only.split(","))
    os.makedirs(OUT_DIR, exist_ok=True)
    driver_log = os.path.join(ROOT, "logs", "ir1_arms_driver.log")

    summary = []
    t_all = time.time()
    for tag, name, ov in ARMS:
        if want is not None and tag not in want:
            continue
        adir = os.path.join(OUT_DIR, name)
        done_train = os.path.exists(os.path.join(adir, "DONE"))
        done_eval = os.path.exists(os.path.join(adir, "test_metrics.json"))
        t0 = time.time()

        if not done_train:
            cmd = [PY, "tcir.py", "train", "-c", CFG,
                   "--set", f"experiment.name={name}"]
            for o in ov:
                cmd += ["--set", o]
            if args.epochs:
                cmd += ["--set", f"train.epochs={args.epochs}"]
            rc = run(cmd, driver_log)
            if rc != 0:
                print(f"[FAIL] {tag} 训练返回码 {rc}", flush=True)
                summary.append({"arm": tag, "name": name, "status": f"train_fail({rc})"})
                continue
            open(os.path.join(adir, "DONE"), "w").write("ok\n")
        else:
            print(f"[skip] {tag} 训练已完成", flush=True)

        if not done_eval:
            cmd = [PY, "tcir.py", "eval", "-c", CFG,
                   "--set", f"experiment.name={name}"]
            rc = run(cmd, driver_log)
            if rc != 0:
                print(f"[FAIL] {tag} 评估返回码 {rc}", flush=True)
                summary.append({"arm": tag, "name": name, "status": f"eval_fail({rc})"})
                continue
        else:
            print(f"[skip] {tag} 评估已完成", flush=True)

        m = json.load(open(os.path.join(adir, "test_metrics.json"), encoding="utf-8"))
        summary.append({"arm": tag, "name": name, "status": "ok",
                        "overrides": ov, "metrics": m.get("metrics"),
                        "minutes": round((time.time() - t0) / 60, 1)})
        print(f"[done] {tag} {name}: {m.get('metrics')}", flush=True)

    print("\n===== 四臂汇总 =====", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    with open(os.path.join(OUT_DIR, "arms_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"总耗时 {(time.time() - t_all) / 60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
