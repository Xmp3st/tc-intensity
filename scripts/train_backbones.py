# -*- coding: utf-8 -*-
"""
train_backbones.py —— 多 backbone 对比训练（统一协议 · 固定风暴划分）

目的
    在完全相同的条件下横向比较不同骨干网络：
      * 输入同为 96x96x3（IR1/WV/PMW）、同一份 data/wpac_96.h5
      * 同一套缺失修正（missing_band=interp）与同一套损坏过滤（damage_filter）
      * 同一套逐通道归一化（norm_mean/norm_std）
      * 同一套风暴级 train/val/test 划分（由 data.tcir.split_seed=42 唯一决定）
      * 同一套超参（batch/lr/optimizer/scheduler/loss），仅 model.backbone 不同
    每个 backbone 单独跑「epochs 轮 + 早停」，各自保存 best.ckpt，
    并把验证集最佳结果（可含测试集）汇总到 logs/backbone_compare.{csv,json}。

为什么能让划分保持固定
    数据划分只取决于 data.tcir.split_seed（基础配置里固定为 42，
    且 split_strategy=by_storm，按风暴 ID 防泄漏切分）；
    experiment.seed 只影响权重初始化与数据加载顺序，不改变风暴归属。
    因此换 backbone / 换 seed 都不会挪动任何一张图所属的子集。

用法（项目根目录，dl-env 的 python）
    python scripts/train_backbones.py \
        -c configs/tcir_wpac_train.yaml \
        --backbones resnet18,resnet34,resnet50,efficientnet_b0 \
        --epochs 40 --patience 10 --missing-band interp --eval-test

断点续跑：每个 backbone 完成后写 <run_dir>/DONE，重跑时自动跳过已完成项。
"""
import argparse
import csv
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tcintens.utils.config import load_config
from tcintens.models.tc_model import build_model
from tcintens.engine.train import train
from tcintens.engine.evaluate import evaluate
from tcintens.engine.common import load_checkpoint
from tcintens.utils.misc import count_parameters

CSV_FIELDS = [
    "backbone", "run_name", "epochs", "patience", "missing_band", "seed",
    "params_M", "best_epoch", "epochs_ran",
    "val_rmse", "val_mae", "val_r2",
    "test_rmse", "test_mae", "test_r2",
    "minutes", "ckpt",
]


def _r(x, nd):
    try:
        return round(float(x), nd)
    except (TypeError, ValueError):
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/tcir_wpac_train.yaml")
    ap.add_argument("--backbones", default="resnet18,resnet34,resnet50,efficientnet_b0",
                    help="逗号分隔的 backbone 列表")
    ap.add_argument("--epochs", type=int, default=40, help="每模型训练轮数")
    ap.add_argument("--patience", type=int, default=10, help="早停耐心（连续无提升轮数）")
    ap.add_argument("--missing-band", default="interp", help="缺失修正策略（与 v2 基线一致）")
    ap.add_argument("--seed", type=int, default=42, help="初始化种子（划分不受其影响）")
    ap.add_argument("--out-dir", default="outputs/tcir_wpac")
    ap.add_argument("--out-name", default="tcir_wpac_bb", help="汇总子目录名")
    ap.add_argument("--eval-test", action="store_true", help="训练后顺带跑测试集评估")
    ap.add_argument("--retries", type=int, default=2, help="单模型失败重试次数")
    ap.add_argument("--no-resume", action="store_true",
                    help="默认：若存在未完成的 last.ckpt 则断点续训；加此参数则一律从头训练")
    ap.add_argument("--set", action="append", default=[], help="额外覆盖 KEY=VAL")
    args = ap.parse_args()

    os.chdir(ROOT)
    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]
    umbrella = os.path.join(args.out_dir, args.out_name)
    os.makedirs(umbrella, exist_ok=True)

    # 与 v2 基线对齐的公共覆盖（数据口径 / 轮数 / 早停）
    base_overrides = list(args.set) + [
        f"data.tcir.missing_band={args.missing_band}",
        f"train.epochs={args.epochs}",
        f"train.early_stopping_patience={args.patience}",
    ]

    results = []
    for bb in backbones:
        cfg = load_config(args.config, base_overrides)
        cfg.set("model.backbone", bb)
        cfg.set("experiment.seed", args.seed)
        run_name = f"bb_{bb}_s{args.seed}"
        cfg.set("experiment.name", run_name)
        cfg.set("experiment.output_dir", umbrella)

        run_dir = os.path.join(umbrella, run_name)
        ckpt = os.path.join(run_dir, "checkpoints", "best.ckpt")
        last_ckpt = os.path.join(run_dir, "checkpoints", "last.ckpt")
        done = os.path.join(run_dir, "DONE")

        print(f"\n{'='*72}\n[{bb}] {args.epochs} epochs / patience={args.patience} "
              f"-> {run_dir}\n{'='*72}", flush=True)
        t0 = time.time()
        if os.path.exists(done) and os.path.exists(ckpt):
            print("  已存在 DONE 标记，跳过训练（复用已有 best.ckpt）", flush=True)
            ok = True
        else:
            # 上次训练被外部中断（如 WSL 重启）时，从 last.ckpt 断点续训，
            # 最多只损失 save_last_every 轮
            if (not args.no_resume) and os.path.exists(last_ckpt):
                cfg.set("experiment.resume", True)
                print("  发现未完成的 last.ckpt -> 断点续训", flush=True)
            ok = False
            for attempt in range(1, args.retries + 1):
                try:
                    train(cfg)
                    ok = True
                    break
                except Exception as e:  # 瞬时 CUDA/IO 错误重试
                    print(f"  [WARN] {bb} 第 {attempt}/{args.retries} 次尝试失败: {e!r}",
                          flush=True)
                    if attempt < args.retries:
                        print("  等待 15s 后重试 ...", flush=True)
                        time.sleep(15)
            if ok:
                open(done, "w").close()
            else:
                print(f"  [ERROR] {bb} 重试仍失败，跳过。", flush=True)
        elapsed = time.time() - t0

        rec = {
            "backbone": bb, "run_name": run_name,
            "epochs": args.epochs, "patience": args.patience,
            "missing_band": args.missing_band, "seed": args.seed,
            "ckpt": os.path.abspath(ckpt) if os.path.exists(ckpt) else None,
            "minutes": round(elapsed / 60.0, 2),
        }

        if os.path.exists(ckpt):
            model = build_model(cfg)
            try:
                n_params = count_parameters(model)
            except Exception:
                n_params = sum(p.numel() for p in model.parameters())
            rec["params_M"] = round(n_params / 1e6, 2)
            c = load_checkpoint(ckpt, model, map_location="cpu")
            m = c.get("metrics", {}) or {}
            rec.update(
                val_rmse=_r(m.get("rmse"), 4), val_mae=_r(m.get("mae"), 4),
                val_r2=_r(m.get("r2"), 4),
                best_epoch=int(c.get("epoch")) + 1 if c.get("epoch") is not None else None,
            )
            print(f"  [VAL] RMSE={rec.get('val_rmse')} MAE={rec.get('val_mae')} "
                  f"R2={rec.get('val_r2')} best_epoch={rec.get('best_epoch')} "
                  f"({rec['minutes']} min)", flush=True)
            if args.eval_test:
                try:
                    tm = evaluate(cfg)
                    rec.update(test_rmse=_r(tm.get("rmse"), 4),
                               test_mae=_r(tm.get("mae"), 4),
                               test_r2=_r(tm.get("r2"), 4))
                    print(f"  [TEST] RMSE={rec.get('test_rmse')} "
                          f"MAE={rec.get('test_mae')} R2={rec.get('test_r2')}", flush=True)
                except Exception as e:
                    print(f"  [WARN] {bb} 测试集评估失败: {e!r}", flush=True)

        results.append(rec)

        # 每完成一个就落盘一次，避免中断丢结果
        _dump(results, umbrella)

    _dump(results, umbrella)

    valid = [r for r in results if r.get("val_rmse") is not None]
    valid.sort(key=lambda r: r["val_rmse"])
    print(f"\n{'='*72}\n汇总（按 val RMSE 升序）\n{'='*72}", flush=True)
    print(f"{'backbone':<20}{'params(M)':>10}{'val_RMSE':>10}{'val_MAE':>9}"
          f"{'val_R2':>8}{'test_RMSE':>11}{'min':>7}", flush=True)
    for r in valid:
        print(f"{r['backbone']:<20}{r.get('params_M', ''):>10}"
              f"{r.get('val_rmse', ''):>10}{r.get('val_mae', ''):>9}"
              f"{r.get('val_r2', ''):>8}{r.get('test_rmse', '-'):>11}"
              f"{r.get('minutes', ''):>7}", flush=True)
    print(f"\n汇总文件: {os.path.join('logs', 'backbone_compare.csv')}", flush=True)
    failed = [r["backbone"] for r in results if r.get("val_rmse") is None]
    if failed:
        print(f"[WARN] 未得到验证指标的 backbone: {failed}", flush=True)


def _dump(results, umbrella):
    """写汇总文件。按 run_name 与既有结果**合并**（同键覆盖、其余保留），
    这样单独补跑某个 backbone 不会清掉之前其它骨干的行。"""
    os.makedirs("logs", exist_ok=True)
    csv_path = os.path.join("logs", "backbone_compare.csv")

    merged, order = {}, []
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                k = row.get("run_name") or row.get("backbone")
                if not k:
                    continue
                if k not in merged:
                    order.append(k)
                merged[k] = {kk: row.get(kk, "") for kk in CSV_FIELDS}
    for r in results:
        k = r.get("run_name") or r["backbone"]
        if k not in merged:
            order.append(k)
        merged[k] = {kk: ("" if r.get(kk) is None else r.get(kk))
                     for kk in CSV_FIELDS}

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for k in order:
            w.writerow(merged[k])

    json_path = os.path.join("logs", "backbone_compare.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"umbrella": os.path.abspath(umbrella),
                   "results": [merged[k] for k in order]},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
