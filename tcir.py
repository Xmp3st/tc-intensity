#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tcir.py —— tc_intensity 统一命令行入口（封装已有功能，便于使用）

设计原则：
  * 不修改任何已有文件；本脚本只是「封装层」。
  * train / eval / predict  直接复用 `tcintens` 引擎函数（同一进程，最干净）。
  * errors / visualize / extract / download / stats  以独立子进程调用 `scripts/` 下
    既有脚本（原样复用，零侵入）。
  * 每个子命令都给了合理默认值（默认配置 = configs/tcir_wpac_train.yaml，VGG16/WPAC），
    不必每次手敲冗长参数；任意子命令都可用 -c/--config 切换配置，用 --set K=V 覆盖。

用法（在项目根目录，用 dl_env 的 python 运行）：
    python tcir.py --help
    python tcir.py train                      # 默认 VGG16 / WPAC
    python tcir.py train --epochs 60 --backbone resnet18
    python tcir.py eval
    python tcir.py predict --image frame.npy
    python tcir.py predict --h5 data/wpac_96.h5 --indices 10260,9603 --out preds.csv
    python tcir.py errors
    python tcir.py visualize --samples 16
    python tcir.py extract --h5 data/TCIR.h5 --csv data/wpac_info.csv \
        --out-h5 data/wpac_96.h5 --out-csv data/wpac_compact.csv --start 27000
    python tcir.py download --out data/
    python tcir.py stats --h5 data/TCIR-ATLN_EPAC_WPAC.h5 --channels IR1 WV PMW

默认配置：configs/tcir_wpac_train.yaml（任意子命令可用 --config 覆盖）。
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SCRIPTS = os.path.join(ROOT, "scripts")
DEFAULT_CONFIG = os.path.join(ROOT, "configs", "tcir_wpac_train.yaml")


def resolve_config(path):
    """解析配置路径：相对路径先按 cwd、再按项目根寻找，最终返回绝对路径。"""
    if os.path.isabs(path):
        return path
    cand = os.path.join(os.getcwd(), path)
    if os.path.exists(cand):
        return os.path.abspath(cand)
    cand2 = os.path.join(ROOT, path)
    if os.path.exists(cand2):
        return os.path.abspath(cand2)
    return os.path.abspath(cand)


def _run_script(script_name, cli_args):
    """以独立子进程运行 scripts/ 下的既有脚本（不修改它们）。"""
    script = os.path.join(SCRIPTS, script_name)
    if not os.path.exists(script):
        print(f"[错误] 找不到脚本: {script}", file=sys.stderr)
        sys.exit(2)
    cmd = [sys.executable, script] + cli_args
    print("> " + " ".join(cmd), file=sys.stderr)
    # cwd 固定为项目根，使脚本里相对路径（data/...）行为一致、可任意目录调用
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    sys.exit(rc)


# --------------------------------------------------------------------------- #
# 直接复用引擎函数的子命令
# --------------------------------------------------------------------------- #
def cmd_train(args):
    from tcintens.engine.train import train
    from tcintens.utils.config import load_config
    overrides = list(args.set or [])
    if args.epochs is not None:
        overrides.append(f"train.epochs={args.epochs}")
    if args.backbone is not None:
        overrides.append(f"model.backbone={args.backbone}")
    cfg = load_config(args.config, overrides)
    train(cfg)


def cmd_eval(args):
    from tcintens.engine.evaluate import evaluate
    from tcintens.utils.config import load_config
    cfg = load_config(args.config, list(args.set or []))
    evaluate(cfg)


def cmd_predict(args):
    from tcintens.engine.predict import predict
    from tcintens.utils.config import load_config
    cfg = load_config(args.config, list(args.set or []))
    predict(cfg, image=args.image, folder=args.folder, h5=args.h5,
            index=args.index, indices=args.indices, out=args.out)


# --------------------------------------------------------------------------- #
# 转发到 scripts/ 既有脚本的子命令
# --------------------------------------------------------------------------- #
def cmd_errors(args):
    cli = ["-c", args.config]
    for s in (args.set or []):
        cli += ["--set", s]
    _run_script("error_analysis.py", cli)


def cmd_visualize(args):
    cli = ["-c", args.config]
    if args.samples is not None:
        cli += ["--samples", str(args.samples)]
    if args.scatter_n is not None:
        cli += ["--scatter-n", str(args.scatter_n)]
    if args.out:
        cli += ["--out", args.out]
    if args.seed is not None:
        cli += ["--seed", str(args.seed)]
    for s in (args.set or []):
        cli += ["--set", s]
    _run_script("visualize_test.py", cli)


def cmd_extract(args):
    cli = ["--h5", args.h5, "--csv", args.csv,
           "--out-h5", args.out_h5, "--out-csv", args.out_csv,
           "--channels", *args.channels, "--resize", str(args.resize),
           "--start", str(args.start), "--batch", str(args.batch),
           "--stats-out", args.stats_out]
    _run_script("extract_wpac.py", cli)


def cmd_download(args):
    cli = ["--out", args.out]
    if args.no_extract:
        cli += ["--no-extract"]
    _run_script("download_tcir.py", cli)


def cmd_stats(args):
    cli = ["--h5", args.h5, "--channels", *args.channels,
           "--sample-every", str(args.sample_every), "--out", args.out]
    if args.csv:
        cli += ["--csv", args.csv, "--index-col", args.index_col]
    _run_script("compute_tcir_stats.py", cli)


# --------------------------------------------------------------------------- #
# 参数解析
# --------------------------------------------------------------------------- #
def _add_cfg(sp):
    sp.add_argument("-c", "--config", default=DEFAULT_CONFIG,
                    help=f"YAML 配置路径（默认 {os.path.relpath(DEFAULT_CONFIG, ROOT)}）")
    sp.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="覆盖配置字段，可多次使用，如 --set train.epochs=60")


def build_parser():
    p = argparse.ArgumentParser(
        prog="tcir.py",
        description="tc_intensity 统一命令入口：训练/评估/推理/误差分析/可视化/抽取/下载/统计",
    )
    sub = p.add_subparsers(dest="cmd", metavar="<命令>")

    sp = sub.add_parser("train", help="训练（默认 VGG16 / WPAC）")
    _add_cfg(sp)
    sp.add_argument("--epochs", type=int, default=None, help="训练轮数")
    sp.add_argument("--backbone", default=None, help="骨干，如 vgg16 / resnet18")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("eval", help="测试集评估（输出 RMSE / MAE / R²）")
    _add_cfg(sp)
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("predict", help="推理（--image / --folder / --h5 三选一）")
    _add_cfg(sp)
    sp.add_argument("--image", default=None, help="单张 .npy（H,W,C，通道同配置）")
    sp.add_argument("--folder", default=None, help="文件夹（含若干 .npy）")
    sp.add_argument("--h5", default=None, help="TCIR matrix h5 路径（配合 --index/--indices）")
    sp.add_argument("--index", type=int, default=None, help="从 h5 取单帧（紧凑行号）")
    sp.add_argument("--indices", default=None, help="从 h5 取多帧，逗号分隔，如 10260,9603")
    sp.add_argument("--out", default=None, help="结果 CSV 输出路径")
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("errors", help="误差分析（逐样本预测 + 误差最大 top-12）")
    _add_cfg(sp)
    sp.set_defaults(func=cmd_errors)

    sp = sub.add_parser("visualize", help="测试集预测可视化报告（图 + 散点）")
    _add_cfg(sp)
    sp.add_argument("--samples", type=int, default=None, help="抽样台风图数量")
    sp.add_argument("--scatter-n", type=int, default=None, help="散点图子样本数")
    sp.add_argument("--out", default=None, help="输出 PNG 路径")
    sp.add_argument("--seed", type=int, default=None, help="抽样随机种子")
    sp.set_defaults(func=cmd_visualize)

    sp = sub.add_parser("extract", help="抽取 WPAC 紧凑 h5 + 标签 CSV")
    sp.add_argument("--h5", required=True, help="原始 TCIR h5（含 matrix + info）")
    sp.add_argument("--csv", required=True, help="wpac_info.csv（matrix_index/Vmax/ID/data_set）")
    sp.add_argument("--out-h5", required=True, help="输出紧凑 h5 路径")
    sp.add_argument("--out-csv", required=True, help="输出标签 CSV 路径")
    sp.add_argument("--channels", nargs="+", default=["IR1", "WV", "PMW"], help="保留通道")
    sp.add_argument("--resize", type=int, default=96, help="下采样尺寸")
    sp.add_argument("--start", type=int, default=27000, help="WPAC 起始帧（跳过前部无关帧）")
    sp.add_argument("--batch", type=int, default=1000, help="读取批大小")
    sp.add_argument("--stats-out", default="data_tcir_wpac_stats.json", help="统计 JSON 输出")
    sp.set_defaults(func=cmd_extract)

    sp = sub.add_parser("download", help="下载 TCIR 数据集（NTU 镜像，约 13GB）")
    sp.add_argument("--out", default="data", help="下载/解压目录")
    sp.add_argument("--no-extract", action="store_true", help="只下载不解压")
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("stats", help="计算逐通道 mean/std 用于归一化")
    sp.add_argument("--h5", required=True, help="TCIR h5 路径")
    sp.add_argument("--channels", nargs="+", default=["IR1", "WV", "PMW"], help="通道")
    sp.add_argument("--sample-every", type=int, default=1, help="每隔多少帧采样一次")
    sp.add_argument("--csv", default=None, help="仅统计该 CSV 指向的子集帧")
    sp.add_argument("--index-col", default="matrix_index", help="CSV 中帧号列名")
    sp.add_argument("--out", default="data/tcir_stats.json", help="输出 JSON 路径")
    sp.set_defaults(func=cmd_stats)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        sys.exit(1)
    if hasattr(args, "config"):
        args.config = resolve_config(args.config)
    args.func(args)


if __name__ == "__main__":
    main()
