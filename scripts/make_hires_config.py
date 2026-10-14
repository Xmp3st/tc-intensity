# -*- coding: utf-8 -*-
"""make_hires_config.py —— 由主配置派生「高分辨率」训练配置

为什么需要：`--set` 的取值解析（utils/config.py::_coerce）只支持标量
（int/float/bool/null/str），**无法传入列表**，因此 norm_mean / norm_std
不能通过命令行覆盖。本脚本从主配置复制一份，替换分辨率相关的 5 个键，
保证其余超参与主配置完全一致（避免手工复制造成漂移）。

用法：
    python scripts/make_hires_config.py \
        --base configs/tcir_wpac_train.yaml --out configs/tcir_wpac_r160.yaml \
        --resize 160 --h5 data/wpac_160.h5 --csv data/wpac_compact.csv \
        --stats data/wpac_160_stats.json --name tcir_wpac_r160
"""
import argparse
import json
import sys

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/tcir_wpac_train.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resize", type=int, required=True)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--csv", default="data/wpac_compact.csv")
    ap.add_argument("--stats", default=None, help="该分辨率的 mean/std json；省略则沿用主配置")
    ap.add_argument("--name", required=True)
    ap.add_argument("--missing-band", default="interp",
                    help="缺失修正策略（默认 interp，与 v2 最终协议一致）")
    args = ap.parse_args()

    with open(args.base, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)

    t = d.setdefault("data", {}).setdefault("tcir", {})
    t["h5_path"] = args.h5
    t["csv_path"] = args.csv
    t["resize"] = args.resize
    t["missing_band"] = args.missing_band
    if args.stats:
        st = json.load(open(args.stats, "r", encoding="utf-8"))
        t["norm_mean"] = [round(float(v), 4) for v in st["mean"]]
        t["norm_std"] = [round(float(v), 4) for v in st["std"]]
        print(f"norm_mean={t['norm_mean']}")
        print(f"norm_std ={t['norm_std']}")

    d.setdefault("experiment", {})["name"] = args.name

    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, allow_unicode=True, sort_keys=False)
    print(f"wrote {args.out}  (resize={args.resize}, h5={args.h5}, name={args.name}, "
          f"missing_band={args.missing_band})")


if __name__ == "__main__":
    sys.exit(main())
