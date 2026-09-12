# -*- coding: utf-8 -*-
"""TCIR 推理入口。

单张 .npy:       python predict.py --image frame.npy -c configs/tcir_wpac_train.yaml
文件夹 .npy:     python predict.py --folder ./frames/ -c ... --out preds.csv
从 TCIR h5 取帧: python predict.py --h5 data/wpac_96.h5 --indices 10260,9603 -c ... --out preds.csv

输出 Vmax（knots 与 m/s 同时给出），回归任务。
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tcintens.engine.predict import predict
from tcintens.utils.config import build_arg_parser, load_config


def main():
    parser = build_arg_parser()
    parser.add_argument("--image", default=None, help="单张 .npy 路径（H,W,C，通道顺序同配置）")
    parser.add_argument("--folder", default=None, help="文件夹路径（内含若干 .npy）")
    parser.add_argument("--h5", default=None, help="TCIR matrix h5 路径，配合 --index/--indices")
    parser.add_argument("--index", type=int, default=None, help="从 h5 取单帧（matrix 行号）")
    parser.add_argument("--indices", default=None, help="从 h5 取多帧，逗号分隔，如 100,200,300")
    parser.add_argument("--out", default=None, help="结果 CSV 输出路径")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    predict(cfg, image=args.image, folder=args.folder, h5=args.h5,
            index=args.index, indices=args.indices, out=args.out)


if __name__ == "__main__":
    main()
