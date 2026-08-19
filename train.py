# -*- coding: utf-8 -*-
"""训练入口。运行: python train.py --config configs/default.yaml"""
from tcintens.engine.train import train
from tcintens.utils.config import parse_config


def main():
    cfg = parse_config()
    train(cfg)


if __name__ == "__main__":
    main()
