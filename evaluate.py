# -*- coding: utf-8 -*-
"""评估入口。运行: python evaluate.py --set experiment.name=xxx"""
from tcintens.engine.evaluate import evaluate
from tcintens.utils.config import parse_config


def main():
    cfg = parse_config()
    evaluate(cfg)


if __name__ == "__main__":
    main()
