# -*- coding: utf-8 -*-
"""推理入口。
运行:
  python predict.py --image path/to/img.png --set experiment.name=xxx
  python predict.py --folder path/to/dir --set experiment.name=xxx --out preds.csv
"""
from tcintens.engine.predict import predict
from tcintens.utils.config import build_arg_parser, load_config


def main():
    parser = build_arg_parser()
    parser.add_argument("--image", default=None, help="单张图片路径")
    parser.add_argument("--folder", default=None, help="图片文件夹路径")
    parser.add_argument("--out", default=None, help="结果 CSV 输出路径")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    predict(cfg, image=args.image, folder=args.folder, out=args.out)


if __name__ == "__main__":
    main()
