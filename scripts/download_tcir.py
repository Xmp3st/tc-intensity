# -*- coding: utf-8 -*-
"""
download_tcir.py —— 下载 TCIR 数据集（Hsuan-Tien Lin 团队）

数据源（NTU 镜像，约 13GB，分 3 个分卷）：
    TCIR-ALL_2017.h5.tar.gz
    TCIR-ATLN_EPAC_WPAC.h5.tar.gz
    TCIR-CPAC_IO_SH.h5.tar.gz

也可从 Google Drive 镜像获取：
    https://drive.google.com/drive/folders/19Jo9iLi2b5qFoPan78SyojBxDHr-egJw

用法：
    python scripts/download_tcir.py --out data/
该脚本仅负责「下载 + 解压」，下载完成后把得到的 *.h5 路径填到
configs/tcir_regression.yaml 的 data.tcir.h5_path 即可。
"""
import argparse
import os
import subprocess
import sys

BASE = "https://learner.csie.ntu.edu.tw/~boyochen/TCIR"
FILES = [
    "TCIR-ALL_2017.h5.tar.gz",
    "TCIR-ATLN_EPAC_WPAC.h5.tar.gz",
    "TCIR-CPAC_IO_SH.h5.tar.gz",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="下载/解压目录")
    ap.add_argument("--no-extract", action="store_true", help="只下载不解压")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for fn in FILES:
        url = f"{BASE}/{fn}"
        tar_path = os.path.join(args.out, fn)
        if os.path.exists(tar_path):
            print(f"[skip] 已存在: {tar_path}")
        else:
            print(f"[download] {url}")
            try:
                subprocess.run(["wget", "-c", url, "-O", tar_path], check=True)
            except FileNotFoundError:
                subprocess.run(["curl", "-L", "-C", "-", "-o", tar_path, url], check=True)
        if not args.no_extract:
            print(f"[extract] {tar_path}")
            subprocess.run(["tar", "-xzf", tar_path, "-C", args.out], check=True)

    print("\n完成。把 configs/tcir_regression.yaml 中的 "
          "data.tcir.h5_path 指向解压出的 .h5 文件（如 data/TCIR-ATLN_EPAC_WPAC.h5）。")


if __name__ == "__main__":
    sys.exit(main())
