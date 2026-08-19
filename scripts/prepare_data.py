# -*- coding: utf-8 -*-
"""
prepare_data.py —— 把真实数据整理成框架要求的 labels.csv

支持两种来源：
  1) 类别子文件夹：--src data/raw  （data/raw/TY/*.png, data/raw/TS/*.png ...）
  2) 已有标注 CSV：--src labels_raw.csv --image-col image --label-col intensity

输出：--out labels.csv，列: image_path, label, wind_speed_ms(可选)

可选：--wind-map wind_map.json  把类别名映射到风速(m/s)
      --copy-to data/processed  顺便把图片按类别复制到标准结构
"""
import argparse
import csv
import json
import os
import shutil


def from_folder(src, wind_map):
    rows = []
    for name in sorted(os.listdir(src)):
        d = os.path.join(src, name)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
                rel = os.path.relpath(os.path.join(d, fn), start=os.getcwd())
                wind = wind_map.get(name, "") if wind_map else ""
                rows.append((rel, name, wind))
    return rows


def from_csv(src, image_col, label_col, wind_col, wind_map):
    rows = []
    with open(src, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img = row[image_col].strip()
            lab = row[label_col].strip()
            if wind_col and row.get(wind_col):
                wind = row[wind_col]
            elif wind_map:
                wind = wind_map.get(lab, "")
            else:
                wind = ""
            rows.append((img, lab, wind))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="类别子文件夹 或 原始标注 CSV")
    ap.add_argument("--out", default="data/labels.csv")
    ap.add_argument("--image-col", default="image")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--wind-col", default="wind_speed_ms")
    ap.add_argument("--wind-map", default=None, help="类别->风速 的 JSON 文件路径")
    ap.add_argument("--copy-to", default=None, help="可选：复制到标准结构目录")
    args = ap.parse_args()

    wind_map = json.load(open(args.wind_map, encoding="utf-8")) if args.wind_map else None

    if args.src.lower().endswith(".csv"):
        rows = from_csv(args.src, args.image_col, args.label_col, args.wind_col, wind_map)
    else:
        rows = from_folder(args.src, wind_map)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "label", "wind_speed_ms"])
        for r in rows:
            w.writerow(r)
    print(f"已写出 {len(rows)} 条标注 -> {args.out}")

    if args.copy_to:
        for rel, lab, _ in rows:
            dst = os.path.join(args.copy_to, lab, os.path.basename(rel))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(rel, dst)
        print(f"已复制到标准结构 -> {args.copy_to}")


if __name__ == "__main__":
    main()
