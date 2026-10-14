# -*- coding: utf-8 -*-
"""analyze_missing_shape.py —— 诊断 WPAC 紧凑集里「局部区域缺失」的真实形态。

目的：为「改进部分缺失图像的补全」提供事实依据。回答：
  1. 有多少帧存在 IR1/WV 局部缺测（占比）
  2. 缺测块的**尺寸分布**（小点状 / 细条带 / 大块）
  3. 缺测块的**形状**（bbox 长宽比、是否贴边、是否整行/整列）
  4. 是否存在「整通道缺失」的帧
  5. 典型样本可视化（掩码 + 原图）

输出
  outputs/missing_shape_analysis.png   直方图 + 案例掩码
  outputs/missing_shape_stats.json     机器可读统计
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

H5 = "data/wpac_96.h5"
CH = ["IR1", "WV", "PMW"]
LOW_THR = [150.0, 150.0, -999.0]     # PMW 用 -999 表示禁用区域检测
SENTINEL = 0.0
MIN_BLOB = 20
DILATE = 1
OUT_PNG = "outputs/missing_shape_analysis.png"
OUT_JSON = "outputs/missing_shape_stats.json"


def analyze():
    import h5py
    from scipy import ndimage

    rng = np.random.default_rng(0)
    stats = {
        "h5": H5, "sentinel": SENTINEL, "low_thr": LOW_THR,
        "min_blob": MIN_BLOB, "dilate": DILATE,
        "per_channel": {},
    }
    blob_sizes = {"IR1": [], "WV": []}
    blob_bbox = {"IR1": [], "WV": []}      # (h, w)
    img_frac = {"IR1": [], "WV": []}
    n_img = 0
    edge_touch = {"IR1": 0, "WV": 0}
    n_blob_total = {"IR1": 0, "WV": 0}
    examples = []                          # (idx, ch, mask) 代表案例

    with h5py.File(H5, "r") as hf:
        mat = hf["matrix"]
        N = mat.shape[0]
        for s in range(0, N, 512):
            e = min(s + 512, N)
            x = mat[s:e][..., :3].astype(np.float64)     # (B,H,W,3)
            for ci in range(2):                          # 只查 IR1 / WV
                ch = x[..., ci]                          # (B,H,W)
                lt = LOW_THR[ci]
                eq = np.isclose(ch, SENTINEL, atol=1e-6) | (ch < lt)
                for k in range(eq.shape[0]):
                    n_img_k = s + k
                    ek = eq[k]
                    if not ek.any():
                        continue
                    lab, nlab = ndimage.label(ek, structure=np.ones((3, 3), bool))
                    sizes = ndimage.sum(np.ones_like(ek, np.int64), lab,
                                        index=range(1, nlab + 1))
                    big = np.where(sizes >= MIN_BLOB)[0]
                    if big.size == 0:
                        continue
                    name = CH[ci]
                    keep = np.zeros(nlab + 1, bool)
                    keep[1:][big] = True
                    mk = keep[lab]                        # 只保留 >=min_blob 的块
                    frac = float(mk.mean())
                    img_frac[name].append(frac)
                    # 逐块几何
                    objs = ndimage.find_objects(lab)
                    for b in big:
                        sl = objs[b]
                        h_, w_ = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
                        blob_sizes[name].append(int(sizes[b]))
                        blob_bbox[name].append((int(h_), int(w_)))
                        n_blob_total[name] += 1
                        if (sl[0].start == 0 or sl[0].stop == 96 or
                                sl[1].start == 0 or sl[1].stop == 96):
                            edge_touch[name] += 1
                    # 收集少量代表案例（张数少、面积中等的）
                    if len(examples) < 24 and rng.random() < 0.004:
                        examples.append((n_img_k, name, ci, mk.copy()))

            n_img = e
            if s % 5120 == 0:
                print(f"  已扫描 {s}/{N}", flush=True)

    for ci in range(2):
        name = CH[ci]
        fr = np.asarray(img_frac[name])
        bs = np.asarray(blob_sizes[name])
        bb = np.asarray(blob_bbox[name]) if blob_bbox[name] else np.zeros((0, 2), int)
        ar = (bb[:, 0] / np.maximum(bb[:, 1], 1)) if len(bb) else np.zeros(0)
        stats["per_channel"][name] = {
            "frames_with_blob": int(len(fr)),
            "frame_ratio": round(len(fr) / max(n_img, 1), 5),
            "n_blobs_total": int(n_blob_total[name]),
            "blob_size": {
                "min": int(bs.min()) if len(bs) else 0,
                "p50": float(np.percentile(bs, 50)) if len(bs) else 0,
                "p90": float(np.percentile(bs, 90)) if len(bs) else 0,
                "max": int(bs.max()) if len(bs) else 0,
                "mean": float(bs.mean()) if len(bs) else 0,
            },
            "frame_missing_frac": {
                "p50": float(np.percentile(fr, 50)) if len(fr) else 0,
                "p90": float(np.percentile(fr, 90)) if len(fr) else 0,
                "max": float(fr.max()) if len(fr) else 0,
            },
            "blob_aspect_h_over_w": {
                "p10": float(np.percentile(ar, 10)) if len(ar) else 0,
                "p50": float(np.percentile(ar, 50)) if len(ar) else 0,
                "p90": float(np.percentile(ar, 90)) if len(ar) else 0,
            },
            "blobs_touching_border": int(edge_touch[name]),
            "blobs_touching_border_ratio": round(
                edge_touch[name] / max(n_blob_total[name], 1), 4),
        }

    stats["n_images"] = int(n_img)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    _plot(stats, blob_sizes, img_frac, examples)
    print(json.dumps(stats["per_channel"], ensure_ascii=False, indent=2))
    print(f"\n已写出: {OUT_JSON} / {OUT_PNG}")
    return stats


def _plot(stats, blob_sizes, img_frac, examples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 8))
    # (1) 块大小分布
    ax1 = fig.add_subplot(2, 3, 1)
    bins = np.logspace(0, 4, 30)
    for name, col in [("IR1", "#1f77b4"), ("WV", "#ff7f0e")]:
        if blob_sizes[name]:
            ax1.hist(blob_sizes[name], bins=bins, alpha=.6, label=name, color=col)
    ax1.set_xscale("log")
    ax1.set_xlabel("blob size (pixels)")
    ax1.set_ylabel("count")
    ax1.set_title("Missing blob size distribution")
    ax1.legend()

    # (2) 每帧缺失面积占比
    ax2 = fig.add_subplot(2, 3, 2)
    for name, col in [("IR1", "#1f77b4"), ("WV", "#ff7f0e")]:
        if img_frac[name]:
            ax2.hist(img_frac[name], bins=40, alpha=.6, label=name, color=col)
    ax2.set_xlabel("missing fraction per frame")
    ax2.set_ylabel("count")
    ax2.set_title("Per-frame missing area")
    ax2.legend()

    # (3) 表格
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.axis("off")
    lines = ["channel   frames  blobs   p50size  p90size  border%"]
    for name in ["IR1", "WV"]:
        d = stats["per_channel"][name]
        lines.append(
            f"{name:7s} {d['frames_with_blob']:6d} {d['n_blobs_total']:6d} "
            f"{d['blob_size']['p50']:7.0f} {d['blob_size']['p90']:8.0f} "
            f"{d['blobs_touching_border_ratio']*100:6.1f}")
    ax3.text(0.0, 0.9, "\n".join(lines), family="monospace", fontsize=9, va="top")

    # (4-6) 案例掩码
    for j, ex in enumerate(examples[:3]):
        idx, name, ci, mk = ex
        ax = fig.add_subplot(2, 3, 4 + j)
        ax.imshow(mk, cmap="hot")
        ax.set_title(f"case idx={idx} {name} area={mk.mean()*100:.1f}%", fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    analyze()
