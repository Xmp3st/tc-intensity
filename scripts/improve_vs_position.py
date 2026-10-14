# -*- coding: utf-8 -*-
"""improve_vs_position.py —— 判断「修复收益」与「缺测位置（中心 vs 边缘）」的关系。

背景：TCIR 图像以风暴中心为画面中心，故「缺测落在画面中心区」≈「缺测盖住了风暴核心」。
对 resnet50 的 637 个 glitch 测试样本，逐样本计算缺测掩码的**位置指标**，
再与其修复收益 improve = |err_raw| - |err_corrected| 做相关/分组分析。

位置指标（均按 96x96 归一化）
  central_frac : 缺测像素中落在「中心圆盘」(半径 0.25*min(H,W)=24px) 内的比例  ← 主指标
  c_min        : 缺测像素到画面中心的**最小**距离 / 半对角线  (0=盖住中心)
  c_cent       : 缺测质心到画面中心的距离 / 半对角线
  row_off      : |质心行 - 中心行| / 中心行  (IR1 是横向条带，衡量条带的垂直位置)
  touches_border: 掩码是否触及任一图像边界

⚠️ 混杂控制：improve 本身与 true_vmax 强相关（见 §5.5），故除原始相关外，
   另报**控制 true_vmax 后的偏相关**，并按强度分层比较中心/边缘组。

产物
  outputs/tcir_wpac/tcir_wpac_bb/improve_vs_position.png
  outputs/tcir_wpac/tcir_wpac_bb/improve_vs_position.csv   (逐样本位置指标)
  outputs/tcir_wpac/tcir_wpac_bb/improve_vs_position.json  (统计)
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage, stats

from tcintens.data.band_health import detect_missing_region
from tcintens.data.dataset import create_datasets
from tcintens.data.tcir import CHANNELS, TCIRDataset
from tcintens.utils.config import load_config
from tcintens.utils.logger import get_logger
from tcintens.utils.plotting import use_cjk_font
from tcintens.utils.repro import align_preprocessing

BB_DIR = "outputs/tcir_wpac/tcir_wpac_bb"
MODEL = "bb_resnet50_s42"
METHODS = ("griddata", "crosschannel")
COLOR = {"griddata": "#2b6cb0", "crosschannel": "#c05621"}


def partial_pearson(x, y, z):
    """控制 z 后 x 与 y 的偏相关。"""
    def _resid(a, b):
        s, i, *_ = stats.linregress(b, a)
        return a - (s * b + i)
    rx, ry = _resid(x, z), _resid(y, z)
    r, p = stats.pearsonr(rx, ry)
    return float(r), float(p)


def partial_pearson_multi(x, y, Z):
    """同时控制多个变量（Z: (n,k)）的偏相关。"""
    Z = np.column_stack([np.ones(len(x))] + [np.asarray(z, float) for z in Z])
    bx, *_ = np.linalg.lstsq(Z, x, rcond=None)
    by, *_ = np.linalg.lstsq(Z, y, rcond=None)
    rx, ry = x - Z @ bx, y - Z @ by
    r, p = stats.pearsonr(rx, ry)
    return float(r), float(p)


def make_dataset(cfg):
    cfg, _ = align_preprocessing(cfg, logger=get_logger("pos"))
    ds, _ = create_datasets(cfg)
    td = ds["test"]
    return cfg, td


def raw_reader(td):
    """同索引、同标签顺序的「原始未修复」读取器（Kelvin，0=缺测）。"""
    return TCIRDataset(
        h5_path=td.h5_path, indices=list(td.indices), channels=td.channels,
        target=td.target, label_unit=td.label_unit, nan_mode=td.nan_mode,
        normalize=False, resize=td.resize, split="test",
        missing_band="none", frame_labels=list(td.labels))


def mask_params(cfg):
    tcfg = cfg.to_dict()["data"]["tcir"]
    mr_ch = tcfg.get("missing_region_channels", ["IR1", "WV"])
    mr_lt = tcfg.get("missing_region_low_thr", [150.0, 150.0, -999.0])
    lt_list = []
    for ci, c in enumerate(CHANNELS):
        if c in mr_ch and ci < len(mr_lt):
            lt_list.append(float(mr_lt[ci]))
        else:
            lt_list.append(-999.0)
    return dict(
        sentinel=float(tcfg.get("missing_region_sentinel", 0.0)),
        low_thr=lt_list,
        min_blob=int(tcfg.get("missing_region_min_blob", 20)),
        dilate=int(tcfg.get("missing_region_dilate", 1)),
    )


def position_metrics(mask, H, W):
    """mask: (H,W) bool → 位置指标 dict。"""
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    halfdiag = float(np.hypot(cy, cx))
    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.hypot(yy - cy, xx - cx)
    central_r = 0.25 * min(H, W)
    out = dict(area_frac=float(mask.mean()), n_px=int(mask.sum()))
    if not mask.any():
        out.update(central_frac=0.0, c_min=1.0, c_cent=1.0, row_off=1.0,
                   col_off=1.0, touches_border=False)
        return out
    d = dist[mask]
    com = ndimage.center_of_mass(mask)
    out["central_frac"] = float((d <= central_r).mean())
    out["c_min"] = float(d.min() / halfdiag)
    out["c_cent"] = float(np.hypot(com[0] - cy, com[1] - cx) / halfdiag)
    out["row_off"] = float(abs(com[0] - cy) / cy)
    out["col_off"] = float(abs(com[1] - cx) / cx)
    out["touches_border"] = bool(mask[0].any() or mask[-1].any()
                                 or mask[:, 0].any() or mask[:, -1].any())
    return out


def compute_positions(cfg, td, rd, test_idx, id_map):
    p = mask_params(cfg)
    rows = []
    for i in test_idx:
        arr, y = rd[int(i)]
        arr = np.asarray(arr, dtype=np.float64)          # (C,H,W) Kelvin
        m = detect_missing_region(arr, sentinel=p["sentinel"],
                                  low_thr=p["low_thr"],
                                  min_blob=p["min_blob"], dilate=p["dilate"])
        C, H, W = m.shape
        per_ch = {c: m[ci].mean() for ci, c in enumerate(CHANNELS[:C])}
        rec = {"test_index": int(i), "true_vmax": float(y),
               "ID": id_map.get(int(i), ""),
               "area_frac_ir1": float(per_ch.get("IR1", 0.0)),
               "area_frac_wv": float(per_ch.get("WV", 0.0)),
               "n_ch_missing": int(sum(1 for c in ("IR1", "WV")
                                       if per_ch.get(c, 0) > 0))}
        rec.update(position_metrics(m.any(0), H, W))
        rows.append(rec)
    return pd.DataFrame(rows)


def analyze(df, tag):
    """返回统计 dict。df 需含 true_vmax / abs_err_improve / central_frac 等。"""
    y = df["abs_err_improve"].to_numpy(float)
    vmax = df["true_vmax"].to_numpy(float)
    st = {"tag": tag, "n": int(len(df))}
    z = df["area_frac"].to_numpy(float)
    for met in ("central_frac", "c_cent", "c_min", "row_off", "area_frac"):
        x = df[met].to_numpy(float)
        if np.ptp(x) == 0:
            st[met] = {"pearson": float("nan"), "p": float("nan"),
                       "spearman": float("nan"), "p_sp": float("nan"),
                       "partial_r": float("nan"), "partial_p": float("nan"),
                       "partial2_r": float("nan"), "partial2_p": float("nan")}
            continue
        r, p = stats.pearsonr(x, y)
        rho, psp = stats.spearmanr(x, y)
        pr, pp = partial_pearson(x, y, vmax)
        pr2, pp2 = partial_pearson_multi(x, y, [vmax, z])
        st[met] = {"pearson": float(r), "p": float(p),
                   "spearman": float(rho), "p_sp": float(psp),
                   "partial_r": float(pr), "partial_p": float(pp),
                   "partial2_r": float(pr2), "partial2_p": float(pp2)}
    # 混杂检查：位置指标本身是否与强度相关
    st["confound_corr_central_vs_vmax"] = float(
        stats.pearsonr(df["central_frac"].to_numpy(float), vmax)[0])

    # 中心组 vs 边缘组（central_frac > 0 视为触及中心区）
    cen = df["central_frac"] > 0
    st["n_central"] = int(cen.sum())
    st["n_edge"] = int((~cen).sum())
    if cen.sum() >= 5 and (~cen).sum() >= 5:
        a, b = y[cen], y[~cen]
        t, pt = stats.ttest_ind(a, b, equal_var=False)
        u, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
        st["central_vs_edge"] = {
            "mean_improve_central": float(a.mean()),
            "mean_improve_edge": float(b.mean()),
            "diff": float(a.mean() - b.mean()),
            "t": float(t), "p_ttest": float(pt),
            "u": float(u), "p_mannwhitney": float(pu),
            "mean_abs_err_raw_central": float(df.loc[cen, "abs_err_raw"].mean()),
            "mean_abs_err_raw_edge": float(df.loc[~cen, "abs_err_raw"].mean()),
            "mean_vmax_central": float(vmax[cen].mean()),
            "mean_vmax_edge": float(vmax[~cen].mean()),
            "mean_area_central": float(df.loc[cen, "area_frac"].mean()),
            "mean_area_edge": float(df.loc[~cen, "area_frac"].mean()),
        }
    # 强度分层
    st["by_vmax_bin"] = []
    edges = [0, 50, 75, 200]
    for k in range(len(edges) - 1):
        m = (vmax >= edges[k]) & (vmax < edges[k + 1])
        if m.sum() < 10:
            continue
        mc = m & cen
        me = m & ~cen
        st["by_vmax_bin"].append(dict(
            lo=edges[k], hi=edges[k + 1], n=int(m.sum()),
            n_central=int(mc.sum()), n_edge=int(me.sum()),
            mean_central=float(y[mc].mean()) if mc.sum() else float("nan"),
            mean_edge=float(y[me].mean()) if me.sum() else float("nan"),
            diff=float(y[mc].mean() - y[me].mean())
            if mc.sum() and me.sum() else float("nan")))
    return st


def binned(x, y, nb=6):
    edges = np.linspace(x.min(), x.max(), nb + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nb - 1)
    out = []
    for k in range(nb):
        m = idx == k
        if m.sum() == 0:
            continue
        out.append(dict(c=0.5 * (edges[k] + edges[k + 1]), n=int(m.sum()),
                        mean=float(y[m].mean()),
                        sem=float(y[m].std(ddof=1) / np.sqrt(m.sum()))
                        if m.sum() > 1 else 0.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/tcir_wpac_train.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", default=os.path.join(BB_DIR, "improve_vs_position"))
    args = ap.parse_args()

    use_cjk_font()
    cfg = load_config(args.config, args.set)
    cfg.set("experiment.name", MODEL)
    cfg.set("experiment.output_dir", BB_DIR)
    cfg, td = make_dataset(cfg)
    rd = raw_reader(td)

    data, stats_all = {}, {}
    for met in METHODS:
        csv = os.path.join(BB_DIR, "corrraw", f"resnet50__{met}",
                           "corrected_vs_raw.csv")
        if not os.path.exists(csv):
            print(f"[跳过] 缺 {csv}")
            continue
        df = pd.read_csv(csv)
        id_map = dict(zip(df["test_index"].astype(int), df["ID"].astype(str)))
        pos = compute_positions(cfg, td, rd, df["test_index"].tolist(), id_map)
        mg = df.merge(pos, on="test_index", suffixes=("", "_pos"))
        # 校验标签一致
        bad = int((np.abs(mg["true_vmax"] - mg["true_vmax_pos"]) > 1e-6).sum())
        if bad:
            print(f"[警告] {met}: {bad} 个样本 true_vmax 不一致（索引口径需检查）")
        id_map2 = dict(zip(pos["test_index"], pos["ID"]))
        mg["ID"] = mg["test_index"].map(id_map2)
        data[met] = mg
        stats_all[met] = analyze(mg, met)
        print(f"\n=== {met} ===")
        print(f"  n={stats_all[met]['n']}  中心组 {stats_all[met]['n_central']} / "
              f"边缘组 {stats_all[met]['n_edge']}")
        for met2 in ("central_frac", "c_cent", "c_min", "row_off", "area_frac"):
            d = stats_all[met][met2]
            print(f"  {met2:<13} r={d['pearson']:+.3f}(p={d['p']:.1e})  "
                  f"rho={d['spearman']:+.3f}  "
                  f"偏(控Vmax) r={d['partial_r']:+.3f}(p={d['partial_p']:.2f})  "
                  f"偏(控Vmax+面积) r={d['partial2_r']:+.3f}(p={d['partial2_p']:.2f})")
        ce = stats_all[met].get("central_vs_edge", {})
        if ce:
            print(f"  中心 vs 边缘 improve: {ce['mean_improve_central']:+.3f} vs "
                  f"{ce['mean_improve_edge']:+.3f}  差 {ce['diff']:+.3f}  "
                  f"t={ce['t']:.2f} p={ce['p_ttest']:.3f}  "
                  f"MannWhitney p={ce['p_mannwhitney']:.3f}")
            print(f"  (中心组 raw|err|={ce['mean_abs_err_raw_central']:.2f} / "
                  f"边缘组 {ce['mean_abs_err_raw_edge']:.2f}；"
                  f"Vmax 中心 {ce['mean_vmax_central']:.1f} / "
                  f"边缘 {ce['mean_vmax_edge']:.1f})")
        for b in stats_all[met]["by_vmax_bin"]:
            print(f"  Vmax[{b['lo']:>3},{b['hi']:>3}) n={b['n']:>3} "
                  f"中心 {b['mean_central']:+.3f}(n={b['n_central']}) / "
                  f"边缘 {b['mean_edge']:+.3f}(n={b['n_edge']})  差 {b['diff']:+.3f}")

    _plot(data, stats_all, args.out)
    if "griddata" in data:
        _plot_cases(data["griddata"], rd, cfg, args.out + "_cases")
    if data:
        pd.concat([d.assign(method=k) for k, d in data.items()]).to_csv(
            args.out + ".csv", index=False)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(stats_all, f, ensure_ascii=False, indent=2)
    print(f"\n[图表] {args.out}.png\n[数据] {args.out}.csv / .json")


def _plot(data, stats_all, out):
    if not data:
        return
    met = "griddata" if "griddata" in data else list(data)[0]
    df = data[met]
    fig = plt.figure(figsize=(15.5, 9.4))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.27,
                          left=0.06, right=0.985, top=0.89, bottom=0.08)

    # (1) 位置分布
    ax = fig.add_subplot(gs[0, 0])
    cf = df["central_frac"].to_numpy(float)
    ax.hist(cf, bins=20, color="#888780", alpha=0.85)
    ax.axvline(0, color="#c00", lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("central_frac（缺测落在中心圆盘内的比例）")
    ax.set_ylabel("样本数 (log)")
    ax.set_title(f"(1) 缺测位置分布（{met}，n={len(df)}）", fontsize=11.5)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.text(0.5, 0.92, f"=0（完全在边缘）: {int((cf == 0).sum())} 张",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#c00")

    # (2) improve vs central_frac
    ax = fig.add_subplot(gs[0, 1])
    y = df["abs_err_improve"].to_numpy(float)
    ax.axhline(0, color="#888", lw=1.0, ls="--")
    ax.scatter(cf, y, s=16, alpha=0.5, color="#2b6cb0", edgecolor="none")
    if np.ptp(cf) > 0:
        bb = binned(cf, y, 5)
        ax.plot([r["c"] for r in bb], [r["mean"] for r in bb], "o-",
                color="black", lw=1.8, ms=5, label="分箱均值(5 箱)")
        d = stats_all[met]["central_frac"]
        ax.set_xlabel("central_frac")
        ax.set_ylabel("improve (knots)")
        ax.set_title(f"(2) improve vs 中心度  r={d['pearson']:+.3f} "
                     f"(偏相关 {d['partial_r']:+.3f})", fontsize=11.5)
        ax.legend(fontsize=8.5)
    ax.grid(ls=":", alpha=0.4)

    # (3) 中心组 vs 边缘组
    ax = fig.add_subplot(gs[0, 2])
    cen = cf > 0
    groups = [y[cen], y[~cen]]
    labels = [f"中心组\nn={int(cen.sum())}", f"边缘组\nn={int((~cen).sum())}"]
    _kw = {"tick_labels": labels}
    try:
        bp = ax.boxplot(groups, widths=0.5, patch_artist=True, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="#c00",
                                       markeredgecolor="#c00", markersize=6),
                        **_kw)
    except TypeError:                       # 旧版 matplotlib 用 labels=
        bp = ax.boxplot(groups, widths=0.5, patch_artist=True, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="#c00",
                                       markeredgecolor="#c00", markersize=6),
                        labels=labels)
    for patch, col in zip(bp["boxes"], ["#F09595", "#B5D4F4"]):
        patch.set_facecolor(col)
    for i, g in enumerate(groups):
        ax.scatter(np.random.default_rng(0).normal(i + 1, 0.06, len(g)), g,
                   s=8, alpha=0.35, color="#333", edgecolor="none")
    ce = stats_all[met].get("central_vs_edge", {})
    if ce:
        ax.set_title(f"(3) 中心 vs 边缘  improve 差异 {ce['diff']:+.2f} kn\n"
                     f"Welch t={ce['t']:.2f}, p={ce['p_ttest']:.3f}",
                     fontsize=11.5)
    ax.axhline(0, color="#888", lw=1.0, ls="--")
    ax.set_ylabel("improve (knots)")
    ax.grid(axis="y", ls=":", alpha=0.4)

    # (4) 强度分层
    ax = fig.add_subplot(gs[1, 0])
    bins = stats_all[met]["by_vmax_bin"]
    xs = np.arange(len(bins))
    w = 0.36
    ax.bar(xs - w / 2, [b["mean_central"] for b in bins], w, color="#E24B4A",
           label="中心组", edgecolor="white")
    ax.bar(xs + w / 2, [b["mean_edge"] for b in bins], w, color="#378ADD",
           label="边缘组", edgecolor="white")
    for i, b in enumerate(bins):
        for xi, v in ((i - w / 2, b["mean_central"]), (i + w / 2, b["mean_edge"])):
            if np.isfinite(v):
                ax.text(xi, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(0, color="#333", lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{b['lo']}-{b['hi']}\nn={b['n']}" for b in bins])
    ax.set_xlabel("true Vmax 分档 (knots)")
    ax.set_ylabel("improve (knots)")
    ax.set_title("(4) 控制强度后：中心 vs 边缘", fontsize=11.5)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.legend(fontsize=8.5)

    # (5) row_off（条带垂直位置）
    ax = fig.add_subplot(gs[1, 1])
    ro = df["row_off"].to_numpy(float)
    ax.axhline(0, color="#888", lw=1.0, ls="--")
    ax.scatter(ro, y, s=16, alpha=0.5, color="#c05621", edgecolor="none")
    bb = binned(ro, y, 5)
    ax.plot([r["c"] for r in bb], [r["mean"] for r in bb], "o-", color="black",
            lw=1.8, ms=5, label="分箱均值(5 箱)")
    d = stats_all[met]["row_off"]
    ax.set_xlabel("row_off（0=条带穿过垂直中线，1=贴上下边缘）")
    ax.set_ylabel("improve (knots)")
    ax.set_title(f"(5) improve vs 条带垂直位置  r={d['pearson']:+.3f} "
                 f"(偏相关 {d['partial_r']:+.3f})", fontsize=11.5)
    ax.legend(fontsize=8.5)
    ax.grid(ls=":", alpha=0.4)

    # (6) 两方法对比
    ax = fig.add_subplot(gs[1, 2])
    for k, m2 in enumerate(METHODS):
        if m2 not in stats_all:
            continue
        ce2 = stats_all[m2].get("central_vs_edge", {})
        if not ce2:
            continue
        ax.bar([0 + k * 0.36, 1 + k * 0.36],
               [ce2["mean_improve_central"], ce2["mean_improve_edge"]],
               0.34, color=COLOR[m2], label=m2, edgecolor="white")
    ax.set_xticks([0.18, 1.18])
    ax.set_xticklabels(["中心组", "边缘组"])
    ax.axhline(0, color="#333", lw=1.0)
    ax.set_ylabel("improve (knots)")
    ax.set_title("(6) 两种补全方法一致性", fontsize=11.5)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.legend(fontsize=8.5)

    fig.suptitle(f"ResNet50 · 修复收益 improve 与缺测位置（中心 vs 边缘）的关系 "
                 f"· glitch 子集 n={len(df)}", fontsize=13.5, y=0.962)
    fig.savefig(out + ".png", dpi=150)
    plt.close(fig)


def _plot_cases(df, rd, cfg, out):
    """中心缺测 vs 边缘缺测 各 3 例：IR1 底图 + 缺测掩码红色叠加 + 误差读数。"""
    p = mask_params(cfg)
    ir1 = rd.channels.index("IR1") if "IR1" in rd.channels else 0
    cen = df["central_frac"] > 0
    picks = [
        ("中心缺测 · 收益最大的 3 例", df[cen].sort_values(
            "abs_err_improve", ascending=False).head(3)),
        ("边缘缺测 · 收益最差（含负）的 3 例", df[~cen].sort_values(
            "abs_err_improve", ascending=True).head(3)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 9.0))
    for r, (title, sub) in enumerate(picks):
        for c in range(3):
            ax = axes[r][c]
            if c >= len(sub):
                ax.axis("off")
                continue
            row = sub.iloc[c]
            arr, _ = rd[int(row["test_index"])]
            arr = np.asarray(arr, dtype=np.float64)
            m = detect_missing_region(arr, sentinel=p["sentinel"],
                                      low_thr=p["low_thr"],
                                      min_blob=p["min_blob"], dilate=p["dilate"])
            ch = arr[ir1]
            valid = ch > 1.0
            vmin, vmax = (np.percentile(ch[valid], [2, 98]) if valid.sum() > 100
                          else (ch.min(), ch.max()))
            g = np.clip((ch - vmin) / max(vmax - vmin, 1e-6), 0, 1)
            rgb = np.dstack([g, g, g])
            mk = m[ir1]
            rgb[mk] = np.array([0.90, 0.20, 0.20])
            ax.imshow(rgb, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            imp = row["abs_err_improve"]
            col = "#A32D2D" if imp < 0 else "#0F6E56"
            ax.set_title(
                f"{row['ID']}  true={row['true_vmax']:.0f} kn  "
                f"缺测 {row['area_frac']*100:.0f}%（IR1 {mk.mean()*100:.0f}%）\n"
                f"raw {row['pred_raw']:.1f} (Δ{row['err_raw']:+.1f}) → "
                f"修复 {row['pred_corrected']:.1f} (Δ{row['err_corrected']:+.1f})  "
                f"improve {imp:+.1f} kn",
                fontsize=8.6, color=col)
            ax.set_xlabel(f"central_frac={row['central_frac']:.2f}  "
                          f"row_off={row['row_off']:.2f}", fontsize=8.4)
        axes[r][0].set_ylabel(title, fontsize=11)
    fig.suptitle("ResNet50 · 中心缺测（上）vs 边缘缺测（下）：IR1 底图 + 缺测掩码（红）"
                 " · 红色像素=被修复区域", fontsize=12.5, y=0.965)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out + ".png", dpi=150)
    plt.close(fig)
    print(f"[图表] {out}.png")


if __name__ == "__main__":
    main()
