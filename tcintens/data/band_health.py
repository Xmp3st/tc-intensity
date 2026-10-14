# -*- coding: utf-8 -*-
"""
band_health.py —— 频带(通道)缺失检测与修正

背景：
    TCIR 的 PMW(被动微波)等通道仅在卫星过顶时有观测，缺测时通常被写成全 0
    （抽取阶段 np.nan_to_num 已把 NaN 置 0）。缺测通道若直接送进网络，会引入
    一个恒定的异常值（归一化后仍是常数偏移），既浪费通道又可能干扰学习。

本模块提供：
    1. detect_missing_channels : 判定 (C,H,W) 中哪些通道整通道缺失
    2. correct_missing         : 按策略修正（none / zero / mean）
    3. scan_dataset            : 向量化扫描整个 h5，统计每通道缺失数
    4. ensure_band_mask        : 取/生成缺失掩码缓存（供 drop 策略复用）

缺失判定（mode）：
    constant : 整通道 min==max（即常数）。缺测通道恒为常数(0)，真实卫星通道必有空间变化，
              因此该判据稳健、默认采用。
    var      : 通道方差 < var_eps。可作为补充。

修正策略（strategy）：
    none : 不处理（保持现状，即 0；作为对照基线）
    zero : 显式把缺失通道置 0（与 none 数值等价，但语义明确）
    mean : 用该通道原始均值填充缺失通道 -> 归一化后变为 0（中性，不引入偏移）
"""
import os

import numpy as np


# --------------------------------------------------------------------------- #
# 单样本检测 / 修正
# --------------------------------------------------------------------------- #
def detect_missing_channels(arr: np.ndarray, mode: str = "constant",
                             var_eps: float = 1e-6) -> np.ndarray:
    """判定 (C, H, W) 数组中哪些通道整通道缺失，返回 (C,) bool。

    arr : (C, H, W) 原始空间（归一化前的浮点数组）
    mode: "constant" -> min==max；"var" -> 方差 < var_eps
    """
    arr = np.asarray(arr, dtype=np.float64)
    if mode == "constant":
        mn = arr.min(axis=(-2, -1))
        mx = arr.max(axis=(-2, -1))
        return (mn == mx)
    elif mode == "var":
        return (arr.var(axis=(-2, -1)) < var_eps)
    else:
        raise ValueError(f"未知 missing_detect mode: {mode}")


def correct_missing(arr: np.ndarray, missing_mask: np.ndarray,
                    strategy: str = "none", fill_values=None) -> np.ndarray:
    """按策略修正缺失通道。

    arr         : (C, H, W) 原始空间浮点
    missing_mask : (C,) bool（detect_missing_channels 的输出）
    strategy    : none | zero | mean
    fill_values : (C,) 各通道原始均值（mean 策略用，归一化后变 0）
    """
    arr = np.asarray(arr, dtype=np.float64)
    out = arr.copy()
    missing = np.where(np.asarray(missing_mask))[0]
    if missing.size == 0 or strategy in ("none",):
        return out
    for c in missing:
        if strategy == "zero":
            out[c] = 0.0
        elif strategy == "mean":
            v = fill_values[c] if fill_values is not None else out[c].mean()
            out[c] = float(v)
        else:
            raise ValueError(f"未知 missing_band strategy: {strategy}")
    return out.astype(arr.dtype) if out.dtype != arr.dtype else out


# --------------------------------------------------------------------------- #
# 数据集级扫描 / 掩码缓存
# --------------------------------------------------------------------------- #
def scan_dataset(h5_path: str, channels, batch: int = 2000,
                 mode: str = "constant") -> tuple:
    """向量化扫描 h5，返回 (mask, stats)。

    mask : (N, C) int8，mask[i,c]==1 表示第 i 帧第 c 通道缺失
    stats: 人类可读摘要（每通道缺失数、含任意缺失的图数、每图缺失数分布等）
    """
    import h5py
    ch_idx = list(range(len(channels)))
    n_ch = len(channels)
    miss_const = np.zeros(n_ch, dtype=np.int64)
    miss_zero = np.zeros(n_ch, dtype=np.int64)
    raw_sum = np.zeros(n_ch, dtype=np.float64)
    raw_sq = np.zeros(n_ch, dtype=np.float64)
    raw_cnt = np.zeros(n_ch, dtype=np.int64)
    per_img = []
    N = 0
    with h5py.File(h5_path, "r") as hf:
        mat = hf["matrix"]
        total = mat.shape[0]
        for s in range(0, total, batch):
            e = min(s + batch, total)
            x = mat[s:e][..., ch_idx].astype(np.float64)   # (B,H,W,C)
            B = x.shape[0]
            N += B
            mn = x.min(axis=(1, 2))
            mx = x.max(axis=(1, 2))
            const = (mn == mx)
            zero = (mn == 0) & (mx == 0)
            miss_const += const.sum(axis=0)
            miss_zero += zero.sum(axis=0)
            raw_sum += x.sum(axis=(0, 1, 2))
            raw_sq += (x ** 2).sum(axis=(0, 1, 2))
            raw_cnt += B * x.shape[1] * x.shape[2]
            per_img.append(const.astype(np.int8))
    mask = np.concatenate(per_img, axis=0)
    mean = raw_sum / raw_cnt
    std = np.sqrt(np.clip(raw_sq / raw_cnt - mean ** 2, 1e-12, None))
    stats = {
        "h5_path": h5_path,
        "n_images": int(N),
        "channels": list(channels),
        "missing_constant": {channels[c]: int(miss_const[c]) for c in ch_idx},
        "missing_allzero": {channels[c]: int(miss_zero[c]) for c in ch_idx},
        "raw_mean": {channels[c]: round(float(mean[c]), 4) for c in ch_idx},
        "raw_std": {channels[c]: round(float(std[c]), 4) for c in ch_idx},
        "images_with_any_missing": int(mask.any(axis=1).sum()),
        "per_image_missing_count": {
            str(k): int((mask.sum(axis=1) == k).sum()) for k in range(n_ch + 1)
        },
    }
    return mask.astype(np.int8), stats


def ensure_band_mask(h5_path: str, channels, mask_path: str,
                     mode: str = "constant", batch: int = 2000) -> np.ndarray:
    """读取缺失掩码缓存；若不存在则用 scan_dataset 生成并保存。

    返回 (N, C) int8 掩码，行号与 h5 帧顺序(紧凑行号)一一对应。
    """
    if os.path.exists(mask_path):
        with np.load(mask_path) as z:
            return z["mask"].astype(np.int8)
    mask, _ = scan_dataset(h5_path, channels, batch=batch, mode=mode)
    os.makedirs(os.path.dirname(os.path.abspath(mask_path)), exist_ok=True)
    np.savez_compressed(mask_path, mask=mask.astype(np.int8),
                        channels=np.array(list(channels), dtype="S"))
    return mask


# --------------------------------------------------------------------------- #
# 局部区域缺失（图像内部某些空间区域的通道缺失）—— 检测 + 插值修复
# --------------------------------------------------------------------------- #
# 背景：TCIR 的 PMW 等被动微波通道为星下条带观测，常有「局部条带/块」缺测；
#       extract_wpac 已把 NaN 置 0 再做 bilinear 下采样，因此缺测区域 = 0 值，
#       并被涂抹成边界渐变。这与「整通道缺失」不同：这里是图像某块区域的
#       某些通道缺数据，应当用【空间插值/inpainting】就地填补，而非整通道丢弃。
# 缺失判定（哨兵值 sentinel + 物理下界 low_thr）：
#       缺测像素 = 精确等于 sentinel(默认 0) 的像素，或低于物理下界 low_thr 的像素。
#       引入 low_thr 是因为 bilinear 下采样把缺失块「涂抹」成了向 0 渐变的过渡带，
#       核心已是精确 0、边缘是异常低值（IR1/WV 亮温不可能 <150K），low_thr 一并捕获。
#       用连通块 + min_blob 过滤掉噪声级孤立点；可选 dilate 进一步覆盖过渡带。
#       注意：PMW 的 0 是合法值（如降雨率=0），默认不对其启用区域检测。
# 修复策略（method）：
#       griddata : scipy 线性插值（最近邻/均值兜底），零额外依赖，平滑自然
#       telea/ns : OpenCV 图像修复（若环境装了 cv2；否则自动回退 griddata）
#       整通道（或近乎整通道）缺失时无法插值 -> 回退为通道均值填充（中性）
# --------------------------------------------------------------------------- #
_STRUCT_CACHE = {}


def _struct(connectivity: int = 8) -> np.ndarray:
    """返回连通域结构元素（8 邻接 / 4 邻接），带缓存。"""
    if connectivity not in _STRUCT_CACHE:
        if connectivity == 4:
            _STRUCT_CACHE[connectivity] = np.array(
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        else:  # 8 邻接（默认，更贴合涂抹后的不规则缺测块）
            _STRUCT_CACHE[connectivity] = np.ones((3, 3), dtype=bool)
    return _STRUCT_CACHE[connectivity]


def _has_cv2() -> bool:
    import importlib.util as u
    return u.find_spec("cv2") is not None


def detect_missing_region(arr: np.ndarray, sentinel: float = 0.0,
                          low_thr=None, min_blob: int = 20, dilate: int = 0,
                          connectivity: int = 8) -> np.ndarray:
    """判定 (C, H, W) 数组中每个通道的「局部区域缺失」像素，返回 (C, H, W) bool。

    arr         : (C, H, W) 原始空间浮点（归一化前）
    sentinel    : 缺测哨兵值（TCIR 经 nan_to_num 后为 0）
    low_thr     : 物理下界（标量，None=不启用）；低于此值的像素也判为缺测，
                  用于捕获 bilinear 下采样造成的「向 0 渐变」涂抹过渡带
                  （IR1/WV 亮温不可能 <150K，故 150 可安全覆盖涂抹带）
    min_blob    : 缺测连通块最小像素数（96x96 空间），小于此视为噪声/合法，忽略
    dilate      : 对缺测掩码做二元膨胀迭代次数，覆盖下采样涂抹过渡带
    connectivity: 连通判定邻接数（4 / 8）
    """
    from scipy import ndimage
    arr = np.asarray(arr, dtype=np.float64)
    C, H, W = arr.shape
    mask = np.zeros((C, H, W), dtype=bool)
    struct = _struct(connectivity)
    lt = np.asarray(low_thr, dtype=np.float64) if low_thr is not None else None
    for c in range(C):
        # 逐通道禁用：low_thr 列表里该通道 <= -900 表示「不参与区域检测」
        if lt is not None and lt.ndim > 0 and c < len(lt) and lt[c] <= -900:
            continue
        eq = np.isclose(arr[c], sentinel, atol=1e-6)
        if lt is not None:
            if lt.ndim == 0:
                eq = eq | (arr[c] < float(lt))
            elif c < len(lt) and np.isfinite(lt[c]):
                eq = eq | (arr[c] < float(lt[c]))
        if not eq.any():
            continue
        lab, nlab = ndimage.label(eq, structure=struct)
        if nlab == 0:
            continue
        sizes = ndimage.sum(
            np.ones_like(eq, dtype=np.int64), lab, index=range(1, nlab + 1))
        keep = sizes >= min_blob
        if not keep.any():
            continue
        keep_lab = np.zeros(nlab + 1, dtype=bool)
        keep_lab[1:][keep] = True
        m = keep_lab[lab]
        if dilate and m.any():
            m = ndimage.binary_dilation(m, iterations=dilate,
                                        structure=struct)
        mask[c] = m
    return mask


def _col_interp(channel: np.ndarray, m: np.ndarray) -> np.ndarray:
    """逐列 1D 线性插值。

    IR1 的缺测以「横向细条带」为主（blob 的长宽比中位数仅 0.027），
    条带内同一行没有有效像素、但**同一列上下方**有 —— 故列向插值对该形态
    几乎就是正解，同时它也是调和插值的极佳初值（迭代次数可降一个量级）。
    """
    out = channel.astype(np.float64).copy()
    H, W = channel.shape
    valid_all = ~m
    gmean = float(channel[valid_all].mean()) if valid_all.any() else 0.0
    for j in range(W):
        cm = m[:, j]
        if not cm.any():
            continue
        cv = ~cm
        nv = int(cv.sum())
        if nv >= 2:
            out[cm, j] = np.interp(np.nonzero(cm)[0], np.nonzero(cv)[0],
                                   channel[cv, j])
        elif nv == 1:
            out[cm, j] = channel[cv, j][0]
        else:
            out[cm, j] = gmean
    return out


def harmonic_fill(channel: np.ndarray, m: np.ndarray, iters: int = 100,
                  init: str = "col") -> np.ndarray:
    """调和插值（harmonic / Laplace inpainting）。

    迭代求解带 Dirichlet 边界的拉普拉斯方程，等价于「在保持已知像素不变的
    前提下最小化缺失区内的梯度能量」—— 结果处处平滑、不会像散点三角剖分
    那样产生折线伪影。

    init="col"：用列向 1D 插值作初值，对细条带几乎已是正解，收敛快 10 倍
    （实测 100 次迭代即达收敛，而均值初值需 300+ 次）。
    """
    from scipy import ndimage
    m = np.asarray(m, dtype=bool)
    known = ~m
    if not known.any():
        return np.zeros_like(channel, dtype=np.float64)
    if init == "col":
        out = _col_interp(channel, m)
    else:
        out = channel.astype(np.float64).copy()
        out[m] = float(channel[known].mean())
    k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float64) / 4.0
    for _ in range(int(iters)):
        out[m] = ndimage.convolve(out, k, mode="nearest")[m]
    return out


def inpaint_regions_crosschannel(arr: np.ndarray, mask: np.ndarray,
                                 deg: int = 2, min_fit: int = 64,
                                 fallback_iters: int = 100,
                                 robust: bool = True,
                                 ref_idx=None, feather: int = 0) -> np.ndarray:
    """跨通道回归补全（+ 空间调和插值兜底）。

    依据（见 scripts/benchmark_inpaint.py 的定量评测）：
      * IR1(窗区) 与 WV(水汽) 亮温高度相关，可用一个通道预测另一个；
      * 两者的缺测**往往不同时发生**（IR1 缺测帧占 ~20%、但多为细条带；
        WV 缺测罕见、但发生时面积很大），因此互为可靠参考；
      * 实测真实数据里 WV 缺测区**外围 1 圈像素已偏低 1.2σ**（bilinear 下采样
        涂抹所致），纯空间插值会把这些被污染的像素当真值使用 —— 而全局回归
        只用「同帧有效像素」拟合，对边缘污染天然鲁棒。合成评测中：
        IR1 RMSE 11.11→7.94 K、WV 10.88→3.33 K；若模拟边缘污染，
        空间插值会崩到 120+ K 而本方法仍保持 ~9 K。

    做法：
      1. 按「缺测面积升序」逐通道修复 —— 先修好的通道可作后修通道的参考；
      2. 以参考通道的全部有效像素拟合多项式 y = f(ref)（默认 2 阶，
         并在归一化坐标上做，避免 260 K 量级导致的病态）；
      3. robust=True 时做一轮残差剔除（3×MAD）后重拟合，抑制被涂抹污染的
         过渡带像素对回归的拉扯；
      4. 若没有可用参考通道（如另一通道在同位置也缺），退回 harmonic_fill。

    feather : >0 时启用**边界羽化**：按「到缺测边界的距离」把跨通道预测与平滑的
              空间调和插值做线性融合 —— 边界处 (距离 0) 完全用调和插值、深度
              >=feather 像素处完全用跨通道预测。
              动因（scripts/seam_diagnostic.py 实测）：跨通道预测的数值精度最高
              （IR1 11.11→8.71 K），但因为它的纹理来自**另一通道**，在缺测区边界
              会留下台阶（接缝比中位数 1.21 vs griddata 1.00、p90 3.13 vs 1.71），
              下游 CNN 对这种伪影敏感。羽化可在保留内部精度的同时消除边界台阶。

    ref_idx : 允许充当「参考通道」的通道索引白名单（默认全部）。
              注意 PMW 的 0 是合法观测值（且缺测时也是 0，无法区分），
              把它当参考会让回归被大量 0 污染 —— 故调用方通常只传 IR1/WV。
    """
    arr = np.asarray(arr, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    C = arr.shape[0]
    out = arr.copy()
    done = np.zeros(C, dtype=bool)
    order = np.argsort([float(mask[c].mean()) for c in range(C)])
    cand = (list(range(C)) if ref_idx is None
            else [int(r) for r in ref_idx if 0 <= int(r) < C])

    for c in order:
        m = mask[c]
        if not m.any():
            done[c] = True
            continue
        ref = None
        for r in cand:
            if r != c and (done[r] or not mask[r].any()):
                ref = r
                break
        if ref is None:
            out[c] = harmonic_fill(arr[c], m, iters=fallback_iters)
            done[c] = True
            continue

        o = np.asarray(out[ref], dtype=np.float64)      # 参考通道（已确保无缺失）
        fit = ~m
        of, yf = o[fit], arr[c][fit]
        if of.size < min_fit or float(np.std(of)) < 1e-6:
            out[c] = harmonic_fill(arr[c], m, iters=fallback_iters)
            done[c] = True
            continue

        mu, sd = float(of.mean()), float(of.std()) + 1e-9
        A = np.vander((of - mu) / sd, deg + 1)
        coef, *_ = np.linalg.lstsq(A, yf, rcond=None)
        if robust:
            resid = yf - A @ coef
            med = float(np.median(resid))
            mad = float(np.median(np.abs(resid - med))) + 1e-6
            keep = np.abs(resid - med) < 3.0 * 1.4826 * mad
            if int(keep.sum()) >= min_fit and int(keep.sum()) < keep.size:
                Ak = np.vander(((of - mu) / sd)[keep], deg + 1)
                coef, *_ = np.linalg.lstsq(Ak, yf[keep], rcond=None)
        pred = np.vander((o[m] - mu) / sd, deg + 1) @ coef
        # 多项式在「拟合样本范围之外」会剧烈外推，而真实缺测区常落在分布边缘
        # （如台风眼墙），故用有效值的极差做一次宽松夹取，杜绝负亮温之类非物理结果。
        lo, hi = float(yf.min()), float(yf.max())
        span = max(hi - lo, 1.0)
        filled = np.clip(pred, lo - 0.25 * span, hi + 0.25 * span)

        if feather and int(feather) > 0:
            # 边界羽化：边界用平滑的空间调和插值，内部（深度>=feather）用跨通道预测
            from scipy import ndimage
            dist = ndimage.distance_transform_edt(m)          # 掩码内各点到边界的距离
            alpha = np.clip(dist / float(int(feather)), 0.0, 1.0)
            harm = harmonic_fill(arr[c], m, iters=fallback_iters)
            out[c][m] = alpha[m] * filled + (1.0 - alpha[m]) * harm[m]
        else:
            out[c][m] = filled
        done[c] = True
    return out


def inpaint_regions(arr: np.ndarray, mask: np.ndarray, method: str = "griddata",
                    fill_values=None, nan_fallback: str = "mean",
                    max_frac: float = 1.0, deg: int = 2,
                    iters: int = 100, ref_idx=None,
                    feather: int = 0) -> np.ndarray:
    """用插值法修复 (C, H, W) 中 mask=True 的缺失像素，返回修复后数组。

    arr         : (C, H, W) 原始空间浮点（在归一化前修复，物理意义更合理）
    mask        : (C, H, W) bool（detect_missing_region 输出）
    method      : "griddata" | "telea" | "ns" | "harmonic" | "crosschannel"
                  （telea/ns 需 cv2；harmonic = 调和插值；crosschannel = 跨通道回归，
                   两者请优先用 crosschannel —— 见 inpaint_regions_crosschannel）
    deg         : crosschannel 的多项式阶数（默认 2）
    iters       : harmonic / crosschannel 兜底用的迭代次数（默认 100）
    fill_values : (C,) 各通道整体均值，用于「整通道/近整通道缺失」(无有效邻域)时填充
    nan_fallback: 线性插值仍留空的像素兜底："nearest"（最近有效像素）或 "mean"
    max_frac    : 若某通道缺测占比超过此值(默认 1.0=不触发)，视为「大面积缺失」，
                  插值退化为无意义外推 -> 改用通道均值填充（中性）
    """
    from scipy.interpolate import griddata
    orig_dtype = np.asarray(arr).dtype          # 保留调用方 dtype（如 float32）
    arr = np.asarray(arr, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    # 跨通道 / 调和插值：需要整帧决策（不是逐通道独立），单独分支
    if method == "crosschannel":
        return inpaint_regions_crosschannel(
            arr, mask, deg=deg, fallback_iters=iters,
            ref_idx=ref_idx, feather=feather).astype(orig_dtype, copy=False)
    if method == "harmonic":
        out = arr.copy()
        for c in range(arr.shape[0]):
            if mask[c].any():
                out[c] = harmonic_fill(arr[c], mask[c], iters=iters)
        return out.astype(orig_dtype, copy=False)

    out = arr.copy()
    C, H, W = arr.shape
    ys, xs = np.indices((H, W))
    coord = np.column_stack([xs.ravel(), ys.ravel()])  # (H*W, 2)

    use_cv2 = method in ("telea", "ns") and _has_cv2()
    if use_cv2:
        import cv2

    for c in range(C):
        m = np.asarray(mask[c], dtype=bool)
        if not m.any():
            continue
        valid = ~m
        n_valid = int(valid.sum())
        if n_valid < 3 or (m.mean() > float(max_frac)):
            # 整通道(或近乎整通道)/大面积缺失：无可插值邻域或外推无意义
            # -> 通道均值兜底（中性，归一化后约为 0）
            if fill_values is not None and fill_values[c] is not None:
                v = float(fill_values[c])
            else:
                v = float(arr[c].mean())
            out[c] = v
            continue

        if use_cv2:
            # cv2.inpaint 需要 uint8 且掩码为 255；对浮点物理值做线性映射后修复
            lo, hi = float(arr[c][valid].min()), float(arr[c][valid].max())
            span = max(hi - lo, 1e-6)
            img8 = np.clip(((arr[c] - lo) / span * 255.0), 0, 255).astype(np.uint8)
            m8 = (m.astype(np.uint8)) * 255
            flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
            rec8 = cv2.inpaint(img8, m8, 3, flag)
            out[c] = rec8.astype(np.float64) / 255.0 * span + lo
            continue

        # 默认 griddata：用有效像素坐标做 2D 线性插值
        pts = coord[valid.ravel()]
        vals = arr[c].ravel()[valid.ravel()]
        tgt = coord[m.ravel()]
        filled = griddata(pts, vals, tgt, method="linear")
        nan_mask = np.isnan(filled)
        if nan_mask.any():
            if nan_fallback == "nearest":
                filled[nan_mask] = griddata(
                    pts, vals, tgt[nan_mask], method="nearest")
            else:
                filled[nan_mask] = float(vals.mean())
        out[c].ravel()[m.ravel()] = filled
    return out.astype(orig_dtype, copy=False)


def correct_region_missing(arr: np.ndarray, sentinel: float = 0.0,
                           low_thr=None, min_blob: int = 20, dilate: int = 0,
                           method: str = "griddata", fill_values=None,
                           connectivity: int = 8, nan_fallback: str = "mean",
                           max_frac: float = 1.0, deg: int = 2,
                           iters: int = 100):
    """一步完成「检测 + 插值修复」，返回 (修复后数组, 缺测掩码)。"""
    mask = detect_missing_region(arr, sentinel, low_thr, min_blob, dilate,
                                 connectivity)
    out = inpaint_regions(arr, mask, method, fill_values, nan_fallback,
                          max_frac=max_frac, deg=deg, iters=iters)
    return out, mask


def scan_region_stats(h5_path: str, channels, sentinel: float = 0.0,
                      low_thr=None, min_blob: int = 20, batch: int = 2000,
                      connectivity: int = 8):
    """扫描 h5，统计每通道「局部区域缺失」情况，返回 (frac, has_blob, stats)。

    frac     : (N, C) float，每图每通道缺测像素占比（0~1）
    has_blob : (N, C) bool，每图每通道是否存在 >=min_blob 的缺测块
    stats    : 人类可读摘要（每通道含缺测块的图数、缺测像素总数/均值等）
    """
    import h5py
    from scipy import ndimage
    ch_idx = list(range(len(channels)))
    n_ch = len(channels)
    imgs_with_blob = np.zeros(n_ch, dtype=np.int64)
    px_missing = np.zeros(n_ch, dtype=np.int64)
    px_total = np.zeros(n_ch, dtype=np.int64)
    frac_list, blob_list = [], []
    struct = _struct(connectivity)
    lt = np.asarray(low_thr, dtype=np.float64) if low_thr is not None else None
    N = 0
    with h5py.File(h5_path, "r") as hf:
        mat = hf["matrix"]
        total = mat.shape[0]
        for s in range(0, total, batch):
            e = min(s + batch, total)
            x = mat[s:e][..., ch_idx].astype(np.float64)   # (B,H,W,C)
            B = x.shape[0]
            N += B
            f = np.zeros((B, n_ch), dtype=np.float32)
            b = np.zeros((B, n_ch), dtype=bool)
            for ci in range(n_ch):
                ch = x[..., ci]
                eq = np.isclose(ch, sentinel, atol=1e-6)
                if lt is not None:
                    if lt.ndim == 0:
                        eq = eq | (ch < float(lt))
                    elif ci < len(lt) and np.isfinite(lt[ci]) and lt[ci] > -900:
                        eq = eq | (ch < float(lt[ci]))
                px_total[ci] += B * ch.shape[1] * ch.shape[2]
                for k in range(B):
                    ek = eq[k]
                    if not ek.any():
                        continue
                    lab, nlab = ndimage.label(ek, structure=struct)
                    if nlab == 0:
                        continue
                    sizes = ndimage.sum(np.ones_like(ek, dtype=np.int64), lab,
                                       index=range(1, nlab + 1))
                    big = sizes >= min_blob
                    if big.any():
                        imgs_with_blob[ci] += 1
                        # 计入被标记缺测的总像素（含膨胀前的核心块）
                        keep_lab = np.zeros(nlab + 1, dtype=bool)
                        keep_lab[1:][big] = True
                        mk = keep_lab[lab]
                        px_missing[ci] += int(mk.sum())
                        f[k, ci] = mk.mean()
                        b[k, ci] = True
            frac_list.append(f)
            blob_list.append(b)
    frac = np.concatenate(frac_list, axis=0)
    has_blob = np.concatenate(blob_list, axis=0).astype(bool)
    stats = {
        "h5_path": h5_path,
        "n_images": int(N),
        "channels": list(channels),
        "sentinel": sentinel,
        "min_blob": min_blob,
        "images_with_region_blob": {
            channels[c]: int(imgs_with_blob[c]) for c in ch_idx},
        "region_missing_pixels": {
            channels[c]: int(px_missing[c]) for c in ch_idx},
        "region_missing_ratio": {
            channels[c]: round(float(px_missing[c] / max(px_total[c], 1)), 6)
            for c in ch_idx},
    }
    return frac, has_blob, stats


# --------------------------------------------------------------------------- #
# 过度损坏样本判定（用于剔除「不可靠修复」的样本）
# --------------------------------------------------------------------------- #
# 动机：局部区域插值只能修复「有限面积」的缺测。若一张图出现下列任一情形，
#       即便插值也难以恢复，属于「不可可靠修复」样本，应剔除而非硬修：
#   (a) 整通道缺失的通道数 >= bad_channels（默认 2，共 3 通道时已损失 2/3 输入）；
#   (b) 全图缺测像素占比 >= image_frac（默认 1/2）；
#   (c) 单个通道缺测像素占比 >= channel_frac（默认 2/3）。
# 说明：每通道「损坏占比」= max(整通道缺失→1.0, 区域缺失掩码像素占比)。
def frame_damage_stats(arr: np.ndarray, sentinel: float = 0.0, low_thr=None,
                       min_blob: int = 20, dilate: int = 0,
                       missing_detect: str = "constant",
                       connectivity: int = 8):
    """单帧 (C,H,W) 原始值 -> (n_whole_missing, n_damaged, image_frac, max_channel_frac)。

    n_whole_missing   : 整通道缺失(detect_missing_channels)的通道数
    n_damaged         : 「有缺测」的通道数（整通道缺失 或 区域掩码非空），
                        即用户口径下的「坏道数」
    image_frac        : 全图缺测像素占比 = 各通道损坏占比的均值（0~1）
    max_channel_frac  : 单通道最大缺测像素占比（0~1）

    每通道「损坏占比」= max(整通道缺失→1.0, 区域缺失掩码像素占比)。
    """
    arr = np.asarray(arr, dtype=np.float64)
    C = arr.shape[0]
    if C == 0:
        return 0, 0, 0.0, 0.0
    ch_frac = np.zeros(C, dtype=np.float64)
    mm = detect_missing_channels(arr, mode=missing_detect)
    n_whole = int(mm.sum())
    if n_whole:
        ch_frac[mm] = 1.0
    mask = detect_missing_region(arr, sentinel=sentinel, low_thr=low_thr,
                                 min_blob=min_blob, dilate=dilate,
                                 connectivity=connectivity)
    for c in range(C):
        if mask[c].any():
            f = float(mask[c].mean())
            if f > ch_frac[c]:
                ch_frac[c] = f
    n_damaged = int((ch_frac > 0).sum())          # 有缺测的通道数（坏道数）
    return n_whole, n_damaged, float(ch_frac.mean()), float(ch_frac.max())


def scan_damage(h5_path: str, frame_indices, ch_idx, sentinel: float = 0.0,
                low_thr_list=None, min_blob: int = 20, dilate: int = 0,
                missing_detect: str = "constant", connectivity: int = 8,
                batch: int = 256):
    """批量扫描指定帧的损坏度量，返回 (n_damaged, n_whole, image_frac, max_ch_frac)。

    frame_indices : 紧凑 h5 的行号（0-based）
    ch_idx        : 目标通道在原始 4 通道中的位置；若 h5 已是紧凑 3 通道，
                    自动按前 len(ch_idx) 列取（与 TCIRDataset.__getitem__ 一致）
    各数组长度 = len(frame_indices)。n_damaged = 有缺测的通道数（坏道数）。
    """
    import h5py
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    M = len(frame_indices)
    ch_idx = list(ch_idx)
    n_damaged = np.zeros(M, dtype=np.int32)
    n_whole = np.zeros(M, dtype=np.int32)
    img_frac = np.zeros(M, dtype=np.float32)
    max_frac = np.zeros(M, dtype=np.float32)
    with h5py.File(h5_path, "r") as hf:
        mat = hf["matrix"]
        for s in range(0, M, batch):
            idx = frame_indices[s:s + batch]
            order = np.argsort(idx, kind="stable")     # h5py 需索引递增
            idx_sorted = idx[order]
            raw = mat[idx_sorted]                       # (b,H,W,C_h5)
            if raw.shape[-1] == len(ch_idx):
                sub = raw[..., :len(ch_idx)]
            else:
                sub = raw[..., ch_idx]
            for k in range(sub.shape[0]):
                a = np.ascontiguousarray(sub[k].transpose(2, 0, 1)).astype(np.float64)
                nw, nd, imf, mxf = frame_damage_stats(
                    a, sentinel, low_thr_list, min_blob, dilate,
                    missing_detect, connectivity)
                pos = s + int(order[k])
                n_damaged[pos] = nd
                n_whole[pos] = nw
                img_frac[pos] = imf
                max_frac[pos] = mxf
    return n_damaged, n_whole, img_frac, max_frac
