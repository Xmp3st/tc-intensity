# -*- coding: utf-8 -*-
"""
repro.py —— 训练 / 评估「配置一致性」守卫（防止静默不一致导致指标失真）

背景（真实踩过的坑）：
  训练时常用 `--set data.tcir.missing_band=interp` 临时开启缺失修正，但这个覆盖
  只作用于那一次训练、**不会写回 YAML**。若评估 / 可视化时忘记带同样的 --set，
  就会拿「未修正」的测试数据去评估「在修正数据上训练」的模型 —— 得到失真的指标，
  而且不会报错，非常隐蔽。

方案：
  训练结束会把「生效配置」保存到 outputs/<output_dir>/<name>/config_used.yaml
  （由 engine/train.py 完成）。评估 / 可视化前调用 align_preprocessing()：
    * 某一致性键「未被本次命令行显式覆盖」→ 采用训练时的取值（保证一致），并打印说明；
    * 某一致性键「被本次显式覆盖」且与训练值不同 → 打印醒目警告（用户有意为之，尊重其选择）。

  `--set` 覆盖列表由 load_config() 记录在 cfg.overrides 上（见 utils/config.py）。
"""
import os
from typing import List, Optional, Tuple

import yaml

from .config import Config

# --------------------------------------------------------------------------- #
# 一致性键：凡是会改变「输入张量如何构造」或「train/val/test 如何划分」的键，
# 训练与评估必须一致，否则指标不可比。这里列全，避免漏项。
# --------------------------------------------------------------------------- #
CONSISTENCY_KEYS: List[str] = [
    # —— 决定测试集「是哪些样本」——
    "experiment.seed",
    "data.split",
    "data.tcir.split_strategy",
    "data.tcir.split_year_test",
    "data.tcir.h5_path",
    "data.tcir.csv_path",
    "data.tcir.label_source",
    "data.tcir.index_col",
    "data.tcir.storm_col",
    "data.tcir.target_col",
    "data.tcir.label_unit",
    "data.tcir.regions",
    # —— 决定「每张图变成什么张量」——
    "data.tcir.channels",
    "data.tcir.nan_mode",
    "data.tcir.normalize",
    "data.tcir.norm_mean",
    "data.tcir.norm_std",
    "data.tcir.resize",
    # —— 缺失修正（最易漏，因为常用 --set 临时开）——
    "data.tcir.missing_band",
    "data.tcir.missing_detect",
    "data.tcir.missing_region_channels",
    "data.tcir.missing_region_sentinel",
    "data.tcir.missing_region_low_thr",
    "data.tcir.missing_region_min_blob",
    "data.tcir.missing_region_dilate",
    "data.tcir.missing_region_method",
    "data.tcir.missing_region_max_frac",
    "data.tcir.missing_region_deg",
    "data.tcir.missing_region_iters",
    "data.tcir.missing_region_feather",
    # —— 离线 U-Net 预修复的 IR1 缓存（改变输入张量，必须对齐）——
    "data.tcir.ir1_repaired_path",
    # —— 过度损坏样本剔除（改变测试集「是哪些样本」）——
    "data.tcir.damage_filter",
    "data.tcir.damage_filter_splits",
    "data.tcir.damage_filter_bad_definition",
    "data.tcir.damage_filter_bad_channels",
    "data.tcir.damage_filter_image_frac",
    "data.tcir.damage_filter_channel_frac",
]


def train_config_path(output_dir: str, name: str) -> str:
    """训练时保存的生效配置路径。"""
    return os.path.join(output_dir, name, "config_used.yaml")


def load_train_config(output_dir: str, name: str) -> Tuple[Optional[Config], str]:
    """读取训练时的生效配置；不存在则返回 (None, path)。"""
    path = train_config_path(output_dir, name)
    if not os.path.exists(path):
        return None, path
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config(data), path


def _emit(logger, msg: str, warning: bool = False) -> None:
    if logger is not None:
        (logger.warning if warning else logger.info)(msg)
    else:
        print(("[警告] " if warning else "[提示] ") + msg)


def align_preprocessing(cfg: Config, overrides: Optional[List[str]] = None,
                        logger=None,
                        keys: Optional[List[str]] = None
                        ) -> Tuple[Config, List[Tuple[str, object]]]:
    """把 cfg 的一致性键对齐到训练时的取值。

    cfg       : 本次评估 / 可视化用的配置（会被就地修改）
    overrides : 本次命令行 --set 列表；其中出现过的键视为「用户显式指定」，不覆盖
    logger    : 可选 logger；为 None 时用 print 输出
    keys      : 只对齐这些键；默认 CONSISTENCY_KEYS 全量（predict 等只用到部分
                预处理键的调用方可以传子集，避免把无效覆盖写进配置）

    返回 (cfg, aligned)：aligned 为实际被对齐的 (键, 新值) 列表。
    """
    exp = cfg.to_dict().get("experiment", {})
    out_dir = exp.get("output_dir", "outputs")
    name = exp.get("name", "exp")

    tcfg, path = load_train_config(out_dir, name)
    if tcfg is None:
        _emit(logger, f"未找到训练配置 {path}，跳过一致性检查"
                      f"（若该模型训练时用过 --set 覆盖，请手动补齐相同参数）。")
        return cfg, []

    # 本次显式覆盖过的键
    overridden = set()
    for item in (overrides or []):
        if "=" in item:
            overridden.add(item.split("=", 1)[0].strip())

    aligned: List[Tuple[str, object]] = []
    conflicts: List[Tuple[str, object, object]] = []
    for k in (keys or CONSISTENCY_KEYS):
        tv = tcfg.get(k)
        if tv is None:
            continue                      # 训练配置里没有该键，无从对齐
        cv = cfg.get(k)
        if tv == cv:
            continue
        if k in overridden:
            conflicts.append((k, tv, cv))  # 用户显式指定且与训练不同
        else:
            cfg.set(k, tv)
            aligned.append((k, tv))

    if aligned:
        _emit(logger, f"已将 {len(aligned)} 个一致性键对齐到训练配置 {path}：")
        for k, v in aligned:
            _emit(logger, f"    {k} -> {v}")
    if conflicts:
        for k, tv, cv in conflicts:
            _emit(logger, f"[配置不一致] {k}: 训练用 {tv!r}，本次显式指定 {cv!r}。"
                          f"若本意是与训练结果可比，请去掉该 --set。", warning=True)
    if not aligned and not conflicts:
        _emit(logger, f"一致性检查通过：预处理/划分配置与训练配置 {path} 一致。")
    return cfg, aligned
