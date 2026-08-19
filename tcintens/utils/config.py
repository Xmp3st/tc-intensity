# -*- coding: utf-8 -*-
"""
config.py —— 层级 YAML 配置 + 命令行覆盖

设计目标：把"调参"这件事收敛到一个文件里。
- 用 YAML 描述所有超参数（嵌套字典）。
- 用命令行 `--set a.b.c=值` 临时覆盖任意字段，无需改文件。
- `Config` 提供 get/set/save，方便在代码里读写。
"""
import argparse
import copy
import os
from typing import Any, Dict, List, Optional

import yaml


# --------------------------------------------------------------------------- #
# 嵌套字典的读写工具
# --------------------------------------------------------------------------- #
def get_by_path(d: Dict, keys: List[str]) -> Any:
    """按 ['a','b','c'] 路径读取嵌套字典中的值。"""
    cur = d
    for k in keys:
        cur = cur[k]
    return cur


def set_by_path(d: Dict, keys: List[str], value: Any) -> None:
    """按路径写入；中间节点不存在会自动创建为字典。"""
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _coerce(value: str) -> Any:
    """把命令行字符串推断成合适的 Python 类型。"""
    v = value.strip()
    low = v.lower()
    if low in ("null", "none", "none"):
        return None
    if low in ("true", "false"):
        return low == "true"
    # 尝试 int / float
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


# --------------------------------------------------------------------------- #
# Config 对象
# --------------------------------------------------------------------------- #
class Config:
    """对嵌套字典的轻量包装，支持路径读写与持久化。"""

    def __init__(self, data: Optional[Dict] = None):
        self._data: Dict = data if data is not None else {}

    # ---- 字典式读写 ----
    def get(self, path: str, default: Any = None) -> Any:
        """按点分路径读取，例如 cfg.get('model.backbone')。"""
        try:
            return get_by_path(self._data, path.split("."))
        except (KeyError, TypeError):
            return default

    def set(self, path: str, value: Any) -> None:
        set_by_path(self._data, path.split("."), value)

    def to_dict(self) -> Dict:
        return copy.deepcopy(self._data)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._data, f, allow_unicode=True, sort_keys=False)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config(keys={list(self._data.keys())})"


# --------------------------------------------------------------------------- #
# 命令行解析
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="北太平洋热带气旋强度识别框架",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "-c", "--config",
        default="configs/default.yaml",
        help="YAML 配置文件路径",
    )
    p.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="覆盖配置，可多次使用，例如 --set model.backbone=resnet50",
    )
    p.add_argument(
        "--print-config", action="store_true",
        help="仅打印最终生效配置后退出（用于检查调参结果）",
    )
    return p


def load_config(config_path: str, overrides: Optional[List[str]] = None) -> Config:
    """加载 YAML 并应用 --set 覆盖。"""
    if not os.path.isabs(config_path):
        # 相对路径基于「当前工作目录」
        cfg_path = os.path.join(os.getcwd(), config_path)
    else:
        cfg_path = config_path
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = Config(data)

    for item in (overrides or []):
        if "=" not in item:
            raise ValueError(f"--set 参数格式应为 KEY=VALUE，收到: {item}")
        key, value = item.split("=", 1)
        cfg.set(key.strip(), _coerce(value))
    return cfg


def parse_config() -> Config:
    """一站式：解析 argv -> 加载配置 -> 应用覆盖。"""
    parser = build_arg_parser()
    args, _ = parser.parse_known_args()
    cfg = load_config(args.config, args.set)
    cfg.set("experiment._config_file", args.config)
    if args.print_config:
        import json
        print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))
        raise SystemExit(0)
    return cfg
