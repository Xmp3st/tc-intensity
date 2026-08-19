# -*- coding: utf-8 -*-
"""
misc.py —— 杂项工具：随机种子、设备选择、模型参数量
"""
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """固定所有随机源，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(mode: str = "auto") -> torch.device:
    """根据配置选择设备。auto 优先 CUDA。"""
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            print("[warn] 配置要求 cuda 但不可用，回退到 cpu")
            return torch.device("cpu")
        return torch.device("cuda")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """返回可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def humanize_count(n: int) -> str:
    """把参数量格式化为可读字符串。"""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)
