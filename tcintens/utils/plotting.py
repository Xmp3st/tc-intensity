# -*- coding: utf-8 -*-
"""
plotting.py —— 绘图通用工具。

【中文字体】项目所在 WSL 里唯一的系统 CJK 字体 `Droid Sans Fallback` **不含拉丁数字**，
且 matplotlib 不会跨字体逐字形回退（只有一个 font.sans-serif 命中就是它，数字会变方块）。
经实测，Windows 侧挂载的以下字体同时含 CJK 与拉丁、渲染 0 缺字，优先级最高：
    /mnt/c/Windows/Fonts/simhei.ttf        -> SimHei
    /mnt/c/Windows/Fonts/NotoSansSC-VF.ttf -> Noto Sans SC
    /mnt/c/Windows/Fonts/msyh.ttc          -> Microsoft YaHei
用法：绘图脚本开头调用 `use_cjk_font()` 即可。
"""
import os

_WIN_FONTS = [
    "/mnt/c/Windows/Fonts/simhei.ttf",
    "/mnt/c/Windows/Fonts/NotoSansSC-VF.ttf",
    "/mnt/c/Windows/Fonts/msyh.ttc",
]
_LINUX_FONTS = [
    ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", "Droid Sans Fallback"),
]


def use_cjk_font(verbose: bool = False) -> str:
    """把 matplotlib 全局字体设为可同时渲染中文与数字的字体，返回字体名。

    找不到合适字体时回退 DejaVu Sans（中文会变方块，但不报错）。
    """
    from matplotlib import font_manager as fm
    import matplotlib.pyplot as plt

    for fp in _WIN_FONTS:
        if os.path.exists(fp):
            try:
                fm.fontManager.addfont(fp)
                name = fm.FontProperties(fname=fp).get_name()
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                if verbose:
                    print(f"[plotting] 使用中文字体: {name} ({fp})")
                return name
            except Exception:
                continue
    # 退回系统 CJK 字体（注意：Droid 无拉丁数字，仅供参考）
    for fp, name in _LINUX_FONTS:
        if os.path.exists(fp):
            try:
                fm.fontManager.addfont(fp)
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                if verbose:
                    print(f"[plotting] 使用中文字体(降级): {name}")
                return name
            except Exception:
                continue
    plt.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans"
