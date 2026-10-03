"""SindyForge · 稀疏非线性动力学系统辨识（SINDy 范式，世界顶级 AI 系统交付）。

作者：晨星
旗舰：StabSINDy（STR-LASSO + 多 λ 稳定性共识 + OLS 重拟合），纯 numpy 离线可跑。
"""

from .core import (
    BenchmarkRow,
    Config,
    DiscoveryResult,
    SindyError,
    SystemSpec,
    available_scipy,
    available_sklearn,
    make_rng,
    set_all,
)
from .data import SYSTEMS, make_dataset, simulate
from .discovery import ALL_METHODS, StabSINDy
from .pipeline import SindyPipeline, summarize

__version__ = "0.1.0"
__author__ = "晨星"

__all__ = [
    # 配置与确定性
    "Config",
    "set_all",
    "make_rng",
    # 旗舰与全部辨识器
    "StabSINDy",
    "ALL_METHODS",
    # 数据
    "SYSTEMS",
    "simulate",
    "make_dataset",
    # 编排
    "SindyPipeline",
    "summarize",
    # 类型与错误
    "DiscoveryResult",
    "BenchmarkRow",
    "SystemSpec",
    "SindyError",
    "available_sklearn",
    "available_scipy",
    "__version__",
    "__author__",
]
