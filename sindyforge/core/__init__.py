"""SindyForge · core 层：全局确定性、错误码、类型、配置、接口、后端探测。"""

from .backends import available_scipy, available_sklearn
from .config import Config
from .errors import (
    ConfigError,
    DataError,
    DeterminismError,
    FitError,
    LibraryError,
    SindyError,
)
from .interfaces import CandidateLibrary, Identifier
from .seed import make_rng, set_all
from .types import BenchmarkRow, DiscoveryResult, SystemSpec

__all__ = [
    "set_all",
    "make_rng",
    "SindyError",
    "ConfigError",
    "DataError",
    "LibraryError",
    "FitError",
    "DeterminismError",
    "DiscoveryResult",
    "BenchmarkRow",
    "SystemSpec",
    "Config",
    "CandidateLibrary",
    "Identifier",
    "available_sklearn",
    "available_scipy",
]
