"""接口契约（Protocol）。调用方向：cli → pipeline → {data, discovery, eval} → core。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .types import DiscoveryResult


@runtime_checkable
class CandidateLibrary(Protocol):
    """候选函数库：把状态 X 映射到特征矩阵 Theta。"""

    feature_names: list

    def build(self, X: np.ndarray) -> np.ndarray:
        """X:(N,d) -> Theta:(N,p)。"""
        ...


@runtime_checkable
class Identifier(Protocol):
    """动力学辨识器：从 (Theta, Xdot) 还原稀疏系数矩阵 Xi。"""

    name: str

    def discover(self, Theta: np.ndarray, Xdot: np.ndarray) -> DiscoveryResult: ...
