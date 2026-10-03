"""对照基线：单阈值 STR-SINDy（Brunton 2016 教科书版，单一 λ + STR 清洗）。

用于消融：验证"稳定性共识 + 多 λ"相对"单阈值"的贡献。
"""

from __future__ import annotations

import numpy as np

from ..core.config import Config
from ..core.interfaces import Identifier
from ..core.types import DiscoveryResult
from .strlasso import numpy_lasso, ols_fit


class SingleThresholdSTR(Identifier):
    name = "SingleSTR"

    def __init__(self, config: Config | None = None):
        self.config = config or Config()

    def discover(self, Theta: np.ndarray, Xdot: np.ndarray) -> DiscoveryResult:
        Theta = np.asarray(Theta, dtype=float)
        Xdot = np.asarray(Xdot, dtype=float)
        xi_ols = ols_fit(Theta, Xdot)
        base = np.max(np.abs(xi_ols), axis=0, keepdims=True)
        base[base == 0] = 1.0
        xi = numpy_lasso(Theta, Xdot, self.config.str_rel * base)
        thr = self.config.str_rel * np.max(np.abs(xi), axis=0, keepdims=True)
        thr[thr == 0] = 1.0
        xi[np.abs(xi) < thr] = 0.0
        support = np.abs(xi) > 1e-12
        return DiscoveryResult(xi=xi, support=support, method=self.name)
