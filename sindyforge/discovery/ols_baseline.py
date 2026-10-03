"""对照基线：纯 OLS（稠密、不过拟合抑制，用作"天花板以下"参照）。"""

from __future__ import annotations

import numpy as np

from ..core.config import Config
from ..core.interfaces import Identifier
from ..core.types import DiscoveryResult
from .strlasso import ols_fit


class OLSBaseline(Identifier):
    name = "OLS"

    def __init__(self, config: Config | None = None):
        self.config = config or Config()

    def discover(self, Theta: np.ndarray, Xdot: np.ndarray) -> DiscoveryResult:
        xi = ols_fit(Theta, Xdot)
        support = np.abs(xi) > 1e-12
        return DiscoveryResult(xi=xi, support=support, method=self.name)
