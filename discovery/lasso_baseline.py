"""强基线：sklearn LassoCV（逐列交叉验证选 λ）。sklearn 缺失时降级为 numpy Lasso。"""

from __future__ import annotations

import numpy as np

from ..core.backends import available_sklearn
from ..core.config import Config
from ..core.interfaces import Identifier
from ..core.types import DiscoveryResult
from .strlasso import numpy_lasso, ols_fit


class LassoBaseline(Identifier):
    name = "Lasso"

    def __init__(self, config: Config | None = None):
        self.config = config or Config()

    def discover(self, Theta: np.ndarray, Xdot: np.ndarray) -> DiscoveryResult:
        Theta = np.asarray(Theta, dtype=float)
        Xdot = np.asarray(Xdot, dtype=float)
        p = Theta.shape[1]
        d = Xdot.shape[1]
        xi = np.zeros((p, d))

        if available_sklearn():
            from sklearn.linear_model import LassoCV

            alphas = list(self.config.lasso_alphas)
            for k in range(d):
                model = LassoCV(alphas=alphas, cv=5, max_iter=5000, tol=1e-4, n_jobs=1)
                model.fit(Theta, Xdot[:, k])
                xi[:, k] = model.coef_
        else:
            # 离线兜底：单 λ numpy Lasso
            xi_ols = ols_fit(Theta, Xdot)
            base = np.max(np.abs(xi_ols), axis=0, keepdims=True)
            base[base == 0] = 1.0
            xi = numpy_lasso(Theta, Xdot, self.config.str_rel * base)
        support = np.abs(xi) > 1e-12
        return DiscoveryResult(xi=xi, support=support, method=self.name)
