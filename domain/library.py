"""候选函数库（SINDy 的 Θ 矩阵）。实现 core.interfaces.CandidateLibrary。

特征顺序（严格对齐 systems.true_terms 的命名）：
  1. 常量 "1"
  2. 单项式 degree 1..degree，字典序 (a1,..,ad) 且 Σai∈[1,degree]
  3. 若 include_trig：sin(x1),cos(x1),... 逐维

单项式命名：x1^a1·x2^a2·... 去掉指数 1，例如 x1*x2、x1*x1*x2、x1*x1*x1。
"""

from __future__ import annotations

import itertools

import numpy as np

from ..core.interfaces import CandidateLibrary


def _monomial_name(exp):
    """指数元组 → 特征名，如 (1,0,0)->"x1"、(2,0,1)->"x1*x1*x3"。"""
    parts = []
    for i, a in enumerate(exp, start=1):
        for _ in range(a):
            parts.append(f"x{i}")
    return "*".join(parts) if parts else "1"


def _monomial_powers(dim, degree):
    """生成所有 Σai∈[1,degree] 的指数元组，字典序。"""
    out = []
    for total in range(1, degree + 1):
        # 非负整数解 a1+...+ad=total，字典序
        for combo in itertools.combinations_with_replacement(range(dim), total):
            exp = [0] * dim
            for idx in combo:
                exp[idx] += 1
            out.append(tuple(exp))
    return out


class PolynomialLibrary(CandidateLibrary):
    def __init__(self, dim, degree=3, include_trig=True):
        self.dim = int(dim)
        self.degree = int(degree)
        self.include_trig = bool(include_trig)
        self.feature_names = self._build_names()

    def _build_names(self):
        names = ["1"]
        for exp in _monomial_powers(self.dim, self.degree):
            names.append(_monomial_name(exp))
        if self.include_trig:
            for i in range(1, self.dim + 1):
                names.append(f"sin(x{i})")
                names.append(f"cos(x{i})")
        return names

    def build(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        N = X.shape[0]
        cols = [np.ones((N, 1))]
        for exp in _monomial_powers(self.dim, self.degree):
            term = np.ones((N, 1))
            for i, a in enumerate(exp):
                if a:
                    term = term * X[:, i : i + 1] ** a
            cols.append(term)
        if self.include_trig:
            for i in range(self.dim):
                cols.append(np.sin(X[:, i : i + 1]))
                cols.append(np.cos(X[:, i : i + 1]))
        return np.hstack(cols)

    def index(self, name):
        return self.feature_names.index(name)

    def true_support_matrix(self, true_terms: dict) -> np.ndarray:
        """把 systems.true_terms 映射到 (p, d) 真实系数矩阵。"""
        d = self.dim
        xi = np.zeros((len(self.feature_names), d), dtype=float)
        for name, coefs in true_terms.items():
            j = self.index(name)
            for k, c in enumerate(coefs):
                xi[j, k] = c
        return xi
