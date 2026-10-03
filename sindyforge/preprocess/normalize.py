"""预处理：状态/导数缩放。默认恒等（保证真值支撑逐位可比）。

Standardizer 为离线可用组件（可选）：对每维做 z-score；当开启时，pipeline 会在
library 构建前缩放状态、并在系数反算时还原。默认关闭，避免引入支撑口径漂移。
"""

from __future__ import annotations

import numpy as np


class NoopScaler:
    """恒等缩放（默认）。fit 为空操作。"""

    def fit(self, X):  # noqa: D401
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)

    def inverse_coef(self, xi):
        return xi


class Standardizer:
    """逐维 z-score。仅当显式启用时使用。"""

    def __init__(self, eps=1e-12):
        self.eps = eps
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + self.eps
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def inverse_coef(self, xi):
        """把在标准化状态上得到的系数还原到原始尺度（库为线性/多项式时近似）。"""
        return np.asarray(xi, dtype=float) / self.std_.reshape(1, -1)


def build_preprocessor(mode: str):
    if mode in ("none", "identity", None):
        return NoopScaler()
    if mode == "standardize":
        return Standardizer()
    raise ValueError(f"未知 preprocess 模式: {mode}")


def build_library_for(X, degree, include_trig, scaler):
    """按 scaler 是否标准化选择库（当前库对原始/标准化状态一致构建）。"""
    dim = X.shape[1]
    from ..domain.library import PolynomialLibrary

    return PolynomialLibrary(dim, degree=degree, include_trig=include_trig)
