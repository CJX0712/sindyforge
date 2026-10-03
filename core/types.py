"""核心数据类型。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DiscoveryResult:
    """一次动力学发现的产出。"""

    xi: np.ndarray  # (n_features, n_dim) 系数矩阵
    support: np.ndarray  # bool (n_features, n_dim) 非零支撑
    method: str = ""


@dataclass
class BenchmarkRow:
    """单条基准记录（一行 = 一个 (系统, 噪声, seed, 方法)）。"""

    system: str
    noise: float
    seed: int
    method: str
    rollout_rmse: float
    support_f1: float
    n_terms: int
    coef_err: float  # 真值支撑上的相对系数误差（越低越好）
    elapsed_sec: float
    chaotic: bool = False  # 混沌系统：长程 rollout 必然发散，不参与 rollout 门禁


@dataclass
class SystemSpec:
    """被辨识系统的元信息（用于文档与基准声明）。"""

    name: str
    dim: int
    description: str
    true_terms: str
    chaotic: bool = False
