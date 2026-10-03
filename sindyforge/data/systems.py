"""合成动力学系统（DGP）：固定 seed 可复现的 ODE 仿真 + 真实稀疏支撑声明。

每个系统同时提供：
  - ``rhs(x)``：解析右端（向量化），用于干净轨迹仿真与"真值 rollout"；
  - ``true_terms``：特征名 → 各维度系数，用于在候选函数库特征空间声明真实支撑（Support-F1）。
特征命名与 ``domain.library.PolynomialLibrary`` 严格对齐（见其 docstring）。
"""

from __future__ import annotations

import numpy as np

from ..core.types import SystemSpec


class System:
    """一个被辨识的非线性动力学系统。"""

    def __init__(self, name, dim, rhs, true_terms, ic, dt_default=0.02, t_burn=5.0, chaotic=False):
        self.name = name
        self.dim = dim
        self._rhs = rhs
        self.true_terms = true_terms  # {feature_name: [coef_dim0, coef_dim1, ...]}
        self.ic = np.asarray(ic, dtype=float)  # 吸引子附近初值（无扰动）
        self.dt_default = dt_default
        self.t_burn = t_burn  # 预演化步数，脱离瞬态
        self.chaotic = chaotic  # 混沌系统：长程 rollout 发散不可逆，单独报告

    def rhs(self, X):
        """X:(N,d) → dX/dt:(N,d)。"""
        return self._rhs(X)

    def spec(self) -> SystemSpec:
        terms = "; ".join(f"{n}={c}" for n, c in self.true_terms.items())
        return SystemSpec(
            name=self.name,
            dim=self.dim,
            description=f"{self.name} nonlinear oscillator (d={self.dim})",
            true_terms=terms,
            chaotic=self.chaotic,
        )


# ----------------------------------------------------------------------------
# 各系统解析右端（x1..xd 对应 state 的第 0..d-1 列）
# ----------------------------------------------------------------------------
def _lorenz(X):
    x1, x2, x3 = X[..., 0], X[..., 1], X[..., 2]
    s, r, b = 10.0, 28.0, 8.0 / 3.0
    return np.stack(
        [
            s * (x2 - x1),
            x1 * (r - x3) - x2,
            x1 * x2 - b * x3,
        ],
        axis=-1,
    )


def _vanderpol(X):
    x1, x2 = X[..., 0], X[..., 1]
    mu = 1.0
    return np.stack(
        [
            x2,
            mu * (1.0 - x1**2) * x2 - x1,
        ],
        axis=-1,
    )


def _duffing(X):
    """保守双阱 Duffing（无阻尼）：ẍ = x − x³。

    刻意取 delta=0：带阻尼的 Duffing 会衰减到不动点 (x1≈±1, x2≈0)，
    导致 x1 长期近乎常数、与截距项近共线，辨识问题退化且不可解（
    实测任何方法都会选出 ±300 量级的爆炸系数）。保守版能量守恒，
    在双阱间作全局轨道，x1 动态范围约 ±1.6，是良态的辨识基准。
    """
    x1, x2 = X[..., 0], X[..., 1]
    alpha, beta = -1.0, 1.0  # ẍ = −alpha·x − beta·x³ = x − x³
    return np.stack(
        [
            x2,
            -alpha * x1 - beta * x1**3,
        ],
        axis=-1,
    )


def _pendulum(X):
    x1, x2 = X[..., 0], X[..., 1]
    g_over_l = 1.0
    return np.stack(
        [
            x2,
            -g_over_l * np.sin(x1),
        ],
        axis=-1,
    )


# true_terms 必须与 library 的特征名对齐
LORENZ = System(
    name="lorenz",
    dim=3,
    rhs=_lorenz,
    true_terms={
        "x1": [-10.0, 28.0, 0.0],
        "x2": [10.0, -1.0, 0.0],
        "x3": [0.0, 0.0, -8.0 / 3.0],
        "x1*x2": [0.0, 0.0, 1.0],
        "x1*x3": [0.0, -1.0, 0.0],
    },
    ic=[1.0, 1.0, 1.0],
    chaotic=True,
)

VANDERPOL = System(
    name="vanderpol",
    dim=2,
    rhs=_vanderpol,
    true_terms={
        "x1": [0.0, -1.0],
        "x2": [1.0, 1.0],
        "x1*x1*x2": [0.0, -1.0],
    },
    ic=[2.0, 0.0],
)

DUFFING = System(
    name="duffing",
    dim=2,
    rhs=_duffing,
    true_terms={
        "x2": [1.0, 0.0],
        "x1": [0.0, 1.0],
        "x1*x1*x1": [0.0, -1.0],
    },
    # 大幅跨阱轨道（|x1|≤~2.7, |x2|≤~3.7）：振幅足够大，使 sin(x1)/sin(x2)
    # 与 x1/x2 显著区分。振幅过小时 sin(x)≈x−x³/6 落在 {x, x³} 张成的空间内，
    # 三角库与多项式库结构退化，任何方法都会把 x1 误替成 sin(x1)。
    ic=[2.5, 0.0],
)

PENDULUM = System(
    name="pendulum",
    dim=2,
    rhs=_pendulum,
    true_terms={
        "x2": [1.0, 0.0],
        "sin(x1)": [0.0, -1.0],
    },
    ic=[2.0, 0.0],
)


SYSTEMS = {
    "lorenz": LORENZ,
    "vanderpol": VANDERPOL,
    "duffing": DUFFING,
    "pendulum": PENDULUM,
}


def simulate(system: System, x0, n_steps, dt, rng=None):
    """固定步长 RK4 仿真（纯 numpy，确定性）。返回 (N,d) 轨迹。"""
    x0 = np.asarray(x0, dtype=float)
    d = system.dim
    X = np.zeros((n_steps, d), dtype=float)
    X[0] = x0
    for i in range(1, n_steps):
        x = X[i - 1]
        k1 = system.rhs(x)
        k2 = system.rhs(x + 0.5 * dt * k1)
        k3 = system.rhs(x + 0.5 * dt * k2)
        k4 = system.rhs(x + dt * k3)
        X[i] = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return X


def make_dataset(system: System, n_steps, dt, noise_level, seed, burn=None):
    """生成 (clean_X, noisy_X, rng)。初值 = ic + 小扰动（seed 固定）。

    - 预演化 ``burn`` 步脱离瞬态；
    - 观测噪声按各维 std 的比例注入。
    """
    rng = np.random.RandomState(int(seed))
    burn = int(burn if burn is not None else system.t_burn / dt)
    x0 = system.ic + rng.normal(0.0, 0.05, size=system.dim)
    full = simulate(system, x0, n_steps + burn, dt, rng)
    clean = full[burn:]
    noisy = clean.copy()
    if noise_level > 0.0:
        for j in range(system.dim):
            sigma = noise_level * (clean[:, j].std() + 1e-12)
            noisy[:, j] += rng.normal(0.0, sigma, size=n_steps)
    return clean, noisy, rng


def rollout_truth(system: System, x0, steps, dt):
    """从 x0 用真值动力学 RK4 仿真 steps 步，返回 (steps,d)。用于评估真值轨迹。"""
    return simulate(system, x0, steps, dt)
