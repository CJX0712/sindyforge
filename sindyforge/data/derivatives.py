"""从含噪轨迹估计时间导数 Xdot。

Tier-0：scipy.signal.savgol_filter（平滑 + 解析求导），对噪声鲁棒。
离线兜底：scipy 不可用时退化为中心差分（纯 numpy），并在文档/报告中标注 skipped 之外仍可读。
"""

from __future__ import annotations

import numpy as np

from ..core.backends import available_scipy
from ..core.errors import DataError


def _finite_diff(X, dt):
    """中心差分（端点单侧）。纯 numpy。"""
    dX = np.gradient(X, dt, axis=0)
    return dX


def estimate_noise_std(X):
    """从观测轨迹稳健估计白噪声 std（无需真值）。

    二阶差分把光滑信号压到 O(h²·x'')，只留下噪声的 6σ² 分量，
    故 σ ≈ std(Δ²x)/√6；用 MAD 代替 std 抵抗轨道上的少量野点。
    """
    X = np.asarray(X, dtype=float)
    if X.shape[0] < 3:
        return np.zeros(X.shape[1] if X.ndim > 1 else 1)
    d2 = X[2:] - 2.0 * X[1:-1] + X[:-2]
    med = np.median(d2, axis=0)
    mad = np.median(np.abs(d2 - med), axis=0)
    return 1.4826 * mad / np.sqrt(6.0)


def _savgol_deriv_norm(window, polyorder, dt):
    """SG 一阶导数滤波器的系数 L2 范数 ‖c_w‖₂ —— 噪声放大倍数。

    白噪声经该滤波器后，导数噪声 std = σ·‖c_w‖₂。
    """
    from scipy.signal import savgol_coeffs

    c = savgol_coeffs(int(window), int(polyorder), deriv=1, delta=float(dt))
    return float(np.sqrt(np.sum(c**2)))


def pick_window(X, dt, polyorder=5, tau=0.10, w_min=5, w_max=101):
    """自适应选择 SG 窗口：满足「导数噪声 ≤ τ·导数信号」的最小窗口。

    为什么必须自适应：固定窗口在噪声-偏置权衡上必有一端崩掉。实测（noise=5%）
    窗口 7 时 VdP/Pendulum 的导数相对误差达 100%~170%，而窗口 41 时 Lorenz
    的截断偏置又涨到 46%——没有任何固定窗口能同时照顾快系统与慢系统。

    做法（纯数据驱动，不用真值）：
      1) 二阶差分稳健估计观测噪声 σ̂；
      2) 用小窗口 w_min 先求一次导数，其功率 = 信号功率 + σ̂²‖c‖²，
         反解出信号功率；
      3) 从 w_min 起递增奇数窗口，取第一个满足 σ̂²‖c_w‖² ≤ τ²·信号功率 的 w。
    取「最小可行窗口」即在满足噪声预算的前提下最小化截断偏置。
    """
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    w_max = int(min(w_max, N if N % 2 == 1 else N - 1))
    w_min = max(int(w_min), polyorder + 2)
    if w_min % 2 == 0:
        w_min += 1
    if w_max < w_min:
        return w_min

    from scipy.signal import savgol_filter

    sigma = estimate_noise_std(X)
    if float(np.max(sigma)) <= 0.0:
        return w_min  # 无噪声：最小偏置

    d0 = np.stack(
        [savgol_filter(X[:, j], w_min, polyorder, deriv=1, delta=dt) for j in range(X.shape[1])],
        axis=1,
    )
    n0 = _savgol_deriv_norm(w_min, polyorder, dt)
    var0 = (sigma * n0) ** 2
    sig_power = np.maximum((d0**2).mean(axis=0) - var0, 1e-12)

    for w in range(w_min, w_max + 1, 2):
        v = (sigma * _savgol_deriv_norm(w, polyorder, dt)) ** 2
        if np.all(v <= (tau**2) * sig_power):
            return w
    return w_max


def estimate_derivative(
    X, dt, window="auto", polyorder=5, method="auto", tau=0.10, return_smoothed=False
):
    """X:(N,d) → Xdot:(N,d)，或 (Xdot, X_smooth)。

    ``return_smoothed=True`` 时一并返回 SG 平滑后的状态。强烈建议用平滑状态
    构建候选库 Θ：含噪状态会让 x1³ 之类非线性项把观测噪声放大 3x1²·σ
    （实测 Duffing 上达 2.4，与真值信号同量级），造成变量误差（EIV）偏差——
    系数符号都会被带翻。平滑后该放大项随窗口长度显著衰减。

    window:
      "auto"     自适应选窗（见 :func:`pick_window`）；
      int        固定窗口（奇数化后使用）。

    method:
      "savgol"  强制 Savitzky-Golay（需 scipy）；
      "fd"      强制中心差分；
      "auto"    scipy 可用走 savgol，否则 fd。
    """
    X = np.asarray(X, dtype=float)
    if X.shape[0] < 3:
        raise DataError("E200: 轨迹过短，无法估计导数")
    if method == "auto":
        method = "savgol" if available_scipy() else "fd"
    if method == "savgol":
        if not available_scipy():
            raise DataError("E200: scipy 不可用，无法使用 savgol，请改用 method='fd'")
        from scipy.signal import savgol_filter

        if isinstance(window, str) and window == "auto":
            win = pick_window(X, dt, polyorder=polyorder, tau=tau)
        else:
            win = int(window)
        if win % 2 == 0:
            win += 1
        if win > X.shape[0]:
            win = X.shape[0] if X.shape[0] % 2 == 1 else X.shape[0] - 1
        if polyorder >= win:
            polyorder = win - 1
        Xdot = np.stack(
            [savgol_filter(X[:, j], win, polyorder, deriv=1, delta=dt) for j in range(X.shape[1])],
            axis=1,
        )
        if return_smoothed:
            Xs = np.stack(
                [savgol_filter(X[:, j], win, polyorder, delta=dt) for j in range(X.shape[1])],
                axis=1,
            )
            return Xdot, Xs
        return Xdot
    if method == "fd":
        Xdot = _finite_diff(X, dt)
        return (Xdot, X.copy()) if return_smoothed else Xdot
    raise DataError(f"E200: 未知 method={method}")
