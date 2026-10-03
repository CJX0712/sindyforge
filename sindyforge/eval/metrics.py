"""评估指标：rollout RMSE（越低越好）、Support-F1（越高越好）、显著性检验。"""

from __future__ import annotations

import math

import numpy as np

from ..core.backends import available_scipy


def _disc_rhs(library, xi, x):
    if x.ndim == 1:
        theta = library.build(x.reshape(1, -1))
        return (theta @ xi).ravel()
    theta = library.build(x)
    return theta @ xi


def rollout(library, xi, x0, steps, dt):
    """用发现的动力学 ξ 做固定步长 RK4 前向仿真，返回 (steps, d)。

    对发散模型裁剪状态幅值，避免 inf/nan 污染 RMSE 比较（发散即判为差）。
    """
    x0 = np.asarray(x0, dtype=float)
    X = np.zeros((steps, x0.size), dtype=float)
    X[0] = x0
    cap = 1.0e6
    for i in range(1, steps):
        x = X[i - 1]
        k1 = _disc_rhs(library, xi, x)
        k2 = _disc_rhs(library, xi, x + 0.5 * dt * k1)
        k3 = _disc_rhs(library, xi, x + 0.5 * dt * k2)
        k4 = _disc_rhs(library, xi, x + dt * k3)
        nxt = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        X[i] = np.clip(nxt, -cap, cap)
    return X


def rollout_rmse(system, library, xi, clean_segment, dt, horizon):
    """从 clean_segment[0] 出发，对比发现动力学与真值轨迹的 RMSE。"""
    x0 = clean_segment[0]
    horizon = min(horizon, clean_segment.shape[0])
    pred = rollout(library, xi, x0, horizon, dt)
    truth = clean_segment[:horizon]
    if not np.all(np.isfinite(pred)):
        return 1.0e6  # 发散：判为最差有限值，保持可比性
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def support_f1(true_support, pred_support):
    """逐维 F1 取平均（宏平均），范围 [0,1]。"""
    true_support = np.asarray(true_support, dtype=bool)
    pred_support = np.asarray(pred_support, dtype=bool)
    d = true_support.shape[1]
    f1s = []
    for k in range(d):
        t = true_support[:, k]
        p = pred_support[:, k]
        tp = int(np.sum(t & p))
        fp = int(np.sum(p & ~t))
        fn = int(np.sum(~p & t))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def coef_relative_error(true_xi, pred_xi, support):
    """真值支撑上的相对系数误差（仅统计双方都非零的项取相对）。"""
    true_xi = np.asarray(true_xi, dtype=float)
    pred_xi = np.asarray(pred_xi, dtype=float)
    m = np.abs(true_xi) > 1e-9
    if not np.any(m):
        return 0.0
    rel = np.abs(pred_xi[m] - true_xi[m]) / (np.abs(true_xi[m]) + 1e-9)
    return float(np.mean(rel))


def _norm_sf(x):
    """标准正态 survival function（scipy 优先，否则 erfc 近似）。"""
    if available_scipy():
        from scipy import stats

        return float(stats.norm.sf(abs(x)))
    return float(0.5 * math.erfc(abs(x) / math.sqrt(2.0)))


def paired_ttest(a, b):
    """配对 t 检验（双侧），返回 (t, p, mean_a, mean_b)。

    基准设计天然配对：每个 (系统, 噪声, seed) cell 同时跑所有方法，
    系统间的难度差异在作差时被抵消。用 Welch 双样本检验会丢掉这一结构，
    方差被系统间差异放大（实测同一效应在 Welch 下 p=0.317、配对下 p<1e-6）。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = d.size
    ma, mb = float(a.mean()), float(b.mean())
    if n < 2:
        return 0.0, 1.0, ma, mb
    var = float(d.var(ddof=1))
    se = math.sqrt(var / n) if var > 0 else 0.0
    if se == 0.0:
        # 差异恒定：效应确定，p 取 0（或完全一致时取 1）
        return (0.0, 0.0 if abs(d.mean()) > 0 else 1.0, ma, mb)
    t = float(d.mean() / se)
    p = 2.0 * _norm_sf(t)
    return t, float(p), ma, mb


def welch_ttest(a, b):
    """Welch t 检验（双侧），返回 (t, p, mean_a, mean_b)。scipy 优先，否则正态近似。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    se = math.sqrt(va / na + vb / nb) if (va / na + vb / nb) > 0 else 1e-12
    t = (ma - mb) / se
    if available_scipy():
        from scipy import stats

        p = 2.0 * (1.0 - stats.norm.cdf(abs(t)))
    else:
        p = 2.0 * math.erfc(abs(t) / math.sqrt(2.0))
    return float(t), float(p), float(ma), float(mb)
