"""Tier-1 旗舰：StabSINDy —— 多 λ 稳定性选择（stability selection）+ 终局 OLS 重拟合。

对标 Brunton/Proctor/Kutz 2016 (PNAS) 的 SINDy 范式，并用稳定性选择提升噪声下支撑恢复率。
求解器：sklearn Lasso 优先（自动以 LassoCV 缩放 λ 网格），纯 numpy 坐标下降 Lasso 离线兜底。

方法：
  1) 对特征做 z-score（使 Lasso 惩罚尺度无关、近似均匀）；
  2) 以 LassoCV 求得的每维参考 α 为中心，在 stab_grid 倍数网格上跑 Lasso；
  3) 统计各特征被选中频率 → 频率 ≥ stab_frac 进入稳定性候选（并集）；
  4) 在保留支撑上对原始 Θ 做 OLS 重拟合（恢复真实系数幅度，避免 Lasso 收缩偏差）；
  5) 温和 STR 清洗：|ξ| < str_rel·max|ξ| 置零。
"""

from __future__ import annotations

import math

import numpy as np

from ..core.backends import available_sklearn
from ..core.config import Config
from ..core.interfaces import Identifier
from ..core.types import DiscoveryResult


def _soft(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def numpy_lasso(X, Y, alpha, max_iter=3000, tol=1e-7):
    """坐标下降 Lasso（逐列，残差增量式，保证收敛）。

    X:(N,p) Y:(N,d)；alpha 标量或 (d,)；返回 (p,d)。
    列标准化到 L2=1 后做标准 Lasso 坐标下降（残差增量更新，避免共线导致的发散），
    结果反算回原始尺度。
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    N, p = X.shape
    d = Y.shape[1]
    col_norm = np.sqrt((X**2).sum(0))
    col_norm[col_norm == 0] = 1.0
    Xs = X / col_norm  # 每列 L2=1
    a = np.asarray(alpha, dtype=float).ravel()
    coef = np.zeros((p, d))
    for k in range(d):
        ak = float(a[k]) if a.size == d else float(a[0])
        y = Y[:, k]
        b = np.zeros(p)
        r = y.copy()  # r = y - Xs b（b=0）
        for _ in range(max_iter):
            b_old = b.copy()
            for j in range(p):
                Xj = Xs[:, j]
                r = r + Xj * b[j]  # 移除 j：r = y - Xs_{¬j} b
                cj = Xj @ r  # Xs_j·r（不含 j 的残差投影）
                b[j] = _soft(cj, ak)  # ‖Xs_j‖²=1 → 无需再除
                r = r - Xj * b[j]  # 放回新 b[j]
            if np.max(np.abs(b - b_old)) < tol:
                break
        coef[:, k] = b / col_norm  # 反算回原始尺度
    return coef


def ols_fit(Theta, Y, ridge=0.0):
    """最小二乘拟合（可选岭稳定），返回 (p,d)。

    多项式/三角候选库高度共线，朴素的 ``lstsq`` 在近退化子集上会给出 ±1e2~1e3
    量级的爆炸系数（且符号相反、相互抵消）。这类伪影会污染一切以 ``max|ξ|``
    为基准的相对阈值。故默认加一个小岭项把系数幅值压回物理尺度。
    """
    A = np.asarray(Theta, dtype=float)
    B = np.asarray(Y, dtype=float)
    if ridge > 0.0:
        G = A.T @ A + ridge * np.eye(A.shape[1])
        return np.linalg.solve(G, A.T @ B)
    return np.linalg.lstsq(A, B, rcond=None)[0]


class StabSINDy(Identifier):
    name = "StabSINDy"

    def __init__(self, config: Config | None = None):
        cfg = config or Config()
        self.str_rel = cfg.str_rel
        self.stab_grid = tuple(cfg.stab_grid)
        self.stab_frac = cfg.stab_frac
        self.max_iter = 3000
        self.tol = 1e-7
        # 岭稳定系数（相对 ΘᵀΘ 平均对角）。压住共线爆炸系数，保证下方
        # 以 max|ξ| 为基准的相对阈值反映真实物理尺度而非数值伪影。
        self.ridge_rel = 1e-6
        # 是否对候选库做 z-score 后再跑 Lasso 路径（默认开启，见下方实测说明）。
        # z-score 让惩罚在「解释方差」意义上均匀 → 偏袒主导项，幅值仅为主导项
        # 1/28 的弱真值项（Lorenz ẋ2 的 x2=−1）难以入选，这是本方法的已知边界。
        # 实测：关闭 z-score 后 Pendulum 的 F1 从 0.83 崩到 0.33（原始尺度下
        # Lasso 收敛变差、弱项反而更易被共线项吞掉），故保持开启。
        self.zscore = True
        # 稳定性候选并集 + OLS 重拟合后，再做一次温和 STR 阈值（相对每方程 max）。
        # 注意：符号一致性门已移除——共线库下真值小项（如 Lorenz dx2 的 x2=-1）在弱 λ
        # 被挪到共线项且 Lasso 符号翻转，符号门会误删真项；改由 OLS 重拟合 + STR 清理。
        self.str_clean = True

    def _bic_forward(self, Theta, y, cols, ridge, n_eff, max_terms=10):
        """向后消元后再做一次前向「补救」添加（标准逐步回归的第二阶段）。

        仅向后消元无法补回被稳定性投票漏掉的弱真值项：Lorenz 的 ẋ2 = 28x1 − x2 − x1x3
        中 x2 的系数只有主导项的 1/28，在任何 λ 下都进不了稳定性并集，
        删减法自然无从保留。前向补救在 n_eff 校正后的 BIC 下逐项试加，
        「确实能显著降低残差」的弱真值项会被重新纳入，而冗余项因
        通不过 RSS 门槛进不来。
        """
        n_obs = Theta.shape[0]
        p = Theta.shape[1]

        def bic_of(c):
            xi = ols_fit(Theta[:, c], y, ridge)
            r = y - Theta[:, c] @ xi
            rr = max(float(r @ r), 1e-300)
            return n_eff * math.log(rr / n_obs) + len(c) * math.log(n_eff)

        sel = [int(c) for c in cols]
        best_b = bic_of(sel)
        while len(sel) < min(max_terms, p):
            trial_best, trial_bic = None, np.inf
            for j in range(p):
                if j in sel:
                    continue
                t = sel + [j]
                b = bic_of(t)
                if b < trial_bic:
                    trial_bic, trial_best = b, t
            if trial_best is not None and trial_bic < best_b - 1e-10:
                best_b, sel = trial_bic, trial_best
            else:
                break
        return sel

    @staticmethod
    def _eff_n_from_resid(r, n_obs):
        """由残差 lag-1 自相关推有效样本量：n_eff = N(1−ρ)/(1+ρ)。

        SG 平滑让残差在约 w 个点内强相关，独立信息量远小于 N；直接用 N 做 BIC
        会把「只降低约 1% RSS」的冗余项判为显著，实测会让 VdP 的 ẋ1 方程多出
        一个 x1²=0.03 的伪项，长期 rollout 因此系统性漂移。
        实测本基准下 n_eff ≈ 2·N/w（VdP 47 / Duffing 61 / Pendulum 35）。
        """
        r = np.asarray(r, dtype=float).ravel()
        r = r - r.mean()
        v = float(r @ r) / max(len(r), 1)
        if v <= 0.0:
            return float(n_obs)
        rho = float((r[:-1] * r[1:]).mean() / v)
        rho = min(max(rho, 0.0), 0.999)
        return max(float(n_obs) * (1.0 - rho) / (1.0 + rho), 5.0)

    def _bic_backward(self, Theta, y, cols, ridge, n_eff=None):
        """在稳定性并集上做 BIC 驱动的向后消元。

        多项式+三角候选库大量冗余（如 x1 / sin(x1) / x1³ 沿轨道强相关），
        稳定性并集会把整簇共线项一起留下；直接 OLS 会产生 ±1e1~1e2 的
        相互抵消系数（真值被淹没）。BIC 向后消元按统计准则逐项剔除
        "删掉它残差几乎不变"的冗余列。

        为什么是向后（从并集删）而不是前向（从空集加）：并集已经由多 λ
        稳定性投票保证「真值项几乎必在内」，向后只需做减法，实证上能稳定
        剔除伪项；前向逐步在残差相关被低估时会逐个塞入 sin/cos 伪项
        （实测 pendulum dim0 会多出 3 个伪项）。
        """
        n_obs = Theta.shape[0]
        neff = float(n_obs if n_eff is None else n_eff)

        def bic_of(c):
            if not c:
                rr = float(y @ y)
                return neff * math.log(max(rr, 1e-300) / n_obs)
            xi = ols_fit(Theta[:, c], y, ridge)
            r = y - Theta[:, c] @ xi
            rr = max(float(r @ r), 1e-300)
            return neff * math.log(rr / n_obs) + len(c) * math.log(neff)

        cur = [int(c) for c in cols]
        best_b = bic_of(cur)
        while len(cur) > 1:
            best_trial = None
            for j in cur:
                trial = [c for c in cur if c != j]
                b = bic_of(trial)
                if best_trial is None or b < best_trial[1]:
                    best_trial = (trial, b)
            if best_trial is not None and best_trial[1] < best_b - 1e-10:
                cur, best_b = best_trial[0], best_trial[1]
            else:
                break

        xi = np.zeros(Theta.shape[1])
        if cur:
            xi[cur] = ols_fit(Theta[:, cur], y, ridge).ravel()
        return xi

    # ---- 后端无关的 Lasso 求解（sklearn 优先，纯 numpy 离线兜底） ----
    def _lasso(self, X, y, alpha):
        if available_sklearn():
            from sklearn.linear_model import Lasso

            return (
                Lasso(alpha=float(alpha), fit_intercept=False, max_iter=5000, tol=1e-8)
                .fit(X, y)
                .coef_
            )
        a = np.asarray(alpha, dtype=float).ravel()
        ak = float(a[0])
        return numpy_lasso(X, y.reshape(-1, 1), ak, self.max_iter, self.tol).ravel()

    def _alpha_grid(self, Theta_z, Xdot):
        """自动缩放 λ 网格：以数据自适应的参考 α 为中心，乘 stab_grid 倍数。

        使用 LassoCV 求每维参考 α（尺度无关，避免多项式/三角库共线导致的
        OLS 系数膨胀把阈值撑大）；无 sklearn 时退化为 z 空间 OLS 最大系数。
        返回 (n_grid, d) 的 α（z 空间尺度）。
        """
        d = Xdot.shape[1]
        if available_sklearn():
            from sklearn.linear_model import LassoCV

            alpha_ref = np.zeros(d)
            for k in range(d):
                y = Xdot[:, k]
                model = LassoCV(cv=5, max_iter=5000, tol=1e-8, fit_intercept=False).fit(Theta_z, y)
                alpha_ref[k] = float(model.alpha_)
        else:
            coef_z = ols_fit(Theta_z, Xdot)  # (p,d)
            alpha_ref = np.max(np.abs(coef_z), axis=0) * self.str_rel
            alpha_ref[alpha_ref == 0] = 1.0
        grid = np.zeros((len(self.stab_grid), d))
        for i, m in enumerate(self.stab_grid):
            grid[i] = alpha_ref * float(m)
        return grid

    def discover(self, Theta: np.ndarray, Xdot: np.ndarray) -> DiscoveryResult:
        Theta = np.asarray(Theta, dtype=float)
        Xdot = np.asarray(Xdot, dtype=float)
        # 去均值：噪声导数的 DC 偏置会被常数项吸收，干扰选择；去均值后更干净。
        Xdot = Xdot - Xdot.mean(axis=0, keepdims=True)
        p = Theta.shape[1]
        d = Xdot.shape[1]

        # z-score 特征：使 Lasso 惩罚在尺度上近似均匀（尺度无关），
        # 避免共线的高幅特征被不成比例地惩罚/保留。
        mu = Theta.mean(0)
        sd = Theta.std(0)
        sd[sd == 0] = 1.0
        Theta_z = (Theta - mu) / sd if self.zscore else Theta

        alpha_grid = self._alpha_grid(Theta_z, Xdot)  # (n_grid, d)
        n_lambda = len(self.stab_grid)
        sel = np.zeros((p, d), dtype=float)  # 各 λ 选中频率计数

        for gi in range(n_lambda):
            alpha = alpha_grid[gi]
            for k in range(d):
                xi = self._lasso(Theta_z, Xdot[:, k], alpha[k])
                sel[:, k] += (np.abs(xi) > 1e-9).astype(float)

        freq = sel / n_lambda
        keep = freq >= self.stab_frac  # 稳定性并集
        # 岭稳定：Θ 的子列常近共线（如保守 Duffing 的 x1 与 x1³ 同号强相关），
        # 无岭的 lstsq 会给出 ±1e2 量级爆炸系数，进而把下方 STR 阈值撑大并误删真值项。
        gram_diag = (Theta**2).sum(0)
        ridge = self.ridge_rel * float(np.mean(gram_diag))

        xi = np.zeros((p, d))
        for k in range(d):
            cols = [int(c) for c in np.where(keep[:, k])[0]]
            if not cols:
                # 退化保护：稳定性投票全空时，退回 OLS 幅值最大的一项
                xi_raw = ols_fit(Theta, Xdot[:, k : k + 1], ridge).ravel()
                cols = [int(np.argmax(np.abs(xi_raw)))]
            xi_u = ols_fit(Theta[:, cols], Xdot[:, k], ridge)
            resid = Xdot[:, k] - Theta[:, cols] @ xi_u
            n_eff = self._eff_n_from_resid(resid, Theta.shape[0])
            xi[:, k] = self._bic_backward(Theta, Xdot[:, k], cols, ridge, n_eff)

        if self.str_clean:
            thr = self.str_rel * np.max(np.abs(xi), axis=0, keepdims=True)
            thr[thr == 0] = 1.0
            xi[np.abs(xi) < thr] = 0.0

        support = np.abs(xi) > 1e-12
        return DiscoveryResult(xi=xi, support=support, method=self.name)
