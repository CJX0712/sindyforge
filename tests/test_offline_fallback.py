import numpy as np

from sindyforge.data.derivatives import estimate_derivative
from sindyforge.data.systems import LORENZ, simulate
from sindyforge.discovery.lasso_baseline import LassoBaseline
from sindyforge.domain.library import PolynomialLibrary


def test_fd_fallback_when_scipy_missing(monkeypatch):
    monkeypatch.setattr("sindyforge.core.backends.available_scipy", lambda: False)
    X = simulate(LORENZ, LORENZ.ic, 200, 0.02)
    est = estimate_derivative(X, 0.02, method="auto")
    assert est.shape == X.shape
    # 用 fd 估计的 Xdot 不应含 NaN
    assert np.all(np.isfinite(est))


def test_lasso_baseline_falls_back_without_sklearn(monkeypatch):
    monkeypatch.setattr("sindyforge.core.backends.available_sklearn", lambda: False)
    X = simulate(LORENZ, LORENZ.ic + 0.01, 400, 0.02)
    Xdot = LORENZ.rhs(X)
    lib = PolynomialLibrary(dim=3, degree=2, include_trig=False)
    Theta = lib.build(X)
    res = LassoBaseline().discover(Theta, Xdot)
    assert res.xi.shape == (len(lib.feature_names), 3)
    assert np.all(np.isfinite(res.xi))
