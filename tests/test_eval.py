import numpy as np
from sindyforge.core.config import Config
from sindyforge.data.derivatives import estimate_derivative
from sindyforge.data.systems import LORENZ, simulate
from sindyforge.discovery.strlasso import StabSINDy
from sindyforge.domain.library import PolynomialLibrary
from sindyforge.eval.metrics import rollout_rmse, support_f1, welch_ttest


def test_rollout_rmse_finite_and_support_f1_range():
    rng = np.random.RandomState(3)
    X = simulate(LORENZ, LORENZ.ic + rng.normal(0, 0.05, 3), 600, 0.02)
    Xdot = estimate_derivative(X, 0.02, method="auto")
    lib = PolynomialLibrary(dim=3, degree=2, include_trig=False)
    Theta = lib.build(X)
    true_xi = lib.true_support_matrix(LORENZ.true_terms)
    true_support = np.abs(true_xi) > 1e-12
    res = StabSINDy(Config()).discover(Theta, Xdot)
    rmse = rollout_rmse(LORENZ, lib, res.xi, X, 0.02, 100)
    assert np.isfinite(rmse) and rmse >= 0.0
    f1 = support_f1(true_support, res.support)
    assert 0.0 <= f1 <= 1.0


def test_welch_ttest_basic():
    a = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
    b = np.array([2.0, 2.1, 1.9, 2.05, 1.95])
    t, p, ma, mb = welch_ttest(a, b)
    assert mb > ma
    assert p < 0.05
