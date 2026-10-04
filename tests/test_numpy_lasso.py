import numpy as np

from sindyforge.discovery.strlasso import numpy_lasso


def test_recover_sparse_vector_noiseless():
    rng = np.random.RandomState(1)
    N, p = 200, 12
    X = rng.randn(N, p)
    true = np.zeros(p)
    true[[1, 4, 7]] = [3.0, -2.0, 1.5]
    y = X @ true
    # 足够大的 alpha 应精确恢复支撑（Lasso 软阈值会轻微收缩，属正常现象）
    coef = numpy_lasso(X, y.reshape(-1, 1), alpha=0.1)
    coef = coef.ravel()
    support = np.abs(coef) > 1e-6
    assert np.array_equal(support, true != 0)  # 支撑完全一致
    rel = np.abs(coef[true != 0] - true[true != 0]) / (np.abs(true[true != 0]) + 1e-9)
    assert np.all(rel < 0.02)  # 真值支撑上相对误差 < 2%


def test_zero_noise_small_alpha_dense():
    rng = np.random.RandomState(2)
    X = rng.randn(100, 5)
    y = X @ np.array([1.0, -1.0, 0.0, 0.0, 2.0])
    coef = numpy_lasso(X, y.reshape(-1, 1), alpha=1e-6).ravel()
    assert np.allclose(coef, [1.0, -1.0, 0.0, 0.0, 2.0], atol=1e-5)
