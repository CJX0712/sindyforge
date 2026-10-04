import numpy as np

from sindyforge.domain.library import PolynomialLibrary, _monomial_powers


def test_monomial_powers_counts():
    # dim=3, degree=3 → C(3,1)+C(4,2)+C(5,3)=3+6+10=19
    assert len(_monomial_powers(3, 3)) == 19
    assert len(_monomial_powers(2, 2)) == 5  # C(2,1)+C(3,2)=2+3=5


def test_feature_names_and_build_shape():
    lib = PolynomialLibrary(dim=2, degree=2, include_trig=True)
    # 1(const) + 5(monomials deg1..2) + 4(trig x2) = 10
    assert lib.feature_names[0] == "1"
    assert len(lib.feature_names) == 10
    X = np.random.RandomState(0).randn(50, 2)
    Theta = lib.build(X)
    assert Theta.shape == (50, 10)
    # 常量列全 1
    assert np.allclose(Theta[:, 0], 1.0)
    # sin/cos 列存在且 named
    assert "sin(x1)" in lib.feature_names
    assert "cos(x2)" in lib.feature_names


def test_true_support_matrix_alignment():
    from sindyforge.data.systems import LORENZ

    lib = PolynomialLibrary(dim=3, degree=2, include_trig=False)
    xi = lib.true_support_matrix(LORENZ.true_terms)
    assert xi.shape == (len(lib.feature_names), 3)
    # lorenz 真支撑：x1,x2,x3,x1*x2,x1*x3
    for name in ["x1", "x2", "x3", "x1*x2", "x1*x3"]:
        j = lib.index(name)
        assert np.any(np.abs(xi[j, :]) > 0)
    # 不应出现 x2*x3（lorenz 无此项）
    assert np.all(np.abs(xi[lib.index("x2*x3"), :]) < 1e-12)
