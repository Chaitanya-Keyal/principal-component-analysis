"""P2: the reconstruction error and the rules for choosing k, recomputed from their
definitions."""

import numpy as np

from src import p2
from src.common.data import D
from src.common.pca import covariance, eigendecomposition
from tests import Z_TEST, Z, numbers

LAM, U = eigendecomposition(covariance(Z))


def test_reconstruction_error_by_hand():
    points = np.array([[3.0, 4.0], [1.0, -2.0]])
    first_axis = np.array([[1.0], [0.0]])  # keeps the first coordinate, loses the second
    np.testing.assert_allclose(p2.reconstruction_error(points, first_axis), (16 + 4) / 1)


def test_training_error_is_the_sum_of_the_discarded_eigenvalues():
    n = numbers("p2")
    np.testing.assert_allclose(n["J_train"], [LAM[k:].sum() for k in p2.KS], atol=1e-12)
    assert n["identity_residual"] < 1e-8
    np.testing.assert_allclose(np.array(n["J_train"]) / LAM.sum(), 1 - np.array(n["R"]), atol=1e-12)


def test_test_error_uses_the_test_divisor_and_never_increases():
    n = numbers("p2")
    scores = Z_TEST @ U  # u_j^T z_n for every test point
    by_residual = [(scores[:, k:] ** 2).sum() / (len(Z_TEST) - 1) for k in p2.KS]
    np.testing.assert_allclose(n["J_test"], by_residual, atol=1e-12)
    assert np.all(np.diff(n["J_test"]) <= 0)
    assert n["J_test"][-1] < 1e-20  # all 13 components reconstruct every point


def test_rules_for_choosing_k():
    n = numbers("p2")
    r = LAM / LAM.sum()
    assert n["k_elbow"] == max(k for k in p2.KS if r[k - 1] >= 0.10)
    assert r[n["k_elbow"] - 1] >= 0.10 > r[n["k_elbow"]]  # the next ratio is the first below 0.10
    for tau in p2.TAUS:
        k = n["k_tau"][f"{tau:.2f}"]
        assert r[:k].sum() >= tau > r[: k - 1].sum()
    assert n["kaiser"] == sum(value > 1 for value in LAM)  # tr(S) / D = 1 for standardised data
    assert len(n["r"]) == len(n["J_train"]) == len(n["J_test"]) == D
