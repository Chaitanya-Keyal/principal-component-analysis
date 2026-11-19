"""The data loader, the PCA building blocks and the number formats, against hand-worked values
and NumPy's own routines."""

import numpy as np

from src.common.data import CLASSES, FEATURES, D
from src.common.fmt import fixed, sci
from src.common.pca import covariance, eigendecomposition, fix_signs, variance_ratio
from tests import X_TEST, Y_TEST, Z_TEST, X, Y, Z


def test_provided_files_load():
    assert X.shape == (124, 13)
    assert X_TEST.shape == (54, 13)
    assert len(FEATURES) == D == 13
    assert set(Y) == set(Y_TEST) == set(CLASSES)


def test_z_score_uses_training_statistics_only():
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(Z.std(axis=0, ddof=1), 1)
    mu, s = X.mean(axis=0), np.sqrt(((X - X.mean(axis=0)) ** 2).sum(axis=0) / (len(X) - 1))
    np.testing.assert_allclose(Z_TEST, (X_TEST - mu) / s)


def test_covariance_uses_the_n_minus_1_divisor():
    np.testing.assert_allclose(covariance(X), np.cov(X, rowvar=False))
    points = np.array([[0.0, 0.0], [2.0, 4.0]])  # centred: (-1, -2) and (1, 2)
    np.testing.assert_allclose(covariance(points), [[2, 4], [4, 8]])


def test_fix_signs_makes_the_largest_entry_positive():
    U = np.array([[0.6, -0.8], [-0.8, -0.6]])
    np.testing.assert_allclose(fix_signs(U), [[-0.6, 0.8], [0.8, 0.6]])


def test_eigendecomposition_by_hand():
    lam, U = eigendecomposition(np.array([[2.0, 1.0], [1.0, 2.0]]))
    np.testing.assert_allclose(lam, [3, 1])
    np.testing.assert_allclose(U, np.array([[1, 1], [1, -1]]) / np.sqrt(2))


def test_eigendecomposition_of_the_training_covariance():
    S = covariance(Z)
    lam, U = eigendecomposition(S)
    assert np.all(np.diff(lam) <= 0)  # descending
    np.testing.assert_allclose(S @ U, U * lam, atol=1e-12)
    np.testing.assert_allclose(U.T @ U, np.eye(D), atol=1e-12)
    assert np.all(U[np.abs(U).argmax(axis=0), np.arange(D)] > 0)  # the sign rule
    np.testing.assert_allclose(variance_ratio(lam).sum(), 1)


def test_number_formats():
    assert fixed(0.35730459) == "0.3573"
    assert fixed(13) == "13.0000"
    assert fixed(-0.00001) == "0.0000"
    assert sci(3.2e-15) == "3.2e-15"
