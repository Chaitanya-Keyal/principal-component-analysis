"""P3: the three preprocessing variants and the numbers reported for them."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from src import p3
from src.common.data import FEATURES
from src.common.pca import covariance
from tests import X_TEST, Y_TEST, X, Y, numbers


def test_centred_and_min_max_use_training_statistics_only():
    np.testing.assert_allclose(p3.centred(X, X).mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(p3.centred(X, X_TEST), X_TEST - X.mean(axis=0))

    scaled = p3.min_max(X, X)
    np.testing.assert_allclose(scaled.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(scaled.max(axis=0) - scaled.min(axis=0), 1)
    low, high = X.min(axis=0), X.max(axis=0)
    expected = (X_TEST - low) / (high - low) - ((X - low) / (high - low)).mean(axis=0)
    np.testing.assert_allclose(p3.min_max(X, X_TEST), expected)


def test_z_scored_covariance_is_the_correlation_matrix():
    z_score = p3.VARIANTS["V3"][1]
    np.testing.assert_allclose(covariance(z_score(X, X)), np.corrcoef(X, rowvar=False))


def test_first_ratio_is_at_least_the_largest_raw_share():
    n = numbers("p3")
    variance = X.var(axis=0, ddof=1)
    np.testing.assert_allclose(n["raw_variance_share"], variance.max() / variance.sum())
    assert n["variants"]["V1"]["r1"] >= n["raw_variance_share"]  # lambda_1 >= S_jj
    top = np.argsort(variance)[::-1][:3]
    assert [d["feature"] for d in n["raw_variance"]] == [FEATURES[j] for j in top]


def test_variants_against_library_pca_and_knn():
    for name, (_, preprocess) in p3.VARIANTS.items():
        reported = numbers("p3")["variants"][name]
        library = PCA(n_components=2, svd_solver="full").fit(preprocess(X, X))
        np.testing.assert_allclose(
            [reported["r1"], reported["r2"]], library.explained_variance_ratio_
        )
        # the projections agree up to the sign of each axis, which nearest neighbours ignore
        knn = KNeighborsClassifier(n_neighbors=5).fit(library.transform(preprocess(X, X)), Y)
        accuracy = knn.score(library.transform(preprocess(X, X_TEST)), Y_TEST)
        np.testing.assert_allclose(reported["accuracy"], accuracy)
