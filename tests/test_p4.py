"""P4: the F-score and forward selection against scikit-learn, and the reported accuracies."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector, f_classif
from threadpoolctl import threadpool_limits

from src import p4
from src.common.data import FEATURES
from tests import Y_TEST, Z_TEST, Y, Z, numbers


def test_f_scores_match_the_one_way_anova_statistic():
    np.testing.assert_allclose(p4.f_scores(Z, Y), f_classif(Z, Y)[0])
    np.testing.assert_allclose(numbers("p4")["f_scores"], f_classif(Z, Y)[0])
    ranking = [FEATURES[j] for j in np.argsort(f_classif(Z, Y)[0])[::-1]]
    assert numbers("p4")["f_ranking"] == ranking


def test_forward_selection_matches_the_library_selector():
    chosen = [FEATURES.index(name) for name in numbers("p4")["sfs_order"]]
    with threadpool_limits(limits=1):
        for k in p4.K_VALUES:
            selector = SequentialFeatureSelector(
                p4.classifier(),
                n_features_to_select=k,
                direction="forward",
                scoring="accuracy",
                cv=p4.FOLDS,
            ).fit(Z, Y)
            assert set(np.flatnonzero(selector.get_support())) == set(chosen[:k])


def test_baseline_and_pca_accuracies():
    n = numbers("p4")
    baseline = p4.classifier().fit(Z, Y).score(Z_TEST, Y_TEST)
    np.testing.assert_allclose(n["baseline"]["accuracy"], baseline)
    for k, reported in zip(p4.K_VALUES, n["accuracy"]["pca"], strict=True):
        library = PCA(n_components=k, svd_solver="full").fit(Z)
        accuracy = (
            p4.classifier().fit(library.transform(Z), Y).score(library.transform(Z_TEST), Y_TEST)
        )
        np.testing.assert_allclose(reported, accuracy)


def test_selected_feature_accuracies_and_shared_counts():
    n = numbers("p4")
    for method, order in (("filter", n["f_ranking"]), ("wrapper", n["sfs_order"])):
        for k, reported in zip(p4.K_VALUES, n["accuracy"][method], strict=True):
            columns = sorted(FEATURES.index(name) for name in order[:k])
            accuracy = p4.classifier().fit(Z[:, columns], Y).score(Z_TEST[:, columns], Y_TEST)
            np.testing.assert_allclose(reported, accuracy)
    for k, sets in n["chosen"].items():
        assert sets["filter"] == n["f_ranking"][: int(k)]
        assert sets["wrapper"] == n["sfs_order"][: int(k)]
        assert set(sets["shared"]) == set(sets["filter"]) & set(sets["wrapper"])


def test_within_one_test_sample_of_the_baseline():
    n = numbers("p4")
    for method, counts in n["correct"].items():
        reached = [
            k for k, c in zip(p4.K_VALUES, counts, strict=True) if c >= n["baseline"]["correct"] - 1
        ]
        assert n["within_one"][method] == (reached[0] if reached else None)
