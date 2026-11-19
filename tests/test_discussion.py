"""What the report's discussion paragraphs state in words, checked on the provided data.

The numbers in those paragraphs come from results/numbers.tex and follow the data by themselves;
the statements below do not, so a failure here names a sentence of report/report.tex to reread.
"""

import numpy as np

from src import p4
from tests import numbers


def test_p2_elbow_agrees_with_kaiser_and_is_below_the_thresholds():
    n = numbers("p2")
    assert n["k_elbow"] == n["kaiser"]
    assert n["k_elbow"] < n["k_tau"]["0.80"] <= n["k_tau"]["0.90"] <= n["k_tau"]["0.95"]


def test_p2_both_errors_fall_at_every_step_and_test_stays_above_train():
    n = numbers("p2")
    J_train, J_test = np.array(n["J_train"]), np.array(n["J_test"])
    assert np.all(np.diff(J_train) < 0)
    assert np.all(np.diff(J_test) < 0)
    assert np.all(J_test[:-1] > J_train[:-1])


def test_p3_one_raw_feature_owns_the_first_component_of_v1():
    n = numbers("p3")
    v1 = n["variants"]["V1"]
    assert v1["pc1_feature"] == n["raw_variance"][0]["feature"]
    assert v1["pc2_feature"] == n["raw_variance"][1]["feature"]
    assert n["raw_variance_share"] > 0.99
    assert v1["pc1_loading"] > 0.99


def test_p3_v1_keeps_the_most_variance_and_classifies_worst():
    n = numbers("p3")
    v = n["variants"]
    assert v["V1"]["r1_plus_r2"] > v["V3"]["r1_plus_r2"]
    assert v["V1"]["correct"] < min(v["V2"]["correct"], v["V3"]["correct"])
    # most of V1's errors are between the one pair of classes the paragraph names
    assert v["V1"]["most_confused"]["errors"] > (n["n_test"] - v["V1"]["correct"]) / 2
    assert 0 < v["V2"]["correct"] - v["V3"]["correct"] <= 2  # V2's small lead over V3


def test_p4_pca_is_never_below_the_selection_methods():
    correct = numbers("p4")["correct"]
    assert all(a >= max(b, c) for a, b, c in zip(*correct.values(), strict=True))


def test_p4_wrapper_is_lowest_from_k_2_and_scores_higher_in_cross_validation():
    n = numbers("p4")
    correct = n["correct"]
    for i, k in enumerate(p4.K_VALUES):
        if k >= 2:
            assert correct["wrapper"][i] < min(correct["pca"][i], correct["filter"][i])
    for k in p4.K_VALUES[-3:]:  # the three k the paragraph quotes
        assert n["sfs_cv_accuracy"][k - 1] > n["accuracy"]["wrapper"][p4.K_VALUES.index(k)]


def test_p4_the_largest_gap_is_more_than_two_samples():
    correct = numbers("p4")["correct"]
    spread = [max(counts) - min(counts) for counts in zip(*correct.values(), strict=True)]
    assert max(spread) > 2
