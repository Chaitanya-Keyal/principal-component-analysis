"""P4: PCA against a filter (F-score) and a wrapper (sequential forward selection) as inputs to
one classifier, on the standardised features.

The classifier and the cross-validation folds are the handout's and are the same for every
method. Forward selection is greedy, so the sets for k = 1, 2, 3, 5, 8 are the first k features
of a single run to k = 8.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from threadpoolctl import threadpool_limits

from src.common.data import FEATURES, TEST, TRAIN, D, load
from src.common.pca import covariance, eigendecomposition, z_score
from src.common.plots import AQUA, BLUE, INK_2, ORANGE

K_VALUES = (1, 2, 3, 5, 8)
LISTED = (3, 8)  # the k for which the chosen features are listed
METHODS = {"pca": "(A) PCA", "filter": "(B) Filter, F-score", "wrapper": "(C) Wrapper, SFS"}
FOLDS = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def classifier():
    return LogisticRegression(C=1.0, max_iter=1000)


def test_correct(A, y, A_test, y_test):
    """How many test samples the classifier trained on (A, y) labels correctly."""
    return int((classifier().fit(A, y).predict(A_test) == y_test).sum())


def f_scores(Z, y):
    """F_j: the between-class variance of feature j over its within-class variance."""
    classes = np.unique(y)
    mean = Z.mean(axis=0)
    between = sum((y == c).sum() * (Z[y == c].mean(axis=0) - mean) ** 2 for c in classes)
    within = sum(((Z[y == c] - Z[y == c].mean(axis=0)) ** 2).sum(axis=0) for c in classes)
    return (between / (len(classes) - 1)) / (within / (len(y) - len(classes)))


def forward_selection(Z, y, n_select):
    """Sequential forward selection by mean cross-validated accuracy over FOLDS.

    Returns the chosen column indices in the order they were added, and the score each reached.
    """
    chosen, scores = [], []
    with threadpool_limits(limits=1):  # hundreds of tiny fits: BLAS threads only add overhead
        for _ in range(n_select):
            best, best_score = None, -np.inf
            for j in range(Z.shape[1]):  # increasing column index
                if j in chosen:
                    continue
                columns = sorted([*chosen, j])
                score = cross_val_score(
                    classifier(), Z[:, columns], y, cv=FOLDS, scoring="accuracy"
                ).mean()
                if score > best_score:  # strictly greater, so the first maximum is kept
                    best, best_score = j, score
            chosen.append(best)
            scores.append(float(best_score))
    return chosen, scores


def compute():
    X, y = load(TRAIN)
    X_test, y_test = load(TEST)
    Z, Z_test = z_score(X, X), z_score(X, X_test)
    n_test = len(y_test)

    _, U = eigendecomposition(covariance(Z))
    F = f_scores(Z, y)
    ranking = [int(j) for j in np.argsort(-F, kind="stable")]  # best first
    sfs, sfs_scores = forward_selection(Z, y, max(K_VALUES))

    def selected(order, k):  # columns in increasing index, as a get_support() mask gives them
        columns = sorted(order[:k])
        return test_correct(Z[:, columns], y, Z_test[:, columns], y_test)

    baseline = test_correct(Z, y, Z_test, y_test)
    correct = {
        "pca": [test_correct(Z @ U[:, :k], y, Z_test @ U[:, :k], y_test) for k in K_VALUES],
        "filter": [selected(ranking, k) for k in K_VALUES],
        "wrapper": [selected(sfs, k) for k in K_VALUES],
    }
    numbers = {
        "n_test": n_test,
        "baseline": {"correct": baseline, "accuracy": baseline / n_test},
        "correct": correct,
        "accuracy": {m: [c / n_test for c in counts] for m, counts in correct.items()},
        # the smallest k in the table within one test sample of the baseline, if any
        "within_one": {
            m: next((k for k, c in zip(K_VALUES, counts, strict=True) if c >= baseline - 1), None)
            for m, counts in correct.items()
        },
        "f_scores": F.tolist(),
        "f_ranking": [FEATURES[j] for j in ranking],
        "sfs_order": [FEATURES[j] for j in sfs],
        "sfs_cv_accuracy": sfs_scores,
        "chosen": {
            str(k): {
                "filter": [FEATURES[j] for j in ranking[:k]],
                "wrapper": [FEATURES[j] for j in sfs[:k]],
                "shared": [FEATURES[j] for j in ranking[:k] if j in sfs[:k]],
            }
            for k in LISTED
        },
    }
    return {"numbers": numbers}


def accuracy_figure(res):
    n = res["numbers"]
    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    # where curves meet, the larger marker stays visible behind the smaller one
    styles = {"pca": (BLUE, "o", 8.5), "filter": (ORANGE, "s", 6.5), "wrapper": (AQUA, "^", 6)}
    for method, label in METHODS.items():
        colour, marker, size = styles[method]
        ax.plot(
            K_VALUES,
            n["accuracy"][method],
            color=colour,
            marker=marker,
            markersize=size,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=label,
        )
    ax.axhline(
        n["baseline"]["accuracy"],
        color=INK_2,
        linestyle="--",
        linewidth=1.2,
        label=f"baseline, all {D} features",
    )
    ax.set(
        xlabel="number of features $k$",
        ylabel="test accuracy",
        title="Test accuracy against the number of features",
        xticks=K_VALUES,
    )
    ax.legend(loc="lower right")
    return fig


def figures(res):
    return {"p4_accuracy": accuracy_figure(res)}
