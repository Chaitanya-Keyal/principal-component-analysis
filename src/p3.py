"""P3: PCA under three preprocessing variants, and what the two-component projection shows.

Every variant is fitted on the training set only and applied unchanged to the test set. The
covariance matrix and its eigendecomposition are P1's code.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.neighbors import KNeighborsClassifier

from src.common.data import CLASSES, FEATURES, TEST, TRAIN, load
from src.common.pca import covariance, eigendecomposition, variance_ratio, z_score
from src.common.plots import AQUA, BLUE, ORANGE

CLASS_COLOURS = dict(zip(CLASSES, (BLUE, ORANGE, AQUA), strict=True))
N_RAW = 3  # how many of the largest raw variances to report for V1


def centred(X_train, X):
    """V1: subtract the training mean, no scaling."""
    return X - X_train.mean(axis=0)


def min_max(X_train, X):
    """V2: scale by the training minima and maxima, then subtract the training mean of the
    scaled features."""
    low, high = X_train.min(axis=0), X_train.max(axis=0)
    return (X - low) / (high - low) - ((X_train - low) / (high - low)).mean(axis=0)


VARIANTS = {"V1": ("centred only", centred), "V2": ("min-max", min_max), "V3": ("z-score", z_score)}


def compute():
    X, y = load(TRAIN)
    X_test, y_test = load(TEST)

    variants, projections = {}, {}
    for name, (_, preprocess) in VARIANTS.items():
        A, A_test = preprocess(X, X), preprocess(X, X_test)
        lam, U = eigendecomposition(covariance(A))
        r = variance_ratio(lam)
        C, C_test = A @ U[:, :2], A_test @ U[:, :2]  # c_n = U_2^T x'_n
        knn = KNeighborsClassifier(n_neighbors=5).fit(C, y)
        correct = int((knn.predict(C_test) == y_test).sum())
        variants[name] = {
            "r1": float(r[0]),
            "r2": float(r[1]),
            "r1_plus_r2": float(r[0] + r[1]),
            "pc1_feature": FEATURES[int(np.abs(U[:, 0]).argmax())],
            "pc2_feature": FEATURES[int(np.abs(U[:, 1]).argmax())],
            "correct": correct,
            "accuracy": correct / len(y_test),
        }
        projections[name] = (C, C_test)

    variance = X.var(axis=0, ddof=1)  # the diagonal of V1's S
    largest = np.argsort(-variance, kind="stable")[:N_RAW]
    numbers = {
        "n_test": len(y_test),
        "variants": variants,
        "raw_variance": [{"feature": FEATURES[j], "variance": float(variance[j])} for j in largest],
        "raw_variance_share": float(variance.max() / variance.sum()),
    }
    return {"numbers": numbers, "projections": projections, "y": y, "y_test": y_test}


def projection_figure(res):
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.7))
    handles = [
        Line2D([], [], linestyle="", marker=marker, color=colour, label=f"class {c}, {split}")
        for c, colour in CLASS_COLOURS.items()
        for marker, split in (("o", "train"), ("x", "test"))
    ]
    for ax, (name, (label, _)) in zip(axes, VARIANTS.items(), strict=True):
        C, C_test = res["projections"][name]
        for c, colour in CLASS_COLOURS.items():
            ax.scatter(*C[res["y"] == c].T, s=16, color=colour, edgecolors="white", linewidths=0.4)
            ax.scatter(
                *C_test[res["y_test"] == c].T, s=24, color=colour, marker="x", linewidths=1.2
            )
        ax.set(xlabel="PC1", ylabel="PC2", title=f"{name}: {label}")
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.19),
            ncol=3,
            fontsize=6.5,
            columnspacing=0.6,
            handletextpad=0.1,
        )
    fig.tight_layout()
    return fig


def figures(res):
    return {"p3_projections": projection_figure(res)}
