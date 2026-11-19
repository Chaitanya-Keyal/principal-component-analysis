"""P2: variance captured and reconstruction error against the number of components k.

The PCA is P1's, on the standardised training features; the test set is standardised with the
training mean and standard deviation and projected on the same U.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.common.data import TEST, TRAIN, D, load
from src.common.pca import covariance, eigendecomposition, variance_ratio, z_score
from src.common.plots import BLUE, INK_2, MUTED, ORANGE

TAUS = (0.80, 0.90, 0.95)  # cumulative-variance thresholds
ELBOW = 0.10  # the elbow is the last component with r_k >= ELBOW
KS = np.arange(1, D + 1)


def reconstruction_error(Z, U_k):
    """J(k) = sum_n ||z_n - U_k U_k^T z_n||^2 / (n - 1), from the reconstructions themselves."""
    Z_hat = Z @ U_k @ U_k.T
    return float(((Z - Z_hat) ** 2).sum() / (len(Z) - 1))


def compute():
    X, _ = load(TRAIN)
    X_test, _ = load(TEST)
    Z, Z_test = z_score(X, X), z_score(X, X_test)
    S = covariance(Z)
    lam, U = eigendecomposition(S)

    r = variance_ratio(lam)
    cumulative = np.cumsum(r)  # R(k)
    k_elbow = int(KS[r >= ELBOW].max())
    J_train = np.array([reconstruction_error(Z, U[:, :k]) for k in KS])
    J_test = np.array([reconstruction_error(Z_test, U[:, :k]) for k in KS])
    tail = np.array([lam[k:].sum() for k in KS])  # sum of lambda_j over j > k

    numbers = {
        "n_train": len(Z),
        "n_test": len(Z_test),
        "r": r.tolist(),
        "R": cumulative.tolist(),
        "J_train": J_train.tolist(),
        "J_test": J_test.tolist(),
        "k_elbow": k_elbow,
        "R_elbow": float(cumulative[k_elbow - 1]),
        "k_tau": {f"{tau:.2f}": int(KS[cumulative >= tau].min()) for tau in TAUS},
        "kaiser": int((lam > np.trace(S) / D).sum()),
        "identity_residual": float(np.abs(J_train - tail).max()),
    }
    return {"numbers": numbers}


def variance_figure(res):
    n = res["numbers"]
    r, R, k_elbow = n["r"], n["R"], n["k_elbow"]
    fig, (scree, cumulative) = plt.subplots(1, 2, figsize=(8.2, 3.4))

    bars = scree.bar(KS, r, width=0.72, color=BLUE, label=r"$r_j$")
    elbow = scree.axvline(
        k_elbow,
        color=INK_2,
        linewidth=1.2,
        linestyle="--",
        label=rf"$k_\mathrm{{elbow}}$ = {k_elbow}",
    )
    threshold = scree.axhline(
        ELBOW, color=MUTED, linewidth=1, linestyle=":", label=f"threshold {ELBOW:.2f}"
    )
    scree.set(
        xlabel="component $j$",
        ylabel=r"explained variance ratio $r_j$",
        title="Scree plot",
        xticks=KS,
    )
    scree.grid(axis="x", visible=False)
    scree.legend(handles=[bars, elbow, threshold], loc="upper right")

    cumulative.plot(KS, R, color=BLUE, marker="o", label="$R(k)$")
    for tau, style in zip(TAUS, (":", "--", "-."), strict=True):
        cumulative.axhline(
            tau, color=INK_2, linewidth=1, linestyle=style, label=rf"$\tau$ = {tau:.2f}"
        )
    cumulative.set(
        xlabel="number of components $k$",
        ylabel="cumulative explained variance $R(k)$",
        title="Cumulative explained variance",
        xticks=KS,
    )
    cumulative.set_ylim(0, 1.03)
    cumulative.legend(loc="lower right")
    fig.tight_layout()
    return fig


def reconstruction_figure(res):
    n = res["numbers"]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(
        KS,
        n["J_train"],
        color=BLUE,
        marker="o",
        label=rf"$J_\mathrm{{train}}(k)$, {n['n_train']} points",
    )
    ax.plot(
        KS,
        n["J_test"],
        color=ORANGE,
        marker="s",
        label=rf"$J_\mathrm{{test}}(k)$, {n['n_test']} points",
    )
    ax.set(
        xlabel="number of components $k$",
        ylabel="reconstruction error $J(k)$",
        title="Reconstruction error against the number of components",
        xticks=KS,
    )
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    return fig


def figures(res):
    return {"p2_variance": variance_figure(res), "p2_reconstruction": reconstruction_figure(res)}
