"""P1: PCA from first principles on the standardised training features.

S is built from its definition and decomposed with a symmetric eigensolver in src/common/pca.py;
scikit-learn's PCA is fitted only to check the eigenvalues and eigenvectors against.
"""

import numpy as np
from sklearn.decomposition import PCA

from src.common.data import FEATURES, TRAIN, D, load
from src.common.fmt import fixed
from src.common.pca import covariance, eigendecomposition, fix_signs, z_score

TOP = 3  # how many of the largest |loadings| to name on each component


def top_features(u, n=TOP):
    """The names of the n features with the largest |loading| in u, largest first."""
    return [FEATURES[j] for j in np.argsort(-np.abs(u), kind="stable")[:n]]


def compute():
    X, _ = load(TRAIN)  # the class column is not used in this problem
    Z = z_score(X, X)
    S = covariance(Z)
    lam, U = eigendecomposition(S)

    library = PCA(n_components=D, svd_solver="full").fit(Z)
    # components_ holds the eigenvectors as rows, so the sign rule goes on the columns of its
    # transpose
    U_library = fix_signs(library.components_.T)

    numbers = {
        "eigenvalues": lam.tolist(),
        "orthonormality": float(np.linalg.norm(U.T @ U - np.eye(D))),  # Frobenius norm
        "trace": float(np.trace(S)),
        "trace_minus_sum": float(abs(np.trace(S) - lam.sum())),
        "library_diff": {
            "eigenvalues": float(np.abs(lam - library.explained_variance_).max()),
            "eigenvectors": float(np.abs(U - U_library).max()),
        },
        "loadings": {"pc1": U[:, 0].tolist(), "pc2": U[:, 1].tolist()},
        "top": {"pc1": top_features(U[:, 0]), "pc2": top_features(U[:, 1])},
    }
    stdout = "\n".join(f"lambda_{j} = {fixed(v)}" for j, v in enumerate(lam, start=1))
    return {"numbers": numbers, "stdout": stdout}


def figures(res):
    """P1 asks for numbers and tables only."""
    return {}
