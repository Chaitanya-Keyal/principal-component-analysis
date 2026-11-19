"""Standardisation, the sample covariance and its eigendecomposition (conventions C2 to C5)."""

import numpy as np


def z_score(X_train, X):
    """X standardised with the training mean and sample standard deviation (divisor N - 1)."""
    return (X - X_train.mean(axis=0)) / X_train.std(axis=0, ddof=1)


def covariance(X):
    """S = Xc^T Xc / (N - 1), with Xc the mean-centred X."""
    Xc = X - X.mean(axis=0)
    return Xc.T @ Xc / (len(X) - 1)


def fix_signs(U):
    """Flip each column so that its entry of largest absolute value is positive."""
    largest = U[np.abs(U).argmax(axis=0), np.arange(U.shape[1])]
    return U * np.sign(largest)


def eigendecomposition(S):
    """Eigenvalues in descending order and their unit eigenvectors as the columns of U."""
    lam, U = np.linalg.eigh(S)  # ascending
    return lam[::-1], fix_signs(U[:, ::-1])


def variance_ratio(lam):
    """r_j = lambda_j / sum_i lambda_i."""
    return lam / lam.sum()
