"""P1: the printed eigenvalues, the checks on the decomposition and the agreement with
scikit-learn."""

import re

import numpy as np

from src import p1
from src.common.data import FEATURES, D
from tests import Z, numbers

TOLERANCE = 1e-8  # the handout's threshold for two quantities to agree


def test_eigenvalues_print_in_the_required_form():
    lines = p1.compute()["stdout"].splitlines()
    assert [line.split(" = ")[0] for line in lines] == [f"lambda_{j}" for j in range(1, D + 1)]
    assert all(re.fullmatch(r"lambda_\d+ = \d\.\d{4}", line) for line in lines)
    values = [float(line.split(" = ")[1]) for line in lines]
    assert values == sorted(values, reverse=True)
    np.testing.assert_allclose(values, numbers("p1")["eigenvalues"], atol=5e-5)


def test_decomposition_checks_and_library_agreement():
    n = numbers("p1")
    assert n["orthonormality"] < TOLERANCE
    assert n["trace_minus_sum"] < TOLERANCE
    assert n["library_diff"]["eigenvalues"] < TOLERANCE
    assert n["library_diff"]["eigenvectors"] < TOLERANCE
    np.testing.assert_allclose(n["trace"], D)  # standardised features each have variance 1


def test_eigenvalues_are_those_of_the_correlation_matrix():
    reference = np.linalg.eigvalsh(np.corrcoef(Z, rowvar=False))[::-1]
    np.testing.assert_allclose(numbers("p1")["eigenvalues"], reference)


def test_top_features():
    u = np.zeros(D)
    u[[2, 5, 9]] = 0.1, -0.9, 0.4
    assert p1.top_features(u) == [FEATURES[5], FEATURES[9], FEATURES[2]]
