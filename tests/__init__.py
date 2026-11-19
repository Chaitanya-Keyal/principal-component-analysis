"""Shared test inputs: the two provided files, standardised with the training statistics, and
each problem's numbers as the report reads them."""

import json
from functools import cache

from src import p1, p2, p3, p4
from src.common.data import TEST, TRAIN, load
from src.common.pca import z_score

X, Y = load(TRAIN)
X_TEST, Y_TEST = load(TEST)
Z, Z_TEST = z_score(X, X), z_score(X, X_TEST)
PROBLEMS = {"p1": p1, "p2": p2, "p3": p3, "p4": p4}


@cache
def numbers(problem):
    """One problem's numbers after the round trip through JSON that results/<p>.json makes."""
    return json.loads(json.dumps(PROBLEMS[problem].compute()["numbers"]))
