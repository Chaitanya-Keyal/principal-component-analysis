"""The two provided CSV files: 13 feature columns in the handout's order, then the class."""

import os
from pathlib import Path

import numpy as np

# WINE_DATA_DIR reads the two files from another folder instead of data/.
DATA_DIR = Path(os.environ.get("WINE_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
TRAIN, TEST = "wine_train.csv", "wine_test.csv"  # N = 124 and N = 54
FEATURES = (
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280/od315_of_diluted_wines",
    "proline",
)
D = len(FEATURES)
CLASSES = (0, 1, 2)


def load(name):
    """X (n x 13) and the integer class labels y from one provided file."""
    path = DATA_DIR / name
    with path.open(encoding="utf-8-sig") as f:
        header = [column.strip().strip('"') for column in f.readline().split(",")]
    if header != [*FEATURES, "class"]:
        raise ValueError(f"{path}: expected the 13 features then class, found {header}")
    table = np.loadtxt(path, delimiter=",", skiprows=1)
    return table[:, :D], table[:, D].astype(int)
