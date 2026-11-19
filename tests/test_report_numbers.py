"""The macros behind results/numbers.tex, and that the report and the generator use exactly the
same keys."""

import json
import re
from pathlib import Path

from src import report_numbers as rn
from tests import PROBLEMS, numbers

ROOT = Path(__file__).resolve().parents[1]


def test_formats():
    assert rn.tex("od280/od315_of_diluted_wines") == r"\texttt{od280/od315\_of\_diluted\_wines}"
    assert rn.residual(3.2e-15) == r"\texttt{3.2e-15}"
    assert rn.listing(["a", "b", "c"]) == "a, b and c"
    assert rn.listing(["a"]) == "a"
    assert rn.listing([]) == "none"


def test_report_uses_every_key_and_only_generated_ones(tmp_path):
    for name in PROBLEMS:
        (tmp_path / f"{name}.json").write_text(json.dumps(numbers(name)))
    report = (ROOT / "report" / "report.tex").read_text()
    used = set(re.findall(r"\\res(?:rows)?\{([^}]+)\}", report))
    assert used == set(rn.macros(tmp_path))


def test_reported_numbers_follow_the_four_decimal_rule(tmp_path):
    for name in PROBLEMS:
        (tmp_path / f"{name}.json").write_text(json.dumps(numbers(name)))
    for key, value in rn.macros(tmp_path).items():
        decimals = re.findall(r"\d+\.(\d+)", re.sub(r"\\texttt\{[^}]*\}", "", value))
        assert all(len(d) == 4 for d in decimals), key
