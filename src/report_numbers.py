r"""results/*.json -> results/numbers.tex: one \resdef{key}{value} per number the report quotes.

The report reads them with \res{key}; a key that is not defined prints a bold ??. Numbers follow
the handout's reporting rule: four decimal places, and scientific notation for the residuals
that are checked against 1e-8.

python -m src.report_numbers
"""

import json
from pathlib import Path

from src import p3, p4
from src.common.data import FEATURES
from src.common.fmt import fixed, sci

RESULTS = Path(__file__).resolve().parents[1] / "results"


def tex(name):
    """A column name in typewriter type."""
    return r"\texttt{" + name.replace("_", r"\_") + "}"


def residual(v):
    """A residual in scientific notation, e.g. 3.2e-15."""
    return r"\texttt{" + sci(v) + "}"


def listing(items):
    """a, b and c."""
    if len(items) < 2:
        return items[0] if items else "none"
    return ", ".join(items[:-1]) + " and " + items[-1]


def p1_macros(n):
    loadings = dict(zip(FEATURES, zip(*n["loadings"].values(), strict=True), strict=True))
    m = {
        "p1-orth": residual(n["orthonormality"]),
        "p1-trace": fixed(n["trace"]),
        "p1-trace-diff": residual(n["trace_minus_sum"]),
        "p1-lib-eigenvalues": residual(n["library_diff"]["eigenvalues"]),
        "p1-lib-eigenvectors": residual(n["library_diff"]["eigenvectors"]),
        "p1-loading-rows": " ".join(
            rf"{tex(name)} & ${fixed(u1)}$ & ${fixed(u2)}$ \\"
            for name, (u1, u2) in loadings.items()
        ),
    }
    for column, pc in enumerate(("pc1", "pc2")):
        m[f"p1-top-{pc}"] = listing(
            [rf"{tex(name)} (${fixed(loadings[name][column])}$)" for name in n["top"][pc]]
        )
    return m


def p2_macros(n):
    rows = zip(n["r"], n["R"], n["J_train"], n["J_test"], strict=True)
    return {
        "p2-n-train": str(n["n_train"]),
        "p2-n-test": str(n["n_test"]),
        "p2-k-elbow": str(n["k_elbow"]),
        "p2-R-elbow": fixed(n["R_elbow"]),
        "p2-tail-elbow": fixed(1 - n["R_elbow"]),
        "p2-k80": str(n["k_tau"]["0.80"]),
        "p2-k90": str(n["k_tau"]["0.90"]),
        "p2-k95": str(n["k_tau"]["0.95"]),
        "p2-kaiser": str(n["kaiser"]),
        "p2-identity": residual(n["identity_residual"]),
        "p2-rows": " ".join(
            rf"{k} & ${'$ & $'.join(fixed(v) for v in row)}$ \\" for k, row in enumerate(rows, 1)
        ),
    }


def p3_macros(n):
    v = n["variants"]
    m = {
        "p3-n-test": str(n["n_test"]),
        "p3-rows": " ".join(
            rf"{name} ({p3.VARIANTS[name][0]}) & ${fixed(d['r1'])}$ & ${fixed(d['r2'])}$ & "
            rf"${fixed(d['r1_plus_r2'])}$ & {tex(d['pc1_feature'])} & ${fixed(d['accuracy'])}$ \\"
            for name, d in v.items()
        ),
        "p3-raw-list": listing(
            [rf"{tex(d['feature'])} (${fixed(d['variance'])}$)" for d in n["raw_variance"]]
        ),
        "p3-raw-share": fixed(n["raw_variance_share"]),
        "p3-V1-pc1": tex(v["V1"]["pc1_feature"]),
        "p3-V1-pc2": tex(v["V1"]["pc2_feature"]),
        "p3-V2-lead": str(v["V2"]["correct"] - v["V3"]["correct"]),
    }
    for name, d in v.items():
        m[f"p3-{name}-acc"] = fixed(d["accuracy"])
    for name in ("V1", "V3"):
        m[f"p3-{name}-r1"] = fixed(v[name]["r1"])
        m[f"p3-{name}-r12"] = fixed(v[name]["r1_plus_r2"])
    return m


def p4_macros(n):
    acc, correct, base = n["accuracy"], n["correct"], n["baseline"]
    at = p4.K_VALUES.index
    m = {
        "p4-n-test": str(n["n_test"]),
        "p4-baseline": fixed(base["accuracy"]),
        "p4-baseline-correct": str(base["correct"]),
        "p4-threshold": fixed((base["correct"] - 1) / n["n_test"]),
        "p4-one-sample": fixed(1 / n["n_test"]),
        "p4-rows": " ".join(
            rf"{k} & ${'$ & $'.join(fixed(acc[method][i]) for method in p4.METHODS)}$ \\"
            for i, k in enumerate(p4.K_VALUES)
        ),
        "p4-f-rows": " ".join(
            rf"{rank} & {tex(name)} & ${fixed(n['f_scores'][FEATURES.index(name)])}$ \\"
            for rank, name in enumerate(n["f_ranking"], 1)
        ),
        "p4-chosen-rows": r" \addlinespace ".join(
            rf"{k} & {', '.join(tex(name) for name in sets['filter'])} & "
            rf"{', '.join(tex(name) for name in sets['wrapper'])} & {len(sets['shared'])} \\"
            for k, sets in n["chosen"].items()
        ),
        "p4-first-pc-acc": fixed(acc["pca"][at(1)]),
        "p4-first-feature": tex(n["f_ranking"][0]),
        "p4-first-feature-acc": fixed(acc["filter"][at(1)]),
        "p4-sfs-cv-last": fixed(n["sfs_cv_accuracy"][-1]),
        "p4-gap-filter-3": str(correct["pca"][at(3)] - correct["filter"][at(3)]),
        "p4-gap-wrapper-3": str(correct["pca"][at(3)] - correct["wrapper"][at(3)]),
    }
    for method, k in n["within_one"].items():
        m[f"p4-within-{method}"] = f"$k = {k}$" if k else "none of the tabulated $k$"
    for k, sets in n["chosen"].items():
        m[f"p4-shared-{k}"] = listing([tex(name) for name in sets["shared"]])
    return m


BUILDERS = {"p1": p1_macros, "p2": p2_macros, "p3": p3_macros, "p4": p4_macros}


def macros(results_dir=RESULTS):
    out = {}
    for name, build in BUILDERS.items():
        path = Path(results_dir) / f"{name}.json"
        if path.exists():
            out.update(build(json.loads(path.read_text())))
    return out


def write(results_dir=RESULTS):
    lines = [
        "% Generated by python -m src.report_numbers from results/*.json; do not edit.",
        *(rf"\resdef{{{key}}}{{{value}}}" for key, value in macros(results_dir).items()),
    ]
    (Path(results_dir) / "numbers.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    write()
