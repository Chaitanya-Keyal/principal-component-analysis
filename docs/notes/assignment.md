# The assignment: what is asked and where it lives

## In short

Four problems on one theme: reduce 13 wine measurements to a few numbers. P1 derives PCA and
builds it from scratch. P2 asks how many components to keep. P3 shows that the answer depends on
how the features are scaled first. P4 compares PCA with picking a subset of the original
features, using one classifier for all three methods. Every deliverable is a table, a figure, a
few numbers or a short paragraph; all of them go in the report, and `python -m src.run` produces
every number and figure from the two provided CSVs. Due 23:55 on 23 November 2026.

## Rules that apply everywhere (C1 to C6)

- **C1, train-only fitting.** Every statistic (mean, standard deviation, minimum, maximum,
  eigenvectors, feature scores) comes from the 124 training wines and is applied unchanged to
  the 54 test wines. The test set is never used to fit or choose anything.
- **C2, z-score.** z = (x − mean) / standard deviation, with the *sample* standard deviation
  (divisor N − 1). In NumPy that is `std(ddof=1)`.
- **C3, covariance.** S = X̃ᵀX̃ / (N − 1) on mean-centred data, the same as `np.cov`.
- **C4, eigendecomposition.** A symmetric solver (`np.linalg.eigh`), eigenvalues sorted largest
  first, unit eigenvectors as the columns of U. **Sign rule:** flip each eigenvector so that its
  entry of largest absolute value is positive. It applies to every eigenvector reported or
  compared, including scikit-learn's.
- **C5, explained variance ratio.** r_j = λ_j / Σλ, and R(k) = r_1 + ... + r_k.
- **C6, reporting.** Numbers to 4 decimal places. Residuals that are compared with 10⁻⁸ are
  written in scientific notation (3.2e-15), because 4 decimals would print them as 0.0000. Two
  quantities "agree" when their largest absolute difference is below 10⁻⁸. Every plot has a
  title, labelled axes, and a legend when it shows more than one series.

## The data

| File | Rows | Used for |
|---|---|---|
| `wine_train.csv` | 124 | fitting everything |
| `wine_test.csv` | 54 | transformed with the training statistics, then scored |

13 numeric feature columns in this order, numbered 0 to 12: `alcohol`, `malic_acid`, `ash`,
`alcalinity_of_ash`, `magnesium`, `total_phenols`, `flavanoids`, `nonflavanoid_phenols`,
`proanthocyanins`, `color_intensity`, `hue`, `od280/od315_of_diluted_wines`, `proline`. Then an
integer `class` in {0, 1, 2}: the grape variety the wine was made from.

## Deliverables and where they are

Figures are `figures/<name>.pdf`; numbers are keys in `results/<p>.json`. In the report, problem
Pn is section n.

| Item | Code | Figure / numbers |
|---|---|---|
| P1 S and its eigendecomposition from scratch | `common/pca.py`: `covariance`, `eigendecomposition` | |
| P1 printed eigenvalues `lambda_j = x.xxxx` | `p1.compute` (`stdout`) | `results/p1.txt`, `eigenvalues` |
| P1 ‖UᵀU − I‖_F, tr(S), \|tr(S) − Σλ\| | `p1.compute` | `orthonormality`, `trace`, `trace_minus_sum` |
| P1 comparison with scikit-learn | `p1.compute` (`library`) | `library_diff` |
| P1 loadings of u₁, u₂ and the top three of each | `p1.top_features` | `loadings`, `top` |
| P2 scree plot with the elbow, cumulative plot | `p2.variance_figure` | `p2_variance` |
| P2 k_elbow, k₀.₈₀, k₀.₉₀, k₀.₉₅, Kaiser count | `p2.compute` | `k_elbow`, `k_tau`, `kaiser` |
| P2 table of r_k, R(k), J_train, J_test | `p2.reconstruction_error` | `r`, `R`, `J_train`, `J_test` |
| P2 identity J_train(k) = Σ_{j>k} λ_j | `p2.compute` | `identity_residual` |
| P2 plot of J_train and J_test | `p2.reconstruction_figure` | `p2_reconstruction` |
| P2 paragraph | | report §2 Discussion |
| P3 table per variant | `p3.compute`, `p3.centred`, `p3.min_max` | `variants` |
| P3 three largest raw variances (V1) | `p3.compute` | `raw_variance` |
| P3 three-panel projection figure | `p3.projection_figure` | `p3_projections` |
| P3 paragraph | | report §3 Discussion |
| P4 accuracy table, k = 1, 2, 3, 5, 8 | `p4.compute`, `p4.test_correct` | `accuracy`, `correct` |
| P4 baseline and accuracy plot | `p4.accuracy_figure` | `baseline`, `p4_accuracy` |
| P4 F-score ranking | `p4.f_scores` | `f_scores`, `f_ranking` |
| P4 features chosen at k = 3 and 8, shared | `p4.forward_selection` | `chosen`, `sfs_order` |
| P4 paragraph | | report §4 Discussion, `within_one` |

## What is where

- `data/`: the two CSVs, unchanged.
- `uv run python -m src.run` writes `results/<p>.json`, `results/p1.txt` (the printed
  eigenvalues), `results/numbers.tex` (the same numbers as LaTeX macros) and
  `figures/<name>.pdf`.
- `report/report.tex` reads every number through `\res{key}` and every figure from `figures/`;
  `cd report && latexmk -pdf report.tex` builds `report/report.pdf`.
- `notebooks/demo.ipynb` calls the same `compute()` and `figures()`; it is committed without
  outputs and run live.

## When the CSVs arrive

1. Copy the two files into `data/` unchanged and commit them. If the loader stops with
   "expected the 13 features then class", the file layout differs from the handout's; fix
   `src/common/data.py`, not the file.
2. `uv run python -m src.run`, then `uv run pytest`. The tests in `tests/test_discussion.py`
   state what the report's three discussion paragraphs say in words. Each one that fails names a
   sentence to rewrite for the real numbers; update the test with the sentence.
3. The borderline values to look at first: r₃ against the 0.10 elbow threshold, and R(5) and
   R(7) against 0.80 and 0.90. On the provisional split they clear their thresholds by 0.0101,
   0.0017 and 0.0007, so k_elbow, k₀.₈₀ and k₀.₉₀ can each move by one.
4. Rebuild the report (`cd report && latexmk -pdf report.tex`), read the three discussion
   paragraphs against the new tables and figures, and check
   `pdftotext report/report.pdf - | grep -c '??'` prints 0.
5. Refresh the numbers in these notes and the test count in `README.txt`.
6. Commit `results/`, `figures/` and `report/report.pdf`, then build the zip with the command in
   the README's Submission section.

## Still open

- The official `wine_train.csv` and `wine_test.csv`.
- The submission format and channel. The handout gives none; assignment 1 used
  `<StudentID>_A1.zip`, so the README's packaging command builds `2023B4A70727H_A2.zip`.
