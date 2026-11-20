# P1: PCA from first principles

## In short

Ask: along which direction are the wines most spread out? Writing that as a maximisation problem
and solving it gives S u = λ u, so the answer is the eigenvector of the covariance matrix with
the largest eigenvalue, and the variance along it is that eigenvalue. Repeating the question at
right angles to the directions already found gives the other components. We build S, take its
eigendecomposition in our own code, print the 13 eigenvalues, and confirm that scikit-learn's
PCA gives the same eigenvalues and eigenvectors to about 10⁻¹⁵. The class labels are not used.

## The idea in plain words

- Each wine is a point in 13-dimensional space. Pick a direction u (a unit vector) and drop every
  point onto it: that gives one number per wine, uᵀx.
- Some directions separate the wines a lot (large variance of those numbers), others hardly at
  all. PCA picks the direction with the largest variance first.
- The length of u is fixed at 1. Otherwise the variance could be made as large as we like just by
  making u longer.

## The derivation, step by step

**1. Variance along u is uᵀS u.** The projections have mean uᵀx̄. Each squared deviation is
(uᵀx_n − uᵀx̄)² = uᵀ(x_n − x̄)(x_n − x̄)ᵀu. Add them up, divide by N − 1, and the middle part is
the definition of S. So the variance is uᵀS u.

**2. Maximise it with the length fixed.** Maximise uᵀS u subject to uᵀu = 1. A Lagrange
multiplier λ turns the constraint into part of the objective:
L = uᵀS u + λ(1 − uᵀu). Setting the gradient to zero: 2S u − 2λu = 0, so **S u = λ u**. Every
candidate is an eigenvector of S.

**3. Which eigenvector?** Multiply S u = λ u on the left by uᵀ: uᵀS u = λ. The variance along an
eigenvector is its eigenvalue, so the best one has the largest eigenvalue λ₁. That is u₁.

**4. The second component.** Now also require uᵀu₁ = 0 (at right angles to the first), with a
second multiplier η. The gradient condition becomes 2S u − 2λu + ηu₁ = 0. Multiply on the left by
u₁ᵀ: the first term is 2λ₁u₁ᵀu = 0, the second is 0, the third is η. So η = 0 and we are back to
S u = λ u. The best direction left is the eigenvector of the second-largest eigenvalue. The same
argument gives u₃ to u₁₃.

**5. The eigenvalues add up to the total variance.** tr(S) = Σλ_j. The trace is also the sum of
the 13 feature variances, which are all 1 after standardising. So the eigenvalues must sum to
13, a check that costs nothing.

## The code

`src/common/pca.py` holds the from-scratch part; `src/p1.py` calls it and does the checks.

```python
Z = z_score(X, X)  # standardise with the training mean and std (ddof=1)
S = covariance(Z)  # Zc.T @ Zc / (N - 1)
lam, U = eigendecomposition(S)  # eigh, largest first, sign rule
```

- `eigendecomposition`: `np.linalg.eigh(S)` returns ascending eigenvalues, so both outputs are
  reversed, then `fix_signs` flips each column so its largest-magnitude entry is positive.
- The printed lines `lambda_1 = 4.6450` ... are built in `p1.compute` and printed by
  `src/run.py`, which also saves them to `results/p1.txt` for the report.
- **The check against scikit-learn:** `PCA(n_components=13, svd_solver="full").fit(Z)`.
  `explained_variance_` holds its eigenvalues and the *rows* of `components_` its eigenvectors,
  so we transpose, apply the sign rule, and take the largest absolute difference.

## Our numbers

Eigenvalues: 4.6450, 2.4972, 1.4309, 0.9426, 0.9065, 0.6944, 0.5922, 0.3140, 0.2653, 0.2570,
0.2191, 0.1413, 0.0947. They sum to 13.

| Check | Value | Meaning |
|---|---|---|
| ‖UᵀU − I‖_F | 4.0e-15 | the eigenvectors are perpendicular and of length 1 |
| tr(S) | 13.0000 | total variance of 13 standardised features |
| \|tr(S) − Σλ\| | 1.8e-15 | the eigenvalues add up to the total variance |
| eigenvalues vs. scikit-learn | 3.6e-15 | same |
| eigenvectors vs. scikit-learn | 9.1e-15 | same |

All are far below the 10⁻⁸ threshold; what is left is floating-point rounding (a computer keeps
about 16 digits). The digits of these residuals are rounding noise, so they can come out
differently with another NumPy or scikit-learn version; only their size matters.

Largest loadings:

- **PC1:** `flavanoids` (0.4260), `total_phenols` (0.3899), `od280/od315_of_diluted_wines`
  (0.3778). `flavanoids` and `total_phenols` measure phenolic content, and
  `od280/od315_of_diluted_wines` rises and falls with them. PC1 is roughly "how phenolic the
  wine is".
- **PC2:** `color_intensity` (0.5263), `alcohol` (0.4941), `proline` (0.3703). PC2 is roughly
  "how strong and dark the wine is".

## Demo

1. `uv run python -m src.run p1 --out /tmp/demo` prints the 13 eigenvalue lines.
2. In `/tmp/demo/results/p1.json`: `orthonormality`, `trace`, `trace_minus_sum`, `library_diff`
   and the two loading vectors.
3. In the notebook, the P1 cell prints the same lines, the checks and the loadings table.

## Likely questions

- *Why does maximising variance lead to eigenvectors?* The gradient condition of the constrained
  problem is S u = λ u, the definition of an eigenvector.
- *Why the constraint uᵀu = 1?* Without it the variance grows without bound as u gets longer.
  Only the direction matters.
- *Why is the variance equal to the eigenvalue?* uᵀS u = uᵀ(λu) = λ uᵀu = λ.
- *Why do the eigenvalues sum to 13?* Their sum is tr(S), the total variance, and each of the 13
  standardised features has variance 1.
- *How does scikit-learn compute PCA?* Through the SVD of the centred data matrix, not through
  S. The squared singular values divided by N − 1 are the eigenvalues, and the right singular
  vectors are the eigenvectors. Same answer, different route, which is what makes it a real
  check.
- *Why apply the sign rule before comparing?* u and −u are both valid, and the two methods need
  not pick the same one. Without the rule the difference could come out near 2 × a loading for a
  column that is in fact the same direction.
- *What does a negative loading mean?* That feature decreases as the score on that component
  increases. On PC1, `malic_acid` (−0.2524) and `nonflavanoid_phenols` (−0.2868) move against the
  phenolic group.
- *Are the eigenvalues the same as for the raw data?* No. These are for standardised data, so
  they are the eigenvalues of the correlation matrix (see P3).

## Likely code changes

- Name the top 5 loadings instead of 3: `TOP` in `src/p1.py`.
- Use the divisor N instead of N − 1: changing `len(X) - 1` to `len(X)` in `covariance`
  (`src/common/pca.py`) multiplies every eigenvalue by 123/124 and leaves U and the r_j alone.
  Changing `ddof=1` to `ddof=0` in `z_score` as well brings the eigenvalues back: S is then the
  correlation matrix again, which does not depend on the divisor.
- Skip the sign rule: return `U[:, ::-1]` without `fix_signs` in `eigendecomposition`. The
  columns whose largest entry came out negative keep that sign (10 of the 13 here), so their
  loadings change sign and the comparison with scikit-learn no longer agrees for them. The
  eigenvalues and every variance stay the same.
