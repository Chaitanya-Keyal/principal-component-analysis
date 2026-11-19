# P2: variance captured vs. number of components

## In short

How many components are enough? Two curves answer it. The cumulative variance R(k) says what
share of the spread the first k components keep. The reconstruction error J(k) says how far the
wines are from their k-component approximation. They are the same information: on the training
set J(k) is exactly the sum of the eigenvalues we threw away. Three rules then pick a k: the
elbow rule and the Kaiser rule both say 3 here, while keeping 80%, 90% or 95% of the variance
needs 5, 7 or 10. The test-set error also falls every time a component is added, and there is a
one-line proof.

## The idea in plain words

- **Explained variance ratio r_j**: the share of the total spread carried by component j. The
  first is the biggest and they shrink from there.
- **Scree plot**: a bar chart of r_j. It drops steeply and then flattens into a tail. The
  "elbow" is where the steep part ends.
- **Cumulative ratio R(k)**: add up the first k bars. It climbs to 1 at k = 13.
- **Projection and reconstruction**: squash a wine from 13 numbers to k scores, then expand it
  back to 13 numbers. Something is lost on the way; the squared distance between the wine and
  its rebuilt version is the reconstruction error.

## The three rules for choosing k

| Rule | Definition | Here |
|---|---|---|
| Elbow | the last component with r_k ≥ 0.10 | 3 |
| Kaiser | components with λ_j above the average eigenvalue, tr(S)/D | 3 |
| Threshold | the smallest k with R(k) ≥ 0.80, 0.90, 0.95 | 5, 7, 10 |

- The handout defines the elbow by a number (10%) so everyone gets the same answer; a real elbow
  is judged by eye.
- For standardised data tr(S)/D = 13/13 = 1, so Kaiser means "keep components with λ > 1". A
  component with λ < 1 carries less variance than one original standardised feature, so it is not
  pulling its weight.
- R(k_elbow) = R(3) = 0.6595: the elbow keeps about two thirds of the variance and throws away
  0.3405. It is the "few strong components" choice. The threshold rules are the "keep most of
  the information" choice, and they need more components.

## Why J_train(k) is the sum of the discarded eigenvalues

1. The 13 eigenvectors are perpendicular unit vectors, so any point is the sum of its pieces
   along them: z = Σ_j (u_jᵀz) u_j.
2. Keeping k components keeps the first k pieces. What is lost is the rest:
   z − ẑ = Σ_{j>k} (u_jᵀz) u_j.
3. Because the u_j are perpendicular, the squared length of that sum is the sum of the squares:
   ‖z − ẑ‖² = Σ_{j>k} (u_jᵀz)².
4. Average (u_jᵀz)² over the training set with the divisor N − 1: that is the variance along
   u_j, which is λ_j.
5. So J_train(k) = λ_{k+1} + ... + λ_13, and dividing by tr(S) gives 1 − R(k).

In words: **the variance you did not keep is exactly the error you make**.

We check it numerically: J_train is computed from the actual reconstructions, and the largest
gap from the eigenvalue sum over k = 1..13 is 3.6e-15.

## Why J_test never goes up

Step 3 above holds for *any* point, not just training points. So

J_test(k) − J_test(k+1) = (1 / 53) Σ_n (u_{k+1}ᵀ z_n)² ≥ 0.

Adding a component removes one sum of squares from the error, and a sum of squares cannot be
negative. The error can stay level only if every test point has a zero score on the new
component, which does not happen with real data. At k = 13 nothing is discarded, so the error is
0 for both sets.

Step 4 does *not* carry over to the test set: the test points were not used to compute S, so
their average squared score along u_j is not λ_j. That is why J_test is not equal to the
eigenvalue sum, and why it sits a little above J_train here (the axes were fitted to the
training points).

## The code

`src/p2.py`:

```python
Z, Z_test = z_score(X, X), z_score(X, X_test)  # test uses the training mean and std
lam, U = eigendecomposition(covariance(Z))


def reconstruction_error(Z, U_k):
    Z_hat = Z @ U_k @ U_k.T  # project, then reconstruct
    return ((Z - Z_hat) ** 2).sum() / (len(Z) - 1)
```

- `k_elbow = KS[r >= ELBOW].max()`, `k_tau = KS[cumulative >= tau].min()`,
  `kaiser = (lam > trace(S) / D).sum()`. Kaiser uses a strict `>`, the elbow a `>=`, as the
  handout writes them.
- The divisor is n − 1 for each set: 123 for training, 53 for test.

## Our numbers

| k | r_k | R(k) | J_train(k) | J_test(k) |
|---|---|---|---|---|
| 1 | 0.3573 | 0.3573 | 8.3550 | 8.5455 |
| 2 | 0.1921 | 0.5494 | 5.8578 | 6.1120 |
| 3 | 0.1101 | 0.6595 | 4.4270 | 4.5980 |
| 4 | 0.0725 | 0.7320 | 3.4844 | 3.8979 |
| 5 | 0.0697 | 0.8017 | 2.5779 | 3.0326 |
| 6 | 0.0534 | 0.8551 | 1.8835 | 2.4541 |
| 7 | 0.0456 | 0.9007 | 1.2913 | 1.8367 |
| 8 | 0.0242 | 0.9248 | 0.9773 | 1.3861 |
| 9 | 0.0204 | 0.9452 | 0.7120 | 1.0353 |
| 10 | 0.0198 | 0.9650 | 0.4550 | 0.7532 |
| 11 | 0.0169 | 0.9819 | 0.2359 | 0.4497 |
| 12 | 0.0109 | 0.9927 | 0.0947 | 0.1726 |
| 13 | 0.0073 | 1.0000 | 0.0000 | 0.0000 |

Check a row by hand: J_train(1) = 13 − 4.6450 = 8.3550, and 8.3550 / 13 = 0.6427 = 1 − 0.3573.

## Demo

1. `uv run python -m src.run p2 --out /tmp/demo`, then open `p2_variance.pdf` (scree plot with
   the dashed elbow line at 3; cumulative curve crossing 0.80, 0.90 and 0.95 at 5, 7 and 10) and
   `p2_reconstruction.pdf` (two falling curves, test above train, both 0 at k = 13).
2. In `results/p2.json`: `k_elbow`, `k_tau`, `kaiser`, `identity_residual`.

## Likely questions

- *What is the difference between r_j and R(k)?* r_j is one component's share; R(k) is the
  running total of the first k.
- *Why does the Kaiser rule use 1?* It is the average eigenvalue for standardised data: a
  component below 1 explains less than a single original feature.
- *Which k would you actually choose?* It depends on the goal. For a compact picture, 3 (elbow
  and Kaiser agree). To keep 80% of the variance, 5. For the classifier in P4, 3 components
  already match the baseline.
- *Why is the training error exactly the eigenvalue sum but the test error is not?* S and the
  eigenvalues were computed from the training points, so λ_j is the training variance along u_j.
  The test points have their own variance along u_j.
- *Can the test error go up when a component is added?* No. The drop is a sum of squares.
- *Why is J_test above J_train?* The directions were chosen to capture the training variance.
  The test points are also centred and scaled with training statistics, so they are not exactly
  mean 0 and variance 1.
- *What would J(0) be?* Reconstructing with no components gives ẑ = 0, so J_train(0) = tr(S) =
  13.
- *Why is the reconstruction U_k U_kᵀ z and not an inverse?* The columns of U_k are orthonormal,
  so the transpose undoes the projection as far as it can be undone.

## Likely code changes

- A different elbow threshold or other variance levels: `ELBOW` and `TAUS` in `src/p2.py`.
- Reconstruct in the original units instead of z-scores: multiply `Z_hat` by the training
  standard deviation and add the training mean.
- Plot J as a share of the total: divide `J_train` by `np.trace(S)`; the curve is then 1 − R(k).
