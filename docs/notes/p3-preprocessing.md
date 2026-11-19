# P3: effect of preprocessing, and the two-dimensional projection

## In short

PCA looks for the directions with the most variance, and variance depends on units. If one
feature has much bigger numbers than the rest, PCA simply reports that feature. On the raw
(centred-only) wine data `proline` holds 99.7% of all the variance, so PC1 is the proline axis
and two components "explain" 99.98% of the variance, yet a 5-NN classifier on that 2-D picture
gets only 0.7222. After min-max scaling or z-scoring, two components explain far less variance
(0.6113 and 0.5494) but the three classes form three separate clouds and the accuracy is
0.9815 and 0.9444. **High variance captured does not mean useful structure.**

## The three variants

Each is fitted on the training set and applied unchanged to the test set (C1).

| Variant | What it does | Code |
|---|---|---|
| V1 centred only | subtract the training mean, keep the original units | `p3.centred` |
| V2 min-max | (x − min) / (max − min) with the training min and max, then subtract the training mean of the scaled feature | `p3.min_max` |
| V3 z-score | subtract the training mean, divide by the training standard deviation | `common/pca.py`: `z_score` |

For each: S, eigendecomposition (the same code as P1), the first two components, the 2-D
scores c = U₂ᵀx′ for training and test points, and a 5-nearest-neighbour classifier trained on
the training scores and scored on the test scores.

## Why one large-scale feature takes over

Multiply one feature by a constant a. Its variance is multiplied by a², its covariances by a.
The total variance tr(S) is then dominated by that one entry. And the largest eigenvalue can
never be smaller than any diagonal entry of S (λ₁ ≥ S_jj, because λ₁ is the largest value of
uᵀS u and u = e_j gives S_jj). So

r₁ ≥ S_jj / tr(S), which goes to 1 as a grows.

On our data, without any artificial scaling:

| Feature | Raw sample variance |
|---|---|
| `proline` | 94906.7109 |
| `magnesium` | 232.0715 |
| `alcalinity_of_ash` | 11.4713 |

`proline` alone is 0.9973 of the total, so r₁ ≥ 0.9973; the actual value is 0.9978. PC2 is
`magnesium`, the runner-up. Nothing about the wines makes proline 400 times more important than
magnesium: it is just measured in bigger numbers.

## Why the z-score fixes it

After z-scoring every feature has variance 1, so none can dominate. The entry S_ji becomes
Cov(x_j, x_i) / (s_j s_i), which is the correlation of the two features. So **PCA on z-scored
data is PCA of the correlation matrix**, and correlations do not change when the units change.
PC1 is then the direction along which correlated features vary together (the phenolic group led
by `flavanoids`), not the units of one feature.

## Our numbers

| Variant | r₁ | r₂ | r₁ + r₂ | Largest loading on PC1 | 5-NN test accuracy |
|---|---|---|---|---|---|
| V1 centred only | 0.9978 | 0.0021 | 0.9998 | `proline` | 0.7222 (39 of 54) |
| V2 min-max | 0.4271 | 0.1841 | 0.6113 | `flavanoids` | 0.9815 (53 of 54) |
| V3 z-score | 0.3573 | 0.1921 | 0.5494 | `flavanoids` | 0.9444 (51 of 54) |

Reading the figure (`p3_projections`): filled circles are training wines, crosses are test wines,
colour is the class.

- **V1:** the horizontal axis is essentially proline. Class 0 sits to the right (high proline)
  but classes 1 and 2 lie on top of each other, so nearest neighbours confuse them: 11 of V1's
  15 test errors are between those two classes.
- **V2 and V3:** three separate clouds in a rough arch. Nearly every test cross falls inside
  the cloud of its own colour (1 test error for V2, 3 for V3).

## The three questions in the paragraph

1. **Why is r₁ so different in V1 and V3?** In V1 the total variance is the sum of the raw
   variances and proline is 0.9973 of it. In V3 every feature has variance 1 and S is the
   correlation matrix.
2. **V1 captures more variance; is it better for classification?** No, it is the worst (0.7222).
   PCA ranks directions by spread only. A direction can have a huge spread because of its units
   and still say little about the classes, while the features that do separate them have small
   raw variances.
3. **Which variant before PCA?** The z-score. The 13 features are in unrelated units and ranges,
   and standardising makes the result independent of them. Min-max also removes the unit
   problem, and it scores 2 test wines higher here, but it sets each feature's scale from just
   its two most extreme training values, so one outlier changes everything, and 2 wines out of
   54 is too small a gap to choose on.

## Demo

1. `uv run python -m src.run p3 --out /tmp/demo`, then open `p3_projections.pdf`.
2. In `results/p3.json`: `variants` (the table), `raw_variance`, `raw_variance_share`.

## Likely questions

- *Does PCA depend on the units of the features?* Yes, unless the data is standardised first.
  Dividing proline by 1000, which is only a change of units, hands V1's first component to
  `magnesium` (r₁ = 0.9176) and leaves V3 exactly as it was.
- *What is the difference between PCA on the covariance and on the correlation matrix?* The
  correlation matrix is the covariance matrix of z-scored data. Use it when features have
  different units.
- *When would you not standardise?* When all features are in the same units and their different
  spreads are meaningful (pixel intensities, the same sensor at different places).
- *Why subtract the mean after min-max scaling?* PCA is defined on centred data. The same shift
  is applied to the test points so that they are projected consistently.
- *Can a test value fall outside [0, 1] under V2?* Yes, if it is beyond the training minimum or
  maximum. We leave it as it is: clipping would change the data, and the handout does not ask
  for it.
- *Why did V2 beat V3 here?* By 2 test wines. Min-max gives features with a small relative
  spread a different weight from the z-score, which moves a few borderline points. It is within
  the noise of 54 samples (see P4).
- *Why k = 5 neighbours and Euclidean distance?* They are the handout's settings; all other
  settings are scikit-learn's defaults.
- *Is the 2-D picture enough to judge the classes?* For V3 it shows 0.5494 of the variance. The
  classes look separable in it, but the other 45% is not shown.

## Likely code changes

- Another number of neighbours: `KNeighborsClassifier(n_neighbors=5)` in `p3.compute`.
- A fourth variant, for example dividing by the range without shifting: add a function with the
  signature `f(X_train, X)` and an entry in `VARIANTS`; the figure needs a fourth panel
  (`plt.subplots(1, 4, ...)`).
- Show what scaling one feature does: multiply the `hue` column of `X` and `X_test` by 10,000
  at the top of `compute`. In V1, PC1 switches from `proline` to `hue` with r₁ = 0.9832; V3 does
  not change at all. A smaller factor is not enough: hue's raw variance is only 0.0529, so it
  has to be scaled a long way before it passes proline's 94906.7.
