# P4: PCA versus feature selection for a downstream classifier

## In short

There are two ways to feed a classifier fewer than 13 numbers. **Extraction** builds new
features from all the old ones (PCA). **Selection** keeps some of the original features and
drops the rest. We compare PCA with two selection methods, a filter (rank features by F-score)
and a wrapper (sequential forward selection), at k = 1, 2, 3, 5, 8, always with the same
logistic-regression classifier. The baseline, all 13 features, gets 0.9815 (53 of 54). PCA is
within one test wine of that from k = 3, the filter from k = 5, the wrapper only at k = 8. With
54 test wines one wine is 0.0185 of accuracy, so most of the gaps are too small to rank the
methods.

## The three methods in plain words

- **(A) PCA.** Use the first k principal component scores. Each score is a weighted mix of all
  13 features. PCA never looks at the class labels.
- **(B) Filter, F-score.** Score each feature on its own: how far apart are the three class
  means, compared with the scatter inside each class? Keep the k best. One pass, no classifier
  involved. It uses the labels.
- **(C) Wrapper, sequential forward selection (SFS).** Start with no features. Try adding each
  remaining feature, train the classifier, measure its cross-validated accuracy, and keep the
  one that helps most. Repeat until there are k. It uses the labels and the classifier itself.

## The F-score

F_j = (between-class variance of feature j) / (within-class variance of feature j)

- Numerator: Σ_c N_c (m_jc − m_j)² / (C − 1). Large when the class means are far from the
  overall mean.
- Denominator: Σ_c Σ_{n in c} (z_nj − m_jc)² / (N − C). Large when the values inside a class are
  scattered.
- C = 3 classes, N = 124, so the divisors are 2 and 121.
- A large F means the classes are well separated on that feature alone. It is the one-way ANOVA
  F statistic, the same number as scikit-learn's `f_classif` (tested).
- Its weakness: it judges features one at a time. Two features that say nearly the same thing
  both get a high score, and picking both adds little.

## Sequential forward selection

- Scoring a candidate set: train on 4 of the 5 folds, test on the fifth, five times, and average
  the five accuracies. The folds are `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)`
  on the training set, built once and reused, so every candidate is judged on the same splits.
  "Stratified" keeps the class proportions the same in every fold.
- Ties (accuracy is a count divided by a number of samples, so ties are common): try candidates
  in increasing column index and keep the first best. That is the strict `>` in the code.
- It is **greedy**: a feature, once added, is never removed. So the sets for k = 1, 2, 3, 5, 8
  are the first k features of one run to k = 8.
- **Cost.** With k features held it scores 13 − k candidates. Running to the end would score
  13 + 12 + ... + 1 = 91 subsets; an exhaustive search would need 2¹³ − 1 = 8191. Stopping at
  k = 8 scores 13 + 12 + ... + 6 = 76 subsets, each with 5 fits: 380 model fits.
- The test set is never involved in the selection.

## The code

`src/p4.py`:

```python
FOLDS = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def classifier():
    return LogisticRegression(C=1.0, max_iter=1000)


for j in range(Z.shape[1]):  # increasing column index
    if j in chosen:
        continue
    columns = sorted([*chosen, j])
    score = cross_val_score(classifier(), Z[:, columns], y, cv=FOLDS, scoring="accuracy").mean()
    if score > best_score:  # strictly greater, so the first maximum is kept
        best, best_score = j, score
```

- `f_scores(Z, y)` is the formula above, written with NumPy sums.
- `test_correct(...)` trains on the training set and returns the *number* of correct test
  predictions; accuracies are that count over 54. Comparing counts avoids rounding trouble in
  "within one test sample of the baseline".
- `tests/test_p4.py` checks that our forward selection returns the same sets as scikit-learn's
  `SequentialFeatureSelector` for every k, and that the F-scores equal `f_classif`.
- The baseline is the same classifier on all 13 standardised features.

## Our numbers

| k | (A) PCA | (B) Filter | (C) Wrapper |
|---|---|---|---|
| 1 | 0.8704 | 0.8519 | 0.8519 |
| 2 | 0.9444 | 0.9074 | 0.8889 |
| 3 | 0.9815 | 0.9074 | 0.8704 |
| 5 | 1.0000 | 0.9815 | 0.9074 |
| 8 | 0.9815 | 0.9815 | 0.9630 |

Baseline: 0.9815 (53 of 54). "Within one test sample" means at least 52 correct, 0.9630.

F-score ranking, best first: `flavanoids` (187.6858), `proline` (149.4335),
`od280/od315_of_diluted_wines` (126.6191), `alcohol` (100.3464), `color_intensity` (83.9793),
`hue` (72.2673), `total_phenols` (59.2558), `alcalinity_of_ash` (26.0528), `malic_acid`
(22.5828), `proanthocyanins` (19.9799), `nonflavanoid_phenols` (17.9053), `ash` (11.5760),
`magnesium` (7.1124).

Forward-selection order: `flavanoids`, `alcohol`, `alcalinity_of_ash`, `proanthocyanins`,
`magnesium`, `ash`, `hue`, `proline`. Its cross-validated accuracy climbs 0.7667, 0.9357,
0.9517, 0.9597, 0.9597, 0.9680, 0.9840, 1.0000.

| k | (B) Filter | (C) Wrapper | Shared |
|---|---|---|---|
| 3 | `flavanoids`, `proline`, `od280/od315_of_diluted_wines` | `flavanoids`, `alcohol`, `alcalinity_of_ash` | 1: `flavanoids` |
| 8 | the top 8 of the ranking | the 8 above | 5: `flavanoids`, `proline`, `alcohol`, `hue`, `alcalinity_of_ash` |

## The three questions in the paragraph

1. **Smallest k within one test sample of the baseline:** PCA 3, filter 5, wrapper 8.
2. **PCA never sees the labels; does the plot show it?** Not as a handicap. The PCA curve is on
   or above the other two at every k. One component built without labels (0.8704) does at least as
   well as `flavanoids`, the best single feature by both supervised methods (0.8519). The reason is
   how the features are built: k components draw on all 13 features, k selected features draw
   on k. And on this data the high-variance directions happen to separate the classes (the V3
   picture in P3). Where the labels do show is the wrapper: it is the lowest of the three on
   the test set from k = 2 on, although its cross-validated accuracy on the training set is
   higher (0.9517, 0.9597 and 1.0000 at k = 3, 5 and 8, against 0.8704, 0.9074 and 0.9630 on the
   test set). It fits its choices to the 124 training wines.
3. **How confident can we be with 54 test wines?** Not very. One wine is 1/54 = 0.0185. The
   standard error of an accuracy near 0.95 is √(0.95 × 0.05 / 54) ≈ 0.03, more than one wine.
   Gaps of one or two wines (PCA and the filter from k = 5 on) mean nothing. Only the gaps of
   four to six wines hint at a real difference: PCA over the filter at k = 3 (4), PCA over the
   wrapper at k = 3 (6) and k = 5 (5), and the filter over the wrapper at k = 5 (4). Even those
   come from a single split.

## Demo

1. `uv run python -m src.run p4 --out /tmp/demo`, then open `p4_accuracy.pdf`: three curves and
   the dashed baseline.
2. In `results/p4.json`: `accuracy`, `correct`, `baseline`, `within_one`, `f_ranking`,
   `sfs_order`, `sfs_cv_accuracy`, `chosen`.

## Likely questions

- *Filter, wrapper, extraction: what is the difference?* A filter scores features with a
  statistic, without the classifier. A wrapper searches subsets by training the classifier. An
  extraction method builds new features as combinations of the old ones.
- *Why can PCA beat methods that see the labels?* A component can combine several weak or
  correlated features into one strong one; a selected feature is only ever itself. It works here
  because class differences are a large part of the total variance.
- *When would PCA lose?* When the direction that separates the classes has small variance. P3's
  V1 is exactly that case: PCA on unscaled data keeps proline and loses the classes.
- *Why does the wrapper do worst on the test set?* It picks whatever maximises cross-validated
  accuracy on 124 wines, and with 76 candidates scored some of that maximum is luck. Its training
  score (1.0000) is optimistic; the test set shows the true level.
- *Why did the wrapper's accuracy drop from k = 2 to k = 3?* It optimises the cross-validated
  accuracy, which did go up (0.9357 to 0.9517). Test accuracy is not what it sees; the change is
  one wine.
- *Why is PCA at k = 5 above the baseline?* 54 against 53 correct: one wine. Dropping the
  low-variance components can remove noise, but one wine is not evidence.
- *Why are the filter and the wrapper different after the first feature?* The filter's second
  and third choices (`proline`, `od280/od315_of_diluted_wines`) are strong alone but partly repeat what
  `flavanoids` already says. The wrapper asks what helps *given* what it already has.
- *Why the same folds for every candidate?* So that differences in score come from the features,
  not from a lucky split.
- *Why standardise before logistic regression?* The L2 penalty treats all coefficients alike,
  which is only fair when the features share a scale.
- *Why does accuracy move in steps?* It is a count out of 54, so it changes by 0.0185 at a time.

## Likely code changes

- Other values of k: `K_VALUES` in `src/p4.py` (forward selection runs to the largest).
- Another classifier: the body of `classifier()`.
- Another number of folds or seed: `FOLDS`.
- Use scikit-learn's selector instead of ours: `SequentialFeatureSelector(classifier(),
  n_features_to_select=k, direction="forward", scoring="accuracy", cv=FOLDS).fit(Z, y)`, then
  `get_support()`; `tests/test_p4.py` already does this.
