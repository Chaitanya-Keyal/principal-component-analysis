# PCA basics: the ideas every problem leans on

## In short

PCA replaces the 13 original measurements with new axes, called principal components, sorted by
how much of the data's spread they carry. The recipe is three steps: put every feature on the
same scale (standardise), measure how the features move together (the covariance matrix S), and
find the special directions of S (its eigenvectors). Each eigenvector is a component; its
eigenvalue is the variance of the data along it. Keep the first few components and you keep most
of the information in far fewer numbers.

## Variance, covariance, correlation

- **Variance** of one feature: the average squared distance from its mean. Large variance means
  the values are spread out. Its square root is the **standard deviation**.
- **Covariance** of two features: whether they move together. Positive when one is high the other
  tends to be high, negative when they move in opposite directions, near zero when unrelated.
- **Correlation**: covariance divided by both standard deviations. It is always between −1 and 1
  and has no units.
- **Why N − 1 and not N.** We have a sample of 124 wines, not all wines. Dividing by N − 1 gives
  the unbiased sample variance; it is what `np.cov` and scikit-learn use, and the handout makes it
  mandatory (C2, C3). Using N in S alone would multiply every eigenvalue by 123/124 and change
  no eigenvector and no ratio r_j.

## Standardising (z-score)

z = (x − mean) / standard deviation, feature by feature.

- After it, every feature has mean 0 and standard deviation 1, and no units.
- It matters because the features are on very different scales: `proline` is in the hundreds to
  thousands, `hue` is around 1. Without it, PCA would just report the feature with the biggest
  numbers (that is P3).
- The mean and standard deviation come from the **training set only**, and the same two numbers
  are then used on the test set (C1). The test set stands in for wines we have not seen, so it
  must not influence anything we fit.

In the code: `z_score(X_train, X)` in `src/common/pca.py`. The first argument always supplies
the statistics; the second is what gets transformed.

## The covariance matrix S

A 13 × 13 table. Entry (i, j) is the covariance of features i and j; the diagonal holds each
feature's variance.

- S = X̃ᵀX̃ / (N − 1), where X̃ is the data with each column's mean subtracted.
- S is **symmetric** (covariance of i with j equals that of j with i).
- For standardised data the diagonal is all ones and S is the **correlation matrix**.
- The **trace** tr(S) is the sum of the diagonal: the total variance. For standardised data it is
  exactly 13.

In the code: `covariance(X)`, two lines.

## Eigenvectors and eigenvalues

An eigenvector of S is a direction u that S only stretches: S u = λ u. The stretch factor λ is
its eigenvalue.

For a covariance matrix they have a concrete meaning:

- Project every wine onto the direction u (compute uᵀz, one number per wine). The variance of
  those numbers is uᵀS u.
- If u is an eigenvector, that variance is exactly λ.
- The eigenvector with the largest λ is the direction along which the wines differ the most.
  That is **PC1**. The next largest, at right angles to it, is **PC2**, and so on.

Because S is symmetric, its eigenvalues are real and non-negative (a variance cannot be
negative) and its eigenvectors can be chosen perpendicular to each other with length 1. Stacked
as columns they form U, and UᵀU = I.

In the code: `eigendecomposition(S)` calls `np.linalg.eigh`, which is made for symmetric
matrices and returns eigenvalues in *ascending* order; we reverse them and apply the sign rule.

## The sign rule

If u is an eigenvector, so is −u: same direction, same variance. A solver may return either, so
two correct programs can disagree by a sign. The handout fixes it: flip each eigenvector so that
its entry of largest absolute value is positive. `fix_signs(U)` does that, and it is applied to
our U and to scikit-learn's vectors before they are compared.

## Vocabulary for the four problems

| Word | Meaning |
|---|---|
| Principal component (PC) | an eigenvector of S, a new axis |
| Loading | the weight of one original feature in a component (one entry of u) |
| Score | a wine's coordinate on a component, uᵀz |
| Explained variance ratio r_j | λ_j / Σλ, the share of the total variance on component j |
| Cumulative ratio R(k) | r_1 + ... + r_k, the share kept by the first k components |
| Projection | going from 13 numbers to k scores: c = U_kᵀ z |
| Reconstruction | going back to 13 numbers from k scores: ẑ = U_k c |

## Likely questions

- *Why not use all 13 components?* Then nothing is reduced: U is just a rotation and no
  information is dropped. The point is that the first few carry most of the variance.
- *Why `eigh` and not `eig`?* S is symmetric. `eigh` uses that: the eigenvalues are guaranteed
  real, the eigenvectors orthonormal, and the result sorted. `eig` is for general matrices and
  can return tiny imaginary parts and an arbitrary order.
- *Does PCA use the class labels?* No. It only looks at the features. That is why it is called
  unsupervised, and it matters in P4.
- *Are the components correlated with each other?* No. The covariance of the scores is
  UᵀSU = Λ, a diagonal matrix, so different components have zero covariance.
