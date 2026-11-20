# The whole assignment in plain language

Part 1 explains every idea without assuming anything. Part 2 is a script for showing the
notebook, and ten likely questions with short answers. Every number is from our run on the
provided data, so it can be quoted.

---

# Part 1: The concepts

## The setup

- **The data.** 178 wines made from three grape varieties (the classes 0, 1 and 2). For each wine
  a lab measured 13 things: alcohol, malic acid, ash, flavanoids, colour intensity, proline and so
  on.
  - 124 wines are for **training**: everything we compute is learned from these.
  - 54 wines are for **testing**: they are only used to check how well things work on wines we
    did not learn from.
- **The problem.** 13 numbers per wine are too many to look at, and several of them say nearly the
  same thing. We want a few numbers that keep most of the information. That is called
  **dimensionality reduction**.
- **The tool.** PCA (principal component analysis) builds new axes out of the old 13, ordered
  from most informative to least. Keep the first few and drop the rest.

## Five words that come up everywhere

- **Variance.** How spread out a set of numbers is. If all wines had the same alcohol, its
  variance would be 0 and it would tell us nothing about which wine is which.
- **Standardising (z-score).** Subtract a feature's mean and divide by its standard deviation.
  Afterwards every feature has mean 0 and spread 1. It is needed because the features are in
  different units: proline is around 750, hue is around 1.
- **Covariance matrix S.** A 13 × 13 table saying how every pair of features moves together.
- **Eigenvector and eigenvalue.** A special direction of S, and a number attached to it. For a
  covariance matrix the number is the variance of the data along that direction.
- **Principal component.** An eigenvector of S. PC1 is the direction along which the wines differ
  most, PC2 the next one at right angles to it, and so on up to PC13.

One rule runs through all four problems: **fit on the training set only**. Means, standard
deviations, the components and any choice of features come from the 124 training wines and are
then applied to the 54 test wines without change.

---

## P1: build PCA from scratch

**The question.** Along which direction are the wines most spread out?

**The answer, in three steps of maths.**
1. The variance along a direction u works out to uᵀS u.
2. Maximising that, with the length of u fixed at 1, gives the equation S u = λ u. So the best
   direction is an eigenvector of S.
3. The variance along an eigenvector is its eigenvalue λ. So take the eigenvector with the
   largest λ. That is PC1. The second largest gives PC2, and so on.

**What we did.**
- Standardised the 13 training features.
- Built S ourselves (two lines of NumPy) and asked NumPy for its eigenvectors.
- Sorted them largest first and fixed the signs. An eigenvector can point either way (u or −u),
  so the handout's rule is: flip it so that its entry of largest absolute value (the biggest
  one, ignoring signs) is positive.
- Printed the 13 eigenvalues: 4.6450, 2.4972, 1.4309, 0.9426, ... down to 0.0947.

**The checks.**
- The eigenvalues add up to 13. They must: the total variance of 13 standardised features is 13.
- The eigenvectors are perpendicular and of length 1 (error 4.0e-15).
- scikit-learn's PCA gives the same eigenvalues and eigenvectors (differences around 10⁻¹⁵).
  Anything below 10⁻⁸ counts as "agree"; what is left is rounding in the 16th digit.

**What the components mean.**
- PC1 is mostly `flavanoids`, `total_phenols` and `od280/od315_of_diluted_wines`: roughly how
  phenolic the wine is.
- PC2 is mostly `color_intensity`, `alcohol` and `proline`: how strong and dark it is.

---

## P2: how many components to keep

**Two ways to measure what k components keep.**
- **Explained variance.** Component j carries the share r_j = λ_j / 13 of the total spread. PC1
  carries 0.3573, PC2 0.1921, PC3 0.1101. Adding them up gives R(k), the share kept by the first
  k.
- **Reconstruction error J(k).** Squash each wine to k numbers, expand it back to 13, and measure
  how far it landed from the original.

They are the same thing: on the training set, J(k) is exactly the sum of the eigenvalues that
were dropped. We verify that to 3.6e-15.

**Three rules for picking k.**

| Rule | Says | Answer |
|---|---|---|
| Elbow | keep components that each explain at least 10% | 3 |
| Kaiser | keep components with eigenvalue above 1 (the average) | 3 |
| Threshold | keep enough to reach 80%, 90%, 95% of the variance | 5, 7, 10 |

- Three components keep 0.6595 of the variance. The elbow and Kaiser rules pick the few strong
  components; the threshold rules keep more of the information and need more components.

**The test set.** Its error also falls every time a component is added, down to 0 at k = 13.
The proof is one line: adding a component removes a sum of squares from the error, and a sum of
squares cannot be negative. The test curve sits above the training curve: the axes are the ones
that make the *training* error as small as possible, and the test wines get no such guarantee.

---

## P3: scaling changes everything

**The experiment.** Run PCA three times, after three kinds of preprocessing, and each time keep
two components, plot the wines, and see how well a 5-nearest-neighbour classifier does on the
2-D picture.

| Preprocessing | Variance in 2 components | PC1 is mostly | 5-NN accuracy |
|---|---|---|---|
| V1: only subtract the mean | 0.9998 | `proline` | 0.7222 |
| V2: min-max to [0, 1] | 0.6113 | `flavanoids` | 0.9815 |
| V3: z-score | 0.5494 | `flavanoids` | 0.9444 |

**What happened in V1.** Proline's raw variance is 94906.7; the next biggest, magnesium, is
232.1. Proline alone is 0.9973 of the total variance, only because it is measured in big
numbers. PCA looks for variance, so PC1 becomes the proline axis and "explains" 0.9978 of
everything. But proline alone cannot tell classes 1 and 2 apart, so the picture is poor.

**What V3 does.** After z-scoring every feature has variance 1, so none can dominate. PCA then
works on the correlation matrix, which does not depend on units, and the three classes show up
as three separate clouds.

**The lesson.** V1 captures almost all the variance and classifies worst. Variance captured is
not the same as useful information. Always put the features on a common scale before PCA when
their units differ. We would use the z-score; min-max scored 2 test wines higher here, but it
depends on each feature's two most extreme values, and 2 wines out of 54 is noise.

---

## P4: PCA against picking features

**Two families.**
- **Extraction** (PCA): make k new features, each a mix of all 13.
- **Selection**: keep k of the original 13.
  - **Filter (F-score):** score each feature by how well it separates the classes on its own,
    keep the top k.
  - **Wrapper (forward selection):** add features one at a time, each time the one that most
    improves the classifier's cross-validated accuracy.

All three feed the same classifier, a logistic regression. The **baseline** is that classifier
on all 13 features: 0.9815, which is 53 of the 54 test wines.

| k | PCA | Filter | Wrapper |
|---|---|---|---|
| 1 | 0.8704 | 0.8519 | 0.8519 |
| 2 | 0.9444 | 0.9074 | 0.8889 |
| 3 | 0.9815 | 0.9074 | 0.8704 |
| 5 | 1.0000 | 0.9815 | 0.9074 |
| 8 | 0.9815 | 0.9815 | 0.9630 |

**What it shows.**
- PCA matches the baseline with 3 components and the filter with 5 features. The wrapper never
  quite does: at 8 features it is one wine short.
- PCA never sees the class labels, yet it is never below the two methods that do. Each component
  mixes all 13 features, while a selected feature is just one of them.
- The wrapper scores well on the training data (cross-validated accuracy 0.9517, 0.9597 and
  1.0000 at k = 3, 5 and 8) and lowest of the three on the test data (0.8704, 0.9074, 0.9630).
  It tuned its choices to the 124 training wines.
- **How far to trust it.** One test wine is 0.0185 of accuracy. Differences of one or two wines
  are noise. Only the largest gaps are worth mentioning, and the widest is 6 wines, between PCA
  and the wrapper at k = 3.

**Why a greedy search.** Trying every subset of 13 features means 8191 subsets. Forward selection
tries 13, then 12, then 11, ...: 76 subsets to reach k = 8.

---

# Part 2: Showing the notebook

## Before

From the repo folder:

```bash
uv sync --group notebook
uv run --group notebook jupyter lab notebooks/demo.ipynb
```

- Run **Run → Restart Kernel and Run All** once beforehand. It takes about 10 s.
- Open `report/report.pdf` beside it.

## Cell by cell

**Opening (title and setup cell)**
> "PCA itself is from scratch in NumPy: the covariance matrix and the eigendecomposition. Each
> problem is one file in `src/`. This notebook just calls them and shows the figures and
> numbers. The same numbers feed the report automatically."

**P1 cell**
- **The 13 `lambda_j` lines:**
  > "These are the eigenvalues of the covariance matrix of the standardised training data,
  > largest first. Each one is the variance along its component. They add up to 13 because there
  > are 13 features with variance 1 each."
- **The checks:**
  > "UᵀU − I is 4e-15, so the eigenvectors are orthonormal. The trace matches the eigenvalue
  > sum. And scikit-learn's PCA, after the same sign rule, agrees with ours to about 10⁻¹⁵, far
  > below the 10⁻⁸ threshold."
- **The loadings table:**
  > "PC1 is led by flavanoids, total phenols and od280: roughly the phenolic content. PC2 is led
  > by colour intensity, alcohol and proline."

**P2 cell**
- **Scree plot (left):**
  > "Each bar is one component's share of the variance. The dashed line is the elbow at 3: the
  > last component that explains at least 10%."
- **Cumulative plot (right):**
  > "The running total. It crosses 80% at 5 components, 90% at 7 and 95% at 10. Kaiser, which
  > keeps eigenvalues above 1, also says 3."
- **Reconstruction plot:**
  > "Reconstruction error for training and test. On the training set it equals the sum of the
  > discarded eigenvalues; we check that to 3.6e-15. The test error also falls at every step,
  > because adding a component removes a sum of squares. Both hit zero at 13."

**P3 cell**
- **The three panels:**
  > "Same PCA code, three preprocessings. With centring only, proline has a variance of about
  > 95,000 and takes over: two components hold 99.98% of the variance but classes 1 and 2
  > overlap, and 5-NN gets 0.72. After min-max or z-score the variance captured drops to about
  > 55 to 61%, yet the classes separate and accuracy is 0.98 and 0.94."
- **The table:**
  > "So variance captured is not the same as useful structure. PCA on unscaled data just reports
  > the feature with the biggest units."

**P4 cell**
- **The plot:**
  > "Test accuracy against the number of features. Dashed is the baseline with all 13: 0.9815.
  > PCA reaches it at k = 3, the F-score filter at 5, forward selection only gets within one
  > sample at 8."
- **The ranking and the chosen sets:**
  > "Both selection methods start with flavanoids. After that they differ: the filter takes the
  > next best features one by one, the wrapper takes what helps given what it already has. They
  > share 1 of 3 and 5 of 8."
- **The caveat:**
  > "One test sample is 0.0185, so one or two samples of difference cannot rank the methods."

## If asked to change something live

- Edit the constant at the top of the relevant file in `src/` (`TAUS` or `ELBOW` in
  `src/p2.py`, `VARIANTS` in `src/p3.py`, `K_VALUES` in `src/p4.py`), then restart the kernel
  and rerun that cell.
- From the terminal, run `uv run python -m src.run p2 --out /tmp/demo` so the repo's results are
  not overwritten.

## Ten likely questions, with short answers

1. **What does PCA do?** It finds perpendicular directions ordered by how much of the data's
   variance lies along them, so that a few of them can stand in for all the features.
2. **Why eigenvectors of the covariance matrix?** Maximising the variance uᵀS u with ‖u‖ = 1
   gives S u = λ u, and the variance along u is λ.
3. **Why standardise first?** Variance depends on units. Unscaled, proline's large numbers take
   over (P3).
4. **Why the divisor N − 1?** It is the unbiased sample variance and the convention of `np.cov`
   and scikit-learn.
5. **Why the sign rule?** u and −u are equally valid, so a fixed rule is needed before two
   results can be compared.
6. **Why is the training reconstruction error the sum of the discarded eigenvalues?** The error
   is the variance left in the dropped directions, and the variance in direction j is λ_j.
7. **Why can the test error never rise with k?** Each added component subtracts a sum of squares.
8. **Is more variance captured always better?** No. V1 captures 99.98% in two components and
   classifies worst.
9. **How can PCA match methods that use the labels?** Its components combine all features, and
   after standardising, the directions of large variance are also the ones that separate the
   classes (unscaled, in P3's V1, they are not).
10. **Can you rank the three methods from these results?** Not reliably. One test wine is 1.85
    percentage points, and most gaps are one or two wines. The largest is 6 wines, between PCA
    and the wrapper at k = 3.

The other files in this folder go deeper on each of these, with more questions and the code
behind every number.
