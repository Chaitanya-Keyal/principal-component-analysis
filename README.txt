CS F320 Foundations of Data Science, Assignment 2: Principal Component Analysis

Group
  Chaitanya Keyal   2023B4A70727H
  Abhishek Verma    2023B4A71256H
  Palash Jain       2023A2B51271H

Report
  report/report.pdf holds every deliverable of P1 to P4: the derivations, the printed
  eigenvalues, the tables, the figures and the discussions. report/report.tex is its source and
  reads every number from results/numbers.tex.
  Repository: https://github.com/Chaitanya-Keyal/principal-component-analysis

Versions used
  Python 3.12.10, numpy 2.5.5, matplotlib 3.11.2, scikit-learn 1.9.1. src/ and tests/ import
  only these, threadpoolctl (installed with scikit-learn) and the Python standard library; the
  notebook also needs JupyterLab. The code also runs on numpy 1.26, matplotlib 3.8 and
  scikit-learn 1.5, with every 4-decimal number unchanged; only the residuals near 1e-15, which
  rounding alone sets, come out with different digits.

How to run
  From this folder, with uv (uv.lock pins every version):

    uv sync
    uv run python -m src.run          all four problems, a few seconds
    uv run python -m src.run p2 p3    some of them
    uv run pytest                     34 tests against hand-worked values and scikit-learn
    uv run --group notebook jupyter lab notebooks/demo.ipynb     the demo (adds JupyterLab)

  or, with any Python 3.12+ that has numpy, matplotlib and scikit-learn: python -m src.run

  P1 prints the 13 eigenvalues as lambda_j = x.xxxx. Each problem writes results/<p>.json (every
  number it reports) and its figures to figures/, and results/numbers.tex is rebuilt from the
  JSON; cd report && latexmk -pdf report.tex then rebuilds the report. --out DIR writes results/
  and figures/ under DIR instead. The only randomness is the shuffle of the P4 cross-validation
  folds (random_state=0, as the handout fixes), so a rerun on the same machine gives the same
  files byte for byte.

What is where
  data/                    the two provided CSVs, unchanged
  src/common/pca.py        z-score, covariance (divisor N - 1), eigendecomposition with the sign
                           rule, explained variance ratio
  src/p1.py ... src/p4.py  one module per problem: compute() for the numbers, figures() for plots
  src/common/              also the data loader, the number formats and the plot style
  src/run.py               the runner above; src/report_numbers.py writes results/numbers.tex
  results/, figures/       the outputs the report uses; results/p1.txt is P1's printed output
  tests/                   checks on the provided data, e.g. that forward selection picks the same
                           features as scikit-learn's SequentialFeatureSelector
  notebooks/demo.ipynb     the demo: calls src/ and shows each problem's figures and numbers

Conventions C1 to C6, as implemented
  - Every statistic comes from the training set: means, standard deviations, minima and maxima,
    eigenvectors, F-scores and the forward-selection order. Test points are only transformed.
  - Standard deviations and S use the divisor N - 1; eigenvalues come from numpy.linalg.eigh,
    sorted in descending order.
  - The sign rule (largest absolute entry positive) is applied to every decomposition, including
    the rows of scikit-learn's components_ before the comparison in P1.
  - J is computed from the reconstructions themselves, with divisor 123 for the training set and
    53 for the test set.

Choices the handout leaves open
  - P2: the scree plot and the cumulative plot are the two panels of one figure, and the scree
    plot also marks the 0.10 threshold that defines the elbow.
  - P3: min-max test values outside [0, 1] are left as they are. Each panel of the projection
    figure carries its own legend.
  - P4: forward selection is our own loop, scored with cross_val_score on the fixed folds; it is
    run once to k = 8 and the smaller sets are its first k features. The selected columns reach
    the classifier in increasing column index. "Within one test sample of the baseline" is
    decided on counts of correct test predictions, not on rounded accuracies. Linear-algebra
    threads are limited to one during forward selection, which changes its speed only.
