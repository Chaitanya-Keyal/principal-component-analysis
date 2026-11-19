# Principal Component Analysis: From Scratch, Choosing k, Preprocessing and Feature Selection

PCA on the 13-feature wine data, built from scratch in NumPy: the covariance matrix and its
eigendecomposition, how much variance and reconstruction accuracy k components keep, what
preprocessing does to the components, and PCA against feature selection as input to a classifier.

| Problem | What it does |
|---------|--------------|
| P1 | PCA as variance maximisation: S and its eigendecomposition in NumPy, checked against scikit-learn's PCA; loadings of PC1 and PC2 |
| P2 | Explained variance and reconstruction error against k: scree and cumulative plots, the elbow, Kaiser and 80/90/95% rules, train and test error |
| P3 | PCA after centring only, min-max scaling and z-scoring: variance captured, the two-dimensional projection, 5-NN accuracy on it |
| P4 | PCA scores vs. F-score ranking vs. sequential forward selection, k = 1, 2, 3, 5, 8, with one logistic-regression classifier |

## 1. Layout

```
assignment.pdf               the handout
data/                        the two provided CSVs (13 features, then class), used as given
src/common/                  data loading, the PCA building blocks, number formats, plot style
src/p1.py ... src/p4.py      one module per problem: compute() for the numbers, figures() for the plots
src/run.py                   runs the problems and writes results/ and figures/
src/report_numbers.py        results/*.json -> results/numbers.tex, the numbers the report quotes
results/                     p1.json ... p4.json (every reported number), p1.txt (the printed
                             eigenvalues) and numbers.tex; written by src.run
figures/                     every figure, as figures/<name>.pdf; written by src.run
tests/                       checks of each problem against hand-worked values and scikit-learn
report/report.tex            the write-up; numbers and figures come in from results/ and figures/
notebooks/demo.ipynb         the demo: each problem's figures and key numbers, by calling src/
docs/notes/                  study notes: the ideas in plain language, the maths, the code, the demo
README.txt                   the submission's readme: group, versions, how to run, open choices
pyproject.toml, uv.lock      dependencies; .python-version pins Python 3.12
```

## 2. Environment

```bash
uv sync                      # numpy, matplotlib and scikit-learn, plus pytest and ruff (the dev group)
uv sync --group notebook     # adds JupyterLab, for the demo notebook
```

`uv.lock` pins every version. Tests, then formatting and lint:

```bash
uv run pytest
uv run ruff format . && uv run ruff check .
```

## 3. Data

The two files the assignment provides go in `data/` unchanged:

```
wine_train.csv               N = 124     fits every statistic, eigenvector and feature score
wine_test.csv                N = 54      only transformed and scored, never used to fit or choose
```

Each has the 13 feature columns in the handout's order (`alcohol` ... `proline`) and then an
integer `class` in {0, 1, 2}. The loader checks the header and stops if the columns differ.
Nothing in this repo generates or modifies data. Setting `WINE_DATA_DIR` reads the two files
from another folder instead.

## 4. Running

```bash
uv run python -m src.run              # all four problems, a few seconds
uv run python -m src.run p2 p3        # some of them
uv run python -m src.run --out DIR    # write results/ and figures/ under DIR instead
```

P1 prints the 13 eigenvalues as `lambda_j = x.xxxx`. Each problem writes `results/<p>.json` and
its figures; `results/numbers.tex` is rebuilt from whatever JSON is present. The only randomness
is the shuffle of the cross-validation folds in P4, with the seed the handout fixes
(`random_state=0`), and PDFs carry no timestamp, so a rerun on the same machine reproduces every
file byte for byte.

## 5. Report and demo

```bash
cd report && latexmk -pdf report.tex                   # report.pdf
uv run --group notebook jupyter lab notebooks/demo.ipynb
```

The report reads every number through `\res{key}` from `results/numbers.tex`, so rerunning
`src.run` and rebuilding updates it. `tests/test_discussion.py` checks the statements its
discussion paragraphs make in words. The notebook has no logic of its own: it calls the same
`compute()` and `figures()` as `src.run`, and is committed without outputs to be run live.

## 6. Submission

```bash
git archive --format=zip -o ../submission/2023B4A70727H_A2.zip HEAD -- README.txt \
  pyproject.toml uv.lock .python-version src tests notebooks data results figures report
```

The zip holds the committed files, flat at its root, dated by the commit, so the same commit
always gives the same bytes. It leaves out this README, the handout and `docs/notes/`.
