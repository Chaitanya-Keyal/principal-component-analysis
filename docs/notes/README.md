# Study notes

The ideas behind each problem, how the code implements them, and what to show and say when
presenting the work. Not part of the submission. Every member may be asked about any problem, so
read them in this order:

- [guide.md](guide.md): the whole assignment in plain language, then a script for showing the
  notebook and ten likely questions; the first file to read
- [assignment.md](assignment.md): what is asked, the rules that apply everywhere (C1 to C6), and
  where each deliverable lives in the repo
- [pca-basics.md](pca-basics.md): variance, covariance, standardising, eigenvectors and the sign
  rule; P1 to P4 all lean on it
- [p1-pca.md](p1-pca.md): PCA as variance maximisation, the from-scratch code and the check
  against scikit-learn
- [p2-choosing-k.md](p2-choosing-k.md): explained variance, reconstruction error, and the elbow,
  Kaiser and threshold rules
- [p3-preprocessing.md](p3-preprocessing.md): why PCA depends on the scale of the features, and
  why variance captured is not the same as useful structure
- [p4-feature-selection.md](p4-feature-selection.md): PCA against a filter and a wrapper, and how
  far 54 test samples can be trusted

Every problem file opens with an In short, then follows the same order: the idea in plain words,
the maths, the code, our numbers, the demo, likely questions with answers, and the code changes
most likely to be asked for (with where to make them). A change is best run in the notebook or
with `src.run --out /tmp/demo`, so that the repo's results and figures are not overwritten.
