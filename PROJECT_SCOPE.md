# Final product decision

## Source audit

The final repository draws from two prior codebases:

- `Soccer-Probability-Engine`: the real StatsBomb-derived match table,
  chronological prematch feature builder, 3,961-row feature dataset, trained
  random-forest experiment, and initial model-comparison results.
- `soccer-analytics`: the submission workflow patterns for version hashes,
  probability metrics, calibration review, cards, tests, and an arXiv-style
  manuscript.

An existing `soccer-analysis-final` repository was empty before this work.

## Product boundary

The final artifact is a narrow research repository, not a merger of the two
source trees. Its sole product is a reproducible empirical study of prematch
soccer outcome probabilities. Code and artifacts are included only when they
are exercised by that study or needed to audit its claims.

## Evaluation contract

- The split is chronological, with the oldest 80% used for training and the
  newest 20% reserved for final evaluation.
- Every predictive feature must be computed before the current match result is
  used to update team state.
- Hyperparameters may not be selected on the final test set. Selection occurs
  within the training period using time-ordered cross-validation.
- The paper reports all material metrics, including weak or negative findings.
- Generated tables and figures are derived from machine-readable experiment
  artifacts rather than hand-edited values.
