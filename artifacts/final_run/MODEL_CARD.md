# Model Card: Prematch Outcome Classifier

## Intended use

Offline research and course-project analysis of three-class soccer outcomes.
This model is not validated for wagering, financial decisions, or deployment.

## Model and selection

- Selected family: `logistic_regression`
- Selected parameters: `{"C": 0.01, "class_weight": null}`
- Temperature: `1.1` (selected on time-ordered out-of-fold training predictions)
- Features: 36 strictly prematch numeric variables

Candidate families and hyperparameters were compared with five-fold expanding-window
cross-validation inside the training period. Mean validation log loss was the primary
selection criterion. The final 20% chronological holdout was not used for selection.

## Held-out performance

| Metric | Training-prior baseline | Final model |
| --- | ---: | ---: |
| Accuracy | 0.4224 | 0.5889 |
| Balanced accuracy | 0.3333 | 0.4906 |
| Macro F1 | 0.1980 | 0.4340 |
| Log loss | 1.0644 | 0.9213 |
| Multiclass Brier score | 0.6457 | 0.5423 |
| Confidence ECE | 0.0321 | 0.0652 |

## Limitations

The selected hard classifier predicted no draws on the holdout. Accuracy therefore
overstates class-balanced performance. Competitions and seasons have heterogeneous
coverage, and the open dataset is not a random sample of world soccer. Temperature
scaling selected on training-period predictions slightly worsened final holdout log
loss and calibration error. These weaknesses are reported, not corrected post hoc.
