# Final Run Summary

The selected `logistic_regression` model used
`{"C": 0.01, "class_weight": null}` and a training-selected
temperature of 1.10. It was trained on 3,168
matches through 2023-10-06 and evaluated once on 793
matches from 2023-10-07 through 2025-07-27.

Held-out accuracy was 0.5889, compared with 0.4224
for the training-prior baseline. Held-out log loss was 0.9213, the
multiclass Brier score was 0.5423, balanced accuracy was
0.4906, and macro F1 was 0.4340.

The uncalibrated model's log loss was 0.9159;
temperature scaling worsened it by
0.0054 on the untouched
holdout. The confusion matrix also shows zero predicted draws. These are central
limitations of the final result.
