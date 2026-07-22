"""Run the final chronological model-selection and evaluation experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from soccer_final.features import FEATURE_COLUMNS


LABELS = [0, 1, 2]
LABEL_NAMES = ["Home win", "Draw", "Away win"]


@dataclass(frozen=True)
class Candidate:
    name: str
    estimator: BaseEstimator
    parameters: dict[str, Any]


def candidates() -> list[Candidate]:
    options: list[Candidate] = []
    for c_value in (0.01, 0.1, 1.0, 10.0):
        for class_weight in (None, "balanced"):
            options.append(Candidate(
                "logistic_regression",
                make_pipeline(StandardScaler(), LogisticRegression(
                    C=c_value, class_weight=class_weight, max_iter=5000, random_state=42
                )),
                {"C": c_value, "class_weight": class_weight},
            ))
    for max_depth in (6, 12, None):
        for min_samples_leaf in (1, 5, 15):
            options.append(Candidate(
                "random_forest",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
                {"max_depth": max_depth, "min_samples_leaf": min_samples_leaf, "n_estimators": 300},
            ))
    for regularization in (0.0, 1.0, 10.0):
        options.append(Candidate(
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(l2_regularization=regularization, random_state=42),
            {"l2_regularization": regularization},
        ))
    return options


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(len(LABELS))[y_true.astype(int)]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def metric_row(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = np.asarray(LABELS)[np.argmax(probabilities, axis=1)]
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, probabilities, labels=LABELS)),
        "brier": multiclass_brier(y_true, probabilities),
    }


def align_probabilities(model: BaseEstimator, values: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(values)
    aligned = np.zeros((len(values), len(LABELS)), dtype=float)
    for source_index, label in enumerate(model.classes_):
        aligned[:, LABELS.index(int(label))] = raw[:, source_index]
    return aligned


def tune_model(x_train: pd.DataFrame, y_train: pd.Series, splits: int = 5) -> tuple[Candidate, pd.DataFrame, np.ndarray]:
    splitter = TimeSeriesSplit(n_splits=splits)
    results: list[dict[str, Any]] = []
    best_candidate: Candidate | None = None
    best_score: tuple[float, float] | None = None
    best_oof: np.ndarray | None = None
    for candidate in candidates():
        fold_metrics: list[dict[str, float]] = []
        oof = np.full((len(x_train), len(LABELS)), np.nan)
        for train_index, validation_index in splitter.split(x_train):
            model = clone(candidate.estimator)
            model.fit(x_train.iloc[train_index], y_train.iloc[train_index])
            probabilities = align_probabilities(model, x_train.iloc[validation_index])
            oof[validation_index] = probabilities
            fold_metrics.append(metric_row(y_train.iloc[validation_index].to_numpy(), probabilities))
        row = {
            "model": candidate.name,
            "parameters": json.dumps(candidate.parameters, sort_keys=True),
            "mean_accuracy": float(np.mean([value["accuracy"] for value in fold_metrics])),
            "sd_accuracy": float(np.std([value["accuracy"] for value in fold_metrics], ddof=1)),
            "mean_balanced_accuracy": float(np.mean([value["balanced_accuracy"] for value in fold_metrics])),
            "mean_macro_f1": float(np.mean([value["macro_f1"] for value in fold_metrics])),
            "mean_log_loss": float(np.mean([value["log_loss"] for value in fold_metrics])),
            "sd_log_loss": float(np.std([value["log_loss"] for value in fold_metrics], ddof=1)),
            "mean_brier": float(np.mean([value["brier"] for value in fold_metrics])),
            "sd_brier": float(np.std([value["brier"] for value in fold_metrics], ddof=1)),
        }
        results.append(row)
        score = (row["mean_log_loss"], -row["mean_accuracy"])
        if best_score is None or score < best_score:
            best_candidate, best_score, best_oof = candidate, score, oof
    if best_candidate is None or best_oof is None:
        raise RuntimeError("no candidates were evaluated")
    return best_candidate, pd.DataFrame(results).sort_values(["mean_log_loss", "mean_accuracy"]), best_oof


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    logits = np.log(clipped) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(logits)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def tune_temperature(y_train: pd.Series, oof_probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    valid = ~np.isnan(oof_probabilities).any(axis=1)
    y_valid = y_train.to_numpy()[valid]
    probability_valid = oof_probabilities[valid]
    rows = []
    for temperature in np.linspace(0.5, 2.0, 31):
        row = {"temperature": float(temperature), **metric_row(y_valid, temperature_scale(probability_valid, temperature))}
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["log_loss", "accuracy"])
    return float(frame.iloc[0]["temperature"]), frame


def calibration_bins(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> pd.DataFrame:
    predicted = np.argmax(probabilities, axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    assigned = np.clip(np.digitize(confidence, edges, right=True) - 1, 0, bins - 1)
    rows = []
    for index in range(bins):
        mask = assigned == index
        if mask.any():
            rows.append({
                "bin_lower": edges[index],
                "bin_upper": edges[index + 1],
                "count": int(mask.sum()),
                "mean_confidence": float(confidence[mask].mean()),
                "empirical_accuracy": float(correct[mask].mean()),
                "calibration_gap": float(confidence[mask].mean() - correct[mask].mean()),
            })
    return pd.DataFrame(rows)


def expected_calibration_error(bins: pd.DataFrame) -> float:
    return float((bins["count"] * bins["calibration_gap"].abs()).sum() / bins["count"].sum())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_figures(
    output_dir: Path,
    bins: pd.DataFrame,
    matrix: np.ndarray,
    importance: pd.DataFrame,
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.2, 4.2))
    plt.plot([0, 1], [0, 1], linestyle="--", color="0.55", label="Perfect calibration")
    plt.plot(bins.mean_confidence, bins.empirical_accuracy, "o-", color="#1f5a94", label="Final model")
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figure_dir / "calibration.pdf", bbox_inches="tight")
    plt.savefig(figure_dir / "calibration.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5.0, 4.2))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks(range(3), LABEL_NAMES, rotation=20)
    plt.yticks(range(3), LABEL_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    for row in range(3):
        for column in range(3):
            plt.text(column, row, str(matrix[row, column]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(figure_dir / "confusion_matrix.pdf", bbox_inches="tight")
    plt.savefig(figure_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close()

    top = importance.head(12).sort_values("importance")
    plt.figure(figsize=(6.0, 4.6))
    plt.barh(top.feature, top.importance, color="#1f5a94")
    plt.xlabel("Accuracy decrease after permutation")
    plt.tight_layout()
    plt.savefig(figure_dir / "feature_importance.pdf", bbox_inches="tight")
    plt.savefig(figure_dir / "feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close()


def write_cards(output_dir: Path, metrics: dict[str, Any], feature_path: Path) -> None:
    final = metrics["final_model"]
    baseline = metrics["baseline"]
    model_card = f"""# Model Card: Prematch Outcome Classifier

## Intended use

Offline research and course-project analysis of three-class soccer outcomes.
This model is not validated for wagering, financial decisions, or deployment.

## Model and selection

- Selected family: `{metrics['selected_model']}`
- Selected parameters: `{json.dumps(metrics['selected_parameters'], sort_keys=True)}`
- Temperature: `{metrics['temperature']}` (selected on time-ordered out-of-fold training predictions)
- Features: {len(FEATURE_COLUMNS)} strictly prematch numeric variables

Candidate families and hyperparameters were compared with five-fold expanding-window
cross-validation inside the training period. Mean validation log loss was the primary
selection criterion. The final 20% chronological holdout was not used for selection.

## Held-out performance

| Metric | Training-prior baseline | Final model |
| --- | ---: | ---: |
| Accuracy | {baseline['accuracy']:.4f} | {final['accuracy']:.4f} |
| Balanced accuracy | {baseline['balanced_accuracy']:.4f} | {final['balanced_accuracy']:.4f} |
| Macro F1 | {baseline['macro_f1']:.4f} | {final['macro_f1']:.4f} |
| Log loss | {baseline['log_loss']:.4f} | {final['log_loss']:.4f} |
| Multiclass Brier score | {baseline['brier']:.4f} | {final['brier']:.4f} |
| Confidence ECE | {baseline['expected_calibration_error']:.4f} | {final['expected_calibration_error']:.4f} |

## Limitations

The selected hard classifier predicted no draws on the holdout. Accuracy therefore
overstates class-balanced performance. Competitions and seasons have heterogeneous
coverage, and the open dataset is not a random sample of world soccer. Temperature
scaling selected on training-period predictions slightly worsened final holdout log
loss and calibration error. These weaknesses are reported, not corrected post hoc.
"""
    data_card = f"""# Data Card: StatsBomb Match Metadata

## Source

StatsBomb Open Data match metadata, normalized from the snapshot previously used by
`Soccer-Probability-Engine` at commit `14a61a6`. Upstream:
<https://github.com/statsbomb/open-data>.

## Contents

- 3,961 completed matches from 1958-06-24 through 2025-07-27
- 24 competitions
- Labels: 1,775 home wins; 879 draws; 1,307 away wins
- 3,168 chronological training rows and 793 chronological test rows
- 36 engineered features computed before updating state with the current result

## Lineage

- Feature table: `{feature_path}`
- Feature-table SHA-256: `{_sha256(feature_path)}`
- Normalized input and upstream-source hashes: `data/input/dataset_manifest.json`

## Known limitations

Coverage is curated and uneven across competitions, seasons, eras, and genders. The
study does not claim population-representative soccer performance. First matches in
each competition-season lack prior state; rest is capped at 30 days. Team state and
Elo reset at competition-season boundaries.
"""
    summary = f"""# Final Run Summary

The selected `{metrics['selected_model']}` model used
`{json.dumps(metrics['selected_parameters'], sort_keys=True)}` and a training-selected
temperature of {metrics['temperature']:.2f}. It was trained on {metrics['train_rows']:,}
matches through {metrics['train_date_max']} and evaluated once on {metrics['test_rows']:,}
matches from {metrics['test_date_min']} through {metrics['test_date_max']}.

Held-out accuracy was {final['accuracy']:.4f}, compared with {baseline['accuracy']:.4f}
for the training-prior baseline. Held-out log loss was {final['log_loss']:.4f}, the
multiclass Brier score was {final['brier']:.4f}, balanced accuracy was
{final['balanced_accuracy']:.4f}, and macro F1 was {final['macro_f1']:.4f}.

The uncalibrated model's log loss was {metrics['uncalibrated_model']['log_loss']:.4f};
temperature scaling worsened it by
{final['log_loss'] - metrics['uncalibrated_model']['log_loss']:.4f} on the untouched
holdout. The confusion matrix also shows zero predicted draws. These are central
limitations of the final result.
"""
    (output_dir / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")
    (output_dir / "DATA_CARD.md").write_text(data_card, encoding="utf-8")
    (output_dir / "RUN_SUMMARY.md").write_text(summary, encoding="utf-8")


def run_experiment(feature_path: Path, output_dir: Path) -> dict[str, Any]:
    frame = pd.read_parquet(feature_path).sort_values(["match_date", "match_id"]).reset_index(drop=True)
    split_index = int(len(frame) * 0.8)
    train, test = frame.iloc[:split_index].copy(), frame.iloc[split_index:].copy()
    x_train, y_train = train[FEATURE_COLUMNS], train.target.astype(int)
    x_test, y_test = test[FEATURE_COLUMNS], test.target.astype(int)

    selected, cv_results, oof_probabilities = tune_model(x_train, y_train)
    temperature, temperature_results = tune_temperature(y_train, oof_probabilities)
    model = clone(selected.estimator).fit(x_train, y_train)
    uncalibrated = align_probabilities(model, x_test)
    calibrated = temperature_scale(uncalibrated, temperature)

    dummy = DummyClassifier(strategy="prior").fit(x_train, y_train)
    baseline_probabilities = align_probabilities(dummy, x_test)
    baseline_metrics = metric_row(y_test.to_numpy(), baseline_probabilities)
    uncalibrated_metrics = metric_row(y_test.to_numpy(), uncalibrated)
    final_metrics = metric_row(y_test.to_numpy(), calibrated)

    predicted = np.argmax(calibrated, axis=1)
    matrix = confusion_matrix(y_test, predicted, labels=LABELS)
    bins = calibration_bins(y_test.to_numpy(), calibrated)
    final_metrics["expected_calibration_error"] = expected_calibration_error(bins)
    uncalibrated_metrics["expected_calibration_error"] = expected_calibration_error(
        calibration_bins(y_test.to_numpy(), uncalibrated)
    )
    baseline_metrics["expected_calibration_error"] = expected_calibration_error(
        calibration_bins(y_test.to_numpy(), baseline_probabilities)
    )
    permutation = permutation_importance(model, x_test, y_test, n_repeats=20, random_state=42, scoring="accuracy")
    importance = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": permutation.importances_mean}).sort_values(
        "importance", ascending=False
    )
    predictions = test[["match_id", "match_date", "competition", "season", "home_team", "away_team", "target"]].copy()
    predictions["prediction"] = predicted
    for index, label in enumerate(("home_win", "draw", "away_win")):
        predictions[f"prob_{label}"] = calibrated[:, index]

    output_dir.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(output_dir / "cv_results.csv", index=False)
    temperature_results.to_csv(output_dir / "temperature_results.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    bins.to_csv(output_dir / "calibration_bins.csv", index=False)
    pd.DataFrame(matrix, index=LABEL_NAMES, columns=LABEL_NAMES).to_csv(output_dir / "confusion_matrix.csv")
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    joblib.dump({"model": model, "temperature": temperature, "features": FEATURE_COLUMNS, "labels": LABELS}, output_dir / "model.joblib")
    write_figures(output_dir, bins, matrix, importance)

    metrics = {
        "experiment": "chronological prematch outcome prediction",
        "random_state": 42,
        "rows": len(frame),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_date_min": train.match_date.min().date().isoformat(),
        "train_date_max": train.match_date.max().date().isoformat(),
        "test_date_min": test.match_date.min().date().isoformat(),
        "test_date_max": test.match_date.max().date().isoformat(),
        "selected_model": selected.name,
        "selected_parameters": selected.parameters,
        "temperature": temperature,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "baseline": baseline_metrics,
        "uncalibrated_model": uncalibrated_metrics,
        "final_model": final_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_cards(output_dir, metrics, feature_path)
    artifacts = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts[str(path.relative_to(output_dir))] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    (output_dir / "artifact_manifest.json").write_text(json.dumps(artifacts, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/derived/prematch_features.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/final_run"))
    args = parser.parse_args()
    metrics = run_experiment(args.features, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
