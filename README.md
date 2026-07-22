# Soccer Outcome Prediction: Final Project

This repository is the submission-focused research artifact for a three-class
prematch soccer outcome study. It consolidates only the relevant work from the
earlier `Soccer-Probability-Engine` and `soccer-analytics` repositories.

The final product will provide one reproducible path from a versioned StatsBomb
Open Data match table through leakage-safe prematch features, chronological
train/test evaluation, model tuning and calibration, and an arXiv-style paper.

## Research question

How accurately, and with what probability calibration, can match outcomes
(home win, draw, away win) be predicted from information available before
kickoff: cumulative team performance, rolling five-match form, Elo strength,
venue history, and rest?

## Scope

Included:

- source-data provenance and SHA-256 manifests;
- chronological, pre-outcome feature generation;
- majority and probabilistic model baselines;
- model comparison and task-specific hyperparameter tuning;
- held-out accuracy, log loss, multiclass Brier score, calibration, and
  confusion-matrix analysis;
- deterministic experiment artifacts and an arXiv-style PDF.

Excluded:

- the unrelated video, YOLO, capture, live-watch, and annotation systems;
- betting, market execution, or real-money actions;
- unlicensed audiovisual data;
- generated caches, IDE settings, and broad framework code not used by the
  final experiment.

## Checkpoint plan

Each completed phase is committed separately:

1. define the final scope and repository contract;
2. add versioned data and leakage-safe feature generation;
3. add model training, fine-tuning, and held-out evaluation;
4. generate and validate reproducible result artifacts;
5. write and render the arXiv-style paper;
6. finish documentation and end-to-end verification.

The source repositories remain unchanged.
