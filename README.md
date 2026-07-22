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

## Final result

The selected model is standardized multinomial logistic regression with strong
regularization (`C=0.01`). Selection used five expanding-window folds confined
to the training period. The newest 20% of matches were evaluated once.

| Metric | Training-prior baseline | Final model |
| --- | ---: | ---: |
| Accuracy | 0.4224 | 0.5889 |
| Balanced accuracy | 0.3333 | 0.4906 |
| Macro F1 | 0.1980 | 0.4340 |
| Log loss | 1.0644 | 0.9213 |
| Multiclass Brier | 0.6457 | 0.5423 |

The model predicted no draws as its most likely class. In addition, temperature
scaling selected on time-ordered training predictions worsened held-out log loss
from 0.9159 to 0.9213. Both findings are treated as central limitations in the
paper rather than hidden behind headline accuracy.

## Deliverables

- `paper/main.pdf`: verified seven-page final paper;
- `paper/main.tex` and `paper/references.bib`: arXiv-ready source;
- `artifacts/final_run/`: metrics, predictions, sweeps, cards, figures, fitted
  model, and SHA-256 manifest;
- `data/input/`: normalized match table and source-lineage manifest;
- `data/derived/`: reproducible prematch feature table;
- `src/soccer_final/`: normalization, feature, tuning, evaluation, calibration,
  plotting, reporting, and manifest code;
- `tests/`: leakage and evaluation tests.
- `presentation/output/`: verified 1080p final-presentation MP4s capped at
  200 MB and 50 MB, tracked with Git LFS.

## Reproduce

Python 3.11 or later is required.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,video]'

# Rebuild features and all experiment artifacts.
make experiment

# Run tests, lint, and artifact-hash verification.
make verify
```

To rebuild the PDF, install [Tectonic](https://tectonic-typesetting.github.io/)
0.16.9 or later and run:

```bash
make paper
```

`make all` runs the experiment, verification, and paper build. Regenerating the
experiment intentionally rewrites `artifacts/final_run/artifact_manifest.json`
to describe the new files.

## Presentation finale

The final presentation renders YOLOv8m player and sports-ball detections over a
130-second, 1080p FIFA highlight clip and displays two probability layers:

- the committed prematch engine at neutral 2026 state: 46.6% Argentina, 20.3%
  draw, and 33.1% Switzerland;
- changing detection-context visualization odds derived from jersey-color,
  player-count, ball-proximity, and four-second smoothing signals.

The changing probabilities are presentation context, not independently validated
live-match odds. The overlay says this on screen. It also labels ambiguous people
as `PERSON`, attributes FIFA/YouTube, and explains that team colors are heuristic.

Final deliverables:

- `presentation/output/argentina_switzerland_prediction_200mb.mp4` — 178.76 MiB
- `presentation/output/argentina_switzerland_prediction_50mb.mp4` — 44.19 MiB

Both are 1920x1080 H.264/AAC MP4s and decode end to end without error. Detailed
inference, compression, hardware, hashes, and rights notes are in
`presentation/RESULTS.md`, `presentation/HARDWARE.md`, and
`presentation/source/source_metadata.json`.

To reproduce locally, first review the source's permission/fair-use requirements,
then run:

```bash
make presentation-download
make presentation-detect
make presentation-master
make presentation-encodes
```

The downloaded source, 50 MB YOLO weights, 4.8 MB detection cache, and 2.15 GiB
lossless annotated master stay local and gitignored. Only the two final delivery
MP4s are stored in Git LFS.

## Repository layout

```text
soccer-analysis-final/
├── artifacts/final_run/       # complete submitted run
├── data/input/                # normalized versioned source
├── data/derived/              # generated prematch features
├── paper/                     # LaTeX, bibliography, rendered PDF
├── presentation/              # video pipeline docs and deliverables
├── scripts/                   # artifact verification
├── src/soccer_final/          # analysis package
└── tests/                     # leakage and metric checks
```

## Responsible-use boundary

This is an offline research artifact. It is not validated for wagering,
financial decisions, or production deployment. It contains no scraped video,
real-money execution, or market integration.
