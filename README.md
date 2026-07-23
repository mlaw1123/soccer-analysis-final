# Match Intelligence 2026

## [Launch the live final presentation →](https://match-intelligence-2026.mattlaw1123.chatgpt.site)

### [Download the final project from GitHub →](https://github.com/mlaw1123/soccer-analysis-final/releases/tag/v1.0.0)

**An end-to-end soccer outcome prediction system—from 67 years of match data to an on-screen World Cup match finale.**

[![Match Intelligence 2026 — before the whistle, know the odds](site/public/og.png)](https://match-intelligence-2026.mattlaw1123.chatgpt.site)

> The hosted presentation is private and may ask you to continue with ChatGPT. The complete website also runs locally from [`site/`](site/).

This repository is the final submission and presentation artifact. It brings together the relevant research, data pipeline, model, evaluation, arXiv-style paper, YOLO match overlay, compressed presentation videos, and polished World Cup 2026-themed website in one reproducible project.

## The story

The project asks a deliberately difficult question:

> How accurately—and with what probability calibration—can an international soccer result be predicted using only information available before kickoff?

The answer is presented in three connected layers:

1. **Prematch intelligence:** leakage-safe form, strength, venue, ranking, and rest features derived chronologically.
2. **Probability evaluation:** an interpretable three-class model assessed on the newest held-out matches, including its calibration failures and draw limitation.
3. **Visual finale:** the Argentina–Switzerland prediction displayed over a full match clip with YOLOv8m player and ball detections.

## Final result

The selected model is standardized multinomial logistic regression with strong regularization (`C=0.01`). Model selection used five expanding-window folds confined to the training period. The newest 20% of matches were evaluated once.

| Metric | Training-prior baseline | Final model |
| --- | ---: | ---: |
| Accuracy | 0.4224 | **0.5889** |
| Balanced accuracy | 0.3333 | **0.4906** |
| Macro F1 | 0.1980 | **0.4340** |
| Log loss ↓ | 1.0644 | **0.9213** |
| Multiclass Brier ↓ | 0.6457 | **0.5423** |

### Experiment at a glance

| Item | Final scope |
| --- | ---: |
| International matches | 3,961 |
| Competitions | 24 |
| Historical window | 1958–2025 |
| Prematch features | 36 |
| Chronological training set | 3,168 matches |
| Held-out test set | 793 matches |

The model predicted no draws as its most likely class. Temperature scaling also worsened held-out log loss from `0.9159` to `0.9213`. The paper and presentation treat both findings as central limitations rather than hiding them behind headline accuracy.

## Argentina vs Switzerland finale

For a neutral 2026 match state, the committed prematch engine produces:

| Argentina | Draw | Switzerland |
| ---: | ---: | ---: |
| **46.59%** | **20.28%** | **33.14%** |

The 130-second presentation combines those model probabilities with YOLOv8m detections and an explicitly labeled visualization of changing match context.

- 1,953 sampled frames processed locally
- 25,090 people detections
- 1,865 ball detections
- ball present in 59.55% of sampled frames
- 13.34 sampled FPS on a 10-core M1 Pro with 16 GB unified memory

The changing detection-context values are a presentation visualization—not independently validated live betting odds. The overlay states this directly and labels ambiguous detections as `PERSON`.

## Final deliverables

- **Live experience:** [match-intelligence-2026.mattlaw1123.chatgpt.site](https://match-intelligence-2026.mattlaw1123.chatgpt.site)
- **GitHub final release:** [Match Intelligence 2026 — Final Project (`v1.0.0`)](https://github.com/mlaw1123/soccer-analysis-final/releases/tag/v1.0.0)
- **Paper:** [`paper/main.pdf`](paper/main.pdf), with arXiv-ready LaTeX and bibliography
- **Model run:** [`artifacts/final_run/`](artifacts/final_run/) with metrics, predictions, sweeps, figures, cards, fitted model, and SHA-256 manifest
- **Data:** normalized source lineage in [`data/input/`](data/input/) and leakage-safe features in [`data/derived/`](data/derived/)
- **Analysis package:** [`src/soccer_final/`](src/soccer_final/)
- **Presentation website:** [`site/`](site/), including the embedded 720p web edition and downloadable paper
- **1080p presentation videos:**
  - [`argentina_switzerland_prediction_200mb.mp4`](presentation/output/argentina_switzerland_prediction_200mb.mp4) — 178.76 MiB
  - [`argentina_switzerland_prediction_50mb.mp4`](presentation/output/argentina_switzerland_prediction_50mb.mp4) — 44.19 MiB

The three presentation MP4s are tracked with Git LFS. The 1080p delivery files decode end to end without error.

## Reproduce the research

Python 3.11 or later is required.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,video]'

make experiment  # rebuild features and experiment artifacts
make verify      # tests, lint, and artifact-hash verification
make paper       # render the paper with Tectonic 0.16.9+
```

`make all` runs the experiment, verification, and paper build.

To rebuild the presentation pipeline after reviewing the source video’s permission and fair-use requirements:

```bash
make presentation-download
make presentation-detect
make presentation-master
make presentation-encodes
```

Large intermediate files—the downloaded source, YOLO weights, detection cache, and lossless annotated master—stay local and are gitignored.

## Run the presentation website

Use Node.js 22 or newer and pnpm:

```bash
cd site
pnpm install
pnpm dev
```

Open `http://localhost:3000`. Use `pnpm test` for a production build plus rendered-content and asset verification, and `pnpm lint` for static analysis.

## Commit checkpoints

The final repository preserves reviewable checkpoints for every major phase:

1. define the final repository contract and data provenance;
2. implement leakage-safe feature generation;
3. train, tune, calibrate, and evaluate the model;
4. generate deterministic result artifacts and the arXiv-style paper;
5. build and verify the YOLO probability-overlay pipeline;
6. encode the 200 MB and 50 MB presentation deliverables;
7. build the World Cup 2026 presentation website;
8. add verified web media and deploy the live experience;
9. finalize and publish the complete repository.

Recent final checkpoints:

```text
b971c3a  Finalize Match Intelligence 2026 project
2d2d1c6  Configure Sites deployment target
3a5ed1d  Add verified web presentation media
38747b9  Build World Cup 2026 presentation experience
3fc6baf  Document and verify presentation finale
ab95e9e  Add verified presentation video deliverables
c92d019  Add YOLO probability overlay pipeline
```

## Repository layout

```text
soccer-analysis-final/
├── artifacts/final_run/       # complete submitted model run
├── data/                       # versioned input and derived features
├── paper/                      # LaTeX, bibliography, rendered PDF
├── presentation/              # detection/video pipeline and final MP4s
├── site/                      # live presentation source and web media
├── src/soccer_final/          # analysis package
└── tests/                     # leakage and metric tests
```

## Responsible-use and rights boundary

This is an offline research and classroom-presentation artifact. It is not validated for wagering, financial decisions, or production deployment. Match frames and video derive from the presentation input; confirm source-video rights before public redistribution. Detailed inference, compression, hardware, hashes, and rights notes are documented in [`presentation/RESULTS.md`](presentation/RESULTS.md), [`presentation/HARDWARE.md`](presentation/HARDWARE.md), and [`presentation/source/source_metadata.json`](presentation/source/source_metadata.json).
