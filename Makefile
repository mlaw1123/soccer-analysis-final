PYTHON ?= .venv/bin/python
TECTONIC ?= tectonic

.PHONY: all experiment paper presentation-download presentation-detect presentation-master presentation-encodes test verify verify-presentation

VIDEO_SOURCE := presentation/source/argentina_vs_switzerland_2026.mp4
VIDEO_MODEL := presentation/models/yolov8m.pt
VIDEO_DETECTIONS := presentation/work/full/detections.jsonl
VIDEO_MASTER := presentation/work/full/annotated_lossless_master.mp4

all: experiment verify paper

experiment:
	$(PYTHON) -m soccer_final.features
	MPLBACKEND=Agg MPLCONFIGDIR=tmp/matplotlib LOKY_MAX_CPU_COUNT=8 $(PYTHON) -m soccer_final.experiment

paper:
	$(TECTONIC) paper/main.tex --outdir paper

presentation-download:
	mkdir -p presentation/source presentation/models
	curl -L --fail https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt -o $(VIDEO_MODEL)
	SSL_CERT_FILE=$$($(PYTHON) -m certifi) .venv/bin/yt-dlp --no-playlist -f '137+140' --merge-output-format mp4 --output 'presentation/source/argentina_vs_switzerland_2026.%%(ext)s' 'https://www.youtube.com/watch?v=zZxxDbLxEi4'

presentation-detect:
	$(PYTHON) -m soccer_final.presentation_video detect --source $(VIDEO_SOURCE) --model $(VIDEO_MODEL) --output $(VIDEO_DETECTIONS) --stride 2 --image-size 960

presentation-master:
	$(PYTHON) -m soccer_final.presentation_video render --source $(VIDEO_SOURCE) --detections $(VIDEO_DETECTIONS) --output $(VIDEO_MASTER)

presentation-encodes:
	$(PYTHON) -m soccer_final.presentation_video encode --source $(VIDEO_MASTER) --output presentation/output/argentina_switzerland_prediction_200mb.mp4 --target-mb 200
	$(PYTHON) -m soccer_final.presentation_video encode --source $(VIDEO_MASTER) --output presentation/output/argentina_switzerland_prediction_50mb.mp4 --target-mb 50

test:
	$(PYTHON) -m pytest -q
	$(PYTHON) -m ruff check src tests scripts

verify: test
	$(PYTHON) scripts/verify_artifacts.py

verify-presentation:
	$(PYTHON) scripts/verify_presentation.py
