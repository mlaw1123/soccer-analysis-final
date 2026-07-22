PYTHON ?= .venv/bin/python
TECTONIC ?= tectonic

.PHONY: all experiment paper test verify

all: experiment verify paper

experiment:
	$(PYTHON) -m soccer_final.features
	MPLBACKEND=Agg MPLCONFIGDIR=tmp/matplotlib LOKY_MAX_CPU_COUNT=8 $(PYTHON) -m soccer_final.experiment

paper:
	$(TECTONIC) paper/main.tex --outdir paper

test:
	$(PYTHON) -m pytest -q
	$(PYTHON) -m ruff check src tests scripts

verify: test
	$(PYTHON) scripts/verify_artifacts.py
