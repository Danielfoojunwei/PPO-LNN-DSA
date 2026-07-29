# Reproduction pipeline.  `make all` targets under 45 minutes on 4 CPU cores and
# reproduces STUDY A.  `make study-b` reproduces STUDY B and takes about 2.5
# hours on the same box, which is why it is not part of `all`.
#
# Every target is a thin wrapper around a script with a real CLI; nothing here
# hides a path or a hyperparameter.  The matrices live in configs/suite.yaml,
# configs/federated.yaml and configs/suite_study_b.yaml, and nowhere else.

PYTHON ?= python
RESULTS ?= results
WORKERS ?= 4

.PHONY: help install test test-fast probe suite federated analyze report all smoke check clean \
        study-b analyze-study-b check-study-b

help:
	@echo "install     pip install -r requirements.txt"
	@echo "test        pytest -q (full unit suite)"
	@echo "test-fast   pytest -q -m 'not slow'"
	@echo "probe       measure real steps/s on this box"
	@echo "suite       run the confirmatory matrix"
	@echo "federated   run the topology study"
	@echo "analyze     build tables, figures and claims.json (Study A)"
	@echo "study-b     run Study B: the same matrix at 16x the budget, 2 scenarios"
	@echo "analyze-study-b  tables, figures, claims and mechanistic endpoints for Study B"
	@echo "report      regenerate RESULTS.md and every generated block in README/docs"
	@echo "all         probe suite federated analyze report"
	@echo "smoke       tiny end-to-end run (the CI budget), into \$$(SMOKE_DIR)"
	@echo "check       verify committed tables, RESULTS.md and README/docs are current"
	@echo "check-study-b    verify Study B's committed tables, claims and endpoints are current"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

test-fast:
	$(PYTHON) -m pytest -q -m "not slow"

probe:
	$(PYTHON) scripts/probe_throughput.py

suite:
	$(PYTHON) scripts/run_suite.py --config configs/suite.yaml --workers $(WORKERS) --output-dir $(RESULTS)

federated:
	$(PYTHON) scripts/run_federated.py --config configs/federated.yaml --workers $(WORKERS) --output-dir $(RESULTS)

analyze:
	$(PYTHON) scripts/analyze.py --results-dir $(RESULTS) --output-dir $(RESULTS)

# Study B: a SEPARATE study at 16x the budget, with its own pre-registration and
# its own output tree.  It never writes results/all_runs.csv, results/tables/ or
# results/claims.json -- scripts/run_study_b.py refuses to -- so Study A stays
# byte-identical committed evidence.  The runner takes ~2.5 hours on 4 cores and
# verifies the frozen pre-registration digest before it executes a single cell.
STUDY_B ?= $(RESULTS)/study_b
PREREG_B ?= configs/preregistration_study_b.yaml

study-b:
	$(PYTHON) scripts/run_study_b.py --output-dir $(STUDY_B) --workers $(WORKERS)

analyze-study-b:
	$(PYTHON) scripts/analyze.py --results-dir $(STUDY_B) --output-dir $(STUDY_B) --prereg $(PREREG_B)
	$(PYTHON) scripts/study_b_mechanistic.py --results-dir $(STUDY_B)

report:
	$(PYTHON) scripts/make_report.py --results-dir $(RESULTS)
	$(PYTHON) scripts/render_docs.py --results-dir $(RESULTS)

all: probe suite federated analyze report

# Deliberately NOT $(RESULTS): a smoke run is a two-seed, two-model, 1024-step
# toy, and writing it over results/all_runs.csv would replace the committed
# confirmatory evidence with it.  SMOKE_DIR is gitignored.
SMOKE_DIR ?= results_smoke

smoke:
	$(PYTHON) scripts/run_suite.py --smoke --output-dir $(SMOKE_DIR)
	$(PYTHON) scripts/run_federated.py --smoke --output-dir $(SMOKE_DIR)
	$(PYTHON) scripts/analyze.py --results-dir $(SMOKE_DIR) --output-dir $(SMOKE_DIR)
	$(PYTHON) scripts/make_report.py --results-dir $(SMOKE_DIR) --output $(SMOKE_DIR)/RESULTS.md

# The gates CI enforces, none of which writes anything: both studies' committed
# tables and claims match a fresh analysis, Study B's mechanistic endpoints match
# a fresh build, RESULTS.md matches a fresh report, and every generated region and
# <!--v:...--> span in README.md and docs/*.md matches a fresh render of BOTH
# studies.
#
# The render gate is the one an adversarial audit proved was missing: two
# falsified headline figures in README.md survived the whole suite because
# nothing regenerated them.
check: check-study-b
	$(PYTHON) scripts/analyze.py --results-dir $(RESULTS) --output-dir $(RESULTS) --check
	$(PYTHON) scripts/make_report.py --results-dir $(RESULTS) --check
	$(PYTHON) scripts/render_docs.py --results-dir $(RESULTS) --check

# Study B's own gate.  README.md and docs/STUDY_B.md cite Study B through the
# same generated regions, so its tables, claims and mechanistic endpoints have to
# be pinned to results/study_b/all_runs.csv exactly as Study A's are to its own.
check-study-b:
	$(PYTHON) scripts/analyze.py --results-dir $(STUDY_B) --output-dir $(STUDY_B) --prereg $(PREREG_B) --check
	$(PYTHON) scripts/study_b_mechanistic.py --results-dir $(STUDY_B) --check

clean:
	rm -rf $(RESULTS)/raw $(RESULTS)/checkpoints $(SMOKE_DIR) .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
