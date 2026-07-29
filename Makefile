# Reproduction pipeline.  `make all` targets under 45 minutes on 4 CPU cores.
#
# Every target is a thin wrapper around a script with a real CLI; nothing here
# hides a path or a hyperparameter.  The matrix lives in configs/suite.yaml and
# configs/federated.yaml, and nowhere else.

PYTHON ?= python
RESULTS ?= results
WORKERS ?= 4

.PHONY: help install test test-fast probe suite federated analyze report all smoke check clean

help:
	@echo "install     pip install -r requirements.txt"
	@echo "test        pytest -q (full unit suite)"
	@echo "test-fast   pytest -q -m 'not slow'"
	@echo "probe       measure real steps/s on this box"
	@echo "suite       run the confirmatory matrix"
	@echo "federated   run the topology study"
	@echo "analyze     build tables, figures and claims.json"
	@echo "report      regenerate RESULTS.md and every generated block in README/docs"
	@echo "all         probe suite federated analyze report"
	@echo "smoke       tiny end-to-end run (the CI budget), into \$$(SMOKE_DIR)"
	@echo "check       verify committed tables, RESULTS.md and README/docs are current"

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

# The three gates CI enforces: committed tables match a fresh analysis,
# RESULTS.md matches a fresh report, and every generated region and <!--v:...-->
# span in README.md and docs/*.md matches a fresh render.  None writes anything.
#
# The third gate is the one an adversarial audit proved was missing: two
# falsified headline figures in README.md survived the whole suite because
# nothing regenerated them.
check:
	$(PYTHON) scripts/analyze.py --results-dir $(RESULTS) --output-dir $(RESULTS) --check
	$(PYTHON) scripts/make_report.py --results-dir $(RESULTS) --check
	$(PYTHON) scripts/render_docs.py --results-dir $(RESULTS) --check

clean:
	rm -rf $(RESULTS)/raw $(RESULTS)/checkpoints $(SMOKE_DIR) .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
