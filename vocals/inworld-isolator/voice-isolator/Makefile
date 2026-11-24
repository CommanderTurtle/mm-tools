# Copyright 2026 Inworld AI — Apache-2.0
.DEFAULT_GOAL := help

PYTHON ?= .venv/bin/python
STYLE_PATHS := src bench/scripts bench/wrappers docs demo tests
ISORT_PATHS := src bench/scripts docs demo tests
# Benchmark source bytes identify the scorer and the rendered outputs.
FORMAT_PATHS := src docs demo tests

.PHONY: help install test lint format format-check

help: ## Show the available commands
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*## / {printf "%-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Python 3.11 lints as CI's 3.10 does and runs every test on a current Mac.
# From 3.12 pylint reads quotes inside f-strings differently and reports
# inconsistent-quotes throughout; on 3.10 the newest SciPy that installs does
# not load on recent macOS, which fails the benchmark tests that import it.
install: ## Create .venv on Python 3.11, or reuse one, and install the dev, eval and docs extras (requires uv)
	@test -x .venv/bin/python || uv venv --python 3.11 .venv
	uv pip install --python $(PYTHON) -e ".[dev,eval,docs]"

test: ## Run the CPU tests and report optional and golden-parity skips
	$(PYTHON) -m pytest -q -rs

lint: ## Check Google Python style and tracked-file hygiene
	PYTHONPATH=src $(PYTHON) -m pylint --rcfile=.pylintrc inworld
	$(PYTHON) -m pylint --rcfile=.pylintrc $(filter-out src,$(STYLE_PATHS))
	@if git grep -nE '(^|[^[:alnum:]_/.])/(home|inworld|mnt)/' -- . ':!AGENTS.md' ':!Makefile' ':!.github/workflows/ci.yml'; then exit 1; fi

format: ## Format Python, preserving the benchmark's digest-bound source files
	$(PYTHON) -m isort $(FORMAT_PATHS)
	$(PYTHON) -m yapf --in-place --recursive --style=.style.yapf $(FORMAT_PATHS)

format-check: ## Check all Python formatting without changing files
	$(PYTHON) -m yapf --diff --recursive --style=.style.yapf $(STYLE_PATHS)
	$(PYTHON) -m isort --check-only --diff $(ISORT_PATHS)
