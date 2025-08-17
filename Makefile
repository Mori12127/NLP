PYTHON = python3
PIP = $(PYTHON) -m pip
VENV ?= .venv
ACTIVATE = . $(VENV)/bin/activate

.DEFAULT_GOAL := help

help:
	@echo "Targets: venv, install, format, lint, typecheck, test, cov, train, eval, all"

venv:
	@test -d $(VENV) || ($(PYTHON) -m venv $(VENV) && $(ACTIVATE) && $(PIP) install -U pip)

install: venv
	$(ACTIVATE) && $(PIP) install -e .[dev]

install-no-venv:
	$(PYTHON) -m pip install --break-system-packages -e .[dev]

format:
	$(ACTIVATE) && ruff check --select I --fix src tests
	$(ACTIVATE) && black src tests
	$(ACTIVATE) && isort src tests

lint:
	$(ACTIVATE) && ruff check src tests

typecheck:
	$(ACTIVATE) && mypy src

pytest_args ?=

test:
	$(ACTIVATE) && pytest -q $(pytest_args)

cov:
	$(ACTIVATE) && pytest --cov=nlp_lab --cov-report=term-missing -q

train:
	$(ACTIVATE) && PYTHONPATH=src $(PYTHON) -m nlp_lab.training.cli train --model baseline

prep:
	$(ACTIVATE) && PYTHONPATH=src $(PYTHON) -m nlp_lab.training.cli prep

eval:
	$(ACTIVATE) && PYTHONPATH=src $(PYTHON) -m nlp_lab.training.cli eval --run models

all: lint typecheck test