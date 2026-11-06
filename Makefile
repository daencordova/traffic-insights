# ==============================================================================
# Traffic Insights — Developer Commands
# ==============================================================================

.DEFAULT_GOAL := help
PYTHON ?= python3
VENV   ?= .venv

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Setup ────────────────────────────────────────────────────────────────────
.PHONY: venv
venv: ## Create virtual environment
	$(PYTHON) -m venv $(VENV)

.PHONY: install
install: ## Install runtime dependencies (editable)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

.PHONY: install-gpu
install-gpu: ## Install with GPU (torch) extras
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev,gpu]"

.PHONY: install-precommit
install-precommit: ## Install pre-commit hooks
	pre-commit install

# ── Lint & Format ───────────────────────────────────────────────────────────
.PHONY: lint
lint: ## Run ruff linter
	ruff check src tests

.PHONY: format
format: ## Apply ruff formatter
	ruff format src tests
	ruff check --fix src tests

.PHONY: typecheck
typecheck: ## Run mypy
	mypy src/

.PHONY: check
check: lint typecheck ## Run all static checks

# ── Test ─────────────────────────────────────────────────────────────────────
.PHONY: test
test: ## Run tests
	pytest

.PHONY: test-fast
test-fast: ## Run tests without coverage
	pytest --no-cov -x

# ── Run ──────────────────────────────────────────────────────────────────────
.PHONY: run
run: ## Run the application (uses config.yaml)
	$(PYTHON) -m src --config config.yaml

.PHONY: run-sync
run-sync: ## Run in synchronous (legacy) mode
	$(PYTHON) -m src --config config.yaml --sync

.PHONY: run-cpu
run-cpu: ## Run forced in CPU mode
	$(PYTHON) -m src --config config.yaml --cpu-mode

# ── Maintenance ──────────────────────────────────────────────────────────────
.PHONY: clean
clean: ## Remove build artifacts and caches
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +