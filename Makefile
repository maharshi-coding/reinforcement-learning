.PHONY: help install test lint train train-dqn train-ppo train-a2c evaluate dashboard \
       docker-build docker-train docker-evaluate docker-dashboard docker-pipeline docker-test clean

PYTHON  := ./venv/bin/python
PIP     := ./venv/bin/pip
PYTEST  := ./venv/bin/pytest
STREAMLIT := ./venv/bin/streamlit

# ── Help ──────────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ─────────────────────────────────────────────────────
install: ## Create venv and install dependencies
	python3 -m venv venv
	$(PIP) install -r requirements.txt

# ── Quality ───────────────────────────────────────────────────
test: ## Run all tests
	$(PYTEST) tests/ -v --tb=short

lint: ## Run linter (ruff)
	$(PYTHON) -m ruff check airs/ scripts/ tests/

format: ## Auto-format code (ruff)
	$(PYTHON) -m ruff format airs/ scripts/ tests/

# ── Training ──────────────────────────────────────────────────
train: ## Train all algorithms (DQN, PPO, A2C)
	$(PYTHON) scripts/train_universal.py --config configs/default.yaml

train-dqn: ## Train DQN only
	$(PYTHON) scripts/train_universal.py --algorithm dqn --config configs/default.yaml

train-ppo: ## Train PPO only
	$(PYTHON) scripts/train_universal.py --algorithm ppo --config configs/default.yaml

train-a2c: ## Train A2C only
	$(PYTHON) scripts/train_universal.py --algorithm a2c --config configs/default.yaml

train-curriculum: ## Train with curriculum learning
	$(PYTHON) scripts/train_universal.py --curriculum --config configs/default.yaml

# ── Evaluation ────────────────────────────────────────────────
evaluate: ## Evaluate all algorithms across all scenarios
	$(PYTHON) scripts/evaluate_all.py --config configs/default.yaml --output_dir results

# ── Dashboard ─────────────────────────────────────────────────
dashboard: ## Launch Streamlit dashboard
	$(STREAMLIT) run scripts/dashboard.py --server.headless true

# ── Full Pipeline ─────────────────────────────────────────────
pipeline: train evaluate ## Train → Evaluate (full pipeline)
	@echo "✅ Pipeline complete. Run 'make dashboard' to view results."

# ── Docker ────────────────────────────────────────────────────
docker-build: ## Build Docker images
	docker compose build

docker-train: ## Train inside Docker
	docker compose run --rm train

docker-evaluate: ## Evaluate inside Docker
	docker compose run --rm evaluate

docker-dashboard: ## Launch dashboard in Docker (port 8501)
	docker compose up dashboard

docker-pipeline: ## Run full pipeline in Docker (train → evaluate)
	docker compose run --rm pipeline

docker-test: ## Run tests in Docker
	docker compose run --rm test

# ── Cleanup ───────────────────────────────────────────────────
clean: ## Remove cached files and __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
