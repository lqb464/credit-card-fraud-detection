.PHONY: help install notebooks train train-all predict test mlflow-ui lint format clean

PYTHON := python
NB_DIR := notebooks

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install all dependencies
	pip install -r requirements.txt
	pip install -e ".[dev,notebooks]"

notebooks:  ## Open all notebooks in Jupyter Lab
	jupyter lab $(NB_DIR)/

eda:  ## Open EDA notebook
	jupyter lab $(NB_DIR)/01_EDA.ipynb

train:  ## Train default model (hist_gradient_boosting)
	$(PYTHON) scripts/train.py

train-all:  ## Train all 4 models sequentially
	$(PYTHON) scripts/train.py --model hist_gradient_boosting
	$(PYTHON) scripts/train.py --model xgboost
	$(PYTHON) scripts/train.py --model lightgbm
	$(PYTHON) scripts/train.py --model logistic_regression

smoke:  ## Quick smoke test (10K rows)
	$(PYTHON) scripts/train.py --smoke-test

predict:  ## Run inference on test set
	$(PYTHON) scripts/predict.py --model-path outputs/models/hist_gradient_boosting_model.joblib

test:  ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing

mlflow-ui:  ## Launch MLflow UI
	mlflow ui --backend-store-uri sqlite:///experiments/mlruns.db --port 5000

lint:  ## Lint with flake8
	flake8 src/ scripts/ tests/ --max-line-length=88 --extend-ignore=E203,W503

format:  ## Format with black + isort
	isort src/ scripts/ tests/
	black src/ scripts/ tests/

clean:  ## Remove outputs (keep data/)
	rm -rf outputs/models/ outputs/reports/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
