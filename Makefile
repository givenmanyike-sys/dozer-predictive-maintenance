PYTHON ?= python

.PHONY: install test lint validate eda dataset features forecast train evaluate robustness clean

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src scripts tests

validate:
	$(PYTHON) -m scripts.01_validate_raw_data

dataset:
	$(PYTHON) -m scripts.02_build_dataset

eda:
	$(PYTHON) -m scripts.03_run_eda

features: dataset
	$(PYTHON) -m scripts.04_build_features

forecast:
	$(PYTHON) -m scripts.05_select_time_series_model

train: features
	$(PYTHON) -m scripts.06_train_models

evaluate: features
	$(PYTHON) -m scripts.07_evaluate_models

robustness: features
	$(PYTHON) -m scripts.08_run_robustness

clean:
	rm -f data/interim/* data/processed/* models/*.joblib outputs/figures/* outputs/metrics/* outputs/reports/*
