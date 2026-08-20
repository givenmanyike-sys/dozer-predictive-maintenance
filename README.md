# Dozer Predictive Maintenance

Production-minded predictive maintenance assessment for a four-machine dozer fleet.

## Objective

Build a leakage-safe early warning system that estimates whether a dozer is likely to experience an **Unplanned Failure within the next N operating hours**. The system must prioritise useful warning time while controlling false alarms.

The project deliberately treats the problem as a **time-series supervised prediction problem**, not as a generic random train/test classification exercise. Sensor forecasting and degradation analysis are included as supporting analyses where they add maintenance value.

## Fleet data

The supplied workbook contains:

- 6 months of hourly telemetry
- 4 dozers: DZ-104, DZ-118, DZ-127 and DZ-142
- Ground-truth failure and maintenance events
- Parameter metadata
- OEM-style warning and critical thresholds

Only `Unplanned Failure` is a positive failure event. `False Alarm` and `Scheduled Maintenance` are not positive labels.

## Repository principles

1. Never use future information to construct a prediction feature.
2. Split and validate data chronologically and by machine.
3. Build the target from the event log, not from OEM thresholds.
4. Treat thresholds as domain-informed features only.
5. Prefer simple, interpretable components before adding complexity.
6. Evaluate operationally, not only with row-level classification metrics.
7. Keep reusable logic in `src/` and thin executable entry points in `scripts/`.
8. Document assumptions, limitations and production implications.

## Repository structure

```text
dozer-predictive-maintenance/
├── config/                 # Experiment and feature configuration
├── data/                   # Raw, interim and processed data
├── docs/                   # Methodology, findings and production design
├── notebooks/              # Exploratory and diagnostic notebooks
├── outputs/                # Generated figures, metrics and reports
├── scripts/                # Thin pipeline entry points
├── src/                    # Reusable project code
│   ├── data/
│   ├── features/
│   ├── targets/
│   ├── models/
│   ├── evaluation/
│   ├── robustness/
│   └── utils/
├── tests/                  # Unit tests
├── models/                 # Saved model artifacts
└── .github/                # Pull request and workflow templates
```

## Quick start

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Validate the supplied workbook:

```bash
python -m scripts.01_validate_raw_data
```

Build the modelling dataset:

```bash
python -m scripts.02_build_dataset
```

Run EDA:

```bash
python -m scripts.03_run_eda
```

Build leakage-safe features:

```bash
python -m scripts.04_build_features
```

Select supporting sensor forecasting baselines:

```bash
python -m scripts.05_select_time_series_model
```

Train candidate models:

```bash
python -m scripts.06_train_models
```

Evaluate them:

```bash
python -m scripts.07_evaluate_models
```

Run robustness experiments:

```bash
python -m scripts.08_run_robustness
```

## Git workflow

The intended branch model is:

```text
prod
  ↑
test
  ↑
dev
  ↑
feature/*
```

Feature branches should be small and focused. Changes are merged into `dev`, promoted to `test` after validation, and promoted to `prod` only after the pull request checklist is satisfied.

Example:

```bash
git checkout dev
git pull

git checkout -b feature/target-definition
```

## Assessment framing

The limited number of genuine failure events is a material statistical limitation. Results should therefore be presented as a proof of concept rather than evidence of production-ready fleet-wide generalisation.

The strongest outcome is a defensible methodology: correct target construction, leakage controls, degradation-aware features, temporal validation, operational metrics, false alarm analysis, robustness testing and a realistic production design.

## Code documentation and maintainability

The project is intentionally written so that another data scientist can take ownership of the pipeline. Reusable functions include docstrings describing their purpose and important assumptions, while inline comments explain decisions that are not obvious from the syntax.

Particular attention is given to documenting:

- temporal and multi-machine leakage prevention
- target construction and the distinction between failures and false alarms
- historical feature windows
- OEM threshold usage as domain features rather than labels
- validation design
- missing-data handling
- model and alert-threshold decisions

See `docs/code_documentation.md` for the project's documentation conventions.
