# Dozer Predictive Maintenance

> **Production-minded early warning predictive maintenance system for a four-machine continuous mining dozer fleet.**

---

## 1. Executive Summary & Objective

In continuous surface mining operations, unscheduled equipment breakdowns cause costly production downtime, secondary component damage, and safety hazards. 

The primary business objective is to build a **leakage-safe early warning classification system** that predicts whether a mining dozer will experience an **Unplanned Mechanical Failure within the next 24 operating hours**:

$$\mathbb{P}(\text{Unplanned Failure} \in [t+1, t+24] \mid \mathcal{H}_t)$$

where $\mathcal{H}_t$ represents the telemetry history strictly observed up to operating hour $t$.

The system prioritizes **actionable advance warning time** while controlling operational false alarm fatigue.

---

## 2. Fleet Telemetry & Ground-Truth Event Audit

The dataset spans 6 months of continuous hourly telemetry across four mining dozers (`DZ-104`, `DZ-118`, `DZ-127`, `DZ-142`), totaling **6,602 observed operating records**.

```
┌─────────────┬─────────────────────┬───────────────────────┬───────────────────────────────┐
│ Machine ID  │ Operating Records   │ Logged Event          │ Pipeline Target Treatment     │
├─────────────┼─────────────────────┼───────────────────────┼───────────────────────────────┤
│ DZ-127      │ 1,640 hrs           │ Unplanned Failure     │ Positive Class (Hydraulics)   │
│ DZ-118      │ 1,671 hrs           │ Unplanned Failure     │ Positive Class (Engine)       │
│ DZ-142      │ 1,595 hrs           │ False Alarm           │ Filtered (Normal Operation)   │
│ DZ-104      │ 1,696 hrs           │ Scheduled Maintenance │ Filtered (Planned Service)    │
└─────────────┴─────────────────────┴───────────────────────┴───────────────────────────────┘
```

### Critical Ground-Truth vs. Domain Distinction
* **Ground Truth**: Only genuine catastrophic breakdowns (`Category == "Unplanned Failure"`) are labeled as positive failure targets.
* **OEM Thresholds as Features**: OEM warning/critical thresholds are treated as domain-informed feature inputs, never as the target definition. Breaches happen frequently during normal peak loading and do not inherently constitute physical failure.
* **Extreme Class Imbalance**: With 48 failure-window hours out of 6,602 total observations, the base failure prevalence is **~0.73% (1 : 137 imbalance)**.

---

## 3. Core Principles & Leakage Prevention

1. **Operating Time over Calendar Time**: Telemetry is recorded when machines run (SMR). Prediction horizons are measured in observed operating hours.
2. **Machine-Isolated Feature Engineering**: All rolling windows, slopes, deltas, and lags group strictly by `Machine ID` so signals never bleed across dozers.
3. **Strict Historical Shift (`shift(1)`)**: Sensor observations at time $t$ are excluded from historical rolling mean/std features at time $t$.
4. **Chronological Out-of-Fold Validation**: Random train/test splits are strictly prohibited. Models are trained on past historical folds and evaluated forward in time on unseen machine failure events.
5. **In-Pipeline Imputation & Scaling**: Preprocessing transformers are encapsulated in Scikit-Learn pipelines and fitted strictly on training folds.

---

## 4. Key Empirical Results

### Chronological Out-of-Fold Classifier Performance

Evaluated on genuine failure event folds (training on `DZ-127` historical breakdown and testing out-of-fold on `DZ-118`):

| Model Architecture | PR-AUC (Average Precision) | ROC-AUC | Operational Alert Recall ($\tau = 0.10$) | Default Recall ($\tau = 0.50$) |
| :--- | :---: | :---: | :---: | :---: |
| **HistGradientBoosting** | **0.7083** | **0.5625** | **Alerts Triggered (Early Warning)** | 0.00 |
| **Random Forest** | **0.6667** | **0.5000** | **Alerts Triggered (Early Warning)** | 0.00 |
| **Logistic Regression** | **0.6026** | **0.2917** | **Alerts Triggered (Early Warning)** | 0.00 |

> **Operational Insight**: Under 0.73% class prevalence, standard classification cutoffs ($\tau = 0.50$) yield zero alerts. Calibrating the operational threshold to $\tau = 0.10$ activates early warnings with actionable lead time while maintaining manageable false alarm rates.

### Sensor Time-Series Forecasting (6-Hour Horizon)

Level + Trend Exponential Smoothing consistently outperforms naive persistence across all machines:
* **Hydraulic Oil Temp Max**: MAE **6.35–7.79 °C** (vs. 8.36–10.78 °C naive, **~25% error reduction**)
* **Coolant Temp Max**: MAE **5.06–5.56 °C** (vs. 7.10–7.56 °C naive, **~28% error reduction**)
* **Blow-by Pressure Max**: MAE **0.34–0.43** (vs. 0.44–0.47 naive, **~20% error reduction**)

---

## 5. Repository Structure

```text
dozer-predictive-maintenance/
├── config/                     # Central YAML configurations
│   ├── config.yaml             # Core pipeline contracts, horizons & parameters
│   ├── features.yaml           # Monitored sensor channels
│   └── thresholds.yaml         # OEM warning & critical parameter thresholds
├── data/                       # Telemetry data (raw immutable workbook & processed tables)
├── docs/                       # Engineering & operational documentation
│   ├── findings.md             # In-depth empirical findings & fleet failure case studies
│   ├── methodology.md          # Formal problem framing, leakage controls & metric choices
│   ├── production_design.md    # Target architecture, alert cooldowns & drift monitoring
│   └── git_workflow.md         # Branching model (dev -> test -> prod -> main)
├── notebooks/                  # Interactive visual diagnostic workflows
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_target_definition.ipynb
│   └── 04_model_diagnostics.ipynb
├── outputs/                    # Exported figures, metrics CSVs, and diagnostic plots
├── scripts/                    # Thin, reproducible pipeline entry points (01 to 08)
├── src/                        # Modular library (ingestion, features, models, evaluation)
├── tests/                      # Pytest unit regression suite
├── Makefile & pyproject.toml   # Project build, linting (ruff/flake8) and test automation
└── README.md
```

---

## 6. Quick Start & Pipeline Execution

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/givenmanyike-sys/dozer-predictive-maintenance.git
cd dozer-predictive-maintenance

# Create and activate virtual environment (Python 3.10+)
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell (or: source .venv/bin/activate on Linux/Mac)

# Install dependencies and local package in editable mode
pip install -r requirements.txt
pip install -e .
```

### 2. Run the Full End-to-End Pipeline
Execute the sequential pipeline stages:

```bash
# 01. Validate raw data schema, timestamps, and key uniqueness
python -m scripts.01_validate_raw_data

# 02. Build modeling base dataset with 24h early warning targets
python -m scripts.02_build_dataset

# 03. Run statistical exploratory data analysis
python -m scripts.03_run_eda

# 04. Generate leakage-safe rolling, slope, delta & threshold features
python -m scripts.04_build_features

# 05. Run rolling-origin time-series forecasting backtests
python -m scripts.05_select_time_series_model

# 06. Train candidate baseline classifiers
python -m scripts.06_train_models

# 07. Run chronological event-centric cross-validation & evaluation
python -m scripts.07_evaluate_models

# 08. Run telemetry missingness & sensor drift robustness experiments
python -m scripts.08_run_robustness
```

### 3. Run Quality Checks
```bash
# Run unit tests
pytest -v

# Run linter
ruff check src scripts tests
```

---

## 7. Interactive Notebooks Walkthrough

For deep visual exploration and stakeholder reviews:

1. **[`01_data_understanding.ipynb`](file:///c:/Users/given/Downloads/dozer-predictive-maintenance/dozer_predictive_maintenance/notebooks/01_data_understanding.ipynb)**: Data contracts, operating hour tracking, and ground-truth event mapping.
2. **[`02_eda.ipynb`](file:///c:/Users/given/Downloads/dozer-predictive-maintenance/dozer_predictive_maintenance/notebooks/02_eda.ipynb)**: Multi-panel sensor degradation trajectories, correlation heatmaps, and outlier boxplots.
3. **[`03_target_definition.ipynb`](file:///c:/Users/given/Downloads/dozer-predictive-maintenance/dozer_predictive_maintenance/notebooks/03_target_definition.ipynb)**: Target window sensitivity, class balance dynamics, and feature importances.
4. **[`04_model_diagnostics.ipynb`](file:///c:/Users/given/Downloads/dozer-predictive-maintenance/dozer_predictive_maintenance/notebooks/04_model_diagnostics.ipynb)**: Out-of-fold probability distributions, ROC/PR curves, confusion matrices, and threshold tuning.

---

## 8. Target Production Architecture

```text
Dozer Fleet Telemetry (SMR & Sensors)
              │
              ▼
    Ingestion & Data Quality Checks (scripts.01)
              │
              ▼
    Leakage-Safe Feature Pipeline (Grouped Shifts & Persistence)
              │
              ▼
    HistGradientBoosting Scoring Pipeline
              │
              ▼
    Decision Layer (Calibrated Threshold tau = 0.10 + Cooldown)
       ┌──────┴──────────────────────────┐
       ▼                                 ▼
Maintenance Alert (CMMS Integration)  Drift & Health Monitoring (PSI)
```

* **Alert Cooldown**: Enforces persistence rules (e.g., minimum 2 consecutive alert hours) to avoid notification spam.
* **Shadow Fleet Deployment**: Recommended initial 60-day shadow deployment to validate warning lead times and gather additional failure episodes before automated work-order dispatch.

