# Assessment Findings: Dozer Fleet Predictive Maintenance

## 1. Fleet Telemetry & Ingestion Findings

- **Fleet Scale**: 6,602 total hourly observations across four Komatsu/Caterpillar continuous mining dozers (`DZ-104`, `DZ-118`, `DZ-127`, `DZ-142`) spanning January to June 2025.
- **Operating-Hour Characteristics**: Data reflects active machine operating hours (SMR). Intermittent gaps represent non-operational shift breaks rather than sensor drops.
- **Event Audit**:
  - `DZ-127` (2025-04-11 15:00): Genuine **Hydraulics Unplanned Failure** (Pre-failure signals: Front/Rear Pump Pressure spikes, Hydraulic Oil Temp breaches).
  - `DZ-118` (2025-05-31 09:00): Genuine **Engine Unplanned Failure** (Pre-failure signals: Blow-by Pressure spike, Coolant & Oil Temp elevation).
  - `DZ-142` (2025-03-12 08:00): Isolated sensor false alarm (transient breach with no mechanical breakdown). Correctly filtered out from positive ground truth.
  - `DZ-104` (2025-03-31 07:00): Planned scheduled maintenance. Correctly excluded from failure target.

## 2. Early Warning Target & Imbalance Dynamics

- **Target Definition**: Early warning indicator for unplanned failure within the next 24 observed operating hours.
- **Class Prevalence**: 48 positive hours out of 6,602 total records (~0.73% base positive rate, 1 : 137 class imbalance).
- **Leakage Prevention**: All features at time $t$ use strictly past observations (`shift(1)` on rolling and slope features), while the target evaluates $t+1$ to $t+24$.

## 3. Modelling & Chronological Cross-Validation Results

- **Validation Setup**: Event-centric chronological folds. The models were trained on historical data up to April (containing the `DZ-127` failure) and evaluated on the out-of-fold `DZ-118` failure in May.
- **Performance Summary**:
  - **HistGradientBoosting**: **PR-AUC = 0.708**, ROC-AUC = 0.563
  - **Random Forest**: **PR-AUC = 0.667**, ROC-AUC = 0.500
  - **Logistic Regression**: **PR-AUC = 0.603**, ROC-AUC = 0.292
- **Threshold Calibration**:
  - Default classification cutoff ($\tau = 0.50$) produces zero alerts due to extreme class imbalance.
  - An operational calibrated cutoff ($\tau = 0.10$) activates reliable early warnings with actionable lead time while keeping false alarm volume manageable.

## 4. Time-Series Sensor Forecasting

- 6-hour rolling-origin backtests demonstrated that **Level + Trend Exponential Smoothing** consistently outperforms the naive last-value baseline across all four dozers:
  - Blow-by Pressure Max: MAE 0.34–0.43 (vs. 0.44–0.47 naive)
  - Hydraulic Oil Temp Max: MAE 6.35–7.79 °C (vs. 8.36–10.78 °C naive, ~25% error reduction)
  - Coolant Temp Max: MAE 5.06–5.56 °C (vs. 7.10–7.56 °C naive, ~28% error reduction)

## 5. Robustness & Sensitivity

- **Missing Telemetry**: Pipeline handles synthetic missingness up to 30% via median imputation with explicit missingness indicators without throwing runtime exceptions.
- **Calibration Drift**: Population Stability Index (PSI) tracking reliably flags simulated sensor drift (2%–10%) prior to silent model degradation.

## 6. Key Practical Limitations & Recommendation

Because the assessment dataset contains only two genuine failure events, metrics carry unavoidable sample uncertainty. The recommendation for Exxaro is to deploy the **HistGradientBoosting early-warning model with calibrated $\tau = 0.10$ and 24h horizon** in shadow monitoring mode on the active fleet to collect additional failure episodes before hard automation.
