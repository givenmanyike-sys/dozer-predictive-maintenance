# Findings

This document is intentionally a template until the full pipeline has been executed and the results have been reviewed.

## Data findings

- Document telemetry coverage by machine.
- Document missingness and irregular operating-hour sampling.
- Document failure and maintenance event timing.
- Describe the DZ-142 false alarm case.

## Target findings

- Compare candidate warning horizons.
- Explain why the selected horizon provides an appropriate maintenance window.
- Demonstrate that scheduled maintenance and false alarms are excluded from the positive class.

## Modelling findings

- Compare threshold, persistence, logistic regression and tree-based baselines.
- Report temporal cross-validation results.
- Report PR-AUC, recall, precision, false positive rate and event-level warning time.

## Limitations

The very small number of genuine failures is the primary limitation. The final report must avoid presenting the model as production-ready without external validation on additional historical failure events.
