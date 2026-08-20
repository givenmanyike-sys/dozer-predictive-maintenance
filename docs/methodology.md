# Methodology

## Problem formulation

The primary problem is framed as early warning classification:

> Will this machine experience an Unplanned Failure within the next N operating hours?

The target is generated exclusively from the ground-truth event log. OEM thresholds are never used to create the target.

## Time-series handling

Telemetry is observed during operating hours rather than on a continuous 24/7 grid. The modelling dataset therefore preserves observed operating rows. Historical features are calculated within machine boundaries using past observations only.

Random row-level splitting is prohibited because it can place adjacent observations from the same degradation episode in both training and test sets.

## Feature engineering

Features are grouped into:

- lagged sensor values
- rolling mean, standard deviation, minimum and maximum
- historical slopes
- sensor deltas
- threshold breach counts
- warning and critical persistence indicators

The current observation is shifted out of rolling features so that the feature represents information available before the prediction timestamp.

## Model selection

The candidate hierarchy starts with domain and statistical baselines before moving to tree-based models. The objective is not to maximise complexity. The selected model should provide useful early warning while controlling false alarms.

## Evaluation

Primary metrics are PR-AUC, recall, precision, false positive rate and event-level warning time. ROC-AUC and balanced accuracy are supplementary.

Because the dataset contains only two genuine unplanned failures, model performance estimates have high uncertainty. This limitation must be carried into the final recommendation.
