# Production Design

## Proposed flow

```text
Machine telemetry
      |
      v
Data ingestion
      |
      v
Data quality checks
      |
      v
Feature generation
      |
      v
Model scoring
      |
      v
Alert decision layer
      |
      +------------------+
      |                  |
      v                  v
Maintenance alert    Monitoring store
      |                  |
      v                  v
Maintenance plan    Data/model drift
```

## Alerting

The model should return a probability and an alert state. The alert state should be based on a validated probability threshold and, where appropriate, a persistence rule or alert cooldown to avoid repeated notifications for the same event.

## Monitoring

Production monitoring should cover:

- missing sensor rates
- sensor range violations
- feature distribution drift
- prediction distribution drift
- alert volume per machine
- false alert rate
- warning lead time
- realised failure rate
- model performance once outcomes become available

## Retraining

Retraining should be triggered by a combination of scheduled review, fleet changes, material data drift and evidence of model performance decay. New failure events should be incorporated only after their timestamps and maintenance outcomes have been validated.

## Governance

Every production model should have a version, training data snapshot, configuration, validation results and deployment date. Changes should be reviewed through pull requests and promoted through dev, test and prod environments.
