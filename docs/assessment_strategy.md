# Assessment Strategy

## What this project is demonstrating

The project is designed to demonstrate production-minded data science rather than only model fitting.

The key decisions are:

1. Define the target from ground-truth failure events.
2. Treat only `Unplanned Failure` as a positive outcome.
3. Measure prediction horizons in operating hours because telemetry is only observed while machines operate.
4. Build temporal features from historical information only.
5. Validate chronologically and make rare-event limitations explicit.
6. Compare simple domain baselines before machine-learning models.
7. Use PR-AUC, recall, precision, false positive rate and warning time as core decision metrics.
8. Analyse the known false alarm as a concrete model diagnostic.
9. Test sensitivity to missing telemetry and sensor drift.
10. Describe how the system would be monitored and promoted through dev, test and prod.

## Expected final recommendation

The final recommendation should not simply be the model with the highest metric. It should identify the model and alert policy that provide a defensible balance between early detection and operational false alarms.

Because only two genuine failures are present, the final recommendation should clearly distinguish proof-of-concept evidence from production evidence.
