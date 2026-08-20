"""Sensor drift simulation and distribution-monitoring utilities."""

import numpy as np
import pandas as pd


def inject_additive_drift(
    df: pd.DataFrame,
    sensors: list[str],
    multiplier: float,
) -> pd.DataFrame:
    """Apply proportional sensor drift to selected columns for sensitivity tests.

    The term additive is retained for compatibility with the existing project
    interface, but the implementation is proportional: a value is multiplied
    by ``1 + multiplier``. This approximates a systematic calibration shift.
    """
    result = df.copy()

    for sensor in sensors:
        if sensor in result.columns:
            result[sensor] = result[sensor] * (1.0 + multiplier)

    return result


def population_stability_index(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10,
) -> float:
    """Calculate the Population Stability Index between two distributions.

    PSI is useful as a production monitoring signal because it compares the
    distribution observed during model development with the distribution seen
    later in production. It is a drift indicator, not proof that model
    performance has deteriorated.
    """
    if bins < 2:
        raise ValueError("bins must be at least 2.")

    expected = expected.dropna()
    actual = actual.dropna()

    if expected.empty or actual.empty:
        return float("nan")

    # Quantile bins are based only on the reference distribution. This keeps the
    # definition of the monitoring bins stable when actual production data move.
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)

    expected_pct = np.clip(expected_counts / expected_counts.sum(), 1e-6, None)
    actual_pct = np.clip(actual_counts / actual_counts.sum(), 1e-6, None)

    return float(
        np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    )
