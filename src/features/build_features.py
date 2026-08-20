"""Orchestration layer for the predictive maintenance feature pipeline."""

import pandas as pd

from src.features.cleaning import prepare_telemetry
from src.features.rolling_features import (
    add_change_features,
    add_lag_features,
    add_rolling_features,
    add_slope_features,
)
from src.features.threshold_features import add_threshold_features


def build_features(
    df: pd.DataFrame,
    sensors: list[str],
    thresholds: dict,
    rolling_windows: list[int],
    slope_windows: list[int],
    lag_periods: list[int],
    threshold_windows: list[int],
    min_periods: int = 3,
) -> pd.DataFrame:
    """Build the complete feature table from raw prepared telemetry.

    This function intentionally acts as an orchestration layer rather than
    containing the individual feature formulas. Keeping each transformation
    separate makes it easier to test, review and replace one feature family
    without changing the rest of the pipeline.
    """
    # Normalise ordering and numeric types before any grouped temporal feature
    # is calculated. All downstream feature functions assume this contract.
    result = prepare_telemetry(df)

    # The order here is deliberate. Features are all constructed from the
    # original historical sensor values, rather than recursively using newly
    # created features as inputs to later feature families.
    result = add_lag_features(result, sensors, lag_periods)
    result = add_change_features(result, sensors, lag_periods)
    result = add_rolling_features(result, sensors, rolling_windows, min_periods)
    result = add_slope_features(result, sensors, slope_windows)
    result = add_threshold_features(result, thresholds, threshold_windows)

    return result
