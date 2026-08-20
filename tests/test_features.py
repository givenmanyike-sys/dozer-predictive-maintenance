"""Regression tests for temporal feature leakage prevention."""

import pandas as pd

from src.features.rolling_features import add_rolling_features


def test_rolling_features_do_not_use_current_observation() -> None:
    """Ensure the rolling feature at time t only uses observations before t."""
    df = pd.DataFrame(
        {
            "Machine ID": ["A"] * 4,
            "Timestamp": pd.date_range("2025-01-01 06:00", periods=4, freq="h"),
            "sensor": [1.0, 2.0, 100.0, 4.0],
        }
    )

    result = add_rolling_features(df, ["sensor"], [2], min_periods=1)

    # If the current value of 100 were included, the mean would be 51.0. The
    # expected value of 1.5 proves that only the preceding two observations are
    # available to the feature at the third row.
    value = result.loc[2, "sensor__roll_mean_2"]
    assert value == 1.5
