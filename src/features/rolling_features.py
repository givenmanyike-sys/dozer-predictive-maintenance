"""Temporal feature engineering utilities.

Rolling statistics in this module are deliberately constructed from past
observations. Slope and change features also use the current observation,
which is available when a prediction is made at time ``t``.
"""

import pandas as pd


def _historical_rolling(
    series: pd.Series,
    groups: pd.Series,
    window: int,
    function: str,
    min_periods: int,
) -> pd.Series:
    """Calculate a grouped rolling statistic using historical observations only.

    The initial ``shift(1)`` is the critical leakage-prevention step. Without
    it, the value at the current timestamp would be included in its own rolling
    feature. That would make the feature look predictive during development
    while changing its meaning in a real-time prediction system.

    Args:
        series (pd.Series): Sensor series to aggregate.
        groups (pd.Series): Grouping series (Machine ID).
        window (int): Size of the rolling observation window.
        function (str): Aggregation function name ('mean' or 'std').
        min_periods (int): Minimum valid observations required in window.

    Returns:
        pd.Series: Calculated rolling statistics series.
    """
    historical = series.groupby(groups, sort=False).shift(1)
    rolling = historical.groupby(groups, sort=False).rolling(
        window=window,
        min_periods=min_periods,
    )
    return getattr(rolling, function)().reset_index(level=0, drop=True)


def add_lag_features(df: pd.DataFrame, sensors: list[str], lags: list[int]) -> pd.DataFrame:
    """Add previous-observation lag features independently for each machine.

    A lag of one represents the previous observed machine-hour. Grouping by
    machine is essential because the previous row globally could belong to a
    different dozer.

    Args:
        df (pd.DataFrame): Input telemetry DataFrame.
        sensors (list[str]): List of sensor column names.
        lags (list[int]): List of lag steps to generate.

    Returns:
        pd.DataFrame: DataFrame augmented with lag features.
    """
    result = df.copy()
    features = {}

    for sensor in sensors:
        if sensor not in result.columns:
            # Configuration may reference a sensor that is absent in a revised
            # workbook. Skipping it allows the pipeline to continue while EDA or
            # validation can report the missing field separately.
            continue

        grouped = result.groupby("Machine ID", sort=False)[sensor]
        for lag in lags:
            features[f"{sensor}__lag_{lag}"] = grouped.shift(lag)

    return pd.concat([result, pd.DataFrame(features, index=result.index)], axis=1)


def add_rolling_features(
    df: pd.DataFrame,
    sensors: list[str],
    windows: list[int],
    min_periods: int = 3,
) -> pd.DataFrame:
    """Add historical rolling mean and standard deviation features.

    The rolling windows are measured in observed operating rows rather than
    assuming that every calendar hour contains a telemetry observation. This
    matches the supplied operating-hour style of the assessment data.

    Args:
        df (pd.DataFrame): Input telemetry DataFrame.
        sensors (list[str]): List of sensor column names.
        windows (list[int]): List of window sizes in operating hours.
        min_periods (int): Minimum observations to compute statistic. Defaults to 3.

    Returns:
        pd.DataFrame: DataFrame augmented with rolling mean and std features.
    """
    result = df.copy()
    features = {}
    groups = result["Machine ID"]

    for sensor in sensors:
        if sensor not in result.columns:
            continue

        series = result[sensor]
        for window in windows:
            for statistic in ("mean", "std"):
                features[f"{sensor}__roll_{statistic}_{window}"] = _historical_rolling(
                    series,
                    groups,
                    window,
                    statistic,
                    min_periods,
                )

    return pd.concat([result, pd.DataFrame(features, index=result.index)], axis=1)


def add_slope_features(df: pd.DataFrame, sensors: list[str], windows: list[int]) -> pd.DataFrame:
    """Estimate historical sensor change per operating-hour window.

    A simple end-to-end slope is used as an interpretable degradation signal.
    The current value is compared with a value from ``window`` observations in
    the past. The calculation remains machine-specific through grouped shifts.

    Args:
        df (pd.DataFrame): Input telemetry DataFrame.
        sensors (list[str]): List of sensor column names.
        windows (list[int]): List of window sizes for slope calculation.

    Returns:
        pd.DataFrame: DataFrame augmented with slope features.
    """
    result = df.copy()
    features = {}

    for sensor in sensors:
        if sensor not in result.columns:
            continue

        grouped = result.groupby("Machine ID", sort=False)[sensor]
        for window in windows:
            features[f"{sensor}__slope_{window}"] = (
                result[sensor] - grouped.shift(window)
            ) / float(window)

    return pd.concat([result, pd.DataFrame(features, index=result.index)], axis=1)


def add_change_features(df: pd.DataFrame, sensors: list[str], lags: list[int]) -> pd.DataFrame:
    """Add signed changes from selected historical sensor lags.

    Args:
        df (pd.DataFrame): Input telemetry DataFrame.
        sensors (list[str]): List of sensor column names.
        lags (list[int]): List of lag periods for delta calculation.

    Returns:
        pd.DataFrame: DataFrame augmented with signed delta features.
    """
    result = df.copy()
    features = {}

    for sensor in sensors:
        if sensor not in result.columns:
            continue

        grouped = result.groupby("Machine ID", sort=False)[sensor]
        for lag in lags:
            # A positive delta means the current reading is above the historical
            # reference. The sign therefore remains meaningful to degradation
            # analysis and is not converted to an absolute value.
            features[f"{sensor}__delta_{lag}"] = result[sensor] - grouped.shift(lag)

    return pd.concat([result, pd.DataFrame(features, index=result.index)], axis=1)
