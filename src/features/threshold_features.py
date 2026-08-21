"""Domain-informed features derived from OEM warning and critical thresholds."""

import pandas as pd


def add_threshold_features(
    df: pd.DataFrame,
    thresholds: dict,
    windows: list[int],
) -> pd.DataFrame:
    """Add threshold breach and historical persistence features.

    OEM thresholds are treated as domain knowledge, not as failure labels. A
    breach can occur during normal operation, so using it as the target would
    create a circular definition of failure.

    Current breach indicators describe the current sensor state. Historical
    count features use a one-step shift so that a feature at time ``t`` only
    summarises threshold behaviour observed before ``t``.

    Args:
        df (pd.DataFrame): Input telemetry DataFrame.
        thresholds (dict): OEM parameter threshold mapping with direction/warning/critical.
        windows (list[int]): Historical window periods (in operating hours).

    Returns:
        pd.DataFrame: DataFrame augmented with breach indicators and count features.

    Raises:
        ValueError: If an unsupported threshold direction is encountered.
    """
    result = df.copy()

    for sensor, spec in thresholds.items():
        if sensor not in result.columns:
            continue

        direction = spec["direction"]
        warning = float(spec["warning"])
        critical = float(spec["critical"])

        if direction == "HIGH":
            warning_breach = result[sensor] > warning
            critical_breach = result[sensor] > critical
        elif direction == "LOW":
            warning_breach = result[sensor] < warning
            critical_breach = result[sensor] < critical
        else:
            raise ValueError(f"Unsupported threshold direction: {direction}")

        # These indicators are features only. The actual target is constructed
        # separately from the ground-truth event log.
        result[f"{sensor}__warning_breach"] = warning_breach.astype(float)
        result[f"{sensor}__critical_breach"] = critical_breach.astype(float)

        for window in windows:
            for level_name, breach in (
                ("warning", warning_breach),
                ("critical", critical_breach),
            ):
                # Excluding the current row prevents a sensor value at time t
                # from being counted as historical evidence at the same time t.
                historical = breach.groupby(result["Machine ID"], sort=False).shift(1).fillna(0)
                counts = (
                    historical.groupby(result["Machine ID"], sort=False)
                    .rolling(window=window, min_periods=1)
                    .sum()
                    .reset_index(level=0, drop=True)
                )
                result[f"{sensor}__{level_name}_count_{window}"] = counts

    return result
