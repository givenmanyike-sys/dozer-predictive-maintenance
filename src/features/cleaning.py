"""Data preparation performed immediately before feature engineering."""

import pandas as pd


def prepare_telemetry(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Sort telemetry and coerce sensor columns to numeric values.

    Missing values are intentionally preserved. Imputation belongs inside the
    model training pipeline so that imputation parameters are learned from the
    training data only. Performing imputation here would risk leaking future
    information into validation data.
    """
    df = telemetry.copy()

    # Timestamp conversion ensures that sorting and temporal comparisons use
    # actual datetime values rather than string ordering.
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Each machine is its own time series. Sorting by machine first makes later
    # group-wise lag and rolling operations deterministic and prevents a window
    # from accidentally crossing from one machine into another.
    df = df.sort_values(["Machine ID", "Timestamp"]).reset_index(drop=True)
    df["Machine ID"] = df["Machine ID"].astype(str).str.strip()

    # Sensor columns may contain strings such as blank cells or formatting
    # artefacts. Converting invalid values to NaN lets downstream validation and
    # model pipelines handle them explicitly instead of silently using strings.
    numeric_columns = [c for c in df.columns if c not in {"Machine ID", "Timestamp"}]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df
