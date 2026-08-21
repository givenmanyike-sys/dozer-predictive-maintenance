"""Utilities for loading and normalising the assessment workbook.

The assessment data are supplied as a multi-sheet Excel workbook rather than
as clean, machine-ready tables. This module keeps all workbook-specific
handling in one place so that the rest of the pipeline can work with ordinary
pandas DataFrames.

Keeping this logic isolated is important for maintainability. If the workbook
layout changes, a future maintainer should only need to update this module
rather than search through the modelling code for Excel-specific assumptions.
"""

from pathlib import Path

import pandas as pd

# The workbook contains explanatory rows above the actual table headers.
# These constants document the layout explicitly instead of hiding the
# assumption inside the read_excel calls below.
TELEMETRY_HEADER = 3
EVENT_HEADER = 3
METADATA_HEADER = 3
THRESHOLD_HEADER = 3


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalised column names and empty Excel columns removed.

    Excel files often contain formatting columns that pandas imports as
    ``Unnamed: ...``. Removing those columns at the ingestion boundary keeps
    downstream feature code focused on actual data fields.
    """
    df = df.copy()

    # Strip accidental whitespace so that downstream code can refer to stable
    # column names such as ``Machine ID`` and ``Timestamp``.
    df.columns = [str(column).strip() for column in df.columns]

    # Pandas names blank Excel columns ``Unnamed: ...``. They have no modelling
    # value and can otherwise accidentally be treated as numeric features.
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def load_workbook(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load and lightly normalise the logical tables in the assessment workbook.

    Args:
        path (str | Path): Path to the supplied Excel workbook.

    Returns:
        dict[str, pd.DataFrame]: Dictionary containing 'telemetry', 'events',
            'metadata', and 'thresholds' DataFrames.

    Notes:
        This function performs ingestion and basic structural cleaning only. It
        deliberately does not impute missing values, create model features, or
        construct labels. Those responsibilities belong to later pipeline stages.
    """
    path = Path(path)

    # Explicit sheet names make the ingestion contract obvious. If a sheet is
    # renamed or removed, this function will fail early rather than silently
    # loading the wrong table.
    telemetry = pd.read_excel(path, sheet_name="Fleet Telemetry", header=TELEMETRY_HEADER)
    events = pd.read_excel(path, sheet_name="Failure Events", header=EVENT_HEADER)
    metadata = pd.read_excel(path, sheet_name="Dataset Metadata", header=METADATA_HEADER)
    thresholds = pd.read_excel(path, sheet_name="Thresholds", header=THRESHOLD_HEADER)

    telemetry = _clean_columns(telemetry)
    events = _clean_columns(events)
    metadata = _clean_columns(metadata)
    thresholds = _clean_columns(thresholds)

    # Rows without machine/time keys cannot participate in a time-series
    # analysis. They are removed here so that every later function receives a
    # consistent temporal key.
    telemetry = telemetry.dropna(subset=["Machine ID", "Timestamp"]).copy()
    events = events.dropna(subset=["Machine ID", "Event Timestamp", "Category"]).copy()

    # Convert timestamps once at the ingestion boundary. This avoids subtle
    # differences between strings and pandas timestamps in comparisons later.
    telemetry["Timestamp"] = pd.to_datetime(telemetry["Timestamp"])
    events["Event Timestamp"] = pd.to_datetime(events["Event Timestamp"])

    return {
        "telemetry": telemetry,
        "events": events,
        "metadata": metadata,
        "thresholds": thresholds,
    }
