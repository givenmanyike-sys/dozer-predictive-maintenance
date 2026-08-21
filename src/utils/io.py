"""Small, centralised input/output helpers for pipeline scripts."""

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_config(path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Load the main YAML configuration file.

    Args:
        path (str | Path): Path to YAML configuration file. Defaults to 'config/config.yaml'.

    Returns:
        dict[str, Any]: Parsed configuration mapping.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_threshold_config(path: str | Path = "config/thresholds.yaml") -> dict[str, Any]:
    """Load the OEM threshold configuration separately from model settings.

    Args:
        path (str | Path): Path to threshold YAML. Defaults to 'config/thresholds.yaml'.

    Returns:
        dict[str, Any]: Parsed threshold mapping.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame to CSV and create missing parent directories.

    Args:
        df (pd.DataFrame): DataFrame to persist.
        path (str | Path): Destination file path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a processed CSV and parse its primary timestamp column.

    Args:
        path (str | Path): Path to CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame with parsed datetime 'Timestamp'.
    """
    return pd.read_csv(path, parse_dates=["Timestamp"])
