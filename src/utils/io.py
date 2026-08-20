"""Small, centralised input/output helpers for pipeline scripts."""

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_config(path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Load the main YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_threshold_config(path: str | Path = "config/thresholds.yaml") -> dict[str, Any]:
    """Load the OEM threshold configuration separately from model settings."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame to CSV and create missing parent directories.

    Keeping directory creation here means pipeline scripts do not each need to
    repeat filesystem setup logic.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a processed CSV and parse its primary timestamp column."""
    return pd.read_csv(path, parse_dates=["Timestamp"])
