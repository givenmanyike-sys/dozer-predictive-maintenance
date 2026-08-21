"""Simple business and statistical baselines."""

import pandas as pd


def threshold_baseline(df: pd.DataFrame, critical_count_threshold: int = 2) -> pd.Series:
    """Flag rows where at least N current critical threshold breaches occur.

    Args:
        df (pd.DataFrame): Feature DataFrame containing '__critical_breach' columns.
        critical_count_threshold (int): Minimum simultaneous breaches required. Defaults to 2.

    Returns:
        pd.Series: Binary alarm indicator series.
    """
    critical_columns = [c for c in df.columns if c.endswith("__critical_breach")]
    if not critical_columns:
        return pd.Series(0, index=df.index, dtype=int)
    return (df[critical_columns].sum(axis=1) >= critical_count_threshold).astype(int)


def persistence_baseline(df: pd.DataFrame, warning_count_threshold: int = 3) -> pd.Series:
    """Flag rows where recent warning breach counts indicate persistence.

    Args:
        df (pd.DataFrame): Feature DataFrame containing '__warning_count_24' columns.
        warning_count_threshold (int): Minimum count of historical breaches. Defaults to 3.

    Returns:
        pd.Series: Binary alarm indicator series.
    """
    count_columns = [c for c in df.columns if "__warning_count_24" in c]
    if not count_columns:
        return pd.Series(0, index=df.index, dtype=int)
    return (df[count_columns].sum(axis=1) >= warning_count_threshold).astype(int)
