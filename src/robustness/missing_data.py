"""Utilities for testing sensitivity to missing telemetry."""

import numpy as np
import pandas as pd


def inject_missingness(df: pd.DataFrame, rate: float, seed: int = 42) -> pd.DataFrame:
    """Randomly mask numeric telemetry at a requested missingness rate.

    This function is intended for sensitivity analysis, not for production data
    cleaning. The purpose is to simulate degraded telemetry and measure how the
    predictive system responds when sensor coverage becomes less reliable.
    """
    if not 0 <= rate <= 1:
        raise ValueError("Missingness rate must be between 0 and 1.")

    result = df.copy()
    rng = np.random.default_rng(seed)

    # Target and SMR are protected because randomly deleting labels would turn
    # a data-quality experiment into a label-corruption experiment.
    protected = {"target_failure", "SMR"}
    numeric = [
        column
        for column in result.columns
        if pd.api.types.is_numeric_dtype(result[column]) and column not in protected
    ]

    if not numeric:
        return result

    mask = rng.random((len(result), len(numeric))) < rate
    result.loc[:, numeric] = result[numeric].mask(mask)
    return result
