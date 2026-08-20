"""Simple sensor forecasting baselines used for degradation analysis.

Forecasting is deliberately kept separate from failure classification. A
forecast can help determine whether a sensor exhibits predictable trajectory
behaviour, but that does not automatically make forecasting the right way to
predict mechanical failure.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def naive_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    """Forecast future values by carrying the latest observed value forward.

    The naive forecast is intentionally simple. It provides a strong baseline
    for telemetry that changes slowly and gives us a reference against which
    more complex forecasting methods can be judged.
    """
    if horizon < 1:
        raise ValueError("Forecast horizon must be positive.")

    clean = series.dropna()
    if clean.empty:
        raise ValueError("Cannot forecast an empty series.")

    return np.repeat(clean.iloc[-1], horizon)


def exponential_smoothing_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    """Forecast a sensor using a simple level-plus-trend exponential smoother.

    The implementation is intentionally modest. We should only introduce more
    complex models such as ARIMA or state-space variants if diagnostics and
    backtesting show that the added complexity provides meaningful benefit.
    """
    if horizon < 1:
        raise ValueError("Forecast horizon must be positive.")

    clean = series.dropna()
    if len(clean) < 5:
        # A very short history does not support reliable trend estimation, so
        # fall back to the naive baseline rather than fitting an unstable model.
        return naive_forecast(clean, horizon)

    model = ExponentialSmoothing(
        clean,
        trend="add",
        seasonal=None,
        initialization_method="estimated",
    )

    # Fixed smoothing parameters make this baseline deterministic and avoid
    # spending the small assessment dataset on unnecessary hyperparameter
    # optimisation. This is a baseline, not the final forecasting model.
    fitted = model.fit(
        optimized=False,
        smoothing_level=0.2,
        smoothing_trend=0.05,
    )
    return np.asarray(fitted.forecast(horizon))
