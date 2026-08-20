"""Rolling-origin backtesting utilities for sensor forecasting baselines."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models.forecasting import exponential_smoothing_forecast, naive_forecast


def backtest_sensor_forecasters(
    series: pd.Series,
    horizon: int = 6,
    minimum_history: int = 24,
) -> pd.DataFrame:
    """Compare simple sensor forecasters using rolling-origin evaluation.

    At each origin, only observations before the origin are provided to the
    forecasting method. The following ``horizon`` observations act as the
    unseen future. Repeating this process produces a more realistic estimate of
    how the forecast behaves as the machine ages.

    Forecasting is treated as a supporting degradation-analysis task. We do not
    assume that ARIMA or another complex model is appropriate merely because
    the source data are time series.
    """
    if horizon < 1 or minimum_history < 1:
        raise ValueError("horizon and minimum_history must be positive.")

    clean = series.dropna().reset_index(drop=True)
    rows = []

    # The step size reduces overlap between adjacent test windows. The intent is
    # not to maximise the number of highly correlated observations, but to get
    # several meaningful origins across the available history.
    step = max(horizon * 6, 24)
    origins = range(minimum_history, len(clean) - horizon + 1, step)

    forecasters = {
        "naive": naive_forecast,
        "exponential_smoothing": exponential_smoothing_forecast,
    }

    for origin in origins:
        history = clean.iloc[:origin]
        actual = clean.iloc[origin : origin + horizon].to_numpy()

        for name, forecaster in forecasters.items():
            forecast = forecaster(history, horizon)
            rows.append(
                {
                    "origin": origin,
                    "model": name,
                    "mae": mean_absolute_error(actual, forecast),
                    "rmse": np.sqrt(mean_squared_error(actual, forecast)),
                }
            )

    return pd.DataFrame(rows)
