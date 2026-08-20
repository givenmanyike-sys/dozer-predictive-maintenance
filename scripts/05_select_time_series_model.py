"""Compare simple sensor forecasting baselines with rolling-origin backtesting."""

from pathlib import Path

import pandas as pd

from src.data.load_data import load_workbook
from src.features.cleaning import prepare_telemetry
from src.models.time_series_selection import backtest_sensor_forecasters
from src.utils.io import load_config


def main() -> None:
    """Evaluate naive and exponential-smoothing baselines for selected sensors."""
    config = load_config()
    workbook = load_workbook(config["project"]["raw_file"])
    telemetry = prepare_telemetry(workbook["telemetry"])

    # These sensors are a small diagnostic subset rather than a claim that they
    # are the final predictors. EDA should determine whether additional signals
    # or different sensors deserve forecasting analysis.
    candidate_sensors = [
        "Blow-by Press Max",
        "Hyd Oil Temp Max",
        "Oil Temp Max",
        "Coolant Temp Max",
    ]

    rows = []
    for machine_id, machine in telemetry.groupby("Machine ID"):
        for sensor in candidate_sensors:
            if sensor not in machine.columns:
                continue

            result = backtest_sensor_forecasters(
                machine[sensor],
                horizon=6,
                minimum_history=48,
            )

            if result.empty:
                continue

            summary = result.groupby("model")[["mae", "rmse"]].mean().reset_index()
            summary.insert(0, "sensor", sensor)
            summary.insert(0, "Machine ID", machine_id)
            rows.append(summary)

    output = Path("outputs/metrics")
    output.mkdir(parents=True, exist_ok=True)

    if rows:
        comparison = pd.concat(rows, ignore_index=True)
        comparison.to_csv(output / "time_series_forecast_comparison.csv", index=False)
        print(comparison.to_string(index=False))
    else:
        print("No valid sensor forecasting backtests were available.")


if __name__ == "__main__":
    main()
