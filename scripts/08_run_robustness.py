"""Run lightweight missing-data and sensor-drift sensitivity experiments."""

from pathlib import Path

import pandas as pd

from src.robustness.drift import inject_additive_drift
from src.robustness.missing_data import inject_missingness
from src.utils.io import load_config


def main() -> None:
    """Create controlled perturbed datasets and record their impact on inputs."""
    config = load_config()
    df = pd.read_csv(config["project"]["featured_file"], parse_dates=["Timestamp"])

    sensor_columns = [
        column
        for column in df.columns
        if column in {
            "Blow-by Press Max",
            "Oil Temp Max",
            "Hyd Oil Temp Max",
            "Coolant Temp Max",
        }
    ]

    rows = []

    for rate in config["robustness"]["missing_rates"]:
        perturbed = inject_missingness(
            df,
            rate,
            seed=config["project"]["random_state"],
        )
        rows.append(
            {
                "experiment": "missingness",
                "level": rate,
                "missing_cells": int(perturbed.isna().sum().sum()),
            }
        )

    for drift in config["robustness"]["drift_multipliers"]:
        perturbed = inject_additive_drift(df, sensor_columns, drift)
        mean_change = float(
            (perturbed[sensor_columns] - df[sensor_columns]).abs().mean().mean()
        )
        rows.append(
            {
                "experiment": "drift",
                "level": drift,
                "mean_change": mean_change,
            }
        )

    # The experiment output is deliberately separate from model evaluation. A
    # robustness experiment describes sensitivity to degraded inputs; it does
    # not itself prove that predictive performance has degraded.
    output = Path("outputs/metrics")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "robustness_experiments.csv", index=False)
    print("Saved outputs/metrics/robustness_experiments.csv")


if __name__ == "__main__":
    main()
