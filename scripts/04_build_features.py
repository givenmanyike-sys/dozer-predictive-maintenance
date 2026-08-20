"""Command-line entry point for leakage-safe feature engineering."""

import pandas as pd
import yaml

from src.features.build_features import build_features
from src.utils.io import load_config, load_threshold_config, save_dataframe


def main() -> None:
    """Build and persist the configured temporal and threshold features."""
    config = load_config()
    threshold_config = load_threshold_config()

    base = pd.read_csv(
        config["project"]["processed_file"],
        parse_dates=["Timestamp"],
    )

    with open("config/features.yaml", encoding="utf-8") as handle:
        sensor_config = yaml.safe_load(handle)["sensors"]

    feature_config = config["features"]

    featured = build_features(
        base,
        sensors=sensor_config,
        thresholds=threshold_config["thresholds"],
        rolling_windows=feature_config["rolling_windows"],
        slope_windows=feature_config["slope_windows"],
        lag_periods=feature_config["lag_periods"],
        threshold_windows=feature_config["threshold_windows"],
        min_periods=feature_config["min_periods"],
    )

    save_dataframe(featured, config["project"]["featured_file"])
    print(f"Saved {featured.shape[0]:,} rows and {featured.shape[1]:,} columns")


if __name__ == "__main__":
    main()
