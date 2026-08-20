"""Train candidate classification models on an early temporal training fold."""

from pathlib import Path

import joblib
import pandas as pd

from src.data.split_data import expanding_time_folds
from src.models.train import train_candidates
from src.utils.io import load_config


def main() -> None:
    """Fit candidate models and save baseline artifacts for inspection."""
    config = load_config()
    df = pd.read_csv(config["project"]["featured_file"], parse_dates=["Timestamp"])

    folds = expanding_time_folds(
        df,
        n_splits=config["validation"]["n_splits"],
        test_size=config["validation"]["test_size_operating_hours"],
        gap=config["validation"].get("gap_operating_hours", 0),
    )
    fold = folds[0]

    # This script intentionally trains only the first fold as a baseline
    # artefact. Final model selection must happen through the full validation
    # process rather than by inspecting a single saved model.
    models = train_candidates(
        fold.train,
        random_state=config["project"]["random_state"],
    )

    Path("models").mkdir(exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, Path("models") / f"{name}.joblib")
        print(f"Saved models/{name}.joblib")


if __name__ == "__main__":
    main()
