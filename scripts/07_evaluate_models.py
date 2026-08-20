"""Run chronological model evaluation and persist the resulting metrics."""

from pathlib import Path

import pandas as pd

from src.data.load_data import load_workbook
from src.evaluation.evaluate import evaluate_candidates
from src.evaluation.plots import plot_metric_by_model
from src.utils.io import load_config


def main() -> None:
    """Evaluate candidate classifiers around the known genuine failure events."""
    config = load_config()
    df = pd.read_csv(config["project"]["featured_file"], parse_dates=["Timestamp"])
    workbook = load_workbook(config["project"]["raw_file"])

    metrics = evaluate_candidates(
        df,
        workbook["events"],
        pre_event_hours=config["validation"].get("event_pre_hours", 72),
        post_event_hours=config["validation"].get("event_post_hours", 24),
        random_state=config["project"]["random_state"],
        threshold=config["model"]["probability_threshold"],
    )

    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    metrics.to_csv("outputs/metrics/temporal_model_metrics.csv", index=False)

    if not metrics.empty:
        plot_metric_by_model(
            metrics,
            "pr_auc",
            "outputs/figures/temporal_pr_auc.png",
        )
        print(
            metrics.groupby("model")["pr_auc"]
            .mean()
            .sort_values(ascending=False)
            .to_string()
        )
    else:
        print("No valid event-centred folds were available for model evaluation.")


if __name__ == "__main__":
    main()
