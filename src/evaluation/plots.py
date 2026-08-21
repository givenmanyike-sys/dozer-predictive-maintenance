"""Reusable plotting helpers for model evaluation outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_by_model(
    metrics: pd.DataFrame,
    metric: str,
    output_path: str | Path,
) -> None:
    """Plot a selected evaluation metric across temporal validation folds.

    Plotting is kept in a small reusable function so scripts can remain focused
    on orchestration and data flow rather than containing repeated Matplotlib
    configuration.

    Args:
        metrics (pd.DataFrame): Evaluation metrics table.
        metric (str): Column name of the metric to plot.
        output_path (str | Path): Destination file path for saved figure.

    Raises:
        ValueError: If metric column is not found in metrics table.
    """
    if metric not in metrics.columns:
        raise ValueError(f"Metric '{metric}' is not present in the evaluation table.")

    fig, ax = plt.subplots(figsize=(9, 5))

    for model, group in metrics.groupby("model"):
        ax.plot(group["fold"], group[metric], marker="o", label=model)

    ax.set_xlabel("Validation fold")
    ax.set_ylabel(metric)
    ax.set_title(f"Temporal validation: {metric}")
    ax.legend()
    fig.tight_layout()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)
