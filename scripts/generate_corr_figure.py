"""Generate sensor correlation heatmap for exploratory data analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.data.load_data import load_workbook
from src.features.cleaning import prepare_telemetry


def main() -> None:
    """Generate and save the sensor correlation heatmap."""
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    wb = load_workbook("data/raw/dozer_predictive_maintenance.xlsx")
    telemetry = prepare_telemetry(wb["telemetry"])

    sensor_cols = [
        c for c in telemetry.columns if c not in ["Machine ID", "Timestamp", "SMR"]
    ]
    corr = telemetry[sensor_cols].corr()

    plt.figure(figsize=(12, 10), dpi=300)
    sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5, annot=False)
    plt.title(
        "Sensor Multicollinearity & Correlation Matrix (Fleet Telemetry)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "05_sensor_correlation_heatmap.png")
    plt.close()
    print("Generated 05_sensor_correlation_heatmap.png successfully!")


if __name__ == "__main__":
    main()
