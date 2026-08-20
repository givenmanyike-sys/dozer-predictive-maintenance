"""Generate reproducible exploratory summaries for the fleet telemetry."""

from pathlib import Path

import matplotlib.pyplot as plt

from src.data.load_data import load_workbook
from src.features.cleaning import prepare_telemetry
from src.utils.io import load_config


def main() -> None:
    """Generate a small baseline set of fleet and event-level EDA outputs."""
    config = load_config()
    workbook = load_workbook(config["project"]["raw_file"])
    telemetry = prepare_telemetry(workbook["telemetry"])

    output = Path("outputs/figures")
    output.mkdir(parents=True, exist_ok=True)

    # Before modelling, verify that all machines contribute observations. An
    # apparently strong model can be misleading if one machine dominates the
    # available history.
    counts = telemetry.groupby("Machine ID").size().sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Telemetry observations by machine")
    ax.set_xlabel("Machine")
    ax.set_ylabel("Observed operating hours")
    fig.tight_layout()
    fig.savefig(output / "telemetry_observations_by_machine.png", dpi=160)
    plt.close(fig)

    # Print the event log alongside the machine counts so the first EDA run has
    # an auditable record of the scarce failure and false-alarm cases.
    print("Machines:")
    print(counts.to_string())
    print("\nEvents:")
    print(
        workbook["events"][
            ["Machine ID", "Event Timestamp", "Category", "Component"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
