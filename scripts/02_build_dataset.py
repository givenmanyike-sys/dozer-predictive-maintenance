"""Command-line entry point for building the labelled base dataset."""

from src.data.load_data import load_workbook
from src.features.cleaning import prepare_telemetry
from src.targets.build_target import add_failure_target
from src.utils.io import load_config, save_dataframe


def main() -> None:
    """Build the base modelling table using only genuine failure events."""
    config = load_config()
    workbook = load_workbook(config["project"]["raw_file"])

    telemetry = prepare_telemetry(workbook["telemetry"])
    target_config = config["target"]

    # The target is created from the ground-truth event log. OEM thresholds are
    # intentionally absent from this step because threshold breaches are not
    # equivalent to mechanical failure.
    modelling = add_failure_target(
        telemetry,
        workbook["events"],
        target_config["default_horizon_operating_hours"],
        target_config["positive_event"],
    )

    save_dataframe(modelling, config["project"]["processed_file"])

    print(f"Saved {len(modelling):,} rows to {config['project']['processed_file']}")
    print(modelling["target_failure"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
