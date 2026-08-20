"""Command-line entry point for validating the supplied raw workbook.

This script intentionally contains very little business logic. The reusable
validation functions live under ``src`` so they can be tested independently
and reused by future pipelines.
"""

from src.data.load_data import load_workbook
from src.data.validate_data import validate_telemetry
from src.utils.io import load_config


def main() -> None:
    """Load the raw workbook, run structural checks, and fail on invalid data."""
    config = load_config()

    # Keep the raw file path in configuration rather than hard-coding it in the
    # script. This makes the same command usable across development and test
    # environments without editing Python code.
    workbook = load_workbook(config["project"]["raw_file"])

    report = validate_telemetry(
        workbook["telemetry"],
        workbook["events"],
        workbook["thresholds"],
    )
    report.print_summary()

    # A non-zero exit code allows CI or another orchestration system to stop the
    # pipeline when the input data violate a required structural assumption.
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
