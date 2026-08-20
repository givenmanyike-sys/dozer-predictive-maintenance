"""Data quality checks for raw telemetry and event data.

Validation is intentionally separated from modelling so that bad inputs fail
before expensive feature engineering or model training begins.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class ValidationReport:
    """Container for named validation checks and explanatory messages."""

    checks: dict[str, bool]
    messages: list[str]

    @property
    def passed(self) -> bool:
        """Return ``True`` only when every registered check passes."""
        return all(self.checks.values())

    def print_summary(self) -> None:
        """Print the validation outcome in a form suitable for CLI execution."""
        for name, passed in self.checks.items():
            print(f"{'PASS' if passed else 'FAIL'}: {name}")

        for message in self.messages:
            print(f"  {message}")


def validate_telemetry(
    telemetry: pd.DataFrame,
    events: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> ValidationReport:
    """Validate minimum structural assumptions required by the pipeline.

    The checks here are intentionally conservative. They verify that the
    tables have the keys and basic data types required by later stages, but
    they do not attempt to decide whether a sensor value is physically valid.
    Domain-specific plausibility checks can be added after the metadata are
    fully understood.
    """
    required = {"Machine ID", "SMR", "Timestamp"}

    # Keep each check separately named so a future maintainer can identify the
    # exact reason a pipeline run failed instead of receiving one generic error.
    checks = {
        "required telemetry columns": required.issubset(telemetry.columns),
        "timestamp is datetime": pd.api.types.is_datetime64_any_dtype(telemetry["Timestamp"]),
        "machine IDs are present": telemetry["Machine ID"].notna().all(),
        "timestamps are present": telemetry["Timestamp"].notna().all(),
        "SMR is numeric": pd.to_numeric(telemetry["SMR"], errors="coerce").notna().all(),
        "events contain categories": "Category" in events.columns,
        "thresholds contain parameters": "Parameter" in thresholds.columns,
    }

    # A machine-hour should identify one unique observation. Duplicates can
    # otherwise double-count operating time and distort rolling features.
    duplicate_keys = telemetry.duplicated(["Machine ID", "Timestamp"]).sum()
    checks["no duplicate machine-timestamps"] = duplicate_keys == 0

    messages = [
        f"Telemetry rows: {len(telemetry):,}",
        f"Machines: {telemetry['Machine ID'].nunique()}",
        f"Duplicate machine-timestamps: {duplicate_keys}",
        f"Event rows: {len(events)}",
    ]

    return ValidationReport(checks=checks, messages=messages)
