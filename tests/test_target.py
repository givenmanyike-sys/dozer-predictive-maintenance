"""Regression tests for ground-truth target construction."""

import pandas as pd

from src.targets.build_target import add_failure_target


def test_failure_target_uses_only_unplanned_failure() -> None:
    """Verify false alarms do not become positive failure labels."""
    telemetry = pd.DataFrame(
        {
            "Machine ID": ["A"] * 5,
            "Timestamp": pd.date_range("2025-01-01 06:00", periods=5, freq="h"),
            "SMR": range(5),
        }
    )

    events = pd.DataFrame(
        {
            "Machine ID": ["A", "A"],
            "Event Timestamp": [
                pd.Timestamp("2025-01-01 08:00"),
                pd.Timestamp("2025-01-01 09:00"),
            ],
            "Category": ["False Alarm", "Unplanned Failure"],
        }
    )

    result = add_failure_target(
        telemetry,
        events,
        horizon_operating_hours=2,
    )

    assert result["target_failure"].sum() >= 1

    # The false alarm at 08:00 must not independently create a positive label
    # at 06:00. This assertion protects the distinction between operational
    # alarm history and the mechanical-failure ground truth.
    assert (
        result.loc[
            result["Timestamp"] == pd.Timestamp("2025-01-01 06:00"),
            "target_failure",
        ].iloc[0]
        == 0
    )
