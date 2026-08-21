"""Construction of failure prediction targets from ground-truth events."""

import pandas as pd


def add_failure_target(
    telemetry: pd.DataFrame,
    events: pd.DataFrame,
    horizon_operating_hours: int,
    positive_event: str = "Unplanned Failure",
) -> pd.DataFrame:
    """Add a binary target indicating an upcoming genuine failure.

    Args:
        telemetry (pd.DataFrame): Machine telemetry containing 'Machine ID' and 'Timestamp'.
        events (pd.DataFrame): Ground-truth event log with 'Machine ID', 'Event Timestamp',
            and 'Category'.
        horizon_operating_hours (int): Number of subsequent observed operating rows
            considered the early-warning horizon window.
        positive_event (str): Event category that represents a true mechanical breakdown.
            Defaults to 'Unplanned Failure'.

    Returns:
        pd.DataFrame: Telemetry DataFrame augmented with 'target_failure' binary column.

    Raises:
        ValueError: If horizon_operating_hours < 1.

    Notes:
        False alarms and scheduled maintenance are intentionally excluded because
        they do not represent the mechanical failure the assessment asks us to
        predict. The event log therefore defines the target, while OEM thresholds
        remain feature inputs only.
    """
    if horizon_operating_hours < 1:
        raise ValueError("horizon_operating_hours must be positive")

    # Work in machine-time order. The target is defined independently for each
    # dozer so that a failure on one machine can never label another machine.
    df = telemetry.sort_values(["Machine ID", "Timestamp"]).copy()
    df["target_failure"] = 0

    # Filter the event log to the one category that represents the target. This
    # is the central distinction between ground truth and domain thresholds.
    failure_events = events.loc[events["Category"].eq(positive_event)].copy()
    failure_events = failure_events.sort_values(["Machine ID", "Event Timestamp"])

    for machine_id, machine_rows in df.groupby("Machine ID", sort=False):
        machine_indices = machine_rows.index.to_list()
        machine_events = failure_events.loc[failure_events["Machine ID"].eq(machine_id)]

        if machine_events.empty:
            continue

        timestamps = machine_rows["Timestamp"]

        for event_time in machine_events["Event Timestamp"]:
            for position, index in enumerate(machine_indices):
                current_time = timestamps.loc[index]

                # Once the event has been reached, later observations cannot be
                # used to predict that same event and we can stop scanning.
                if current_time >= event_time:
                    break

                # The current observation is excluded. The future slice starts
                # at the next observed machine-hour because a prediction made at
                # time t cannot use information from t + 1 when defining the
                # information available at t.
                future_slice = timestamps.iloc[
                    position + 1 : position + 1 + horizon_operating_hours
                ]

                # A positive label is assigned when the failure falls inside the
                # next ``horizon_operating_hours`` observed machine-hours.
                if len(future_slice) == 0:
                    continue

                if future_slice.iloc[-1] >= event_time:
                    df.loc[index, "target_failure"] = 1

    return df
