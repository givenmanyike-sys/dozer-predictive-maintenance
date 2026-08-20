"""Event-level metrics for evaluating maintenance alert usefulness."""

import pandas as pd


def first_alert_warning_time(
    predictions: pd.DataFrame,
    events: pd.DataFrame,
    machine_column: str = "Machine ID",
) -> pd.DataFrame:
    """Calculate the earliest alert lead time for each genuine failure.

    Event-level warning time is more operationally meaningful than counting
    every correctly classified machine-hour. A maintenance planner needs to
    know whether the first useful alert arrived early enough to act.
    """
    failures = events.loc[events["Category"].eq("Unplanned Failure")].copy()
    rows = []

    for _, event in failures.iterrows():
        machine = event[machine_column]
        event_time = event["Event Timestamp"]

        candidate = predictions.loc[
            predictions[machine_column].eq(machine)
            & predictions["alert"].eq(1)
            & predictions["Timestamp"].lt(event_time)
        ]

        if candidate.empty:
            rows.append(
                {
                    machine_column: machine,
                    "event_timestamp": event_time,
                    "warning_hours": None,
                }
            )
            continue

        # The first alert in chronological order is the earliest warning. We
        # sort ascending and take the first row rather than the final alert,
        # because repeated alerts after the first warning do not increase the
        # available maintenance lead time.
        first_alert = candidate.sort_values("Timestamp").iloc[0]
        warning_hours = (event_time - first_alert["Timestamp"]).total_seconds() / 3600

        rows.append(
            {
                machine_column: machine,
                "event_timestamp": event_time,
                "warning_hours": warning_hours,
            }
        )

    return pd.DataFrame(rows)


def false_alert_rate(predictions: pd.DataFrame, observed_hours: int | None = None) -> float:
    """Return alert count per observed machine-hour.

    This is a simple alert-rate measure, not a true false-positive rate. A true
    false-positive rate requires ground-truth labels. The distinction matters
    in production because an alert on a genuinely deteriorating machine may be
    correct even if a failure has not yet occurred.
    """
    alert_count = int(predictions["alert"].sum())
    hours = observed_hours or len(predictions)
    return alert_count / hours if hours else 0.0
