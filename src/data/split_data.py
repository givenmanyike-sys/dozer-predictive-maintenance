"""Leakage-aware temporal splitting utilities.

Predictive maintenance data cannot be validated with ordinary random splits.
The model is intended to predict future machine behaviour, so validation must
preserve the ordering that would exist in production.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class TemporalFold:
    """Container describing one chronological train/test split.

    Attributes:
        train (pd.DataFrame): Observations available to the model during training.
        test (pd.DataFrame): Observations that simulate future observations.
        fold_number (int): Human-readable fold identifier used in reports.
        description (str): Short explanation of how the fold was constructed.
    """

    train: pd.DataFrame
    test: pd.DataFrame
    fold_number: int
    description: str = ""


def expanding_time_folds(
    df: pd.DataFrame,
    n_splits: int = 4,
    test_size: int = 24,
    gap: int = 0,
) -> list[TemporalFold]:
    """Create expanding chronological validation folds.

    Each fold grows the training window forward in time while keeping the test
    window later than the training observations. This approximates repeated
    model retraining as new telemetry becomes available.

    Args:
        df (pd.DataFrame): Data containing a 'Timestamp' column.
        n_splits (int): Number of chronological validation windows. Defaults to 4.
        test_size (int): Number of observations in each test window. Defaults to 24.
        gap (int): Optional number of observations between training and test windows.
            A gap can be useful when feature windows or operational delays create a
            risk of information bleeding across the split. Defaults to 0.

    Returns:
        list[TemporalFold]: List of chronological train/test fold containers.

    Raises:
        ValueError: If n_splits or test_size < 1, or gap < 0, or dataset too small.
    """
    if n_splits < 1 or test_size < 1 or gap < 0:
        raise ValueError("n_splits and test_size must be positive; gap cannot be negative.")

    # Sorting by timestamp first preserves the real temporal order. Machine ID
    # is used only as a deterministic tie-breaker when multiple machines have
    # observations at the same timestamp.
    ordered = df.sort_values(["Timestamp", "Machine ID"]).reset_index(drop=True)

    if len(ordered) < (n_splits + 1) * test_size:
        raise ValueError("Dataset is too small for the requested temporal folds.")

    folds: list[TemporalFold] = []
    first_train_end = len(ordered) - n_splits * test_size

    for fold in range(n_splits):
        train_end = first_train_end + fold * test_size
        test_start = train_end + gap
        test_end = test_start + test_size

        # ``iloc`` is used deliberately because the fold boundaries are based
        # on ordered positions, not on the original DataFrame index.
        folds.append(
            TemporalFold(
                train=ordered.iloc[:train_end].copy(),
                test=ordered.iloc[test_start:test_end].copy(),
                fold_number=fold + 1,
                description="expanding chronological window",
            )
        )

    return folds


def event_centric_folds(
    df: pd.DataFrame,
    events: pd.DataFrame,
    event_category: str = "Unplanned Failure",
    pre_event_hours: int = 72,
    post_event_hours: int = 24,
    embargo_hours: int = 0,
) -> list[TemporalFold]:
    """Create validation folds around genuine failure events.

    This splitter is useful when failures are extremely rare. A generic time
    split can easily produce test periods containing no positive failures,
    making metrics such as PR-AUC uninformative.

    The training set contains observations before the evaluation window. The
    test set contains the affected machine's pre-event and post-event context.
    An optional embargo can widen the separation between training and testing.

    Args:
        df (pd.DataFrame): Telemetry DataFrame with 'Timestamp' and 'Machine ID'.
        events (pd.DataFrame): Event log with 'Event Timestamp', 'Machine ID', and 'Category'.
        event_category (str): Category representing failure. Defaults to 'Unplanned Failure'.
        pre_event_hours (int): Hours prior to event included in test window. Defaults to 72.
        post_event_hours (int): Hours after event included in test window. Defaults to 24.
        embargo_hours (int): Separation buffer between train and test. Defaults to 0.

    Returns:
        list[TemporalFold]: Chronological event-centric folds.

    Raises:
        ValueError: If window parameters are invalid.
    """
    if pre_event_hours < 1 or post_event_hours < 0 or embargo_hours < 0:
        raise ValueError("Event window sizes must be non-negative, with pre_event_hours positive.")

    failures = events.loc[events["Category"].eq(event_category)].sort_values("Event Timestamp")
    folds: list[TemporalFold] = []

    for fold_number, (_, event) in enumerate(failures.iterrows(), start=1):
        machine = event["Machine ID"]
        event_time = event["Event Timestamp"]
        test_start = event_time - pd.Timedelta(hours=pre_event_hours)
        test_end = event_time + pd.Timedelta(hours=post_event_hours)
        train_end = test_start - pd.Timedelta(hours=embargo_hours)

        # Training observations are restricted to the past. This is the key
        # protection against training on data that would only be known after
        # the model's simulated prediction time.
        train = df.loc[df["Timestamp"] < train_end].copy()

        # The test window is restricted to the machine that experienced the
        # event. Other machines are not incorrectly treated as observations of
        # the same physical failure event.
        test = df.loc[
            df["Machine ID"].eq(machine)
            & df["Timestamp"].ge(test_start)
            & df["Timestamp"].le(test_end)
        ].copy()

        if train.empty or test.empty:
            continue

        folds.append(
            TemporalFold(
                train=train,
                test=test,
                fold_number=fold_number,
                description=f"event-centred fold for {machine} at {event_time}",
            )
        )

    return folds
