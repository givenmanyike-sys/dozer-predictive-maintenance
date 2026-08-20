"""Orchestration for candidate model evaluation."""

import pandas as pd

from src.data.split_data import event_centric_folds
from src.evaluation.metrics import classification_metrics
from src.models.predict import predict_probability
from src.models.train import train_candidates


def evaluate_candidates(
    df: pd.DataFrame,
    events: pd.DataFrame,
    pre_event_hours: int = 72,
    post_event_hours: int = 24,
    random_state: int = 42,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Evaluate candidate classifiers on chronological event-centred folds.

    The function deliberately skips folds where the training data contain only
    one target class. A classifier cannot learn a meaningful binary decision
    boundary from a training set containing only failures or only healthy rows.
    """
    rows = []

    folds = event_centric_folds(
        df,
        events,
        pre_event_hours=pre_event_hours,
        post_event_hours=post_event_hours,
    )

    for fold in folds:
        # With only one class in training, fitting the candidate classifiers is
        # not a valid binary classification experiment. Skipping is preferable
        # to silently changing the modelling problem.
        if fold.train["target_failure"].nunique() < 2:
            continue

        models = train_candidates(fold.train, random_state=random_state)

        for name, model in models.items():
            probability = predict_probability(model, fold.test)
            metrics = classification_metrics(
                fold.test["target_failure"],
                probability,
                threshold,
            )
            rows.append(
                {
                    "fold": fold.fold_number,
                    "description": fold.description,
                    "model": name,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)
