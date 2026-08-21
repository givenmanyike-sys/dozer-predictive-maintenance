"""Prediction and alert-threshold helpers."""

import pandas as pd

from src.models.train import TrainedModel


def predict_probability(model: TrainedModel, df: pd.DataFrame) -> pd.Series:
    """Return the model probability assigned to an impending failure.

    The feature columns are taken from the fitted model bundle rather than
    inferred from the prediction DataFrame. This protects the prediction path
    from column-order changes and accidental inclusion of new fields.

    Args:
        model (TrainedModel): Fitted model wrapper bundle.
        df (pd.DataFrame): Data to score.

    Returns:
        pd.Series: Probability scores aligned with input index.
    """
    probability = model.pipeline.predict_proba(df[model.feature_columns])[:, 1]
    return pd.Series(probability, index=df.index, name="failure_probability")


def apply_alert_threshold(probability: pd.Series, threshold: float = 0.5) -> pd.Series:
    """Convert failure probabilities into binary maintenance alerts.

    The threshold should be selected using validation data and operational cost,
    not simply defaulted to 0.5 because that is the conventional classification
    cutoff. A later assessment stage should explicitly tune this decision rule.

    Args:
        probability (pd.Series): Predicted failure probabilities.
        threshold (float): Decision probability threshold. Defaults to 0.5.

    Returns:
        pd.Series: Binary alerts (1 = Alert, 0 = Normal).

    Raises:
        ValueError: If threshold is not between 0 and 1.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("Alert threshold must be between 0 and 1.")

    return (probability >= threshold).astype(int)
