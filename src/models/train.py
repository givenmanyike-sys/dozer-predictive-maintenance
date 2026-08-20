"""Training utilities for a compact set of baseline classifiers."""

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainedModel:
    """Bundle a fitted pipeline with the exact feature columns used by it."""

    name: str
    pipeline: Pipeline
    feature_columns: list[str]


def prepare_xy(
    df: pd.DataFrame,
    target: str = "target_failure",
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Separate numeric model inputs from identifiers and the target column.

    Machine ID and Timestamp are excluded because they are identifiers and can
    allow the model to memorise machine-specific or temporal structure rather
    than learning generalisable sensor relationships.
    """
    excluded = {target, "Machine ID", "Timestamp", "Event Timestamp"}

    # Restricting to numeric fields makes the model contract explicit and avoids
    # accidental conversion of identifiers or categorical text into arbitrary
    # numeric encodings.
    feature_columns = [
        column
        for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
    ]

    if not feature_columns:
        raise ValueError("No numeric model features were found.")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is missing.")

    X = df[feature_columns]
    y = df[target].astype(int)
    return X, y, feature_columns


def train_candidates(
    train_df: pd.DataFrame,
    random_state: int = 42,
) -> dict[str, TrainedModel]:
    """Fit a compact, defensible set of candidate classifiers.

    The candidates deliberately cover different modelling assumptions:

    * Logistic regression provides an interpretable linear baseline.
    * Random forest captures non-linear interactions with limited tuning.
    * HistGradientBoosting provides a stronger non-linear baseline.

    Imputation is performed inside each sklearn pipeline so that the imputer is
    fitted on training data only. This is important for temporal validation.
    """
    X, y, feature_columns = prepare_xy(train_df)

    candidates = {
        "logistic_regression": Pipeline(
            [
                # The imputer is part of the pipeline so future test values do
                # not influence the training medians.
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=500,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=150,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=150,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

    trained: dict[str, TrainedModel] = {}
    for name, pipeline in candidates.items():
        # Keeping fitting in one loop ensures every candidate sees exactly the
        # same training rows and feature columns.
        pipeline.fit(X, y)
        trained[name] = TrainedModel(
            name=name,
            pipeline=pipeline,
            feature_columns=feature_columns,
        )

    return trained
