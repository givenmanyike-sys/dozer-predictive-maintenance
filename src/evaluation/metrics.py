"""Classification metrics chosen for rare-event maintenance prediction."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, probability, threshold: float = 0.5) -> dict[str, float]:
    """Calculate probability and threshold-based metrics for one evaluation set.

    Precision-recall metrics are included because genuine failures are rare and
    a high number of healthy machine-hours can make accuracy look impressive
    even when the alerting system is operationally poor.

    The probability threshold is kept explicit because alert generation is a
    business decision. The final threshold should be selected from validation
    results using false-alarm cost and desired warning coverage.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()

    # Ranking metrics are undefined when the evaluation set contains only one
    # class. Returning NaN makes that limitation visible instead of silently
    # producing a misleading score.
    has_both_classes = len(np.unique(y_true)) > 1

    return {
        "pr_auc": float(average_precision_score(y_true, probability))
        if has_both_classes
        else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, probability)) if has_both_classes else float("nan"),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "true_negative_rate": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }
