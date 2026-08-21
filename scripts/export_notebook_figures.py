"""Export exact figures and tables from all 4 notebooks into outputs/figures/."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns
from scipy.spatial.distance import squareform
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.data.load_data import load_workbook
from src.data.split_data import event_centric_folds
from src.features.cleaning import prepare_telemetry
from src.models.predict import predict_probability
from src.models.train import train_candidates
from src.utils.io import load_config


def main() -> None:
    """Generate all figures and tables from notebooks."""
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    config = load_config()
    wb = load_workbook(config["project"]["raw_file"])
    telemetry = prepare_telemetry(wb["telemetry"])
    events = wb["events"]
    df_features = pd.read_csv(config["project"]["featured_file"], parse_dates=["Timestamp"])

    non_sensor_cols = {"Machine ID", "Timestamp", "Event Timestamp", "target_failure", "SMR"}
    sensor_cols = [
        c
        for c in telemetry.columns
        if c not in non_sensor_cols and pd.api.types.is_numeric_dtype(telemetry[c])
    ]

    # =========================================================================
    # Notebook 02: EDA Figures & Tables
    # =========================================================================

    # 1. Exact Correlation Heatmap from 02_eda.ipynb
    corr_matrix = telemetry[sensor_cols].corr()
    plt.figure(figsize=(14, 11), dpi=300)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot_kws={"size": 7.5},
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Sensor Pearson Correlation Heatmap", fontsize=14, pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "eda_01_sensor_correlation_heatmap.png")
    plt.close()

    # 2. Exact High Correlation Pairs Table from 02_eda.ipynb
    high_corr_data = [
        ["Engine Speed Avg", "Engine Speed Max", "0.9084"],
        ["LB Front Exh Temp", "LB Rear Exh Temp", "0.8872"],
        ["RB Front Exh Temp", "RB Rear Exh Temp", "0.8814"],
        ["LB Front Exh Temp", "RB Front Exh Temp", "0.8655"],
        ["LB Rear Exh Temp", "RB Rear Exh Temp", "0.8540"],
        ["Front Pump Press Max", "Rear Pump Press Max", "0.7632"],
    ]
    high_corr_df = pd.DataFrame(
        high_corr_data, columns=["Feature 1", "Feature 2", "Correlation (r)"]
    )

    fig, ax = plt.subplots(figsize=(9, 4), dpi=300)
    ax.axis("off")
    table = ax.table(
        cellText=high_corr_df.values,
        colLabels=high_corr_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="black")
            cell.set_facecolor("#E0E0E0")
        else:
            cell.set_facecolor("white")
    plt.title(
        "Highly Correlated Sensor Pairs (|r| > 0.75)", fontsize=12, weight="bold", pad=15
    )
    plt.tight_layout()
    plt.savefig(out_dir / "eda_02_multicollinearity_table.png")
    plt.close()

    # 3. Exact Hierarchical Clustering Dendrogram from 02_eda.ipynb
    dist_matrix = 1.0 - corr_matrix.abs()
    dist_condensed = squareform(dist_matrix.fillna(0), checks=False)
    linkage = sch.linkage(dist_condensed, method="average")

    plt.figure(figsize=(12, 6), dpi=300)
    sch.dendrogram(
        linkage,
        labels=sensor_cols,
        orientation="top",
        leaf_rotation=45,
        leaf_font_size=9,
    )
    plt.title(
        "Hierarchical Feature Clustering Dendrogram (Distance = 1 - |r|)",
        fontsize=13,
        pad=12,
    )
    plt.ylabel("Cluster Distance", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "eda_03_clustering_dendrogram.png")
    plt.close()

    # =========================================================================
    # Notebook 04: Model Diagnostics, ROC/PR Curves & Threshold Calibration
    # =========================================================================

    folds = event_centric_folds(df_features, events, pre_event_hours=72, post_event_hours=24)
    valid_folds = [f for f in folds if f.train["target_failure"].nunique() >= 2]
    test_fold = valid_folds[0]

    models = train_candidates(test_fold.train, random_state=42)
    y_test = test_fold.test["target_failure"].astype(int).values

    results = {}
    for name, model_wrapper in models.items():
        probs = predict_probability(model_wrapper, test_fold.test)
        pr_auc = average_precision_score(y_test, probs)
        roc_auc = roc_auc_score(y_test, probs)
        results[name] = {"probs": probs, "pr_auc": pr_auc, "roc_auc": roc_auc}

    display_names = {
        "logistic_regression": "LogisticRegression",
        "random_forest": "RandomForest",
        "hist_gradient_boosting": "HistGradientBoosting",
    }

    colors = {
        "logistic_regression": "#4C72B0",
        "random_forest": "#DD8452",
        "hist_gradient_boosting": "#55A868",
    }

    # 4. ROC & PR Curves Comparison (Notebook 04)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    for name, data in results.items():
        fpr, tpr, _ = roc_curve(y_test, data["probs"])
        dname = display_names.get(name, name)
        roc_val = data["roc_auc"]
        ax1.plot(
            fpr,
            tpr,
            label=f"{dname} (AUC = {roc_val:.4f})",
            linewidth=2.2,
            color=colors.get(name),
        )

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Chance (AUC = 0.5000)")
    ax1.set_title("Out-Of-Fold ROC Curves", fontsize=13, weight="bold", pad=12)
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax1.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(loc="lower right", framealpha=0.95, fontsize=10)
    ax1.grid(True, alpha=0.3)

    baseline = y_test.mean()
    for name, data in results.items():
        prec, rec, _ = precision_recall_curve(y_test, data["probs"])
        dname = display_names.get(name, name)
        pr_val = data["pr_auc"]
        ax2.plot(
            rec,
            prec,
            label=f"{dname} (PR-AUC = {pr_val:.4f})",
            linewidth=2.2,
            color=colors.get(name),
        )

    ax2.axhline(
        baseline,
        color="k",
        linestyle="--",
        alpha=0.6,
        label=f"Test Prevalence Baseline ({baseline:.4f})",
    )
    ax2.set_title("Out-Of-Fold Precision-Recall Curves", fontsize=13, weight="bold", pad=12)
    ax2.set_xlabel("Recall (Detection Rate)", fontsize=11)
    ax2.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(0.35, 1.05)
    ax2.legend(loc="lower left", framealpha=0.95, fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "diagnostics_01_roc_pr_curves.png")
    plt.close()

    # 5. Threshold Calibration Curve (Notebook 04)
    hgb_probs = results["hist_gradient_boosting"]["probs"]
    p_min = hgb_probs.min()
    p_max = hgb_probs.max()
    calibrated_probs = (hgb_probs - p_min) / (p_max - p_min + 1e-12)

    sweep = []
    for thresh in np.linspace(0.01, 0.99, 100):
        preds = (calibrated_probs >= thresh).astype(int)
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        sweep.append({"Threshold": thresh, "Recall": recall, "FPR": fpr})
    sweep_df = pd.DataFrame(sweep)

    plt.figure(figsize=(10, 5.5), dpi=300)
    plt.plot(
        sweep_df["Threshold"],
        sweep_df["Recall"],
        label="Recall (Detection Rate)",
        color="green",
        linewidth=2.5,
    )
    plt.plot(
        sweep_df["Threshold"],
        sweep_df["FPR"],
        label="False Positive Rate (Alarm Fatigue)",
        color="crimson",
        linewidth=2.5,
    )
    plt.axvline(
        0.10, color="black", linestyle="--", linewidth=1.5, label="Operational Cutoff (tau = 0.10)"
    )
    plt.ylim(-0.05, 1.05)
    plt.xlim(0.0, 1.0)
    plt.title(
        "Operational Decision Threshold Calibration (HistGradientBoosting)",
        fontsize=13,
        weight="bold",
        pad=12,
    )
    plt.xlabel("Probability Threshold", fontsize=11)
    plt.ylabel("Rate (0.0 to 1.0)", fontsize=11)
    plt.legend(loc="center right", framealpha=0.9, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "diagnostics_02_threshold_calibration_curve.png")
    plt.close()

    # 6. Confusion Matrix at tau = 0.10 (Notebook 04)
    preds_01 = (calibrated_probs >= 0.10).astype(int)
    cm = confusion_matrix(y_test, preds_01, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Normal (0)", "Failure (1)"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(
        "Confusion Matrix: HistGradientBoosting (tau = 0.10)",
        fontsize=12,
        weight="bold",
        pad=12,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "diagnostics_03_confusion_matrix_calibrated.png")
    plt.close()

    print("Successfully exported all notebook figures & tables to outputs/figures/!")


if __name__ == "__main__":
    main()
