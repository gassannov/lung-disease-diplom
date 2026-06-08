"""Classification metrics and report plots."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)


def compute_metrics(y_true: list[int], y_pred: list[int], labels: list[str]) -> dict[str, Any]:
    """Compute core classification metrics.

    Args:
        y_true: Ground-truth class ids.
        y_pred: Predicted class ids.
        labels: Class names ordered by class id.

    Returns:
        Metrics dictionary with macro F1, weighted F1, balanced accuracy, and recalls.
    """
    recall_values = recall_score(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        average=None,
        zero_division=0,
    )
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class_recall": {
            label: float(value) for label, value in zip(labels, recall_values, strict=True)
        },
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    labels: list[str],
    path: str | Path,
) -> None:
    """Save a confusion matrix heatmap.

    Args:
        y_true: Ground-truth class ids.
        y_pred: Predicted class ids.
        labels: Class names ordered by class id.
        path: Destination image path.

    Returns:
        None.
    """
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    frame = pd.DataFrame(matrix, index=labels, columns=labels)
    plt.figure(figsize=(7, 6))
    sns.heatmap(frame, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("true")
    plt.xlabel("predicted")
    plt.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(destination, dpi=180)
    plt.close()


def save_training_curves(metrics_csv: str | Path, path: str | Path) -> None:
    """Save train and validation curves from a metrics CSV file.

    Args:
        metrics_csv: CSV file produced during training.
        path: Destination image path.

    Returns:
        None.
    """
    frame = pd.read_csv(metrics_csv)
    plt.figure(figsize=(8, 5))
    for column in ["train_loss", "val_loss", "val_macro_f1", "val_balanced_accuracy"]:
        if column in frame:
            plt.plot(frame["epoch"], frame[column], label=column)
    plt.legend()
    plt.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(destination, dpi=180)
    plt.close()


def class_weights(labels: list[int], num_classes: int) -> np.ndarray:
    """Compute inverse-frequency class weights.

    Args:
        labels: Class ids from the training split.
        num_classes: Total number of classes.

    Returns:
        Weight array ordered by class id.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    return counts.sum() / (num_classes * counts)
