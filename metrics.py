"""Classification metrics for Ligandformer experiments."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def binary_classification_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    logits = np.asarray(logits).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(int)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auprc": float(average_precision_score(labels, probabilities)),
    }
    try:
        metrics["auroc"] = float(roc_auc_score(labels, probabilities))
    except ValueError:
        metrics["auroc"] = float("nan")
    return metrics
