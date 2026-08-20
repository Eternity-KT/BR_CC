"""
Evaluation Metrics for Multi-Label Classification
Implements standard metrics from:
- Zhang et al. (FCS 2018)
- Read et al. (Machine Learning 2011)
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    accuracy_score,
    precision_score,
    recall_score
)

METRICS_INFO = {
    "Macro-F1": {
        "type": "Label-based",
        "direction": "higher_better",
        "description": "Macro-averaged F1 score across all labels (Mandatory)"
    },
    "Micro-F1": {
        "type": "Label-based",
        "direction": "higher_better",
        "description": "Micro-averaged F1 score globally across all label-instance pairs"
    },
    "Hamming Loss": {
        "type": "Example-based",
        "direction": "lower_better",
        "description": "Fraction of misclassified labels (lower is better)"
    },
    "Subset Accuracy": {
        "type": "Example-based",
        "direction": "higher_better",
        "description": "Exact match ratio where predicted label set strictly equals ground truth"
    },
    "Example-F1": {
        "type": "Example-based",
        "direction": "higher_better",
        "description": "Instance-averaged F1 score across all samples"
    },
    "Macro Precision": {
        "type": "Label-based",
        "direction": "higher_better",
        "description": "Macro-averaged precision across all labels"
    },
    "Macro Recall": {
        "type": "Label-based",
        "direction": "higher_better",
        "description": "Macro-averaged recall across all labels"
    }
}


def compute_example_f1(y_true, y_pred):
    """
    Compute example-based F1 score.
    For each sample i:
        F1_i = 2 * |Y_i ∩ Ŷ_i| / (|Y_i| + |Ŷ_i|)
    If both Y_i and Ŷ_i are empty, F1_i = 1.0.
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    n_samples = y_true.shape[0]

    if n_samples == 0:
        return 0.0

    true_sums = y_true.sum(axis=1)
    pred_sums = y_pred.sum(axis=1)
    intersections = np.logical_and(y_true, y_pred).sum(axis=1)

    example_f1s = np.zeros(n_samples, dtype=np.float64)

    both_empty = (true_sums == 0) & (pred_sums == 0)
    example_f1s[both_empty] = 1.0

    non_empty = ~both_empty
    denominators = true_sums[non_empty] + pred_sums[non_empty]
    example_f1s[non_empty] = (2.0 * intersections[non_empty]) / denominators

    return float(np.mean(example_f1s))


def compute_all_metrics(y_true, y_pred):
    """
    Compute all 7 evaluation metrics for multi-label predictions.

    Parameters:
        y_true (np.ndarray): Binary ground truth matrix of shape (n_samples, n_labels)
        y_pred (np.ndarray): Binary prediction matrix of shape (n_samples, n_labels)

    Returns:
        results (dict): Dictionary mapping metric name to float value.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)

    results = {
        "Macro-F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Micro-F1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "Hamming Loss": float(hamming_loss(y_true, y_pred)),
        "Subset Accuracy": float(accuracy_score(y_true, y_pred)),
        "Example-F1": compute_example_f1(y_true, y_pred),
        "Macro Precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Macro Recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    }

    return results
