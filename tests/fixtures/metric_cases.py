"""Small matrices with hand-computed expectations for metric contract tests."""

import numpy as np


def complete_metric_case():
    """Return a 4x2 case covering errors and an empty-label instance."""

    y_true = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 1],
            [0, 0],
        ],
        dtype=np.int32,
    )
    y_full = np.array(
        [
            [1, 0],
            [1, 0],
            [1, 1],
            [0, 0],
        ],
        dtype=np.int32,
    )
    expected = {
        "Macro-F1": 11.0 / 15.0,
        "Micro-F1": 3.0 / 4.0,
        "Hamming Accuracy": 3.0 / 4.0,
        "Hamming Loss": 1.0 / 4.0,
        "Subset Accuracy": 3.0 / 4.0,
        "Instance-F1": 3.0 / 4.0,
        "Instance Jaccard": 3.0 / 4.0,
        "Macro Precision": 5.0 / 6.0,
        "Macro Recall": 3.0 / 4.0,
    }
    return y_true, y_full, expected


def partial_metric_case():
    """Return complete and partial outputs with both decisions and abstentions."""

    y_true, y_full, _ = complete_metric_case()
    y_partial = np.array(
        [
            [1, -1],
            [-1, 0],
            [1, 1],
            [-1, -1],
        ],
        dtype=np.int32,
    )
    return y_true, y_full, y_partial


def all_abstain_case():
    """Return a partial output that abstains on every label position."""

    y_true, y_full, _ = complete_metric_case()
    return y_true, y_full, np.full_like(y_true, -1)


def zero_positive_support_case():
    """Return a case whose second label has no positive truth support."""

    y_true = np.array([[1, 0], [0, 0], [1, 0]], dtype=np.int32)
    y_full = np.array([[1, 0], [0, 1], [0, 0]], dtype=np.int32)
    return y_true, y_full


LABEL_GROUPS = {"IL": (0,), "DL": (1,), "EMPTY": ()}
