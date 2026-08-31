"""Complete/immediate schema-v3 metrics for multi-label predictions."""

import numpy as np

from .metric_contract import COMPLETE_METRIC_NAMES
from .metric_utils import (
    instance_set_scores,
    positive_class_counts,
    positive_precision_recall_f1,
    validate_complete_inputs,
)


def compute_complete_metrics(y_true, y_full):
    """Compute the canonical complete metrics for binary ``N x K`` matrices.

    Positive-class precision, recall and F1 use ``zero_division=0``.  For
    instance-F1 and instance Jaccard, a both-empty true/predicted label set
    scores one, as locked by the metric contract.
    """

    truth, full = validate_complete_inputs(y_true, y_full)
    label_precision = []
    label_recall = []
    label_f1 = []
    for label_index in range(truth.shape[1]):
        precision, recall, f1 = positive_precision_recall_f1(
            truth[:, label_index], full[:, label_index]
        )
        label_precision.append(precision)
        label_recall.append(recall)
        label_f1.append(f1)

    micro_tp, micro_fp, micro_fn = positive_class_counts(truth, full)
    micro_denominator = 2 * micro_tp + micro_fp + micro_fn
    micro_f1 = (
        2 * micro_tp / micro_denominator if micro_denominator else 0.0
    )
    hamming_loss = float(np.mean(truth != full))
    instance_f1, instance_jaccard = instance_set_scores(truth, full)

    results = {
        "Macro-F1": float(np.mean(label_f1)),
        "Micro-F1": float(micro_f1),
        "Hamming Accuracy": float(1.0 - hamming_loss),
        "Hamming Loss": hamming_loss,
        "Subset Accuracy": float(np.mean(np.all(truth == full, axis=1))),
        "Instance-F1": instance_f1,
        "Instance Jaccard": instance_jaccard,
        "Macro Precision": float(np.mean(label_precision)),
        "Macro Recall": float(np.mean(label_recall)),
    }
    if tuple(results) != COMPLETE_METRIC_NAMES:
        raise RuntimeError("Complete metric output no longer matches the contract order.")
    return results
