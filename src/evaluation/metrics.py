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
    "Generalized Loss": {
        "type": "Partial-abstention",
        "direction": "lower_better",
        "description": "Decided-label errors plus the configured abstention penalty"
    },
    "Selective Hamming Loss": {
        "type": "Partial-abstention",
        "direction": "lower_better",
        "description": "Hamming error rate restricted to decided labels"
    },
    "Selective Macro-F1": {
        "type": "Partial-abstention",
        "direction": "higher_better",
        "description": "Macro-F1 computed only over decided examples per label"
    },
    "Selective Micro-F1": {
        "type": "Partial-abstention",
        "direction": "higher_better",
        "description": "Micro-F1 computed only over decided label-instance positions"
    },
    "Selective Instance-F1": {
        "type": "Partial-abstention",
        "direction": "higher_better",
        "description": "Instance-based F1 computed only over decided labels per instance"
    },
    "Coverage": {
        "type": "Partial-abstention",
        "direction": "descriptive",
        "description": "Fraction of instance-label positions receiving a decision"
    },
    "Abstention Rate": {
        "type": "Partial-abstention",
        "direction": "descriptive",
        "description": "Fraction of instance-label positions receiving abstention"
    },
    "ABS": {
        "type": "Partial-abstention",
        "direction": "lower_better",
        "description": "Fraction of samples with at least one abstained label"
    },
    "AABS": {
        "type": "Partial-abstention",
        "direction": "lower_better",
        "description": "Fraction of all label-instance positions abstained"
    },
    "Independent Label Count": {
        "type": "GSI-MLC-PA",
        "direction": "descriptive",
        "description": "Number of labels selected for direct BR inference"
    },
    "Dependent Label Count": {
        "type": "GSI-MLC-PA",
        "direction": "descriptive",
        "description": "Number of labels using conditional CC inference"
    },
    "Validation Full Macro-F1": {
        "type": "GSI-MLC-PA",
        "direction": "higher_better",
        "description": "Final rejection-free Macro-F1 on internal validation"
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
    Compute the five complete-prediction metrics used by the benchmark.

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
    }

    return results


def compute_f1_with_abstentions_as_zero(y_true, y_partial, abstain_value=-1):
    """Compute standard multilabel F1 after mapping abstentions to zero."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_partial = np.asarray(y_partial, dtype=np.int32)
    if y_true.shape != y_partial.shape or y_true.ndim != 2:
        raise ValueError("y_true and y_partial must be identically shaped 2D arrays.")

    zero_filled = np.where(y_partial == abstain_value, 0, y_partial)
    return {
        "Macro-F1": float(
            f1_score(y_true, zero_filled, average="macro", zero_division=0)
        ),
        "Micro-F1": float(
            f1_score(y_true, zero_filled, average="micro", zero_division=0)
        ),
    }


def compute_selective_macro_f1(y_true, y_partial, abstain_value=-1):
    """Macro-F1 computed label-wise only over non-abstained predictions.

    A label with no decided examples contributes zero.  This conservative
    convention prevents an all-abstain output from receiving an undefined or
    artificially favorable selective score.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_partial = np.asarray(y_partial, dtype=np.int32)
    if y_true.shape != y_partial.shape or y_true.ndim != 2:
        raise ValueError("y_true and y_partial must be identically shaped 2D arrays.")

    label_scores = np.zeros(y_true.shape[1], dtype=np.float64)
    for label_index in range(y_true.shape[1]):
        decided = y_partial[:, label_index] != abstain_value
        if np.any(decided):
            label_scores[label_index] = f1_score(
                y_true[decided, label_index],
                y_partial[decided, label_index],
                zero_division=0,
            )
    return float(np.mean(label_scores)) if label_scores.size else 0.0


def compute_selective_micro_f1(y_true, y_partial, abstain_value=-1):
    """Micro-F1 over decided label-instance positions only."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_partial = np.asarray(y_partial, dtype=np.int32)
    if y_true.shape != y_partial.shape or y_true.ndim != 2:
        raise ValueError("y_true and y_partial must be identically shaped 2D arrays.")

    decided = y_partial != abstain_value
    if not np.any(decided):
        return 0.0
    return float(
        f1_score(y_true[decided], y_partial[decided], zero_division=0)
    )


def compute_selective_instance_f1(y_true, y_partial, abstain_value=-1):
    """Compute instance-based F1 restricted to decided labels for each sample.

    For each sample i:
        F1_i = 2 * |Y_{i, D} ∩ Ŷ_{i, D}| / (|Y_{i, D}| + |Ŷ_{i, D}|)
    Conventions:
        - If all labels on sample i are abstained (|D| = 0): F1_i = 0.0
        - If both true and predicted sets are empty on decided labels: F1_i = 1.0
        - Otherwise, standard harmonic mean of precision and recall on decided positions.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_partial = np.asarray(y_partial, dtype=np.int32)
    if y_true.shape != y_partial.shape or y_true.ndim != 2:
        raise ValueError("y_true and y_partial must be identically shaped 2D arrays.")
    n_samples = y_true.shape[0]
    if n_samples == 0:
        return 0.0

    decided = y_partial != abstain_value
    has_decisions = np.any(decided, axis=1)

    pred_pos = (y_partial == 1) & decided
    true_pos = (y_true == 1) & decided

    pred_sums = pred_pos.sum(axis=1)
    true_sums = true_pos.sum(axis=1)
    intersections = (pred_pos & true_pos).sum(axis=1)

    scores = np.zeros(n_samples, dtype=np.float64)
    both_empty = has_decisions & (true_sums == 0) & (pred_sums == 0)
    scores[both_empty] = 1.0

    non_empty = has_decisions & ~both_empty
    denominators = true_sums[non_empty] + pred_sums[non_empty]
    scores[non_empty] = (2.0 * intersections[non_empty]) / denominators

    return float(np.mean(scores))


def compute_partial_abstention_metrics(
    y_true, y_partial, cost, abstain_value=-1, penalty="linear"
):
    """Evaluate partial predictions under generalized Hamming loss.

    The implementation follows Eqs. (30)-(31) of Nguyen and Huellermeier
    (2021).  An incorrect decided label costs 1.  With ``penalty="linear"``
    (SEP), ``a`` abstentions cost ``a*c``.  With ``penalty="concave"`` (PAR),
    they cost ``a*K*c/(K+a)``.  The generalized loss is normalized by
    ``n_samples * n_labels`` so it remains on the Hamming-loss scale.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_partial = np.asarray(y_partial, dtype=np.int32)
    if y_true.shape != y_partial.shape:
        raise ValueError("y_true and y_partial must have identical shapes.")
    if y_true.ndim != 2:
        raise ValueError("Partial-abstention metrics require 2D label matrices.")
    if not 0.0 <= float(cost) <= 1.0:
        raise ValueError("cost must lie in [0, 1].")
    if penalty not in ("linear", "concave"):
        raise ValueError("penalty must be either 'linear' or 'concave'.")

    valid_outputs = (y_partial == 0) | (y_partial == 1) | (
        y_partial == abstain_value
    )
    if not np.all(valid_outputs):
        raise ValueError("y_partial contains values other than 0, 1, and abstain_value.")

    abstained = y_partial == abstain_value
    decided = ~abstained
    errors = decided & (y_partial != y_true)
    total_count = y_true.size
    decided_count = int(decided.sum())
    abstention_count = int(abstained.sum())
    error_count = int(errors.sum())

    coverage = decided_count / total_count if total_count else 0.0
    sample_abstention_rate = (
        float(np.mean(np.any(abstained, axis=1))) if y_true.shape[0] else 0.0
    )
    average_abstention_rate = (
        abstention_count / total_count if total_count else 0.0
    )
    selective_hamming = error_count / decided_count if decided_count else 0.0
    if total_count:
        errors_per_sample = errors.sum(axis=1, dtype=np.float64)
        abstentions_per_sample = abstained.sum(axis=1, dtype=np.float64)
        if penalty == "linear":
            abstention_penalties = float(cost) * abstentions_per_sample
        else:
            n_labels = float(y_true.shape[1])
            abstention_penalties = (
                abstentions_per_sample
                * n_labels
                * float(cost)
                / (n_labels + abstentions_per_sample)
            )
        generalized_hamming = float(
            np.sum(errors_per_sample + abstention_penalties) / total_count
        )
    else:
        generalized_hamming = 0.0

    zero_filled_f1 = compute_f1_with_abstentions_as_zero(
        y_true, y_partial, abstain_value=abstain_value
    )
    return {
        # Keep this diagnostic convention under explicit names.  In particular,
        # never overwrite Macro-F1/Micro-F1 computed from the complete output.
        "Abstentions-as-Zero Macro-F1": zero_filled_f1["Macro-F1"],
        "Abstentions-as-Zero Micro-F1": zero_filled_f1["Micro-F1"],
        "Generalized Loss": generalized_hamming,
        # Backward-compatible alias used by older result tables.
        "Generalized Hamming Loss": generalized_hamming,
        "Selective Hamming Loss": float(selective_hamming),
        "Selective Macro-F1": compute_selective_macro_f1(
            y_true, y_partial, abstain_value=abstain_value
        ),
        "Selective Micro-F1": compute_selective_micro_f1(
            y_true, y_partial, abstain_value=abstain_value
        ),
        "Selective Instance-F1": compute_selective_instance_f1(
            y_true, y_partial, abstain_value=abstain_value
        ),
        "Coverage": float(coverage),
        "ABS": sample_abstention_rate,
        "AABS": float(average_abstention_rate),
        # Backward-compatible alias: the previous implementation's abstention
        # rate was the fraction of abstained label-instance positions (AABS).
        "Abstention Rate": float(average_abstention_rate),
    }
