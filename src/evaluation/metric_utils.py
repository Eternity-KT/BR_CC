"""Internal validation and binary-confusion helpers for schema-v3 metrics."""

import numpy as np


def as_binary_matrix(value, name):
    """Return a non-empty 2D int32 matrix containing only zero and one."""

    matrix = np.asarray(value)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must have positive N and K dimensions.")
    if not np.all(np.isin(matrix, (0, 1))):
        raise ValueError(f"{name} must contain only binary values 0 and 1.")
    return matrix.astype(np.int32, copy=False)


def validate_complete_inputs(y_true, y_full):
    """Validate and return identically shaped complete binary matrices."""

    truth = as_binary_matrix(y_true, "y_true")
    full = as_binary_matrix(y_full, "y_full")
    if truth.shape != full.shape:
        raise ValueError("y_true and y_full must have identical shapes.")
    return truth, full


def validate_partial_inputs(y_true, y_partial, y_full, abstain_value=-1):
    """Validate complete matrices plus a partial matrix using one sentinel."""

    truth, full = validate_complete_inputs(y_true, y_full)
    if abstain_value in (0, 1):
        raise ValueError("abstain_value must differ from binary outputs 0 and 1.")

    partial = np.asarray(y_partial)
    if partial.ndim != 2 or partial.shape != truth.shape:
        raise ValueError(
            "y_partial must be a 2D matrix with the same shape as y_true."
        )
    if not np.all(np.isin(partial, (0, 1, abstain_value))):
        raise ValueError(
            "y_partial must contain only 0, 1, and the configured abstain_value."
        )
    return truth, partial.astype(np.int32, copy=False), full


def positive_class_counts(y_true, y_pred):
    """Return positive-class TP, FP and FN for equally shaped arrays."""

    truth = np.asarray(y_true, dtype=bool)
    predicted = np.asarray(y_pred, dtype=bool)
    if truth.shape != predicted.shape:
        raise ValueError("Positive-class count inputs must have identical shapes.")
    tp = int(np.count_nonzero(truth & predicted))
    fp = int(np.count_nonzero(~truth & predicted))
    fn = int(np.count_nonzero(truth & ~predicted))
    return tp, fp, fn


def positive_precision_recall_f1(y_true, y_pred):
    """Return positive-class precision, recall and F1 with zero_division=0."""

    tp, fp, fn = positive_class_counts(y_true, y_pred)
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    precision = tp / precision_denominator if precision_denominator else 0.0
    recall = tp / recall_denominator if recall_denominator else 0.0
    f1 = 2 * tp / f1_denominator if f1_denominator else 0.0
    return float(precision), float(recall), float(f1)


def instance_set_scores(y_true, y_pred):
    """Return mean instance-F1 and Jaccard with both-empty instances scoring 1."""

    truth = np.asarray(y_true, dtype=bool)
    predicted = np.asarray(y_pred, dtype=bool)
    if truth.ndim != 2 or truth.shape != predicted.shape:
        raise ValueError("Instance-set inputs must be identically shaped 2D matrices.")

    intersections = np.count_nonzero(truth & predicted, axis=1)
    true_sizes = np.count_nonzero(truth, axis=1)
    predicted_sizes = np.count_nonzero(predicted, axis=1)
    f1_denominators = true_sizes + predicted_sizes
    union_sizes = true_sizes + predicted_sizes - intersections

    f1_scores = np.ones(truth.shape[0], dtype=np.float64)
    jaccard_scores = np.ones(truth.shape[0], dtype=np.float64)
    nonempty_f1 = f1_denominators > 0
    nonempty_union = union_sizes > 0
    f1_scores[nonempty_f1] = (
        2.0 * intersections[nonempty_f1] / f1_denominators[nonempty_f1]
    )
    jaccard_scores[nonempty_union] = (
        intersections[nonempty_union] / union_sizes[nonempty_union]
    )
    return float(np.mean(f1_scores)), float(np.mean(jaccard_scores))


def validate_label_names(label_names, n_labels):
    """Return stable display names for every label."""

    if label_names is None:
        return tuple(f"label_{index}" for index in range(n_labels))
    names = tuple(str(name) for name in label_names)
    if len(names) != n_labels:
        raise ValueError("label_names must contain exactly K entries.")
    if len(set(names)) != len(names):
        raise ValueError("label_names must be unique.")
    return names
