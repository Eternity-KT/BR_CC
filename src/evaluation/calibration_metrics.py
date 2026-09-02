"""Probability calibration metrics and reliability-bin audit data."""

import numpy as np


CALIBRATION_METRIC_NAMES = ("Brier Score", "Log Loss", "ECE")


def _validate_inputs(y_true, probabilities, n_bins):
    truth = np.asarray(y_true, dtype=np.int32)
    predicted = np.asarray(probabilities, dtype=np.float64)
    if truth.ndim != 2 or predicted.shape != truth.shape:
        raise ValueError("y_true and probabilities must share a two-dimensional shape.")
    if truth.shape[0] == 0 or truth.shape[1] == 0:
        raise ValueError("Calibration metrics require a non-empty matrix.")
    if not np.all(np.isin(truth, (0, 1))):
        raise ValueError("y_true must contain only binary values 0 and 1.")
    if not np.all(np.isfinite(predicted)) or np.any(
        (predicted < 0.0) | (predicted > 1.0)
    ):
        raise ValueError("probabilities must be finite values in [0, 1].")
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(
        n_bins, (int, np.integer)
    ) or int(n_bins) < 2:
        raise ValueError("n_bins must be an integer of at least 2.")
    return truth, predicted, int(n_bins)


def _reliability_records(truth, predicted, n_bins):
    flat_truth = np.ravel(truth).astype(np.float64)
    flat_predicted = np.ravel(predicted)
    bin_indices = np.minimum(
        (flat_predicted * n_bins).astype(np.int64), n_bins - 1
    )
    records = []
    weighted_gap = 0.0
    for bin_index in range(n_bins):
        selected = bin_indices == bin_index
        count = int(np.sum(selected))
        lower = float(bin_index / n_bins)
        upper = float((bin_index + 1) / n_bins)
        if count:
            mean_probability = float(np.mean(flat_predicted[selected]))
            observed_rate = float(np.mean(flat_truth[selected]))
            gap = float(abs(mean_probability - observed_rate))
            weighted_gap += count * gap
        else:
            mean_probability = float("nan")
            observed_rate = float("nan")
            gap = float("nan")
        records.append({
            "Bin": int(bin_index),
            "Lower Bound": lower,
            "Upper Bound": upper,
            "Count": count,
            "Mean Predicted Probability": mean_probability,
            "Observed Positive Rate": observed_rate,
            "Absolute Gap": gap,
        })
    return records, float(weighted_gap / len(flat_truth))


def _scalar_metrics(truth, predicted, n_bins):
    clipped = np.clip(predicted, 1e-15, 1.0 - 1e-15)
    brier = float(np.mean((predicted - truth) ** 2))
    log_loss = float(
        -np.mean(truth * np.log(clipped) + (1 - truth) * np.log(1.0 - clipped))
    )
    reliability, ece = _reliability_records(truth, predicted, n_bins)
    return {
        "Brier Score": brier,
        "Log Loss": log_loss,
        "ECE": ece,
    }, reliability


def compute_calibration_metrics(
    y_true,
    probabilities,
    *,
    n_bins=10,
    label_names=None,
):
    """Return aggregate/per-label metrics plus raw reliability-bin records."""

    truth, predicted, n_bins = _validate_inputs(y_true, probabilities, n_bins)
    if label_names is None:
        names = [str(index) for index in range(truth.shape[1])]
    else:
        names = [str(name) for name in label_names]
        if len(names) != truth.shape[1]:
            raise ValueError("label_names length must match the label count.")
    metrics, reliability = _scalar_metrics(truth, predicted, n_bins)
    per_label = []
    for label_index, label_name in enumerate(names):
        label_metrics, _ = _scalar_metrics(
            truth[:, [label_index]], predicted[:, [label_index]], n_bins
        )
        per_label.append({
            "Label Index": int(label_index),
            "Label Name": label_name,
            **label_metrics,
        })
    return {
        "Metrics": metrics,
        "Per Label": per_label,
        "Reliability": reliability,
        "Bin Count": int(n_bins),
        "Sample Label Count": int(truth.size),
    }
