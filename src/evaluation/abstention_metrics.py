"""Selective, rejected and optimistic schema-v3 metrics."""

import numpy as np

from .complete_metrics import compute_complete_metrics
from .metric_contract import (
    OPTIMISTIC_METRIC_NAMES,
    ORACLE_GAIN_METRIC_NAMES,
    REJECTED_METRIC_NAMES,
    SELECTIVE_METRIC_NAMES,
)
from .metric_utils import (
    positive_precision_recall_f1,
    validate_partial_inputs,
)


def _validate_cost_and_penalty(cost, penalty):
    try:
        normalized_cost = float(cost)
    except (TypeError, ValueError) as exc:
        raise ValueError("cost must be a finite number in [0, 1].") from exc
    if not np.isfinite(normalized_cost) or not 0.0 <= normalized_cost <= 1.0:
        raise ValueError("cost must be a finite number in [0, 1].")
    if penalty not in ("linear", "concave"):
        raise ValueError("penalty must be either 'linear' or 'concave'.")
    return normalized_cost


def _compute_aurc(y_true, y_full, acceptance_confidence):
    if acceptance_confidence is None:
        return float("nan")
    try:
        confidence = np.asarray(acceptance_confidence, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("acceptance_confidence must be a finite numeric matrix.") from exc
    if confidence.shape != y_true.shape:
        raise ValueError("acceptance_confidence must have the same shape as y_true.")
    if not np.all(np.isfinite(confidence)):
        raise ValueError("acceptance_confidence must contain only finite values.")

    # Stable sorting makes ties deterministic in row-major flattened order.
    order = np.argsort(-confidence.ravel(), kind="stable")
    ordered_errors = (y_full != y_true).ravel()[order].astype(np.float64)
    prefix_risk = np.cumsum(ordered_errors) / np.arange(
        1, ordered_errors.size + 1, dtype=np.float64
    )
    return float(np.mean(prefix_risk))


def compute_abstention_metrics(
    y_true,
    y_partial,
    y_full,
    *,
    cost,
    penalty="linear",
    abstain_value=-1,
    acceptance_confidence=None,
):
    """Compute partial-abstention metrics under explicitly separated scopes.

    ``y_full`` is the counterfactual complete output used for rejected-set
    diagnostics.  ``acceptance_confidence`` is optional; without it AURC is
    undefined and returned as ``NaN`` while all operating-point metrics remain
    available.
    """

    truth, partial, full = validate_partial_inputs(
        y_true, y_partial, y_full, abstain_value=abstain_value
    )
    normalized_cost = _validate_cost_and_penalty(cost, penalty)
    abstained = partial == abstain_value
    decided = ~abstained
    decided_count = int(np.count_nonzero(decided))
    abstained_count = int(np.count_nonzero(abstained))
    total_count = int(truth.size)
    decided_errors = decided & (partial != truth)
    decided_error_count = int(np.count_nonzero(decided_errors))

    coverage = decided_count / total_count
    selective_hamming_accuracy = (
        1.0 - decided_error_count / decided_count
        if decided_count
        else float("nan")
    )
    label_f1 = []
    for label_index in range(truth.shape[1]):
        label_decided = decided[:, label_index]
        if not np.any(label_decided):
            label_f1.append(0.0)
            continue
        _, _, f1 = positive_precision_recall_f1(
            truth[label_decided, label_index],
            partial[label_decided, label_index],
        )
        label_f1.append(f1)
    if decided_count:
        _, _, selective_micro_f1 = positive_precision_recall_f1(
            truth[decided], partial[decided]
        )
    else:
        selective_micro_f1 = 0.0

    abstentions_per_instance = np.count_nonzero(abstained, axis=1).astype(np.float64)
    errors_per_instance = np.count_nonzero(decided_errors, axis=1).astype(np.float64)
    if penalty == "linear":
        abstention_penalty = normalized_cost * abstentions_per_instance
    else:
        n_labels = float(truth.shape[1])
        abstention_penalty = (
            normalized_cost
            * n_labels
            * abstentions_per_instance
            / (n_labels + abstentions_per_instance)
        )
    generalized_loss = float(
        np.sum(errors_per_instance + abstention_penalty) / total_count
    )

    selective = {
        "Selective Hamming Accuracy": float(selective_hamming_accuracy),
        "Selective Macro-F1": float(np.mean(label_f1)),
        "Selective Micro-F1": float(selective_micro_f1),
        "Generalized Loss": generalized_loss,
        "Coverage": float(coverage),
        "ABS": float(np.mean(np.any(abstained, axis=1))),
        "AABS": float(abstained_count / total_count),
        "Risk at Coverage": (
            float(decided_error_count / decided_count)
            if decided_count
            else float("nan")
        ),
        "AURC": _compute_aurc(truth, full, acceptance_confidence),
    }

    rejected_errors = abstained & (full != truth)
    rejected_count = abstained_count
    rejected_error_count = int(np.count_nonzero(rejected_errors))
    full_error_count = int(np.count_nonzero(full != truth))
    rejected_label_f1 = []
    for label_index in range(truth.shape[1]):
        label_rejected = abstained[:, label_index]
        if not np.any(label_rejected):
            continue
        _, _, f1 = positive_precision_recall_f1(
            truth[label_rejected, label_index], full[label_rejected, label_index]
        )
        rejected_label_f1.append(f1)
    rejected = {
        "Rejected Counterfactual Macro-F1": (
            float(np.mean(rejected_label_f1))
            if rejected_label_f1
            else float("nan")
        ),
        "Rejected Error Rate": (
            float(rejected_error_count / rejected_count)
            if rejected_count
            else float("nan")
        ),
        "Error Capture Rate": (
            float(rejected_error_count / full_error_count)
            if full_error_count
            else float("nan")
        ),
    }

    oracle = np.where(abstained, truth, partial)
    full_metrics = compute_complete_metrics(truth, full)
    oracle_metrics = compute_complete_metrics(truth, oracle)
    optimistic_bases = tuple(
        name.replace("Optimistic ", "", 1) for name in OPTIMISTIC_METRIC_NAMES
    )
    optimistic = {
        f"Optimistic {name}": oracle_metrics[name] for name in optimistic_bases
    }
    optimistic.update(
        {
            f"Oracle Gain {name}": oracle_metrics[name] - full_metrics[name]
            for name in optimistic_bases
        }
    )
    diagnostics = {
        "Total Position Count": total_count,
        "Decided Position Count": decided_count,
        "Abstained Position Count": abstained_count,
        "Full Error Count": full_error_count,
        "Rejected Position Count": rejected_count,
        "Rejected Error Count": rejected_error_count,
        "Errors Avoided": rejected_error_count,
    }

    if tuple(selective) != SELECTIVE_METRIC_NAMES:
        raise RuntimeError("Selective metric output no longer matches the contract.")
    if tuple(rejected) != REJECTED_METRIC_NAMES:
        raise RuntimeError("Rejected metric output no longer matches the contract.")
    if tuple(optimistic) != OPTIMISTIC_METRIC_NAMES + ORACLE_GAIN_METRIC_NAMES:
        raise RuntimeError("Optimistic metric output no longer matches the contract.")
    return {
        "Selective": selective,
        "Rejected": rejected,
        "Optimistic": optimistic,
        "Diagnostics": diagnostics,
    }
