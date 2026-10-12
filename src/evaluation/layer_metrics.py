"""v6.3 selective metrics and IL/DL layer audit rows."""

import json
from typing import Iterable, List

import numpy as np

from .abstention_metrics import compute_abstention_metrics
from .complete_metrics import compute_complete_metrics
from .metrics import compute_selective_instance_f1


def compute_selective_subset_accuracy(
    y_true, y_partial, *, abstain_value: int = -1
) -> float:
    """Exact correctness on decided positions, excluding all-abstain rows."""
    truth = np.asarray(y_true, dtype=np.int32)
    partial = np.asarray(y_partial, dtype=np.int32)
    if truth.ndim != 2 or truth.shape != partial.shape:
        raise ValueError("y_true and y_partial must share a two-dimensional shape.")
    decided = partial != abstain_value
    eligible = np.any(decided, axis=1)
    if not np.any(eligible):
        return float("nan")
    correct = np.all((partial == truth) | ~decided, axis=1)
    return float(np.mean(correct[eligible]))


def compute_v6_3_selective_metrics(
    y_true,
    y_partial,
    y_full,
    *,
    cost: float,
    penalty: str = "linear",
    abstain_value: int = -1,
):
    """Return the exact selective table schema required by the v6.3 spec."""
    result = compute_abstention_metrics(
        y_true,
        y_partial,
        y_full,
        cost=cost,
        penalty=penalty,
        abstain_value=abstain_value,
    )["Selective"]
    return {
        "selective_hamming_accuracy": result["Selective Hamming Accuracy"],
        "selective_subset_accuracy": compute_selective_subset_accuracy(
            y_true, y_partial, abstain_value=abstain_value
        ),
        "selective_micro_f1": result["Selective Micro-F1"],
        "selective_macro_f1": result["Selective Macro-F1"],
        "selective_instance_f1": compute_selective_instance_f1(
            y_true, y_partial, abstain_value=abstain_value
        ),
        "generalized_loss": result["Generalized Loss"],
        "coverage": result["Coverage"],
        "abs": result["ABS"],
        "aabs": result["AABS"],
    }


def _group_metrics(y_true, y_full, y_partial, labels, abstain_value):
    labels = list(labels)
    if not labels:
        return None
    truth = np.asarray(y_true)[:, labels]
    full = np.asarray(y_full)[:, labels]
    partial = np.asarray(y_partial)[:, labels]
    complete = compute_complete_metrics(truth, full)
    decided = partial != abstain_value
    coverage = float(np.mean(decided))
    from .metrics import compute_selective_macro_f1

    return {
        "full_macro_f1": complete["Macro-F1"],
        "selective_macro_f1": compute_selective_macro_f1(
            truth, partial, abstain_value=abstain_value
        ),
        "coverage": coverage,
    }


def build_layer_rows(
    *,
    common: dict,
    y_true,
    y_full,
    y_partial,
    independent_labels: Iterable[int],
    dependent_labels: Iterable[int],
    abstain_value: int = -1,
) -> List[dict]:
    """Build IL_1 and DL layer rows, including cumulative metrics."""
    il = [int(label) for label in independent_labels]
    dl = [int(label) for label in dependent_labels]
    rows = []
    groups = (("IL", 1, il, il), ("DL", 0, dl, il + dl))
    for layer_type, layer_index, labels, cumulative_labels in groups:
        group = _group_metrics(
            y_true, y_full, y_partial, labels, abstain_value
        )
        cumulative = _group_metrics(
            y_true, y_full, y_partial, cumulative_labels, abstain_value
        )
        rows.append(
            {
                **common,
                "layer_type": layer_type,
                "layer_index": layer_index,
                "label_indices": json.dumps(labels, separators=(",", ":")),
                "label_count": len(labels),
                "cumulative_label_count": len(cumulative_labels),
                "full_macro_f1": None if group is None else group["full_macro_f1"],
                "selective_macro_f1": (
                    None if group is None else group["selective_macro_f1"]
                ),
                "coverage": None if group is None else group["coverage"],
                "cumulative_full_macro_f1": (
                    None if cumulative is None else cumulative["full_macro_f1"]
                ),
                "cumulative_selective_macro_f1": (
                    None if cumulative is None else cumulative["selective_macro_f1"]
                ),
                "cumulative_coverage": (
                    None if cumulative is None else cumulative["coverage"]
                ),
                "status": "empty_group" if group is None else "ok",
            }
        )
    return rows


__all__ = [
    "build_layer_rows",
    "compute_selective_subset_accuracy",
    "compute_v6_3_selective_metrics",
]
