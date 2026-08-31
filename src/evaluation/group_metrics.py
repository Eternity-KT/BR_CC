"""Per-label and named-group schema-v3 metrics."""

from collections.abc import Mapping

import numpy as np

from .metric_contract import group_metric_names
from .metric_utils import (
    instance_set_scores,
    positive_precision_recall_f1,
    validate_complete_inputs,
    validate_label_names,
    validate_partial_inputs,
)


def compute_per_label_metrics(
    y_true,
    y_full,
    y_partial=None,
    *,
    abstain_value=-1,
    label_names=None,
):
    """Return support, quality, coverage and rejected counts for each label."""

    if y_partial is None:
        truth, full = validate_complete_inputs(y_true, y_full)
        partial = full
    else:
        truth, partial, full = validate_partial_inputs(
            y_true, y_partial, y_full, abstain_value=abstain_value
        )
    names = validate_label_names(label_names, truth.shape[1])
    records = []
    for label_index, label_name in enumerate(names):
        true_label = truth[:, label_index]
        full_label = full[:, label_index]
        partial_label = partial[:, label_index]
        precision, recall, f1 = positive_precision_recall_f1(
            true_label, full_label
        )
        decided = partial_label != abstain_value
        rejected = ~decided
        if np.any(decided):
            selective_precision, selective_recall, selective_f1 = (
                positive_precision_recall_f1(
                    true_label[decided], partial_label[decided]
                )
            )
        else:
            selective_precision = selective_recall = selective_f1 = 0.0
        positive_support = int(np.count_nonzero(true_label == 1))
        records.append(
            {
                "Label Index": label_index,
                "Label Name": label_name,
                "Positive Support": positive_support,
                "Negative Support": int(true_label.size - positive_support),
                "Prevalence": float(positive_support / true_label.size),
                "Full Precision": precision,
                "Full Recall": recall,
                "Full F1": f1,
                "Full Hamming Accuracy": float(np.mean(true_label == full_label)),
                "Selective Precision": selective_precision,
                "Selective Recall": selective_recall,
                "Selective F1": selective_f1,
                "Coverage": float(np.mean(decided)),
                "Rejected Position Count": int(np.count_nonzero(rejected)),
                "Rejected Error Count": int(
                    np.count_nonzero(rejected & (full_label != true_label))
                ),
            }
        )
    return records


def _validate_label_groups(label_groups, n_labels):
    if not isinstance(label_groups, Mapping):
        raise ValueError("label_groups must be a mapping of name to label indices.")
    normalized = {}
    for raw_name, raw_indices in label_groups.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("Group names must be non-empty.")
        if name in normalized:
            raise ValueError(f"Duplicate group name after normalization: {name}")
        try:
            indices = tuple(raw_indices)
        except TypeError as exc:
            raise ValueError(f"Group {name} indices must be iterable.") from exc
        normalized_indices = []
        seen_indices = set()
        for index in indices:
            if isinstance(index, (bool, np.bool_)) or not isinstance(
                index, (int, np.integer)
            ):
                raise ValueError(f"Group {name} label indices must be integers.")
            normalized_index = int(index)
            if normalized_index in seen_indices:
                raise ValueError(f"Group {name} contains duplicate label indices.")
            if not 0 <= normalized_index < n_labels:
                raise ValueError(f"Group {name} contains out-of-range label {index}.")
            normalized_indices.append(normalized_index)
            seen_indices.add(normalized_index)
        normalized[name] = tuple(normalized_indices)
    return normalized


def compute_group_metrics(
    y_true,
    y_full,
    y_partial,
    label_groups,
    *,
    abstain_value=-1,
):
    """Aggregate full quality and partial coverage over named label groups."""

    if y_partial is None:
        truth, full = validate_complete_inputs(y_true, y_full)
        partial = full
    else:
        truth, partial, full = validate_partial_inputs(
            y_true, y_partial, y_full, abstain_value=abstain_value
        )
    groups = _validate_label_groups(label_groups, truth.shape[1])
    output = {}
    for group_name, indices in groups.items():
        keys = group_metric_names(group_name)
        if not indices:
            output[group_name] = {
                keys[0]: 0,
                **{key: float("nan") for key in keys[1:]},
            }
            continue

        precision_scores = []
        recall_scores = []
        f1_scores = []
        for label_index in indices:
            precision, recall, f1 = positive_precision_recall_f1(
                truth[:, label_index], full[:, label_index]
            )
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        _, instance_jaccard = instance_set_scores(
            truth[:, indices], full[:, indices]
        )
        output[group_name] = {
            keys[0]: len(indices),
            keys[1]: float(np.mean(f1_scores)),
            keys[2]: float(np.mean(precision_scores)),
            keys[3]: float(np.mean(recall_scores)),
            keys[4]: instance_jaccard,
            keys[5]: float(np.mean(partial[:, indices] != abstain_value)),
        }
    return output
