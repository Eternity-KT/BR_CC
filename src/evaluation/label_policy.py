"""Domain-provided critical-label and deployment-cost configuration."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


DEFAULT_LABEL_POLICY_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "label_policy.json"
)
DEFAULT_REVIEWER_ACCURACIES = (0.80, 0.90, 0.95, 1.00)


def load_label_policy_config(path=None):
    """Load a policy document; a missing file means no domain policy."""

    policy_path = DEFAULT_LABEL_POLICY_PATH if path is None else Path(path)
    if not policy_path.exists():
        return {
            "schema_version": 1,
            "datasets": {},
            "source_path": str(policy_path),
            "status": "missing",
        }
    with policy_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("label_policy root must be a JSON object.")
    if "datasets" in payload:
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported label_policy schema version.")
        datasets = payload["datasets"]
    else:
        # Accept the meeting-note shape where dataset names are root keys.
        datasets = payload
        payload = {"schema_version": 1, "datasets": datasets}
    if not isinstance(datasets, dict):
        raise ValueError("label_policy datasets must be a JSON object.")
    result = deepcopy(payload)
    result["source_path"] = str(policy_path)
    result["status"] = "available"
    return result


def _numeric_mapping(value, field_name, label_names):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object keyed by label name.")
    output = {}
    for raw_label, raw_cost in value.items():
        label = str(raw_label)
        if label_names is not None and label not in label_names:
            raise ValueError(f"{field_name} references unknown label: {label}")
        if isinstance(raw_cost, bool) or not isinstance(raw_cost, (int, float)):
            raise ValueError(f"{field_name}[{label}] must be numeric.")
        cost = float(raw_cost)
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError(f"{field_name}[{label}] must be finite and nonnegative.")
        output[label] = cost
    return output


def validate_dataset_label_policy(dataset_name, policy, label_names=None):
    """Validate one dataset policy without consulting labels or predictions."""

    if not isinstance(policy, dict):
        raise ValueError(f"Policy for {dataset_name} must be a JSON object.")
    known_labels = None if label_names is None else tuple(str(name) for name in label_names)
    critical = policy.get("critical_labels", [])
    if not isinstance(critical, list) or any(
        not isinstance(label, str) or not label.strip() for label in critical
    ):
        raise ValueError("critical_labels must be a list of non-empty names.")
    critical = [label.strip() for label in critical]
    if len(critical) != len(set(critical)):
        raise ValueError("critical_labels cannot contain duplicates.")
    if known_labels is not None:
        unknown = sorted(set(critical) - set(known_labels))
        if unknown:
            raise ValueError(f"critical_labels contains unknown labels: {unknown}")

    review_cost = policy.get("review_cost")
    if review_cost is not None:
        if isinstance(review_cost, bool) or not isinstance(review_cost, (int, float)):
            raise ValueError("review_cost must be numeric.")
        review_cost = float(review_cost)
        if not math.isfinite(review_cost) or review_cost < 0.0:
            raise ValueError("review_cost must be finite and nonnegative.")

    reviewer_accuracies = policy.get(
        "reviewer_accuracies", list(DEFAULT_REVIEWER_ACCURACIES)
    )
    if not isinstance(reviewer_accuracies, list) or not reviewer_accuracies:
        raise ValueError("reviewer_accuracies must be a non-empty list.")
    normalized_accuracies = []
    for accuracy in reviewer_accuracies:
        if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)):
            raise ValueError("reviewer accuracies must be numeric.")
        accuracy = float(accuracy)
        if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
            raise ValueError("reviewer accuracies must lie in [0, 1].")
        normalized_accuracies.append(accuracy)
    if len(normalized_accuracies) != len(set(normalized_accuracies)):
        raise ValueError("reviewer_accuracies cannot contain duplicates.")

    return {
        "dataset": str(dataset_name),
        "critical_labels": critical,
        "weights": _numeric_mapping(policy.get("weights"), "weights", known_labels),
        "false_negative_cost": _numeric_mapping(
            policy.get("false_negative_cost"),
            "false_negative_cost",
            known_labels,
        ),
        "false_positive_cost": _numeric_mapping(
            policy.get("false_positive_cost"),
            "false_positive_cost",
            known_labels,
        ),
        "review_cost": review_cost,
        "reviewer_accuracies": sorted(normalized_accuracies),
    }


def get_dataset_label_policy(config, dataset_name, label_names=None):
    datasets = config.get("datasets", {})
    if dataset_name not in datasets:
        return None
    return validate_dataset_label_policy(
        dataset_name,
        datasets[dataset_name],
        label_names=label_names,
    )


def label_policy_hash(policy):
    if policy is None:
        return None
    encoded = json.dumps(
        policy,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
