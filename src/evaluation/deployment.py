"""Deployment metrics and leakage-safe operating-point selection."""

import math

import numpy as np

from .abstention_metrics import compute_abstention_metrics
from .complete_metrics import compute_complete_metrics
from .metric_utils import (
    positive_precision_recall_f1,
    validate_label_names,
    validate_partial_inputs,
)


OPERATING_POINT_RULES = (
    "min_generalized_loss",
    "max_utility_at_coverage",
    "max_coverage_at_risk",
)


def _critical_metrics(truth, full, partial, policy, label_names, abstain_value):
    if policy is None:
        return {
            "Status": "N/A",
            "Reason": "label policy is not configured by the domain",
            "Metrics": {},
        }
    critical_names = tuple(policy.get("critical_labels", ()))
    if not critical_names:
        return {
            "Status": "N/A",
            "Reason": "label policy contains no critical labels",
            "Metrics": {},
        }
    by_name = {name: index for index, name in enumerate(label_names)}
    indices = [by_name[name] for name in critical_names]
    decided = partial[:, indices] != abstain_value
    full_errors = full[:, indices] != truth[:, indices]
    rejected_errors = full_errors & ~decided
    full_error_count = int(np.count_nonzero(full_errors))
    rejected_error_count = int(np.count_nonzero(rejected_errors))
    coverage = float(np.mean(decided))

    selective_recall = []
    selective_f1 = []
    optimistic_f1 = []
    for local_index, _ in enumerate(indices):
        label_decided = decided[:, local_index]
        if np.any(label_decided):
            _, recall, f1 = positive_precision_recall_f1(
                truth[label_decided, indices[local_index]],
                partial[label_decided, indices[local_index]],
            )
        else:
            recall, f1 = 0.0, 0.0
        oracle = np.where(
            decided[:, local_index],
            partial[:, indices[local_index]],
            truth[:, indices[local_index]],
        )
        _, _, oracle_f1 = positive_precision_recall_f1(
            truth[:, indices[local_index]], oracle
        )
        selective_recall.append(recall)
        selective_f1.append(f1)
        optimistic_f1.append(oracle_f1)

    return {
        "Status": "Available",
        "Reason": None,
        "Critical Labels": list(critical_names),
        "Metrics": {
            "Critical Selective Recall": float(np.mean(selective_recall)),
            "Critical Selective F1": float(np.mean(selective_f1)),
            "Critical Coverage": coverage,
            "Critical Error Capture Rate": (
                float(rejected_error_count / full_error_count)
                if full_error_count
                else float("nan")
            ),
            "Optimistic Critical F1": float(np.mean(optimistic_f1)),
            "Critical Full Error Count": full_error_count,
            "Critical Rejected Error Count": rejected_error_count,
        },
    }


def _cost_vectors(policy, label_names):
    weights = policy.get("weights", {})
    false_negative = policy.get("false_negative_cost", {})
    false_positive = policy.get("false_positive_cost", {})
    return (
        np.asarray([weights.get(name, 1.0) for name in label_names], dtype=np.float64),
        np.asarray(
            [false_negative.get(name, 1.0) for name in label_names],
            dtype=np.float64,
        ),
        np.asarray(
            [false_positive.get(name, 1.0) for name in label_names],
            dtype=np.float64,
        ),
    )


def _reviewer_scenarios(
    truth,
    partial,
    policy,
    label_names,
    abstain_value,
):
    if policy is None or policy.get("review_cost") is None:
        return {
            "Status": "N/A",
            "Reason": "review cost/domain assumptions are not configured",
            "Records": [],
        }
    weights, fn_cost, fp_cost = _cost_vectors(policy, label_names)
    decided = partial != abstain_value
    abstained = ~decided
    predicted_positive = partial == 1
    predicted_negative = partial == 0
    decided_cost = float(np.sum(
        weights
        * (
            fn_cost * np.sum(decided & predicted_negative & (truth == 1), axis=0)
            + fp_cost * np.sum(decided & predicted_positive & (truth == 0), axis=0)
        )
    ))
    review_cost = float(policy["review_cost"])
    review_total = float(np.count_nonzero(abstained) * review_cost)
    decided_correct = int(np.count_nonzero(decided & (partial == truth)))
    abstained_count = int(np.count_nonzero(abstained))
    records = []
    for accuracy in policy["reviewer_accuracies"]:
        reviewer_error_cost = float((1.0 - accuracy) * np.sum(
            weights
            * (
                fn_cost * np.sum(abstained & (truth == 1), axis=0)
                + fp_cost * np.sum(abstained & (truth == 0), axis=0)
            )
        ))
        total_cost = decided_cost + review_total + reviewer_error_cost
        records.append({
            "Reviewer Accuracy": float(accuracy),
            "Expected Hamming Accuracy": float(
                (decided_correct + accuracy * abstained_count) / truth.size
            ),
            "Expected Error Count": float(
                np.count_nonzero(decided & (partial != truth))
                + (1.0 - accuracy) * abstained_count
            ),
            "Prediction Error Cost": decided_cost,
            "Review Cost": review_total,
            "Reviewer Error Cost": reviewer_error_cost,
            "Total Expected Cost": total_cost,
            "Cost-Sensitive Utility": float(-total_cost / truth.size),
        })
    return {"Status": "Available", "Reason": None, "Records": records}


def compute_deployment_metrics(
    y_true,
    y_full,
    y_partial,
    *,
    cost,
    penalty="linear",
    abstain_value=-1,
    label_names=None,
    label_policy=None,
    acceptance_confidence=None,
):
    """Evaluate workload, triage, optimistic gain and reviewer scenarios."""

    truth, partial, full = validate_partial_inputs(
        y_true, y_partial, y_full, abstain_value=abstain_value
    )
    names = validate_label_names(label_names, truth.shape[1])
    immediate = compute_complete_metrics(truth, full)
    abstention = compute_abstention_metrics(
        truth,
        partial,
        full,
        cost=cost,
        penalty=penalty,
        abstain_value=abstain_value,
        acceptance_confidence=acceptance_confidence,
    )
    selective = abstention["Selective"]
    rejected = abstention["Rejected"]
    optimistic = abstention["Optimistic"]
    coverage = float(selective["Coverage"])
    aabs = float(selective["AABS"])
    selective_accuracy = float(selective["Selective Hamming Accuracy"])
    risk = (
        float(1.0 - selective_accuracy)
        if math.isfinite(selective_accuracy)
        else float("nan")
    )
    error_capture = float(rejected["Error Capture Rate"])
    rejected_error_rate = float(rejected["Rejected Error Rate"])
    capture_lift = (
        float(error_capture / aabs)
        if aabs > 0.0 and math.isfinite(error_capture)
        else float("nan")
    )
    metrics = {
        "Immediate Hamming Accuracy": float(immediate["Hamming Accuracy"]),
        "Immediate Macro-F1": float(immediate["Macro-F1"]),
        "Immediate Instance-F1": float(immediate["Instance-F1"]),
        "Coverage": coverage,
        "Selective Risk": risk,
        "Risk at Coverage": risk,
        "Selective Hamming Accuracy": selective_accuracy,
        "Generalized Loss": float(selective["Generalized Loss"]),
        "AURC": float(selective["AURC"]),
        "Review Load Position": aabs,
        "Review Load Case": float(selective["ABS"]),
        "Rejected Error Rate": rejected_error_rate,
        "Error Capture Rate": error_capture,
        "Random-Rejection Expected Capture": aabs,
        "Error Capture Lift": capture_lift,
        "Optimistic Hamming Accuracy": float(
            optimistic["Optimistic Hamming Accuracy"]
        ),
        "Optimistic Macro-F1": float(optimistic["Optimistic Macro-F1"]),
        "Optimistic Gain Hamming Accuracy": float(
            optimistic["Oracle Gain Hamming Accuracy"]
        ),
        "Optimistic Gain Macro-F1": float(
            optimistic["Oracle Gain Macro-F1"]
        ),
        "Full Error Count": int(abstention["Diagnostics"]["Full Error Count"]),
        "Reviewed Position Count": int(
            abstention["Diagnostics"]["Rejected Position Count"]
        ),
        "Captured Error Count": int(
            abstention["Diagnostics"]["Rejected Error Count"]
        ),
    }
    critical = _critical_metrics(
        truth, full, partial, label_policy, names, abstain_value
    )
    reviewers = _reviewer_scenarios(
        truth, partial, label_policy, names, abstain_value
    )
    if reviewers["Status"] == "Available":
        oracle = next(
            (
                row for row in reviewers["Records"]
                if np.isclose(row["Reviewer Accuracy"], 1.0)
            ),
            None,
        )
        metrics["Cost-Sensitive Utility"] = (
            float(oracle["Cost-Sensitive Utility"])
            if oracle is not None
            else float("nan")
        )
    else:
        metrics["Cost-Sensitive Utility"] = float("nan")
    return {
        "Metrics": metrics,
        "Critical Labels": critical,
        "Reviewer Scenarios": reviewers,
        "Policy Status": "Available" if label_policy is not None else "N/A",
    }


def operating_point_record(cost, deployment, *, data_scope):
    metrics = deployment["Metrics"]
    return {
        "Cost": float(cost),
        "Data Scope": str(data_scope),
        "Generalized Loss": float(metrics["Generalized Loss"]),
        "Coverage": float(metrics["Coverage"]),
        "Selective Risk": float(metrics["Selective Risk"]),
        "Cost-Sensitive Utility": float(metrics["Cost-Sensitive Utility"]),
    }


def select_operating_point(
    records,
    rule,
    *,
    data_scope,
    coverage_gamma=None,
    risk_epsilon=None,
):
    """Choose an operating point only from declared inner-validation records."""

    if data_scope != "inner_validation":
        raise ValueError(
            "Operating points may only be selected on inner_validation data."
        )
    candidates = [dict(record) for record in records]
    if not candidates:
        raise ValueError("Operating-point records cannot be empty.")
    if any(record.get("Data Scope") != data_scope for record in candidates):
        raise ValueError("Every operating-point record must use inner_validation scope.")
    normalized_rule = str(rule).strip().lower().replace("-", "_")
    aliases = {
        "min_loss": "min_generalized_loss",
        "max_utility": "max_utility_at_coverage",
        "max_coverage": "max_coverage_at_risk",
    }
    normalized_rule = aliases.get(normalized_rule, normalized_rule)
    if normalized_rule not in OPERATING_POINT_RULES:
        raise ValueError(f"Unknown operating-point rule: {rule}.")

    if normalized_rule == "min_generalized_loss":
        feasible = [
            row for row in candidates
            if math.isfinite(float(row["Generalized Loss"]))
        ]
        ordering = lambda row: (
            float(row["Generalized Loss"]),
            -float(row["Coverage"]),
            float(row["Cost"]),
        )
    elif normalized_rule == "max_utility_at_coverage":
        if coverage_gamma is None or not 0.0 <= float(coverage_gamma) <= 1.0:
            raise ValueError("coverage_gamma in [0, 1] is required for this rule.")
        feasible = [
            row for row in candidates
            if float(row["Coverage"]) >= float(coverage_gamma)
            and math.isfinite(float(row["Cost-Sensitive Utility"]))
        ]
        ordering = lambda row: (
            -float(row["Cost-Sensitive Utility"]),
            -float(row["Coverage"]),
            float(row["Cost"]),
        )
    else:
        if risk_epsilon is None or not 0.0 <= float(risk_epsilon) <= 1.0:
            raise ValueError("risk_epsilon in [0, 1] is required for this rule.")
        feasible = [
            row for row in candidates
            if math.isfinite(float(row["Selective Risk"]))
            and float(row["Selective Risk"]) <= float(risk_epsilon)
        ]
        ordering = lambda row: (
            -float(row["Coverage"]),
            float(row["Selective Risk"]),
            float(row["Cost"]),
        )
    if not feasible:
        return {
            "Status": "Infeasible",
            "Rule": normalized_rule,
            "Data Scope": data_scope,
            "Selected Cost": None,
            "Selected Record": None,
            "Candidate Count": len(candidates),
        }
    selected = min(feasible, key=ordering)
    return {
        "Status": "Selected",
        "Rule": normalized_rule,
        "Data Scope": data_scope,
        "Selected Cost": float(selected["Cost"]),
        "Selected Record": selected,
        "Candidate Count": len(candidates),
        "Feasible Count": len(feasible),
        "Coverage Gamma": (
            None if coverage_gamma is None else float(coverage_gamma)
        ),
        "Risk Epsilon": None if risk_epsilon is None else float(risk_epsilon),
    }
