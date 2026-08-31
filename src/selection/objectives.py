"""Leakage-safe objective registry for GSI candidate partitions."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from ..decision import create_configured_policy
from ..decision.base import abstention_penalty


@dataclass(frozen=True)
class _ObjectiveSpec:
    name: str
    policy: str
    score_kind: str
    allow_abstention: bool
    beta: float | None = None


@dataclass(frozen=True)
class SelectionObjectiveResult:
    """One candidate score plus audit metadata and its decision output."""

    objective: str
    score: float
    predictions: np.ndarray
    policy_config: dict
    diagnostics: dict
    beta: float | None

    def history_metadata(self):
        """Return the JSON-safe fields persisted for every candidate."""

        return {
            "selection_objective": self.objective,
            "selection_policy": self.policy_config["name"],
            "beta": None if self.beta is None else float(self.beta),
            "policy_config": dict(self.policy_config),
            "objective_diagnostics": dict(self.diagnostics),
        }


_OBJECTIVES = {
    "full_macro_f1": _ObjectiveSpec(
        "full_macro_f1", "hamming", "macro_f1", False
    ),
    "immediate_instance_f1": _ObjectiveSpec(
        "immediate_instance_f1", "fbeta", "fbeta", False, 1.0
    ),
    "bop_instance_f1": _ObjectiveSpec(
        "bop_instance_f1", "fbeta", "fbeta", True, 1.0
    ),
    "bop_jaccard": _ObjectiveSpec(
        "bop_jaccard", "jaccard", "jaccard", True
    ),
    "macro_precision": _ObjectiveSpec(
        "macro_precision", "hamming", "macro_precision", False
    ),
    "macro_recall": _ObjectiveSpec(
        "macro_recall", "hamming", "macro_recall", False
    ),
    "f_beta_0_5": _ObjectiveSpec(
        "f_beta_0_5", "fbeta", "fbeta", True, 0.5
    ),
    "f_beta_2": _ObjectiveSpec(
        "f_beta_2", "fbeta", "fbeta", True, 2.0
    ),
}
_OBJECTIVE_ALIASES = {"complete_macro_f1": "full_macro_f1"}


def available_selection_objectives():
    """Return canonical objective IDs in deterministic order."""

    return tuple(_OBJECTIVES)


def canonical_selection_objective(name):
    """Resolve one canonical objective ID."""

    normalized = str(name).strip().lower()
    canonical = _OBJECTIVE_ALIASES.get(normalized, normalized)
    if canonical not in _OBJECTIVES:
        available = ", ".join(available_selection_objectives())
        raise ValueError(
            f"Unknown selection objective '{name}'. Available: {available}."
        )
    return canonical


def _validate_inputs(y_true, probabilities):
    truth = np.asarray(y_true, dtype=np.int32)
    matrix = np.asarray(probabilities, dtype=np.float64)
    if truth.ndim != 2 or truth.shape[0] == 0 or truth.shape[1] == 0:
        raise ValueError("y_true must be a non-empty two-dimensional matrix.")
    if matrix.shape != truth.shape:
        raise ValueError("probabilities must have the same shape as y_true.")
    if not np.all(np.isin(truth, (0, 1))):
        raise ValueError("y_true must contain only binary values 0 and 1.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("probabilities must contain only finite values.")
    return truth, np.clip(matrix, 0.0, 1.0)


def _instance_fbeta_scores(truth, predictions, beta, abstain_value):
    decided = predictions != abstain_value
    true_positive = np.sum(
        (truth == 1) & (predictions == 1) & decided, axis=1
    ).astype(np.float64)
    false_positive = np.sum(
        (truth == 0) & (predictions == 1) & decided, axis=1
    ).astype(np.float64)
    false_negative = np.sum(
        (truth == 1) & (predictions == 0) & decided, axis=1
    ).astype(np.float64)
    beta_squared = float(beta) ** 2
    denominator = (
        (1.0 + beta_squared) * true_positive
        + beta_squared * false_negative
        + false_positive
    )
    scores = np.ones(truth.shape[0], dtype=np.float64)
    nonempty = denominator > 0.0
    scores[nonempty] = (
        (1.0 + beta_squared) * true_positive[nonempty]
        / denominator[nonempty]
    )
    return scores


def _instance_jaccard_scores(truth, predictions, abstain_value):
    decided = predictions != abstain_value
    true_positive = np.sum(
        (truth == 1) & (predictions == 1) & decided, axis=1
    ).astype(np.float64)
    false_positive = np.sum(
        (truth == 0) & (predictions == 1) & decided, axis=1
    ).astype(np.float64)
    false_negative = np.sum(
        (truth == 1) & (predictions == 0) & decided, axis=1
    ).astype(np.float64)
    union = true_positive + false_positive + false_negative
    scores = np.ones(truth.shape[0], dtype=np.float64)
    nonempty = union > 0.0
    scores[nonempty] = true_positive[nonempty] / union[nonempty]
    return scores


def evaluate_selection_objective(
    name,
    y_true,
    probabilities,
    *,
    cost=0.3,
    penalty="linear",
    abstain_value=-1,
):
    """Apply the objective's policy, then score one inner-validation candidate."""

    truth, matrix = _validate_inputs(y_true, probabilities)
    canonical = canonical_selection_objective(name)
    specification = _OBJECTIVES[canonical]
    policy_cost = 0.5 if specification.policy == "hamming" else cost
    policy = create_configured_policy(
        specification.policy,
        cost=policy_cost,
        penalty=penalty,
        beta=1.0 if specification.beta is None else specification.beta,
        allow_abstention=specification.allow_abstention,
        abstain_value=abstain_value,
        hamming_boundary="symmetric_thresholds",
    )
    predictions = policy.predict(matrix)
    threshold_predictions = (matrix >= 0.5).astype(np.int32)
    full_macro_f1 = float(
        f1_score(
            truth, threshold_predictions, average="macro", zero_division=0
        )
    )

    decided = predictions != abstain_value
    coverage = float(np.mean(decided))
    predicted_positive_rate = float(np.mean(predictions == 1))
    abstention_counts = np.sum(~decided, axis=1)
    if specification.allow_abstention:
        penalties = np.asarray(
            abstention_penalty(
                abstention_counts, truth.shape[1], cost, penalty
            ),
            dtype=np.float64,
        )
    else:
        penalties = np.zeros(truth.shape[0], dtype=np.float64)

    if specification.score_kind == "macro_f1":
        base_scores = None
        score = full_macro_f1
    elif specification.score_kind == "macro_precision":
        base_scores = None
        score = float(
            precision_score(
                truth, predictions, average="macro", zero_division=0
            )
        )
    elif specification.score_kind == "macro_recall":
        base_scores = None
        score = float(
            recall_score(
                truth, predictions, average="macro", zero_division=0
            )
        )
    elif specification.score_kind == "fbeta":
        base_scores = _instance_fbeta_scores(
            truth, predictions, specification.beta, abstain_value
        )
        score = float(np.mean(base_scores - penalties))
    elif specification.score_kind == "jaccard":
        base_scores = _instance_jaccard_scores(
            truth, predictions, abstain_value
        )
        score = float(np.mean(base_scores - penalties))
    else:  # pragma: no cover - registry definitions are module constants
        raise RuntimeError(f"Unhandled score kind: {specification.score_kind}")

    diagnostics = {
        "full_macro_f1": full_macro_f1,
        "coverage": coverage,
        "predicted_positive_rate": predicted_positive_rate,
        "mean_abstention_count": float(np.mean(abstention_counts)),
        "mean_abstention_penalty": float(np.mean(penalties)),
        "mean_base_utility": (
            float(score) if base_scores is None else float(np.mean(base_scores))
        ),
    }
    return SelectionObjectiveResult(
        objective=canonical,
        score=float(score),
        predictions=predictions,
        policy_config=policy.get_config(),
        diagnostics=diagnostics,
        beta=specification.beta,
    )
