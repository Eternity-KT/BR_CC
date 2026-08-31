"""Bayes-optimal SEP/PAR decisions for generalized Hamming loss."""

import numpy as np

from .base import (
    DecisionPolicy,
    abstention_penalty,
    validate_cost,
    validate_penalty,
    validate_probability_matrix,
)


class HammingBOPPolicy(DecisionPolicy):
    """Wrap the repository's original generalized-Hamming decision rules."""

    policy_name = "hamming_bop"

    def __init__(
        self,
        cost=0.3,
        penalty="linear",
        abstain_value=-1,
        linear_boundary="minimum_loss",
    ):
        self.cost = validate_cost(cost)
        self.penalty = validate_penalty(penalty)
        if abstain_value in (0, 1):
            raise ValueError("abstain_value must be different from 0 and 1.")
        if linear_boundary not in ("minimum_loss", "symmetric_thresholds"):
            raise ValueError(
                "linear_boundary must be 'minimum_loss' or "
                "'symmetric_thresholds'."
            )
        self.abstain_value = int(abstain_value)
        self.linear_boundary = linear_boundary

    def _settings(self, cost, penalty):
        resolved_cost = self.cost if cost is None else validate_cost(cost)
        resolved_penalty = (
            self.penalty if penalty is None else validate_penalty(penalty)
        )
        return resolved_cost, resolved_penalty

    def _linear_mask(self, probabilities, cost):
        if self.linear_boundary == "symmetric_thresholds":
            if cost >= 0.5:
                return np.ones(probabilities.shape, dtype=bool)
            return (probabilities <= cost) | (probabilities >= 1.0 - cost)
        expected_label_losses = np.minimum(probabilities, 1.0 - probabilities)
        return expected_label_losses <= cost

    @staticmethod
    def _concave_mask(probabilities, cost):
        n_samples, n_labels = probabilities.shape
        expected_label_losses = np.minimum(probabilities, 1.0 - probabilities)
        masks = np.zeros((n_samples, n_labels), dtype=bool)
        abstention_counts = n_labels - np.arange(n_labels + 1, dtype=np.float64)
        penalties = abstention_penalty(
            abstention_counts, n_labels, cost, "concave"
        )

        for row_index in range(n_samples):
            order = np.argsort(expected_label_losses[row_index], kind="stable")
            sorted_losses = expected_label_losses[row_index, order]
            cumulative_losses = np.concatenate(
                ([0.0], np.cumsum(sorted_losses, dtype=np.float64))
            )
            risks = cumulative_losses + penalties
            minimum = np.min(risks)
            # Existing PAR convention: among tied risks, decide more labels.
            best_decided = int(np.flatnonzero(np.isclose(risks, minimum))[-1])
            masks[row_index, order[:best_decided]] = True
        return masks

    def decision_mask(self, probabilities, *, cost=None, penalty=None):
        matrix = validate_probability_matrix(probabilities)
        resolved_cost, resolved_penalty = self._settings(cost, penalty)
        if resolved_penalty == "linear":
            return self._linear_mask(matrix, resolved_cost)
        return self._concave_mask(matrix, resolved_cost)

    def predict(self, probabilities, *, cost=None, penalty=None):
        matrix = validate_probability_matrix(probabilities)
        resolved_cost, resolved_penalty = self._settings(cost, penalty)
        complete = (matrix >= 0.5).astype(np.int32)
        if resolved_penalty == "linear":
            decided = self._linear_mask(matrix, resolved_cost)
        else:
            decided = self._concave_mask(matrix, resolved_cost)
        partial = np.full(complete.shape, self.abstain_value, dtype=np.int32)
        partial[decided] = complete[decided]
        return partial

    def expected_utility(self, probabilities, *, cost=None, penalty=None):
        """Return negative expected generalized-Hamming loss for each row."""

        matrix = validate_probability_matrix(probabilities)
        resolved_cost, resolved_penalty = self._settings(cost, penalty)
        predictions = self.predict(
            matrix, cost=resolved_cost, penalty=resolved_penalty
        )
        decided = predictions != self.abstain_value
        predicted_positive = predictions == 1
        expected_errors = np.where(predicted_positive, 1.0 - matrix, matrix)
        decided_risk = np.sum(expected_errors * decided, axis=1)
        abstentions = np.sum(~decided, axis=1)
        total_risk = decided_risk + abstention_penalty(
            abstentions, matrix.shape[1], resolved_cost, resolved_penalty
        )
        return -np.asarray(total_risk, dtype=np.float64)

    def get_config(self):
        return {
            "name": self.policy_name,
            "objective": "generalized_hamming",
            "cost": float(self.cost),
            "penalty": self.penalty,
            "abstain_value": int(self.abstain_value),
            "linear_boundary": self.linear_boundary,
            "tie_breaking": "more_decisions_then_stable_label_index",
        }
