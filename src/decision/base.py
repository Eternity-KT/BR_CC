"""Common interfaces and validation helpers for decision-time policies."""

from abc import ABC, abstractmethod

import numpy as np


class DecisionPolicy(ABC):
    """Map marginal label probabilities to complete or partial decisions."""

    abstain_value = -1

    @abstractmethod
    def predict(self, probabilities, *, cost=None, penalty=None):
        """Return one decision row per probability row."""

    def predict_from_proba(self, probabilities, *, cost=None, penalty=None):
        """Compatibility alias used by the existing model APIs."""

        return self.predict(probabilities, cost=cost, penalty=penalty)

    def decision_mask(self, probabilities, *, cost=None, penalty=None):
        """Return ``True`` for every label position decided by the policy."""

        return self.predict(
            probabilities, cost=cost, penalty=penalty
        ) != self.abstain_value

    @abstractmethod
    def expected_utility(self, probabilities, *, cost=None, penalty=None):
        """Return the conditional expected generalized utility per row."""

    @abstractmethod
    def get_config(self):
        """Return a JSON-serializable scientific configuration."""


def validate_probability_matrix(probabilities):
    """Validate and clip a two-dimensional marginal-probability matrix."""

    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError(
            "probabilities must have shape (n_samples, n_labels) with at "
            "least one label."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("probabilities must contain only finite values.")
    return np.clip(matrix, 0.0, 1.0)


def validate_cost(cost):
    """Return a finite abstention cost in the closed unit interval."""

    value = float(cost)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("cost must lie in [0, 1].")
    return value


def validate_penalty(penalty):
    """Validate the two generalized-loss penalty families used in the repo."""

    if penalty not in ("linear", "concave"):
        raise ValueError("penalty must be either 'linear' or 'concave'.")
    return penalty


def abstention_penalty(abstention_count, n_labels, cost, penalty):
    """Evaluate SEP ``a*c`` or PAR ``a*K*c/(K+a)`` elementwise."""

    count = np.asarray(abstention_count, dtype=np.float64)
    cost = validate_cost(cost)
    penalty = validate_penalty(penalty)
    if penalty == "linear":
        return count * cost
    return count * float(n_labels) * cost / (float(n_labels) + count)
