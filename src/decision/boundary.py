"""Shared mechanics for boundary-set Bayes decision policies."""

from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .base import (
    DecisionPolicy,
    abstention_penalty,
    validate_cost,
    validate_penalty,
    validate_probability_matrix,
)


class BoundarySetBOPPolicy(DecisionPolicy):
    """Common API, action construction and tie rules for set utilities."""

    def __init__(
        self,
        cost=0.3,
        penalty="linear",
        allow_abstention=True,
        abstain_value=-1,
    ):
        if abstain_value in (0, 1):
            raise ValueError("abstain_value must be different from 0 and 1.")
        self.cost = validate_cost(cost)
        self.penalty = validate_penalty(penalty)
        self.allow_abstention = bool(allow_abstention)
        self.abstain_value = int(abstain_value)

    def _settings(self, cost, penalty):
        resolved_cost = self.cost if cost is None else validate_cost(cost)
        resolved_penalty = (
            self.penalty if penalty is None else validate_penalty(penalty)
        )
        return resolved_cost, resolved_penalty

    def _action(self, order, positive_count, negative_start):
        n_labels = order.size
        sorted_action = np.full(
            n_labels, self.abstain_value, dtype=np.int32
        )
        sorted_action[:positive_count] = 1
        sorted_action[negative_start:] = 0
        action = np.empty(n_labels, dtype=np.int32)
        action[order] = sorted_action
        return action

    @staticmethod
    def _tie_signature(action):
        # After maximizing decided count, prefer the larger action at the
        # smallest original label index. Stable probability sorting makes
        # equal-probability label ties reproducible across runs.
        return tuple(int(value) for value in action)

    def _is_better(self, utility, action, best_utility, best_action):
        if best_action is None:
            return True
        if utility > best_utility and not np.isclose(
            utility, best_utility, rtol=1e-12, atol=1e-12
        ):
            return True
        if not np.isclose(utility, best_utility, rtol=1e-12, atol=1e-12):
            return False
        decided = int(np.count_nonzero(action != self.abstain_value))
        best_decided = int(
            np.count_nonzero(best_action != self.abstain_value)
        )
        if decided != best_decided:
            return decided > best_decided
        return self._tie_signature(action) > self._tie_signature(best_action)

    def _best_boundary_candidate(self, order, utilities):
        """Select one ``(positive_count, negative_start)`` utility exactly.

        ``utilities`` is a square table with ``-inf`` at ineligible
        boundaries.  Numerical ties use the same tolerance and deterministic
        rules as :meth:`_is_better`, without a Python loop over every
        candidate.  This is important for the O(K^3) F/Jaccard policies at
        large K, where the probability recurrences are vectorized.
        """

        table = np.asarray(utilities, dtype=np.float64)
        if table.ndim != 2 or table.shape[0] != table.shape[1]:
            raise ValueError("utilities must be a square boundary table.")
        maximum = float(np.max(table))
        if not np.isfinite(maximum):
            raise RuntimeError("No eligible decision boundary was evaluated.")
        tied = np.argwhere(
            np.isfinite(table)
            & np.isclose(table, maximum, rtol=1e-12, atol=1e-12)
        )
        decided_counts = order.size - (tied[:, 1] - tied[:, 0])
        tied = tied[decided_counts == np.max(decided_counts)]

        best_action = None
        best_utility = None
        for positive_count, negative_start in tied:
            action = self._action(
                order, int(positive_count), int(negative_start)
            )
            if (
                best_action is None
                or self._tie_signature(action)
                > self._tie_signature(best_action)
            ):
                best_action = action
                best_utility = float(
                    table[int(positive_count), int(negative_start)]
                )
        return best_action, best_utility

    @staticmethod
    def _generalized_utility(
        expected_score, abstentions, n_labels, cost, penalty
    ):
        return float(expected_score) - float(
            abstention_penalty(abstentions, n_labels, cost, penalty)
        )

    @abstractmethod
    def _solve_row(self, probabilities, cost, penalty):
        """Return the best action and utility for one probability row."""

    def _solve(self, probabilities, cost, penalty):
        matrix = validate_probability_matrix(probabilities)
        predictions = np.empty(matrix.shape, dtype=np.int32)
        utilities = np.empty(matrix.shape[0], dtype=np.float64)
        # The vectorized O(K^3) row solvers release the GIL in NumPy/BLAS.
        # Two workers improve the large-label validation workload without the
        # oversubscription observed with wider pools.  map() preserves row
        # order, so actions and tie-breaking remain deterministic.
        use_workers = matrix.shape[0] >= 64 and matrix.shape[1] >= 64
        if use_workers:
            with ThreadPoolExecutor(max_workers=2) as executor:
                solved = executor.map(
                    lambda row: self._solve_row(row, cost, penalty),
                    matrix,
                )
                for row_index, (action, utility) in enumerate(solved):
                    predictions[row_index] = action
                    utilities[row_index] = utility
        else:
            for row_index, row in enumerate(matrix):
                predictions[row_index], utilities[row_index] = self._solve_row(
                    row, cost, penalty
                )
        return predictions, utilities

    def predict(self, probabilities, *, cost=None, penalty=None):
        resolved_cost, resolved_penalty = self._settings(cost, penalty)
        predictions, _ = self._solve(
            probabilities, resolved_cost, resolved_penalty
        )
        return predictions

    def predict_with_utility(self, probabilities, *, cost=None, penalty=None):
        """Return both the BOP action and expected generalized utility."""

        resolved_cost, resolved_penalty = self._settings(cost, penalty)
        return self._solve(probabilities, resolved_cost, resolved_penalty)

    def expected_utility(self, probabilities, *, cost=None, penalty=None):
        resolved_cost, resolved_penalty = self._settings(cost, penalty)
        _, utilities = self._solve(
            probabilities, resolved_cost, resolved_penalty
        )
        return utilities
