"""Complete and partial Bayes-optimal policies for instance Jaccard."""

import numpy as np

from .boundary import BoundarySetBOPPolicy
from .base import abstention_penalty
from .count_distribution import suffix_count_distributions


class JaccardBOPPolicy(BoundarySetBOPPolicy):
    """Algorithm-3 Jaccard BOP under conditional label independence.

    After a stable descending probability sort, candidates predict a positive
    prefix, abstain on a middle block and predict a negative suffix. For a
    candidate with ``l`` predicted positives and ``B`` positives in the
    decided-negative suffix, Jaccard is ``TP / (l + B)``. Conditional label
    independence therefore factorizes its expectation into
    ``E[TP] * E[1 / (l + B)]`` and permits the Algorithm-3 recurrence.

    The repository defines an empty union as Jaccard one. Negative-only
    boundary candidates are consequently included as an explicit extension
    to presentations of Algorithm 3 that assign zero in this case.
    """

    policy_name = "jaccard_bop"

    def _solve_row(self, probabilities, cost, penalty):
        n_labels = probabilities.size
        order = np.argsort(-probabilities, kind="stable")
        sorted_probabilities = probabilities[order]
        suffix_counts = suffix_count_distributions(sorted_probabilities)
        suffix_table = np.zeros(
            (n_labels + 1, n_labels + 1), dtype=np.float64
        )
        for start, distribution in enumerate(suffix_counts):
            suffix_table[start, : distribution.size] = distribution

        count_axis = np.arange(n_labels + 1, dtype=np.float64)
        expected_kernel = np.zeros_like(suffix_table)
        cumulative_true_positives = np.cumsum(sorted_probabilities)
        for positive_count in range(1, n_labels + 1):
            expected_kernel[positive_count] = (
                cumulative_true_positives[positive_count - 1]
                / (positive_count + count_axis)
            )
        expected_scores = expected_kernel @ suffix_table.T
        expected_scores[0] = suffix_table[:, 0]

        utilities = np.full_like(expected_scores, -np.inf)
        for positive_count in range(n_labels + 1):
            starts = (
                np.arange(positive_count, n_labels + 1, dtype=np.int64)
                if self.allow_abstention
                else np.array([positive_count], dtype=np.int64)
            )
            abstentions = starts - positive_count
            utilities[positive_count, starts] = (
                expected_scores[positive_count, starts]
                - np.asarray(
                    abstention_penalty(
                        abstentions, n_labels, cost, penalty
                    ),
                    dtype=np.float64,
                )
            )

        return self._best_boundary_candidate(order, utilities)

    def get_config(self):
        return {
            "name": self.policy_name,
            "objective": "generalized_instance_jaccard",
            "cost": float(self.cost),
            "penalty": self.penalty,
            "allow_abstention": bool(self.allow_abstention),
            "abstain_value": int(self.abstain_value),
            "probability_assumption": "conditional_label_independence",
            "dependent_marginal_interpretation": "BOP under CLI approximation",
            "algorithm": "Nguyen-Huellermeier Algorithm 3 with vectorized count tables and empty-union extension",
            "inference_complexity": "O(K^3) time, O(K^2) count cache",
            "empty_union_jaccard": 1.0,
            "tie_breaking": "more_decisions_then_stable_label_index",
        }
