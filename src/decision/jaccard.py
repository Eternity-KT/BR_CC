"""Complete and partial Bayes-optimal policies for instance Jaccard."""

import numpy as np

from .boundary import BoundarySetBOPPolicy
from .count_distribution import prefix_count_distributions


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
        prefix_counts = prefix_count_distributions(sorted_probabilities)

        best_utility = -np.inf
        best_action = None

        # When no positive is predicted, Jaccard is one only if all truth
        # labels in the decided-negative suffix are also zero.
        if self.allow_abstention:
            probability_empty_union = 1.0
            for negative_start in range(n_labels, -1, -1):
                if negative_start < n_labels:
                    probability_empty_union *= (
                        1.0 - sorted_probabilities[negative_start]
                    )
                utility = self._generalized_utility(
                    probability_empty_union,
                    negative_start,
                    n_labels,
                    cost,
                    penalty,
                )
                action = self._action(order, 0, negative_start)
                if self._is_better(
                    utility, action, best_utility, best_action
                ):
                    best_utility, best_action = utility, action
        else:
            probability_empty_union = float(
                np.prod(1.0 - sorted_probabilities, dtype=np.float64)
            )
            best_utility = probability_empty_union
            best_action = self._action(order, 0, 0)

        count_axis = np.arange(n_labels + 1, dtype=np.float64)
        for positive_count in range(1, n_labels + 1):
            positive_distribution = prefix_counts[positive_count]
            expected_true_positives = float(
                np.dot(
                    np.arange(positive_count + 1, dtype=np.float64),
                    positive_distribution,
                )
            )
            inverse_union = 1.0 / (positive_count + count_axis)

            # Start with no decided-negative suffix, then add suffix labels
            # from right to left using the Algorithm-3 expectation recurrence.
            negative_start = n_labels
            while True:
                if self.allow_abstention or negative_start == positive_count:
                    expected_jaccard = (
                        expected_true_positives * inverse_union[0]
                    )
                    abstentions = negative_start - positive_count
                    utility = self._generalized_utility(
                        expected_jaccard,
                        abstentions,
                        n_labels,
                        cost,
                        penalty,
                    )
                    action = self._action(
                        order, positive_count, negative_start
                    )
                    if self._is_better(
                        utility, action, best_utility, best_action
                    ):
                        best_utility, best_action = utility, action

                if negative_start == positive_count:
                    break
                negative_start -= 1
                probability = sorted_probabilities[negative_start]
                previous = inverse_union
                updated = previous.copy()
                updated[:-1] = (
                    (1.0 - probability) * previous[:-1]
                    + probability * previous[1:]
                )
                inverse_union = updated

        return best_action, float(best_utility)

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
            "algorithm": "Nguyen-Huellermeier Algorithm 3 with empty-union extension",
            "inference_complexity": "O(K^3) time, O(K^2) count cache",
            "empty_union_jaccard": 1.0,
            "tie_breaking": "more_decisions_then_stable_label_index",
        }
