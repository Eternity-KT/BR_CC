"""Complete and partial Bayes-optimal policies for instance F-beta."""

import numpy as np

from .boundary import BoundarySetBOPPolicy
from .count_distribution import prefix_count_distributions


class FbetaBOPPolicy(BoundarySetBOPPolicy):
    """Algorithm-2 F-beta BOP under conditional label independence.

    Marginal probabilities are sorted stably in descending order.  A candidate
    predicts the first ``l`` labels as positive, abstains on the middle block,
    and predicts the last block as negative.  Expected utilities for all such
    candidates are evaluated in ``O(K^3)`` by reusing count distributions.

    When ``allow_abstention`` is false, only complete candidates are eligible.
    The empty/empty F-beta convention is one, matching the metric contract in
    this repository.  This adds the negative-only boundary candidates omitted
    by presentations of Algorithm 2 that define zero-positive F-beta as zero.
    """

    policy_name = "fbeta_bop"

    def __init__(
        self,
        beta=1.0,
        cost=0.3,
        penalty="linear",
        allow_abstention=True,
        abstain_value=-1,
    ):
        beta = float(beta)
        if not np.isfinite(beta) or beta <= 0.0:
            raise ValueError("beta must be a finite positive number.")
        self.beta = beta
        super().__init__(
            cost=cost,
            penalty=penalty,
            allow_abstention=allow_abstention,
            abstain_value=abstain_value,
        )

    def _solve_row(self, probabilities, cost, penalty):
        n_labels = probabilities.size
        order = np.argsort(-probabilities, kind="stable")
        sorted_probabilities = probabilities[order]
        prefix_counts = prefix_count_distributions(sorted_probabilities)

        best_utility = -np.inf
        best_action = None

        # Empty predicted-positive set. Under the repository's empty-set
        # convention, F-beta is one exactly when every decided-negative truth
        # is also zero.  Include these candidates so exhaustive optimality and
        # the published metric contract use the same utility.
        probability_all_negative = 1.0
        negative_starts = (
            range(n_labels, -1, -1)
            if self.allow_abstention
            else (0,)
        )
        if self.allow_abstention:
            for negative_start in negative_starts:
                if negative_start < n_labels:
                    probability_all_negative *= (
                        1.0 - sorted_probabilities[negative_start]
                    )
                abstentions = negative_start
                utility = self._generalized_utility(
                    probability_all_negative,
                    abstentions,
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
            probability_all_negative = float(
                np.prod(1.0 - sorted_probabilities, dtype=np.float64)
            )
            best_utility = probability_all_negative
            best_action = self._action(order, 0, 0)

        beta_squared = self.beta * self.beta
        beta_factor = 1.0 + 1.0 / beta_squared
        count_axis = np.arange(n_labels + 1, dtype=np.float64)

        for positive_count in range(1, n_labels + 1):
            positive_distribution = prefix_counts[positive_count]
            inverse_denominator = 1.0 / (
                positive_count / beta_squared + count_axis
            )

            # First evaluate a candidate with no decided-negative labels, then
            # add them from right to left using Algorithm 2's S recurrence.
            negative_start = n_labels
            while True:
                if self.allow_abstention or negative_start == positive_count:
                    expected_fbeta = beta_factor * float(
                        np.dot(
                            np.arange(positive_count + 1, dtype=np.float64)
                            * positive_distribution,
                            inverse_denominator[: positive_count + 1],
                        )
                    )
                    abstentions = negative_start - positive_count
                    utility = self._generalized_utility(
                        expected_fbeta,
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
                previous = inverse_denominator
                updated = previous.copy()
                updated[:-1] = (
                    (1.0 - probability) * previous[:-1]
                    + probability * previous[1:]
                )
                inverse_denominator = updated

        return best_action, float(best_utility)

    def get_config(self):
        return {
            "name": self.policy_name,
            "objective": "generalized_instance_fbeta",
            "beta": float(self.beta),
            "cost": float(self.cost),
            "penalty": self.penalty,
            "allow_abstention": bool(self.allow_abstention),
            "abstain_value": int(self.abstain_value),
            "probability_assumption": "conditional_label_independence",
            "dependent_marginal_interpretation": "BOP under CLI approximation",
            "algorithm": "Nguyen-Huellermeier Algorithm 2 with empty-set extension",
            "inference_complexity": "O(K^3) time, O(K^2) count cache",
            "empty_set_fbeta": 1.0,
            "tie_breaking": "more_decisions_then_stable_label_index",
        }
