"""Complete and partial Bayes-optimal policies for instance F-beta."""

import numpy as np

from .boundary import BoundarySetBOPPolicy
from .base import abstention_penalty
from .count_distribution import (
    prefix_count_distributions,
    suffix_count_distributions,
)


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
        if not self.allow_abstention:
            return self._solve_complete_row_fast(
                order, sorted_probabilities, prefix_counts[-1]
            )
        suffix_counts = suffix_count_distributions(sorted_probabilities)

        # Padded prefix/suffix count tables let NumPy/BLAS evaluate the exact
        # Algorithm-2 expectations.  This preserves O(K^3) arithmetic while
        # removing the Python-level candidate/count recurrences that dominate
        # selection time for datasets with hundreds of labels.
        suffix_table = np.zeros(
            (n_labels + 1, n_labels + 1), dtype=np.float64
        )
        for start, distribution in enumerate(suffix_counts):
            suffix_table[start, : distribution.size] = distribution

        expected_kernel = np.zeros_like(suffix_table)

        beta_squared = self.beta * self.beta
        beta_factor = 1.0 + 1.0 / beta_squared
        count_axis = np.arange(n_labels + 1, dtype=np.float64)

        for positive_count in range(1, n_labels + 1):
            positive_distribution = prefix_counts[positive_count]
            positive_axis = np.arange(
                positive_count + 1, dtype=np.float64
            )
            denominator = (
                positive_count / beta_squared
                + positive_axis[:, None]
                + count_axis[None, :]
            )
            expected_kernel[positive_count] = beta_factor * np.sum(
                (
                    positive_axis
                    * positive_distribution
                )[:, None]
                / denominator,
                axis=0,
            )

        expected_scores = expected_kernel @ suffix_table.T
        # Empty predicted-positive candidates follow the repository's
        # empty/empty convention: score one iff every decided-negative truth
        # in the suffix is zero.
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

    def _solve_complete_row_fast(
        self, order, sorted_probabilities, total_distribution
    ):
        """Exact complete F-beta BOP in O(K^2) under label independence.

        Let ``S`` be the total number of true positive labels.  For a prefix
        of ``k`` predicted positives, expected F-beta is

        ``(1 + beta^2) * sum_s P(TP_prefix, S=s) / (beta^2*s + k)``.

        The cumulative joint distribution is updated one prefix label at a
        time.  Each leave-one-out count distribution is recovered from the
        total Poisson-binomial polynomial in O(K), avoiding Algorithm 2's
        unnecessary O(K^3) boundary table when abstention is disallowed.
        """

        n_labels = sorted_probabilities.size
        total = np.asarray(total_distribution, dtype=np.float64)
        beta_squared = self.beta * self.beta
        scores = np.empty(n_labels + 1, dtype=np.float64)
        scores[0] = total[0]
        count_axis = np.arange(1, n_labels + 1, dtype=np.float64)
        without_label = self._leave_one_out_count_distributions(
            total, sorted_probabilities
        )
        joint = np.zeros((n_labels, n_labels + 1), dtype=np.float64)
        joint[:, 1:] = sorted_probabilities[:, None] * without_label
        cumulative_joint = np.cumsum(joint, axis=0)
        positive_counts = np.arange(1, n_labels + 1, dtype=np.float64)
        denominators = 1.0 / (
            beta_squared * count_axis[None, :]
            + positive_counts[:, None]
        )
        scores[1:] = (1.0 + beta_squared) * np.sum(
            cumulative_joint[:, 1:] * denominators, axis=1
        )

        utilities = np.full((n_labels + 1, n_labels + 1), -np.inf)
        indices = np.arange(n_labels + 1)
        utilities[indices, indices] = scores
        return self._best_boundary_candidate(order, utilities)

    @staticmethod
    def _leave_one_out_count_distributions(total_distribution, probabilities):
        """Recover every P(S_-i) from P(S) with vectorized recurrences."""

        total = np.asarray(total_distribution, dtype=np.float64)
        n_labels = total.size - 1
        values = np.asarray(probabilities, dtype=np.float64)
        result = np.empty((n_labels, n_labels), dtype=np.float64)
        complement = 1.0 - values
        forward = complement >= values
        backward = ~forward

        if np.any(forward):
            selected = values[forward]
            selected_complement = complement[forward]
            result[forward, 0] = total[0] / selected_complement
            for count in range(1, n_labels):
                result[forward, count] = (
                    total[count] - selected * result[forward, count - 1]
                ) / selected_complement
        if np.any(backward):
            selected = values[backward]
            selected_complement = complement[backward]
            result[backward, -1] = total[-1] / selected
            for count in range(n_labels - 1, 0, -1):
                result[backward, count - 1] = (
                    total[count] - selected_complement * result[backward, count]
                ) / selected
        # Round-off can create values around -1e-16.  Clipping only this
        # numerical artifact preserves the probability distribution and keeps
        # the exact tie logic stable for deterministic 0/1 marginals.
        return np.clip(result, 0.0, 1.0)

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
            "algorithm": "Nguyen-Huellermeier Algorithm 2 with vectorized count tables and empty-set extension",
            "inference_complexity": "O(K^2) complete; O(K^3) partial, O(K^2) count cache",
            "empty_set_fbeta": 1.0,
            "tie_breaking": "more_decisions_then_stable_label_index",
        }
