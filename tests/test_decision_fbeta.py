"""Phase-Q4 exhaustive validation for complete and partial F-beta BOP."""

import itertools
import json
import time
import unittest

import numpy as np

from src.decision import FbetaBOPPolicy, available_policies, create_policy
from src.decision.count_distribution import (
    prefix_count_distributions,
    suffix_count_distributions,
)


def _penalty(abstentions, n_labels, cost, penalty):
    if penalty == "linear":
        return abstentions * cost
    return abstentions * n_labels * cost / (n_labels + abstentions)


def _fbeta_score(truth, action, beta, abstain_value=-1):
    decided = action != abstain_value
    decided_truth = truth[decided]
    decided_action = action[decided]
    true_positive = np.count_nonzero(
        (decided_truth == 1) & (decided_action == 1)
    )
    false_positive = np.count_nonzero(
        (decided_truth == 0) & (decided_action == 1)
    )
    false_negative = np.count_nonzero(
        (decided_truth == 1) & (decided_action == 0)
    )
    beta_squared = beta * beta
    denominator = (
        (1.0 + beta_squared) * true_positive
        + beta_squared * false_negative
        + false_positive
    )
    if denominator == 0.0:
        return 1.0
    return (1.0 + beta_squared) * true_positive / denominator


def _expected_action_utility(
    probabilities,
    action,
    beta=1.0,
    cost=0.3,
    penalty="linear",
    abstain_value=-1,
):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    expected = 0.0
    for truth_values in itertools.product((0, 1), repeat=probabilities.size):
        truth = np.asarray(truth_values, dtype=np.int32)
        truth_probability = float(
            np.prod(
                np.where(truth == 1, probabilities, 1.0 - probabilities),
                dtype=np.float64,
            )
        )
        expected += truth_probability * _fbeta_score(
            truth, action, beta, abstain_value=abstain_value
        )
    abstentions = int(np.count_nonzero(action == abstain_value))
    return expected - _penalty(
        abstentions, probabilities.size, cost, penalty
    )


def _exhaustive_optimum(
    probabilities,
    beta=1.0,
    cost=0.3,
    penalty="linear",
    allow_abstention=True,
    abstain_value=-1,
):
    alphabet = (abstain_value, 0, 1) if allow_abstention else (0, 1)
    best = -np.inf
    for values in itertools.product(alphabet, repeat=len(probabilities)):
        action = np.asarray(values, dtype=np.int32)
        utility = _expected_action_utility(
            probabilities,
            action,
            beta=beta,
            cost=cost,
            penalty=penalty,
            abstain_value=abstain_value,
        )
        best = max(best, utility)
    return float(best)


class FbetaBOPTests(unittest.TestCase):
    def test_count_distributions_are_normalized_and_match_enumeration(self):
        probabilities = np.array([0.2, 0.6, 0.9], dtype=np.float64)
        prefix = prefix_count_distributions(probabilities)
        suffix = suffix_count_distributions(probabilities)
        for distribution in (*prefix, *suffix):
            self.assertAlmostEqual(float(np.sum(distribution)), 1.0)
            self.assertTrue(np.all(distribution >= 0.0))
        np.testing.assert_allclose(
            prefix[2],
            [0.8 * 0.4, 0.2 * 0.4 + 0.8 * 0.6, 0.2 * 0.6],
        )
        np.testing.assert_allclose(
            suffix[1],
            [0.4 * 0.1, 0.6 * 0.1 + 0.4 * 0.9, 0.6 * 0.9],
        )

    def test_complete_and_partial_policies_match_exhaustive_optimum(self):
        fixtures = [
            (np.array([0.05]), 1.0, 0.20, "linear", False),
            (np.array([0.40, 0.40]), 1.0, 0.10, "linear", False),
            (np.array([0.15, 0.55, 0.85]), 0.5, 0.12, "linear", True),
            (np.array([0.10, 0.35, 0.65, 0.90]), 1.0, 0.18, "concave", True),
            (np.array([0.08, 0.28, 0.52, 0.73, 0.94]), 2.0, 0.09, "linear", True),
            (
                np.array([0.04, 0.22, 0.41, 0.59, 0.78, 0.96]),
                1.0,
                0.11,
                "concave",
                True,
            ),
        ]
        for probabilities, beta, cost, penalty, allow_abstention in fixtures:
            with self.subTest(
                labels=len(probabilities),
                beta=beta,
                penalty=penalty,
                allow_abstention=allow_abstention,
            ):
                policy = FbetaBOPPolicy(
                    beta=beta,
                    cost=cost,
                    penalty=penalty,
                    allow_abstention=allow_abstention,
                )
                prediction, reported = policy.predict_with_utility(
                    probabilities[None, :]
                )
                direct = _expected_action_utility(
                    probabilities,
                    prediction[0],
                    beta=beta,
                    cost=cost,
                    penalty=penalty,
                )
                exhaustive = _exhaustive_optimum(
                    probabilities,
                    beta=beta,
                    cost=cost,
                    penalty=penalty,
                    allow_abstention=allow_abstention,
                )
                self.assertAlmostEqual(reported[0], direct, places=11)
                self.assertAlmostEqual(reported[0], exhaustive, places=11)
                if not allow_abstention:
                    self.assertFalse(np.any(prediction == -1))

    def test_deterministic_random_fixtures_match_exhaustive_through_k5(self):
        generator = np.random.default_rng(314159)
        checked = 0
        for n_labels in range(1, 6):
            for allow_abstention in (False, True):
                for penalty in ("linear", "concave"):
                    probabilities = generator.uniform(0.01, 0.99, n_labels)
                    beta = (0.5, 1.0, 2.0)[checked % 3]
                    cost = float(generator.uniform(0.0, 0.25))
                    policy = FbetaBOPPolicy(
                        beta=beta,
                        cost=cost,
                        penalty=penalty,
                        allow_abstention=allow_abstention,
                    )
                    _, utility = policy.predict_with_utility(
                        probabilities[None, :]
                    )
                    exhaustive = _exhaustive_optimum(
                        probabilities,
                        beta=beta,
                        cost=cost,
                        penalty=penalty,
                        allow_abstention=allow_abstention,
                    )
                    self.assertAlmostEqual(utility[0], exhaustive, places=11)
                    checked += 1
        self.assertEqual(checked, 20)

    def test_empty_set_extension_matches_exhaustive_all_zero_action(self):
        probabilities = np.array([[0.01, 0.02, 0.03]], dtype=np.float64)
        policy = FbetaBOPPolicy(allow_abstention=False)
        prediction, utility = policy.predict_with_utility(probabilities)
        np.testing.assert_array_equal(prediction, [[0, 0, 0]])
        self.assertAlmostEqual(
            utility[0], float(np.prod(1.0 - probabilities[0])), places=12
        )

    def test_f1_bop_differs_from_half_threshold_and_hamming_action(self):
        probabilities = np.array([[0.4, 0.4]], dtype=np.float64)
        threshold_action = (probabilities >= 0.5).astype(np.int32)
        policy = FbetaBOPPolicy(beta=1.0, allow_abstention=False)
        prediction, utility = policy.predict_with_utility(probabilities)
        np.testing.assert_array_equal(threshold_action, [[0, 0]])
        np.testing.assert_array_equal(prediction, [[1, 1]])
        self.assertAlmostEqual(utility[0], 0.48, places=12)

    def test_partial_mode_ties_prefer_more_decisions(self):
        probabilities = np.array([[0.0, 1.0]], dtype=np.float64)
        policy = FbetaBOPPolicy(cost=0.0, allow_abstention=True)
        first = policy.predict(probabilities)
        second = policy.predict(probabilities)
        np.testing.assert_array_equal(first, [[0, 1]])
        np.testing.assert_array_equal(second, first)
        self.assertTrue(np.all(policy.decision_mask(probabilities)))

    def test_registry_config_and_cli_approximation_metadata(self):
        self.assertIn("fbeta", available_policies())
        policy = create_policy(
            "f1_bop",
            beta=1,
            cost=0.15,
            penalty="concave",
            allow_abstention=False,
        )
        self.assertIsInstance(policy, FbetaBOPPolicy)
        config = policy.get_config()
        self.assertEqual(
            config["dependent_marginal_interpretation"],
            "BOP under CLI approximation",
        )
        self.assertEqual(json.loads(json.dumps(config)), config)

    def test_complete_mode_is_cost_invariant_and_runtime_smoke(self):
        probabilities = np.linspace(0.01, 0.99, 40, dtype=np.float64)[None, :]
        policy = FbetaBOPPolicy(allow_abstention=False)
        started = time.perf_counter()
        low_cost = policy.predict(probabilities, cost=0.0)
        elapsed = time.perf_counter() - started
        high_cost = policy.predict(probabilities, cost=1.0, penalty="concave")
        np.testing.assert_array_equal(low_cost, high_cost)
        self.assertFalse(np.any(low_cost == -1))
        self.assertLess(elapsed, 5.0)

    def test_parameter_and_probability_validation(self):
        with self.assertRaises(ValueError):
            FbetaBOPPolicy(beta=0)
        with self.assertRaises(ValueError):
            FbetaBOPPolicy(cost=-0.1)
        with self.assertRaises(ValueError):
            FbetaBOPPolicy(penalty="unknown")
        with self.assertRaises(ValueError):
            FbetaBOPPolicy().predict(np.array([[0.2, np.nan]]))


if __name__ == "__main__":
    unittest.main()
