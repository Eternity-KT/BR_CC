"""Phase-Q5 exhaustive validation for complete and partial Jaccard BOP."""

import itertools
import json
import time
import tracemalloc
import unittest

import numpy as np

from src.decision import (
    DecisionPolicy,
    FbetaBOPPolicy,
    HammingBOPPolicy,
    JaccardBOPPolicy,
    available_policies,
    create_policy,
)
from src.evaluation.cache_v3 import compute_config_hash


def _penalty(abstentions, n_labels, cost, penalty):
    if penalty == "linear":
        return abstentions * cost
    return abstentions * n_labels * cost / (n_labels + abstentions)


def _jaccard_score(truth, action, abstain_value=-1):
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
    union = true_positive + false_positive + false_negative
    return 1.0 if union == 0 else true_positive / union


def _expected_action_utility(
    probabilities,
    action,
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
        expected += truth_probability * _jaccard_score(
            truth, action, abstain_value=abstain_value
        )
    abstentions = int(np.count_nonzero(action == abstain_value))
    return expected - _penalty(
        abstentions, probabilities.size, cost, penalty
    )


def _exhaustive_optimum(
    probabilities,
    cost=0.3,
    penalty="linear",
    allow_abstention=True,
    abstain_value=-1,
):
    alphabet = (abstain_value, 0, 1) if allow_abstention else (0, 1)
    best = -np.inf
    for values in itertools.product(alphabet, repeat=len(probabilities)):
        action = np.asarray(values, dtype=np.int32)
        best = max(
            best,
            _expected_action_utility(
                probabilities,
                action,
                cost=cost,
                penalty=penalty,
                abstain_value=abstain_value,
            ),
        )
    return float(best)


class JaccardBOPTests(unittest.TestCase):
    def test_complete_and_partial_policies_match_exhaustive_optimum(self):
        fixtures = [
            (np.array([0.05]), 0.20, "linear", False),
            (np.array([0.40, 0.40]), 0.10, "linear", False),
            (np.array([0.15, 0.55, 0.85]), 0.12, "linear", True),
            (np.array([0.10, 0.35, 0.65, 0.90]), 0.18, "concave", True),
            (np.array([0.08, 0.28, 0.52, 0.73, 0.94]), 0.09, "linear", True),
            (
                np.array([0.04, 0.22, 0.41, 0.59, 0.78, 0.96]),
                0.11,
                "concave",
                True,
            ),
        ]
        for probabilities, cost, penalty, allow_abstention in fixtures:
            with self.subTest(
                labels=len(probabilities),
                penalty=penalty,
                allow_abstention=allow_abstention,
            ):
                policy = JaccardBOPPolicy(
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
                    cost=cost,
                    penalty=penalty,
                )
                exhaustive = _exhaustive_optimum(
                    probabilities,
                    cost=cost,
                    penalty=penalty,
                    allow_abstention=allow_abstention,
                )
                self.assertAlmostEqual(reported[0], direct, places=11)
                self.assertAlmostEqual(reported[0], exhaustive, places=11)
                if not allow_abstention:
                    self.assertFalse(np.any(prediction == -1))

    def test_deterministic_random_fixtures_match_exhaustive_through_k5(self):
        generator = np.random.default_rng(271828)
        checked = 0
        for n_labels in range(1, 6):
            for allow_abstention in (False, True):
                for penalty in ("linear", "concave"):
                    probabilities = generator.uniform(0.01, 0.99, n_labels)
                    cost = float(generator.uniform(0.0, 0.25))
                    policy = JaccardBOPPolicy(
                        cost=cost,
                        penalty=penalty,
                        allow_abstention=allow_abstention,
                    )
                    _, utility = policy.predict_with_utility(
                        probabilities[None, :]
                    )
                    exhaustive = _exhaustive_optimum(
                        probabilities,
                        cost=cost,
                        penalty=penalty,
                        allow_abstention=allow_abstention,
                    )
                    self.assertAlmostEqual(utility[0], exhaustive, places=11)
                    checked += 1
        self.assertEqual(checked, 20)

    def test_empty_union_extension_and_counterexample(self):
        rare_probabilities = np.array([[0.01, 0.02, 0.03]], dtype=np.float64)
        complete = JaccardBOPPolicy(allow_abstention=False)
        prediction, utility = complete.predict_with_utility(rare_probabilities)
        np.testing.assert_array_equal(prediction, [[0, 0, 0]])
        self.assertAlmostEqual(
            utility[0], float(np.prod(1.0 - rare_probabilities[0])), places=12
        )

        counterexample = np.array([[0.4, 0.4]], dtype=np.float64)
        threshold_action = (counterexample >= 0.5).astype(np.int32)
        prediction, utility = complete.predict_with_utility(counterexample)
        np.testing.assert_array_equal(threshold_action, [[0, 0]])
        np.testing.assert_array_equal(prediction, [[1, 1]])
        self.assertAlmostEqual(utility[0], 0.4, places=12)

    def test_partial_ties_and_complete_cost_invariance(self):
        deterministic = np.array([[0.0, 1.0]], dtype=np.float64)
        partial = JaccardBOPPolicy(cost=0.0, allow_abstention=True)
        np.testing.assert_array_equal(partial.predict(deterministic), [[0, 1]])
        self.assertTrue(np.all(partial.decision_mask(deterministic)))

        probabilities = np.array([[0.2, 0.45, 0.8]], dtype=np.float64)
        complete = JaccardBOPPolicy(allow_abstention=False)
        np.testing.assert_array_equal(
            complete.predict(probabilities, cost=0.0),
            complete.predict(probabilities, cost=1.0, penalty="concave"),
        )

    def test_registry_api_and_config_parity_for_all_policy_families(self):
        self.assertEqual(
            set(available_policies()), {"fbeta", "hamming", "jaccard"}
        )
        policies = (
            create_policy("hamming", cost=0.2),
            create_policy("f1", cost=0.2),
            create_policy("iou", cost=0.2),
        )
        self.assertIsInstance(policies[0], HammingBOPPolicy)
        self.assertIsInstance(policies[1], FbetaBOPPolicy)
        self.assertIsInstance(policies[2], JaccardBOPPolicy)
        probabilities = np.array([[0.2, 0.8]], dtype=np.float64)
        configs = []
        for policy in policies:
            self.assertIsInstance(policy, DecisionPolicy)
            self.assertEqual(policy.predict(probabilities).shape, (1, 2))
            self.assertEqual(policy.expected_utility(probabilities).shape, (1,))
            config = policy.get_config()
            configs.append(config)
            self.assertEqual(json.loads(json.dumps(config)), config)
            self.assertIn("name", config)
            self.assertIn("objective", config)
            self.assertIn("tie_breaking", config)
        self.assertEqual(
            policies[2].get_config()["dependent_marginal_interpretation"],
            "BOP under CLI approximation",
        )
        self.assertEqual(
            len({compute_config_hash(config) for config in configs}), 3
        )

    def test_runtime_and_memory_smoke_at_larger_k(self):
        probabilities = np.linspace(0.01, 0.99, 100, dtype=np.float64)[None, :]
        policy = JaccardBOPPolicy(
            cost=0.1, penalty="concave", allow_abstention=True
        )
        tracemalloc.start()
        started = time.perf_counter()
        prediction, utility = policy.predict_with_utility(probabilities)
        elapsed = time.perf_counter() - started
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.assertEqual(prediction.shape, probabilities.shape)
        self.assertTrue(np.isfinite(utility[0]))
        self.assertLess(elapsed, 5.0)
        self.assertLess(peak_bytes, 64 * 1024 * 1024)

    def test_parameter_and_probability_validation(self):
        with self.assertRaises(ValueError):
            JaccardBOPPolicy(cost=1.1)
        with self.assertRaises(ValueError):
            JaccardBOPPolicy(penalty="unknown")
        with self.assertRaises(ValueError):
            JaccardBOPPolicy(abstain_value=0)
        with self.assertRaises(ValueError):
            JaccardBOPPolicy().predict(np.array([[0.2, np.inf]]))


if __name__ == "__main__":
    unittest.main()
