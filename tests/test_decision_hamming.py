"""Phase-Q3 tests for the decision-policy interface and Hamming parity."""

import json
import unittest

import numpy as np

from main import _evaluate_model
from src.decision import (
    HammingBOPPolicy,
    available_policies,
    create_policy,
)
from src.evaluation.pipeline_v3 import _evaluation_policy_config
from src.models.gsi_mlc_pa import GSIMLCPartialAbstentionClassifier
from src.models.mlc_pa import MLCPartialAbstentionClassifier


def _legacy_hamming_prediction(
    probabilities, cost, penalty="linear", abstain_value=-1
):
    """Frozen pre-Q3 implementation used only as a regression oracle."""

    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
    full = (probabilities >= 0.5).astype(np.int32)
    losses = np.minimum(probabilities, 1.0 - probabilities)
    if penalty == "linear":
        decided = losses <= float(cost)
    else:
        n_samples, n_labels = probabilities.shape
        decided = np.zeros((n_samples, n_labels), dtype=bool)
        abstentions = n_labels - np.arange(n_labels + 1, dtype=np.float64)
        penalty_values = (
            abstentions
            * float(n_labels)
            * float(cost)
            / (float(n_labels) + abstentions)
        )
        for row_index in range(n_samples):
            order = np.argsort(losses[row_index], kind="stable")
            cumulative = np.concatenate(
                ([0.0], np.cumsum(losses[row_index, order], dtype=np.float64))
            )
            risks = cumulative + penalty_values
            best_decided = int(
                np.flatnonzero(np.isclose(risks, np.min(risks)))[-1]
            )
            decided[row_index, order[:best_decided]] = True
    partial = np.full(full.shape, abstain_value, dtype=np.int32)
    partial[decided] = full[decided]
    return partial


def _legacy_gsi_prediction(probabilities, cost, abstain_value=-1):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if cost >= 0.5:
        return (probabilities >= 0.5).astype(np.int32)
    partial = np.full(probabilities.shape, abstain_value, dtype=np.int32)
    partial[probabilities <= cost] = 0
    partial[probabilities >= 1.0 - cost] = 1
    return partial


class _EvaluatorClassifier:
    cost = 0.3
    penalty = "linear"
    abstain_value = -1

    def predict_proba(self, x_data):
        return np.array([[0.1, 0.5], [0.8, 0.2]], dtype=np.float64)[: len(x_data)]

    def predict_full_from_proba(self, probabilities):
        return (probabilities >= 0.5).astype(np.int32)

    def predict_from_proba(self, probabilities, cost=None):
        raise AssertionError("schema-v3 evaluator must use the policy registry")


class HammingPolicyTests(unittest.TestCase):
    def test_registry_and_config_are_deterministic_and_json_serializable(self):
        self.assertIn("hamming", available_policies())
        policy = create_policy(
            "hamming_bop", cost=0.2, penalty="concave", abstain_value=-9
        )
        self.assertIsInstance(policy, HammingBOPPolicy)
        config = policy.get_config()
        self.assertEqual(config["name"], "hamming_bop")
        self.assertEqual(config["penalty"], "concave")
        self.assertEqual(json.loads(json.dumps(config)), config)

    def test_sep_boundaries_and_expected_utility(self):
        probabilities = np.array(
            [[0.10, 0.20, 0.21, 0.50, 0.79, 0.80, 0.90]],
            dtype=np.float64,
        )
        policy = HammingBOPPolicy(cost=0.20)
        np.testing.assert_array_equal(
            policy.predict(probabilities),
            np.array([[0, 0, -1, -1, -1, 1, 1]], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            policy.decision_mask(probabilities),
            np.array([[True, True, False, False, False, True, True]]),
        )
        np.testing.assert_array_equal(
            policy.predict(np.array([[0.5]]), cost=0.5),
            np.array([[1]], dtype=np.int32),
        )
        utility = HammingBOPPolicy(cost=0.2).expected_utility(
            np.array([[0.1, 0.5, 0.9]])
        )
        np.testing.assert_allclose(utility, [-0.4])

    def test_policy_and_delegating_models_match_frozen_pre_q3_rules(self):
        rng = np.random.default_rng(20260831)
        random_rows = rng.uniform(-0.2, 1.2, size=(12, 7))
        boundaries = np.array(
            [[0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0]], dtype=np.float64
        )
        probabilities = np.vstack((boundaries, random_rows))
        for penalty in ("linear", "concave"):
            classifier = MLCPartialAbstentionClassifier(
                cost=0.3, penalty=penalty
            )
            classifier.n_labels_ = probabilities.shape[1]
            for cost in (0.0, 0.2, 0.3, 0.5, 1.0):
                expected = _legacy_hamming_prediction(
                    probabilities, cost, penalty=penalty
                )
                policy_actual = HammingBOPPolicy(
                    cost=0.3, penalty=penalty
                ).predict(probabilities, cost=cost)
                model_actual = classifier.predict_from_proba(
                    probabilities, cost=cost
                )
                np.testing.assert_array_equal(policy_actual, expected)
                np.testing.assert_array_equal(model_actual, expected)

        gsi = GSIMLCPartialAbstentionClassifier(cost=0.3)
        for cost in (0.0, 0.2, 0.3, 0.5, 1.0):
            np.testing.assert_array_equal(
                gsi._apply_bop(probabilities, cost=cost),
                _legacy_gsi_prediction(probabilities, cost),
            )

    def test_v3_evaluator_uses_registry_and_records_policy(self):
        result = _evaluate_model(
            "MLC_PA",
            _EvaluatorClassifier(),
            np.zeros((2, 1)),
            np.array([[0, 1], [1, 0]], dtype=np.int32),
            [0.2],
            metric_schema=3,
            label_names=("a", "b"),
        )
        self.assertEqual(
            result["Model Metadata"]["Decision Policy"]["name"],
            "hamming_bop",
        )
        np.testing.assert_equal(result["Costs"]["0.20"]["Selective"]["Coverage"], 0.75)

    def test_v3_cache_policy_config_distinguishes_sep_and_par(self):
        sep = _evaluation_policy_config("MLC_PA", 0.3, "linear")
        par = _evaluation_policy_config("MLC_PA", 0.3, "concave")
        gsi = _evaluation_policy_config("GSI_MLC_PA", 0.3, "concave")
        self.assertNotEqual(sep, par)
        self.assertEqual(sep["partial_policy"]["name"], "hamming_bop")
        self.assertEqual(gsi["partial_policy"]["penalty"], "linear")
        self.assertEqual(
            gsi["partial_policy"]["linear_boundary"], "symmetric_thresholds"
        )


if __name__ == "__main__":
    unittest.main()
