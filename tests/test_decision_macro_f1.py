"""Tests for PerLabelMacroF1Policy and Selective Instance-F1."""

import json
import unittest

import numpy as np

from src.decision import (
    available_policies,
    canonical_policy_name,
    create_configured_policy,
    create_policy,
)
from src.decision.macro_f1 import PerLabelMacroF1Policy
from src.evaluation.metrics import (
    compute_partial_abstention_metrics,
    compute_selective_instance_f1,
)


class PerLabelMacroF1PolicyTests(unittest.TestCase):
    def test_registry_registration_and_aliases(self):
        self.assertIn("macro_f1", available_policies())
        self.assertEqual(canonical_policy_name("per_label_macro_f1"), "macro_f1")
        self.assertEqual(canonical_policy_name("macro_f1_bop"), "macro_f1")
        self.assertEqual(canonical_policy_name("per_label_f1"), "macro_f1")

        policy = create_configured_policy("macro_f1", cost=0.25, penalty="linear")
        self.assertIsInstance(policy, PerLabelMacroF1Policy)
        self.assertEqual(policy.cost, 0.25)
        self.assertEqual(policy.penalty, "linear")

        config = policy.get_config()
        self.assertEqual(config["name"], "per_label_macro_f1")
        self.assertEqual(config["objective"], "selective_macro_f1")
        self.assertEqual(json.loads(json.dumps(config)), config)

    def test_unfitted_default_behavior(self):
        policy = PerLabelMacroF1Policy(cost=0.3, penalty="linear", allow_abstention=True)
        probs = np.array(
            [[0.10, 0.29, 0.30, 0.50, 0.70, 0.71, 0.90]],
            dtype=np.float64,
        )
        preds = policy.predict(probs)
        # Default thresholds: tau_low = 0.3, tau_high = 0.7
        # p <= 0.3 -> 0, p >= 0.7 -> 1, 0.3 < p < 0.7 -> -1
        expected = np.array([[0, 0, 0, -1, 1, 1, 1]], dtype=np.int32)
        np.testing.assert_array_equal(preds, expected)

        # Allow abstention = False -> threshold at 0.5
        policy_complete = PerLabelMacroF1Policy(allow_abstention=False)
        preds_complete = policy_complete.predict(probs)
        expected_complete = np.array([[0, 0, 0, 1, 1, 1, 1]], dtype=np.int32)
        np.testing.assert_array_equal(preds_complete, expected_complete)

    def test_fitting_optimal_thresholds(self):
        np.random.seed(42)
        n_samples = 300
        n_labels = 3

        val_probs = np.random.uniform(0.0, 1.0, size=(n_samples, n_labels))
        # Label 0: correlates with p > 0.4
        # Label 1: correlates with p > 0.7 (rare label)
        # Label 2: all zeros (degenerate)
        val_y = np.zeros((n_samples, n_labels), dtype=np.int32)
        val_y[:, 0] = (val_probs[:, 0] > 0.4).astype(np.int32)
        val_y[:, 1] = (val_probs[:, 1] > 0.7).astype(np.int32)
        # Label 2 stays all 0

        policy = PerLabelMacroF1Policy(cost=0.3, penalty="linear")
        policy.fit(val_probs, val_y)

        self.assertIsNotNone(policy.tau_low)
        self.assertIsNotNone(policy.tau_high)
        self.assertEqual(len(policy.tau_low), n_labels)
        self.assertEqual(len(policy.tau_high), n_labels)

        # All thresholds must satisfy tau_low <= tau_high
        self.assertTrue(np.all(policy.tau_low <= policy.tau_high + 1e-9))

        # Degenerate all-zero label should have tau_low = tau_high = 1.0 (always predict 0)
        self.assertAlmostEqual(policy.tau_low[2], 1.0)
        self.assertAlmostEqual(policy.tau_high[2], 1.0)

        # Predictions on test data
        test_probs = np.array([[0.5, 0.8, 0.9], [0.1, 0.2, 0.5]], dtype=np.float64)
        preds = policy.predict(test_probs)
        self.assertEqual(preds.shape, (2, 3))
        # Label 2 should always predict 0
        self.assertEqual(preds[0, 2], 0)
        self.assertEqual(preds[1, 2], 0)

    def test_expected_utility_is_finite_and_bounded(self):
        policy = PerLabelMacroF1Policy(cost=0.3, penalty="linear")
        probs = np.array([[0.2, 0.8], [0.5, 0.5]], dtype=np.float64)
        utility = policy.expected_utility(probs)
        self.assertEqual(utility.shape, (2,))
        self.assertTrue(np.all(np.isfinite(utility)))


class SelectiveInstanceF1Tests(unittest.TestCase):
    def test_exact_match_and_partial_abstention(self):
        # Sample 0: true=[1, 0, 1], partial=[1, -1, 1] -> decided=[1, 1], true=[1, 1], pred=[1, 1] -> F1=1.0
        # Sample 1: true=[0, 1, 0], partial=[-1, -1, -1] -> all abstained -> F1=0.0
        # Sample 2: true=[0, 0, 0], partial=[0, 0, -1] -> decided=[0, 0], both empty -> F1=1.0
        y_true = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=np.int32)
        y_partial = np.array([[1, -1, 1], [-1, -1, -1], [0, 0, -1]], dtype=np.int32)

        score = compute_selective_instance_f1(y_true, y_partial)
        # Expected mean: (1.0 + 0.0 + 1.0) / 3 = 2/3
        self.assertAlmostEqual(score, 2.0 / 3.0)

    def test_partial_mismatch(self):
        # Sample 0: true=[1, 1, 0], pred=[1, 0, -1] -> decided labels 0,1: TP=1, FP=0, FN=1 -> F1 = 2*1 / (2*1 + 0 + 1) = 2/3
        y_true = np.array([[1, 1, 0]], dtype=np.int32)
        y_partial = np.array([[1, 0, -1]], dtype=np.int32)
        score = compute_selective_instance_f1(y_true, y_partial)
        self.assertAlmostEqual(score, 2.0 / 3.0)

    def test_integration_with_partial_abstention_metrics(self):
        y_true = np.array([[1, 0], [0, 1]], dtype=np.int32)
        y_partial = np.array([[1, -1], [0, 1]], dtype=np.int32)
        metrics = compute_partial_abstention_metrics(y_true, y_partial, cost=0.3)

        self.assertIn("Selective Instance-F1", metrics)
        self.assertIsInstance(metrics["Selective Instance-F1"], float)
        self.assertGreater(metrics["Selective Instance-F1"], 0.0)


if __name__ == "__main__":
    unittest.main()
