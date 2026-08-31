"""Phase-Q1 reference and edge-case tests for schema-v3 metric modules."""

import math
import unittest

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

from src.evaluation.abstention_metrics import compute_abstention_metrics
from src.evaluation.complete_metrics import compute_complete_metrics
from src.evaluation.group_metrics import (
    compute_group_metrics,
    compute_per_label_metrics,
)
from src.evaluation.metric_facade import compute_metric_bundle
from src.evaluation.metric_contract import PER_LABEL_FIELD_NAMES
from tests.fixtures.metric_cases import (
    LABEL_GROUPS,
    all_abstain_case,
    complete_metric_case,
    partial_metric_case,
    zero_positive_support_case,
)


class CompleteMetricTests(unittest.TestCase):
    def test_complete_metrics_match_hand_values_and_sklearn(self):
        y_true, y_full, expected = complete_metric_case()
        actual = compute_complete_metrics(y_true, y_full)
        for name, value in expected.items():
            self.assertAlmostEqual(actual[name], value, msg=name)

        self.assertAlmostEqual(
            actual["Macro-F1"],
            f1_score(y_true, y_full, average="macro", zero_division=0),
        )
        self.assertAlmostEqual(
            actual["Micro-F1"],
            f1_score(y_true, y_full, average="micro", zero_division=0),
        )
        self.assertAlmostEqual(
            actual["Macro Precision"],
            precision_score(y_true, y_full, average="macro", zero_division=0),
        )
        self.assertAlmostEqual(
            actual["Macro Recall"],
            recall_score(y_true, y_full, average="macro", zero_division=0),
        )
        self.assertAlmostEqual(actual["Hamming Loss"], hamming_loss(y_true, y_full))
        self.assertAlmostEqual(
            actual["Subset Accuracy"], accuracy_score(y_true, y_full)
        )

    def test_complete_metrics_match_sklearn_on_deterministic_random_cases(self):
        generator = np.random.default_rng(20260831)
        # K>=2 forces sklearn to use its multilabel-indicator path.  For K=1,
        # sklearn may infer a binary-class target and include the negative class
        # in average="macro", which is not this contract's positive-label F1.
        for n_samples, n_labels in ((2, 2), (3, 2), (9, 5), (17, 8)):
            y_true = generator.integers(0, 2, size=(n_samples, n_labels))
            y_full = generator.integers(0, 2, size=(n_samples, n_labels))
            actual = compute_complete_metrics(y_true, y_full)
            with self.subTest(shape=y_true.shape):
                self.assertAlmostEqual(
                    actual["Macro-F1"],
                    f1_score(y_true, y_full, average="macro", zero_division=0),
                )
                self.assertAlmostEqual(
                    actual["Micro-F1"],
                    f1_score(y_true, y_full, average="micro", zero_division=0),
                )
                self.assertAlmostEqual(
                    actual["Macro Precision"],
                    precision_score(
                        y_true, y_full, average="macro", zero_division=0
                    ),
                )
                self.assertAlmostEqual(
                    actual["Macro Recall"],
                    recall_score(y_true, y_full, average="macro", zero_division=0),
                )
                self.assertAlmostEqual(
                    actual["Hamming Accuracy"] + actual["Hamming Loss"], 1.0
                )

        single_negative = np.zeros((1, 1), dtype=np.int32)
        self.assertEqual(
            compute_complete_metrics(single_negative, single_negative)["Macro-F1"],
            0.0,
        )

    def test_perfect_prediction_and_both_empty_instance(self):
        y_true = np.array([[1, 0], [0, 1], [0, 0]], dtype=np.int32)
        actual = compute_complete_metrics(y_true, y_true)
        self.assertEqual(actual["Hamming Loss"], 0.0)
        for name, value in actual.items():
            if name != "Hamming Loss":
                self.assertEqual(value, 1.0, msg=name)

    def test_complete_input_validation(self):
        valid = np.array([[0, 1]], dtype=np.int32)
        invalid_cases = (
            (np.empty((0, 2), dtype=np.int32), np.empty((0, 2), dtype=np.int32)),
            (valid, np.array([[0, 2]], dtype=np.int32)),
            (valid, np.array([0, 1], dtype=np.int32)),
            (valid, np.array([[0], [1]], dtype=np.int32)),
        )
        for y_true, y_full in invalid_cases:
            with self.subTest(y_true_shape=y_true.shape, y_full_shape=y_full.shape):
                with self.assertRaises(ValueError):
                    compute_complete_metrics(y_true, y_full)


class AbstentionMetricTests(unittest.TestCase):
    def test_partial_operating_point_metrics_and_raw_counts(self):
        y_true, y_full, y_partial = partial_metric_case()
        result = compute_abstention_metrics(
            y_true, y_partial, y_full, cost=0.25
        )
        selective = result["Selective"]
        self.assertAlmostEqual(selective["Selective Hamming Accuracy"], 0.75)
        self.assertAlmostEqual(selective["Selective Macro-F1"], 5.0 / 6.0)
        self.assertAlmostEqual(selective["Selective Micro-F1"], 6.0 / 7.0)
        self.assertAlmostEqual(selective["Generalized Loss"], 0.25)
        self.assertAlmostEqual(selective["Coverage"], 0.5)
        self.assertAlmostEqual(selective["ABS"], 0.75)
        self.assertAlmostEqual(selective["AABS"], 0.5)
        self.assertAlmostEqual(selective["Risk at Coverage"], 0.25)
        self.assertTrue(math.isnan(selective["AURC"]))

        rejected = result["Rejected"]
        self.assertEqual(rejected["Rejected Counterfactual Macro-F1"], 0.0)
        self.assertAlmostEqual(rejected["Rejected Error Rate"], 0.25)
        self.assertAlmostEqual(rejected["Error Capture Rate"], 0.5)
        self.assertEqual(
            result["Diagnostics"],
            {
                "Total Position Count": 8,
                "Decided Position Count": 4,
                "Abstained Position Count": 4,
                "Full Error Count": 2,
                "Rejected Position Count": 4,
                "Rejected Error Count": 1,
                "Errors Avoided": 1,
            },
        )

    def test_concave_generalized_loss(self):
        y_true, y_full, y_partial = partial_metric_case()
        result = compute_abstention_metrics(
            y_true,
            y_partial,
            y_full,
            cost=0.25,
            penalty="concave",
        )
        self.assertAlmostEqual(
            result["Selective"]["Generalized Loss"],
            (1.0 + 1.0 / 6.0 + 1.0 / 6.0 + 1.0 / 4.0) / 8.0,
        )

    def test_all_abstain_conventions_and_oracle_completion(self):
        y_true, y_full, y_partial = all_abstain_case()
        result = compute_abstention_metrics(
            y_true, y_partial, y_full, cost=0.25
        )
        selective = result["Selective"]
        self.assertEqual(selective["Coverage"], 0.0)
        self.assertEqual(selective["ABS"], 1.0)
        self.assertEqual(selective["AABS"], 1.0)
        self.assertTrue(math.isnan(selective["Selective Hamming Accuracy"]))
        self.assertTrue(math.isnan(selective["Risk at Coverage"]))
        self.assertEqual(selective["Selective Macro-F1"], 0.0)
        self.assertEqual(selective["Selective Micro-F1"], 0.0)
        self.assertAlmostEqual(selective["Generalized Loss"], 0.25)
        self.assertAlmostEqual(
            result["Rejected"]["Rejected Counterfactual Macro-F1"], 11.0 / 15.0
        )
        self.assertAlmostEqual(result["Rejected"]["Rejected Error Rate"], 0.25)
        self.assertEqual(result["Rejected"]["Error Capture Rate"], 1.0)
        for name, value in result["Optimistic"].items():
            if name.startswith("Optimistic "):
                self.assertEqual(value, 1.0, msg=name)

        zero_true, zero_full = zero_positive_support_case()
        zero_partial = np.full_like(zero_true, -1)
        zero_result = compute_abstention_metrics(
            zero_true, zero_partial, zero_full, cost=0.25
        )
        self.assertAlmostEqual(
            zero_result["Optimistic"]["Optimistic Macro-F1"], 0.5
        )

    def test_no_abstain_and_no_error_denominators(self):
        y_true, y_full, _ = partial_metric_case()
        no_abstain = compute_abstention_metrics(
            y_true, y_full, y_full, cost=0.4
        )
        self.assertEqual(no_abstain["Selective"]["Coverage"], 1.0)
        self.assertEqual(no_abstain["Selective"]["AABS"], 0.0)
        self.assertAlmostEqual(no_abstain["Selective"]["Generalized Loss"], 0.25)
        self.assertTrue(
            math.isnan(no_abstain["Rejected"]["Rejected Error Rate"])
        )
        self.assertTrue(
            math.isnan(no_abstain["Rejected"]["Rejected Counterfactual Macro-F1"])
        )
        self.assertEqual(no_abstain["Rejected"]["Error Capture Rate"], 0.0)
        for name, value in no_abstain["Optimistic"].items():
            if name.startswith("Oracle Gain "):
                self.assertEqual(value, 0.0, msg=name)

        perfect_partial = y_true.copy()
        perfect_partial[0, 0] = -1
        no_full_errors = compute_abstention_metrics(
            y_true, perfect_partial, y_true, cost=0.25
        )
        self.assertEqual(no_full_errors["Rejected"]["Rejected Error Rate"], 0.0)
        self.assertTrue(
            math.isnan(no_full_errors["Rejected"]["Error Capture Rate"])
        )
        self.assertEqual(no_full_errors["Diagnostics"]["Full Error Count"], 0)

    def test_aurc_uses_stable_acceptance_ranking(self):
        y_true, y_full, y_partial = partial_metric_case()
        confidence = np.array(
            [[8.0, 7.0], [2.0, 1.0], [6.0, 5.0], [4.0, 3.0]]
        )
        result = compute_abstention_metrics(
            y_true,
            y_partial,
            y_full,
            cost=0.25,
            acceptance_confidence=confidence,
        )
        expected = (1.0 / 7.0 + 2.0 / 8.0) / 8.0
        self.assertAlmostEqual(result["Selective"]["AURC"], expected)

    def test_abstention_validation(self):
        y_true, y_full, y_partial = partial_metric_case()
        invalid_calls = (
            {"cost": -0.1},
            {"cost": float("nan")},
            {"cost": 0.2, "penalty": "unknown"},
            {"cost": 0.2, "acceptance_confidence": np.ones((1, 1))},
        )
        for kwargs in invalid_calls:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    compute_abstention_metrics(y_true, y_partial, y_full, **kwargs)

        invalid_partial = y_partial.copy()
        invalid_partial[0, 0] = 2
        with self.assertRaises(ValueError):
            compute_abstention_metrics(
                y_true, invalid_partial, y_full, cost=0.2
            )


class LabelGroupAndFacadeTests(unittest.TestCase):
    def test_per_label_support_coverage_and_zero_support(self):
        y_true, y_full, y_partial = partial_metric_case()
        records = compute_per_label_metrics(
            y_true,
            y_full,
            y_partial,
            label_names=("first", "second"),
        )
        self.assertEqual([record["Label Name"] for record in records], ["first", "second"])
        self.assertEqual(records[0]["Positive Support"], 2)
        self.assertEqual(records[0]["Negative Support"], 2)
        self.assertEqual(records[0]["Coverage"], 0.5)
        self.assertEqual(records[0]["Rejected Error Count"], 1)
        self.assertEqual(tuple(records[0]), PER_LABEL_FIELD_NAMES)

        zero_true, zero_full = zero_positive_support_case()
        zero_records = compute_per_label_metrics(zero_true, zero_full)
        self.assertEqual(zero_records[1]["Positive Support"], 0)
        self.assertEqual(zero_records[1]["Full Recall"], 0.0)
        self.assertEqual(zero_records[1]["Full F1"], 0.0)

    def test_named_groups_and_empty_group(self):
        y_true, y_full, y_partial = partial_metric_case()
        groups = compute_group_metrics(
            y_true, y_full, y_partial, LABEL_GROUPS
        )
        self.assertEqual(groups["IL"]["IL Label Count"], 1)
        self.assertAlmostEqual(groups["IL"]["IL Full Macro-F1"], 0.8)
        self.assertAlmostEqual(groups["DL"]["DL Full Macro-F1"], 2.0 / 3.0)
        self.assertEqual(groups["IL"]["IL Coverage"], 0.5)
        self.assertEqual(groups["EMPTY"]["EMPTY Label Count"], 0)
        for name, value in groups["EMPTY"].items():
            if name != "EMPTY Label Count":
                self.assertTrue(math.isnan(value), msg=name)

    def test_group_validation(self):
        y_true, y_full, y_partial = partial_metric_case()
        for groups in ({"bad": (2,)}, {"bad": (0, 0)}, [("IL", (0,))]):
            with self.subTest(groups=groups):
                with self.assertRaises(ValueError):
                    compute_group_metrics(y_true, y_full, y_partial, groups)

    def test_facade_alias_critical_hook_and_no_duplicate_column(self):
        y_true, y_full, y_partial = partial_metric_case()
        bundle = compute_metric_bundle(
            y_true,
            y_full,
            y_partial=y_partial,
            cost=0.25,
            label_groups=LABEL_GROUPS,
            label_names=("first", "second"),
        )
        self.assertNotIn("Example-F1", bundle["Full"].keys())
        self.assertEqual(bundle["Full"]["Example-F1"], bundle["Full"]["Instance-F1"])
        self.assertEqual(
            bundle["Full"].get("Example-F1"), bundle["Full"]["Instance-F1"]
        )
        self.assertEqual(
            bundle["Selective"]["Generalized Hamming Loss"],
            bundle["Selective"]["Generalized Loss"],
        )
        self.assertEqual(bundle["Critical Labels"]["Status"], "N/A")

        critical = compute_metric_bundle(
            y_true,
            y_full,
            y_partial=y_partial,
            cost=0.25,
            label_names=("first", "second"),
            critical_labels=("second", 0),
        )["Critical Labels"]
        self.assertEqual(critical["Status"], "Available")
        self.assertEqual(
            [record["Label Index"] for record in critical["Records"]], [1, 0]
        )

    def test_facade_full_only_is_no_abstention_and_partial_requires_cost(self):
        y_true, y_full, y_partial = partial_metric_case()
        full_only = compute_metric_bundle(y_true, y_full)
        self.assertEqual(full_only["Selective"]["Coverage"], 1.0)
        self.assertEqual(full_only["Selective"]["AABS"], 0.0)
        with self.assertRaises(ValueError):
            compute_metric_bundle(y_true, y_full, y_partial=y_partial)


if __name__ == "__main__":
    unittest.main()
