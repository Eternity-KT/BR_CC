"""Phase-Q0 tests for names, conventions, fixtures and migration boundaries."""

import inspect
import unittest

import numpy as np

from src.evaluation import compute_all_metrics
from src.evaluation.abstention_metrics import compute_abstention_metrics
from src.evaluation.complete_metrics import compute_complete_metrics
from src.evaluation.group_metrics import (
    compute_group_metrics,
    compute_per_label_metrics,
)
from src.evaluation.metric_facade import compute_metric_bundle
from src.evaluation.metric_contract import (
    COMPLETE_METRIC_NAMES,
    DEFAULT_ABSTAIN_VALUE,
    DIAGNOSTIC_FIELD_NAMES,
    EDGE_CASE_CONVENTIONS,
    LEGACY_ALIASES,
    METRIC_DEFINITIONS,
    PER_LABEL_FIELD_NAMES,
    SCOPES,
    canonical_metric_names,
    group_metric_names,
    validate_metric_contract,
)
from tests.fixtures.metric_cases import (
    LABEL_GROUPS,
    all_abstain_case,
    complete_metric_case,
    partial_metric_case,
    zero_positive_support_case,
)


class MetricContractTests(unittest.TestCase):
    def test_contract_names_and_aliases_are_valid(self):
        self.assertTrue(validate_metric_contract())
        names = canonical_metric_names()
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(LEGACY_ALIASES["Example-F1"], "Instance-F1")
        self.assertIn("Hamming Accuracy", COMPLETE_METRIC_NAMES)
        self.assertEqual(SCOPES, ("Full", "Selective", "Rejected", "Optimistic", "Group"))
        self.assertGreaterEqual(len(EDGE_CASE_CONVENTIONS), 10)
        self.assertEqual(len(METRIC_DEFINITIONS), len(names))
        self.assertTrue(all(item.formula for item in METRIC_DEFINITIONS))
        self.assertIn("Full Error Count", DIAGNOSTIC_FIELD_NAMES)
        self.assertIn("Positive Support", PER_LABEL_FIELD_NAMES)
        self.assertEqual(group_metric_names("IL")[0], "IL Label Count")
        self.assertIn("IL Full Macro-F1", group_metric_names("IL"))
        self.assertNotIn(DEFAULT_ABSTAIN_VALUE, (0, 1))

    def test_hand_computed_complete_fixture_is_consistent(self):
        y_true, y_full, expected = complete_metric_case()
        self.assertEqual(y_true.shape, (4, 2))
        self.assertEqual(y_true.shape, y_full.shape)
        self.assertTrue(np.all(np.isin(y_true, (0, 1))))
        self.assertTrue(np.all(np.isin(y_full, (0, 1))))
        self.assertAlmostEqual(
            expected["Hamming Accuracy"] + expected["Hamming Loss"], 1.0
        )
        self.assertEqual(set(expected), set(COMPLETE_METRIC_NAMES))

    def test_partial_and_group_fixtures_cover_contract_edges(self):
        y_true, y_full, y_partial = partial_metric_case()
        self.assertEqual(y_true.shape, y_partial.shape)
        self.assertTrue(np.all(np.isin(y_partial, (-1, 0, 1))))
        self.assertGreater(np.count_nonzero(y_partial == -1), 0)
        _, _, all_abstain = all_abstain_case()
        self.assertTrue(np.all(all_abstain == -1))
        self.assertEqual(LABEL_GROUPS["EMPTY"], ())
        self.assertEqual(y_full.shape[1], 2)

        zero_true, zero_full = zero_positive_support_case()
        self.assertFalse(np.any(zero_true[:, 1]))
        self.assertEqual(zero_true.shape, zero_full.shape)

    def test_q0_skeleton_api_signatures_are_stable(self):
        self.assertEqual(
            list(inspect.signature(compute_complete_metrics).parameters),
            ["y_true", "y_full"],
        )
        self.assertEqual(
            list(inspect.signature(compute_abstention_metrics).parameters)[:3],
            ["y_true", "y_partial", "y_full"],
        )
        self.assertEqual(
            list(inspect.signature(compute_per_label_metrics).parameters)[:3],
            ["y_true", "y_full", "y_partial"],
        )
        self.assertEqual(
            list(inspect.signature(compute_group_metrics).parameters)[:4],
            ["y_true", "y_full", "y_partial", "label_groups"],
        )
        self.assertEqual(
            list(inspect.signature(compute_metric_bundle).parameters)[:2],
            ["y_true", "y_full"],
        )

    def test_production_facade_is_unchanged_during_q0(self):
        y_true, y_full, expected = complete_metric_case()
        actual = compute_all_metrics(y_true, y_full)
        self.assertEqual(
            set(actual),
            {"Macro-F1", "Micro-F1", "Hamming Loss", "Subset Accuracy", "Example-F1"},
        )
        self.assertAlmostEqual(actual["Macro-F1"], expected["Macro-F1"])
        self.assertAlmostEqual(actual["Micro-F1"], expected["Micro-F1"])
        self.assertAlmostEqual(actual["Hamming Loss"], expected["Hamming Loss"])
        self.assertAlmostEqual(actual["Subset Accuracy"], expected["Subset Accuracy"])
        self.assertAlmostEqual(actual["Example-F1"], expected["Instance-F1"])


if __name__ == "__main__":
    unittest.main()
