"""Phase-Q10 deployment, policy, selection, export and plot tests."""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.evaluation.deployment import (
    compute_deployment_metrics,
    select_operating_point,
)
from src.evaluation.export_v3 import export_v3_artifacts
from src.evaluation.label_policy import (
    get_dataset_label_policy,
    label_policy_hash,
    load_label_policy_config,
)
from src.evaluation.pipeline_v3 import _select_inner_operating_point
from src.visualization.deployment import generate_deployment_plots


def _fixture_arrays():
    truth = np.array([[1, 0], [0, 1]], dtype=np.int32)
    full = np.array([[1, 1], [0, 0]], dtype=np.int32)
    partial = np.array([[1, -1], [0, -1]], dtype=np.int32)
    confidence = np.array([[0.9, 0.1], [0.8, 0.2]], dtype=np.float64)
    return truth, full, partial, confidence


def _policy():
    return {
        "dataset": "tiny",
        "critical_labels": ["b"],
        "weights": {"b": 2.0},
        "false_negative_cost": {"b": 3.0},
        "false_positive_cost": {"b": 4.0},
        "review_cost": 0.25,
        "reviewer_accuracies": [0.8, 1.0],
    }


class DeploymentMetricTests(unittest.TestCase):
    def test_hand_computed_workload_capture_optimistic_and_aurc(self):
        truth, full, partial, confidence = _fixture_arrays()
        result = compute_deployment_metrics(
            truth,
            full,
            partial,
            cost=0.25,
            label_names=("a", "b"),
            acceptance_confidence=confidence,
        )
        metrics = result["Metrics"]
        self.assertAlmostEqual(metrics["Coverage"], 0.5)
        self.assertAlmostEqual(metrics["Selective Risk"], 0.0)
        self.assertAlmostEqual(metrics["Review Load Position"], 0.5)
        self.assertAlmostEqual(metrics["Review Load Case"], 1.0)
        self.assertAlmostEqual(metrics["Generalized Loss"], 0.125)
        self.assertAlmostEqual(metrics["Error Capture Rate"], 1.0)
        self.assertAlmostEqual(metrics["Error Capture Lift"], 2.0)
        self.assertAlmostEqual(metrics["Optimistic Gain Hamming Accuracy"], 0.5)
        self.assertAlmostEqual(metrics["AURC"], (0.0 + 0.0 + 1 / 3 + 0.5) / 4)

    def test_policy_enables_critical_and_reviewer_utility_scenarios(self):
        truth, full, partial, confidence = _fixture_arrays()
        result = compute_deployment_metrics(
            truth,
            full,
            partial,
            cost=0.25,
            label_names=("a", "b"),
            label_policy=_policy(),
            acceptance_confidence=confidence,
        )
        self.assertEqual(result["Critical Labels"]["Status"], "Available")
        critical = result["Critical Labels"]["Metrics"]
        self.assertEqual(critical["Critical Coverage"], 0.0)
        self.assertEqual(critical["Critical Error Capture Rate"], 1.0)
        reviewers = result["Reviewer Scenarios"]["Records"]
        self.assertEqual([row["Reviewer Accuracy"] for row in reviewers], [0.8, 1.0])
        self.assertAlmostEqual(reviewers[0]["Total Expected Cost"], 3.3)
        self.assertAlmostEqual(reviewers[0]["Cost-Sensitive Utility"], -0.825)
        self.assertAlmostEqual(reviewers[1]["Cost-Sensitive Utility"], -0.125)
        self.assertAlmostEqual(result["Metrics"]["Cost-Sensitive Utility"], -0.125)

    def test_missing_policy_marks_domain_metrics_na_without_crashing(self):
        truth, full, partial, _ = _fixture_arrays()
        result = compute_deployment_metrics(
            truth,
            full,
            partial,
            cost=0.5,
            label_names=("a", "b"),
        )
        self.assertEqual(result["Policy Status"], "N/A")
        self.assertEqual(result["Critical Labels"]["Status"], "N/A")
        self.assertEqual(result["Reviewer Scenarios"]["Status"], "N/A")
        self.assertTrue(math.isnan(result["Metrics"]["Cost-Sensitive Utility"]))


class LabelPolicyTests(unittest.TestCase):
    def test_missing_file_and_meeting_note_shape_are_supported(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "missing.json"
            config = load_label_policy_config(missing)
            self.assertEqual(config["status"], "missing")
            self.assertIsNone(get_dataset_label_policy(config, "tiny", ["a", "b"]))

            path = Path(temporary_directory) / "policy.json"
            path.write_text(
                json.dumps({"tiny": {
                    "critical_labels": ["b"],
                    "weights": {"b": 2},
                    "review_cost": 0.25,
                }}),
                encoding="utf-8",
            )
            loaded = load_label_policy_config(path)
            policy = get_dataset_label_policy(loaded, "tiny", ["a", "b"])
            self.assertEqual(policy["critical_labels"], ["b"])
            self.assertEqual(policy["reviewer_accuracies"], [0.8, 0.9, 0.95, 1.0])
            self.assertEqual(label_policy_hash(policy), label_policy_hash(dict(policy)))

    def test_policy_validation_uses_declared_names_not_test_prevalence(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "policy.json"
            path.write_text(
                json.dumps({"tiny": {"critical_labels": ["not-a-label"]}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unknown labels"):
                get_dataset_label_policy(
                    load_label_policy_config(path), "tiny", ["a", "b"]
                )


class OperatingPointTests(unittest.TestCase):
    @staticmethod
    def _records():
        return [
            {"Cost": 0.2, "Data Scope": "inner_validation", "Generalized Loss": 0.3,
             "Coverage": 0.6, "Selective Risk": 0.05, "Cost-Sensitive Utility": -0.4},
            {"Cost": 0.3, "Data Scope": "inner_validation", "Generalized Loss": 0.2,
             "Coverage": 0.8, "Selective Risk": 0.10, "Cost-Sensitive Utility": -0.2},
            {"Cost": 0.5, "Data Scope": "inner_validation", "Generalized Loss": 0.25,
             "Coverage": 1.0, "Selective Risk": 0.20, "Cost-Sensitive Utility": -0.1},
        ]

    def test_all_selection_rules_are_deterministic_and_inner_only(self):
        records = self._records()
        self.assertEqual(
            select_operating_point(
                records, "min_generalized_loss", data_scope="inner_validation"
            )["Selected Cost"],
            0.3,
        )
        self.assertEqual(
            select_operating_point(
                records,
                "max_utility_at_coverage",
                data_scope="inner_validation",
                coverage_gamma=0.8,
            )["Selected Cost"],
            0.5,
        )
        self.assertEqual(
            select_operating_point(
                records,
                "max_coverage_at_risk",
                data_scope="inner_validation",
                risk_epsilon=0.1,
            )["Selected Cost"],
            0.3,
        )
        with self.assertRaisesRegex(ValueError, "inner_validation"):
            select_operating_point(
                records, "min_generalized_loss", data_scope="outer_test"
            )

    def test_inner_selector_never_accepts_outer_test_data(self):
        fit_sizes = []

        class Selector:
            abstain_value = -1
            decision_policy = "hamming"
            cost = 0.3
            penalty = "linear"
            beta = 1.0

            def fit(self, x_data, y_data):
                fit_sizes.append((len(x_data), len(y_data)))
                return self

            def predict_proba(self, x_data):
                base = np.linspace(0.1, 0.9, len(x_data))
                return np.column_stack((base, base[::-1]))

            @staticmethod
            def predict_full_from_proba(probabilities):
                return (probabilities >= 0.5).astype(np.int32)

        x_data = np.arange(40, dtype=np.float64).reshape(10, 4)
        y_data = np.column_stack((
            np.arange(10) % 2,
            (np.arange(10) // 2) % 2,
        )).astype(np.int32)
        selection = _select_inner_operating_point(
            model_name="MLC_PA_Logistic",
            x_train=x_data,
            y_train=y_data,
            abstention_costs=[0.2, 0.5],
            rule="min_generalized_loss",
            coverage_gamma=0.8,
            risk_epsilon=0.1,
            validation_size=0.2,
            random_state=42,
            model_factory=lambda *args, **kwargs: Selector(),
            model_factory_kwargs={},
            label_names=("a", "b"),
            label_policy=None,
        )
        self.assertEqual(fit_sizes, [(8, 8)])
        self.assertEqual(selection["Data Scope"], "inner_validation")
        self.assertEqual(selection["Inner Validation Size"], 2)
        self.assertFalse(selection["Outer Test Access"])
        self.assertTrue(all(
            row["Data Scope"] == "inner_validation"
            for row in selection["Candidate Records"]
        ))


class DeploymentArtifactTests(unittest.TestCase):
    @staticmethod
    def _document():
        raw = {
            "Fold": 1,
            "Coverage": 0.5,
            "Selective Risk": 0.1,
            "Review Load Position": 0.5,
            "Optimistic Gain Macro-F1": 0.2,
            "Error Capture Rate": 0.8,
        }
        return {
            "Schema Version": 3,
            "Results": {"tiny": {"MLC_PA_Logistic": {
                "Full": {"Raw Folds": [{"Fold": 1, "Macro-F1": 0.4}]},
                "Costs": {"0.30": {
                    "Selective": {"Raw Folds": [{"Fold": 1, "Coverage": 0.5}]},
                    "Rejected": {"Raw Folds": []},
                    "Optimistic": {"Raw Folds": []},
                    "Diagnostics": {"Raw Folds": []},
                    "Deployment": {
                        "Raw Folds": [raw],
                        "Reviewer Scenarios": [],
                        "Critical Labels": [{"Fold": 1, "Status": "N/A", "Metrics": {}}],
                    },
                }},
            }}},
        }

    def test_exports_and_plots_keep_full_and_selective_scopes_separate(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            document = self._document()
            plot_artifacts = generate_deployment_plots(document, root / "figures")
            document["Artifacts"] = plot_artifacts
            artifacts = export_v3_artifacts(document, root / "tables")
            for path in artifacts.values():
                self.assertTrue(Path(path).exists(), path)
            with Path(artifacts["deployment_csv"]).open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row["Scope"] for row in rows}, {"Operating Point"})
            self.assertEqual(rows[0]["Coverage"], "0.5")
            complete_text = Path(artifacts["complete_csv"]).read_text(encoding="utf-8")
            self.assertIn("Macro-F1", complete_text)
            self.assertNotIn("Selective Risk", complete_text)

    def test_plots_tolerate_empty_partial_and_nan_records(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            empty = {"Results": {}}
            artifacts = generate_deployment_plots(empty, temporary_directory)
            self.assertEqual(len(artifacts), 7)
            self.assertTrue(all(Path(path).stat().st_size > 0 for path in artifacts.values()))

            partial = self._document()
            partial["Results"]["tiny"]["MLC_PA_Logistic"]["Costs"]["0.30"][
                "Deployment"
            ]["Raw Folds"][0]["Selective Risk"] = float("nan")
            generate_deployment_plots(partial, temporary_directory)


if __name__ == "__main__":
    unittest.main()
