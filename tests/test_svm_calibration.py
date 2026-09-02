"""Q9 tests for calibrated SVM baselines and calibration metrics."""

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from main import _create_model, _evaluate_model
from src.evaluation.calibration_metrics import compute_calibration_metrics
from src.evaluation.export_v3 import export_v3_artifacts
from src.evaluation.pipeline_v3 import summarize_v3_checkpoint
from src.models.probability_adapter import ProbabilityAdapter
from src.models.registry import MATCHED_MODEL_IDS, model_family


class ProbabilityAdapterTests(unittest.TestCase):
    @staticmethod
    def _features(n_samples=12):
        return np.linspace(-1.0, 1.0, n_samples * 2, dtype=np.float32).reshape(
            n_samples, 2
        )

    def test_balanced_labels_use_nested_sigmoid_calibration(self):
        features = self._features()
        target = np.array([0, 1] * 6, dtype=np.int32)
        adapter = ProbabilityAdapter(cv=5, random_state=7).fit(features, target)
        audit = adapter.calibration_audit_
        self.assertEqual(audit["strategy"], "calibrated_sigmoid_cv")
        self.assertEqual(audit["effective_cv"], 5)
        self.assertIsNone(audit["fallback"])
        self.assertEqual(audit["fit_sample_count"], 12)
        self.assertFalse(audit["outer_test_access"])
        probabilities = adapter.predict_proba(features[:3])
        self.assertEqual(probabilities.shape, (3, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_rare_and_constant_fallbacks_are_deterministic(self):
        features = self._features()
        fixtures = (
            (
                np.array([1] + [0] * 11, dtype=np.int32),
                "smoothed_prior",
                2.0 / 14.0,
            ),
            (np.zeros(12, dtype=np.int32), "constant_label", 0.0),
            (np.ones(12, dtype=np.int32), "constant_label", 1.0),
        )
        for target, strategy, expected_probability in fixtures:
            with self.subTest(strategy=strategy, positive_count=int(target.sum())):
                first = ProbabilityAdapter(random_state=9).fit(features, target)
                second = ProbabilityAdapter(random_state=9).fit(features, target)
                self.assertEqual(first.calibration_audit_, second.calibration_audit_)
                self.assertEqual(first.calibration_audit_["strategy"], strategy)
                self.assertIsNotNone(first.calibration_audit_["fallback"])
                np.testing.assert_allclose(
                    first.predict_proba(features[:2])[:, 1],
                    expected_probability,
                )
                np.testing.assert_array_equal(
                    first.predict(features[:2]), second.predict(features[:2])
                )


class CalibrationMetricTests(unittest.TestCase):
    def test_brier_log_loss_ece_and_reliability_match_direct_values(self):
        truth = np.array([[0, 1], [1, 0]], dtype=np.int32)
        probabilities = np.array([[0.1, 0.8], [0.7, 0.4]], dtype=np.float64)
        result = compute_calibration_metrics(
            truth,
            probabilities,
            n_bins=2,
            label_names=("a", "b"),
        )
        expected_brier = float(np.mean((probabilities - truth) ** 2))
        expected_log_loss = float(
            -np.mean(
                truth * np.log(probabilities)
                + (1 - truth) * np.log(1.0 - probabilities)
            )
        )
        self.assertAlmostEqual(result["Metrics"]["Brier Score"], expected_brier)
        self.assertAlmostEqual(result["Metrics"]["Log Loss"], expected_log_loss)
        self.assertEqual(sum(row["Count"] for row in result["Reliability"]), 4)
        self.assertEqual(len(result["Reliability"]), 2)
        self.assertEqual(len(result["Per Label"]), 2)
        self.assertTrue(0.0 <= result["Metrics"]["ECE"] <= 1.0)
        with self.assertRaises(ValueError):
            compute_calibration_metrics(truth, probabilities[:, :1])
        with self.assertRaises(ValueError):
            compute_calibration_metrics(truth, probabilities, n_bins=1)


class CalibratedSVMRegistryTests(unittest.TestCase):
    @staticmethod
    def _data():
        generator = np.random.default_rng(123)
        features = generator.normal(size=(48, 4)).astype(np.float32)
        truth = np.column_stack((
            features[:, 0] + features[:, 1] > 0.0,
            features[:, 2] - 0.25 * features[:, 0] > 0.0,
            features[:, 3] + 0.4 * features[:, 1] > 0.0,
        )).astype(np.int32)
        return features, truth

    def test_registry_has_twelve_ids_and_four_calibrated_svm_families(self):
        svm_ids = (
            "BR_SVM",
            "CC_SVM",
            "MLC_PA_SVM",
            "GSI_MLC_PA_SVM",
        )
        self.assertEqual(len(MATCHED_MODEL_IDS), 12)
        self.assertEqual(MATCHED_MODEL_IDS[-4:], svm_ids)
        self.assertEqual(
            {model_family(model_id) for model_id in svm_ids},
            {"BR", "CC", "MLC_PA", "GSI_MLC_PA"},
        )
        for model_id in svm_ids:
            model = _create_model(model_id)
            base_manifest = model.experiment_manifest_["base_learner"]
            self.assertEqual(base_manifest["name"], "svm_calibrated")
            self.assertEqual(base_manifest["calibration"]["method"], "sigmoid")
            self.assertEqual(
                base_manifest["calibration"]["scope"],
                "nested_in_training_fold",
            )

    def test_four_svm_families_smoke_and_export_calibration_scopes(self):
        features, truth = self._data()
        train_x, test_x = features[:36], features[36:]
        train_y, test_y = truth[:36], truth[36:]
        summaries = {}
        for model_id in MATCHED_MODEL_IDS[-4:]:
            with self.subTest(model_id=model_id):
                model = _create_model(
                    model_id,
                    random_state=17,
                    gsi_validation_size=0.25,
                    gsi_partition_mode="all_il",
                    gsi_final_order="natural",
                ).fit(train_x, train_y)
                result = _evaluate_model(
                    model_id,
                    model,
                    test_x,
                    test_y,
                    [0.3],
                    metric_schema=3,
                    label_names=("a", "b", "c"),
                )
                calibration = result["Calibration"]
                self.assertTrue(
                    all(
                        math.isfinite(calibration["Metrics"][name])
                        for name in ("Brier Score", "Log Loss", "ECE")
                    )
                )
                self.assertEqual(len(calibration["Reliability"]), 10)
                self.assertIn("Calibration Audit", result["Model Metadata"])
                checkpoint = {
                    "status": "complete",
                    "config_hash": f"q9-{model_id}",
                    "config": {"n_splits": 1},
                    "folds": {"1": {"metrics": result, "metadata": {}}},
                }
                summaries[model_id] = summarize_v3_checkpoint(checkpoint)

        with tempfile.TemporaryDirectory() as temporary_directory:
            document = {
                "Schema Version": 3,
                "Results": {"tiny": summaries},
            }
            artifacts = export_v3_artifacts(document, Path(temporary_directory))
            self.assertGreater(
                len(Path(artifacts["calibration_csv"]).read_text(encoding="utf-8").splitlines()),
                1,
            )
            self.assertGreater(
                len(Path(artifacts["reliability_csv"]).read_text(encoding="utf-8").splitlines()),
                1,
            )

    def test_svm_br_and_mlc_pa_have_identical_marginals(self):
        features, truth = self._data()
        br = _create_model("BR_SVM", random_state=29).fit(features[:36], truth[:36])
        mlc = _create_model("MLC_PA_SVM", random_state=29).fit(
            features[:36], truth[:36]
        )
        np.testing.assert_allclose(
            br.predict_proba(features[36:]),
            mlc.predict_proba(features[36:]),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            br.predict(features[36:]),
            mlc.predict_full(features[36:]),
        )

    def test_rare_multilabel_fallback_is_recorded_and_repeatable(self):
        features, truth = self._data()
        rare_truth = truth[:20].copy()
        rare_truth[:, 1] = 0
        rare_truth[0, 1] = 1
        rare_truth[:, 2] = 0
        first = _create_model("BR_SVM", random_state=5).fit(
            features[:20], rare_truth
        )
        repeated = _create_model("BR_SVM", random_state=5).fit(
            features[:20], rare_truth
        )
        strategies = [
            row["strategy"] for row in first.calibration_audit_["labels"]
        ]
        self.assertEqual(
            strategies,
            ["calibrated_sigmoid_cv", "smoothed_prior", "constant_label"],
        )
        self.assertEqual(first.calibration_audit_["fallback_label_count"], 2)
        self.assertEqual(first.calibration_audit_, repeated.calibration_audit_)
        np.testing.assert_allclose(
            first.predict_proba(features[20:24]),
            repeated.predict_proba(features[20:24]),
            rtol=0.0,
            atol=0.0,
        )

    def test_spy_confirms_calibrators_never_receive_outer_test_features(self):
        features, truth = self._data()
        outer_x = np.full((3, 4), 999.0, dtype=np.float32)
        outer_y = np.zeros((3, 3), dtype=np.int32)
        observed = []
        real_fit = ProbabilityAdapter.fit

        def fit_spy(adapter, x_data, y_data):
            observed.append(
                (np.asarray(x_data).copy(), np.asarray(y_data).copy())
            )
            return real_fit(adapter, x_data, y_data)

        classifier = _create_model(
            "GSI_MLC_PA_SVM",
            random_state=41,
            gsi_validation_size=0.25,
            gsi_partition_mode="all_il",
            gsi_final_order="natural",
        )
        with mock.patch.object(ProbabilityAdapter, "fit", new=fit_spy):
            classifier.fit(features[:36], truth[:36])
        observed_count = len(observed)
        _evaluate_model(
            "GSI_MLC_PA_SVM",
            classifier,
            outer_x,
            outer_y,
            [0.3],
            metric_schema=3,
            label_names=("a", "b", "c"),
        )
        self.assertEqual(len(observed), observed_count)
        self.assertTrue(observed)
        for fitted_x, fitted_y in observed:
            self.assertLess(float(np.max(fitted_x[:, :4])), 999.0)
            self.assertLessEqual(len(fitted_y), 36)
        self.assertFalse(classifier.calibration_audit_["outer_test_access"])


if __name__ == "__main__":
    unittest.main()
