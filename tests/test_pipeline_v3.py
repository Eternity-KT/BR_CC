"""Phase-Q2 integration tests for schema, export and fold-level resume."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import main as benchmark_main
from main import _evaluate_model
from src.evaluation.cache_v3 import (
    IncompatibleV3CacheError,
    atomic_json_dump_v3,
    completed_fold_indices,
    compute_config_hash,
    fold_checkpoint_path,
    load_or_create_fold_checkpoint,
    mark_checkpoint_complete,
    save_completed_fold,
)
from src.evaluation.export_v3 import export_v3_artifacts
from src.evaluation.metric_contract import (
    COMPLETE_METRIC_NAMES,
    REJECTED_METRIC_NAMES,
    SELECTIVE_METRIC_NAMES,
)
from src.evaluation.pipeline_v3 import (
    V3OutputIsolationError,
    ensure_v3_output_directory,
    run_experiment_v3,
    summarize_v3_checkpoint,
)


class FakeSelectiveClassifier:
    abstain_value = -1
    penalty = "linear"
    independent_labels_ = (0,)
    dependent_labels_ = (1,)
    validation_objective_ = 0.75

    def predict_proba(self, x_data):
        probabilities = np.array(
            [[0.90, 0.40], [0.60, 0.10], [0.20, 0.80], [0.55, 0.45]],
            dtype=np.float64,
        )
        return probabilities[: len(x_data)]

    def predict_full_from_proba(self, probabilities):
        return (probabilities >= 0.5).astype(np.int32)

    def predict_from_proba(self, probabilities, cost=None):
        return np.where(
            probabilities <= cost,
            0,
            np.where(probabilities >= 1.0 - cost, 1, self.abstain_value),
        ).astype(np.int32)


class FakeCompleteClassifier:
    def __init__(self, fit_calls):
        self.fit_calls = fit_calls
        self.n_labels = None

    def fit(self, x_data, y_data):
        self.fit_calls.append(len(self.fit_calls) + 1)
        self.n_labels = y_data.shape[1]
        return self

    def predict(self, x_data):
        return np.zeros((len(x_data), self.n_labels), dtype=np.int32)


class FixedThreeFoldCV:
    def split(self, x_data, y_data):
        all_indices = np.arange(len(x_data))
        for start in (0, 2, 4):
            test_indices = all_indices[start : start + 2]
            train_indices = np.setdiff1d(all_indices, test_indices)
            yield train_indices, test_indices


def tiny_dataset_loader(dataset_name):
    if dataset_name != "tiny":
        raise ValueError(dataset_name)
    x_data = np.arange(18, dtype=np.float64).reshape(6, 3)
    y_data = np.array(
        [[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1]],
        dtype=np.int32,
    )
    return x_data, y_data, ["x0", "x1", "x2"], ["first", "second"]


def fixed_cv_factory(n_splits, random_state):
    if n_splits != 3:
        raise ValueError("The tiny test requires three folds.")
    return FixedThreeFoldCV()


class CacheV3Tests(unittest.TestCase):
    def test_config_hash_is_order_stable_and_setting_sensitive(self):
        first = {"model": "BR", "settings": {"seed": 42, "folds": 3}}
        reordered = {"settings": {"folds": 3, "seed": 42}, "model": "BR"}
        changed = {"model": "BR", "settings": {"seed": 43, "folds": 3}}
        self.assertEqual(compute_config_hash(first), compute_config_hash(reordered))
        self.assertNotEqual(compute_config_hash(first), compute_config_hash(changed))

    def test_schema_v2_at_v3_path_is_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = {"seed": 42, "n_splits": 2}
            path = fold_checkpoint_path(
                temporary_directory, "MODEL", "dataset", config
            )
            path.parent.mkdir(parents=True)
            original = '{"schema_version": 2, "datasets": {}}\n'
            path.write_text(original, encoding="utf-8")
            with self.assertRaises(IncompatibleV3CacheError):
                load_or_create_fold_checkpoint(
                    temporary_directory, "MODEL", "dataset", config
                )
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_strict_json_maps_nonfinite_values_to_null(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "strict.json"
            atomic_json_dump_v3(
                {"nan": float("nan"), "inf": float("inf"), "array": np.array([1])},
                path,
            )
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("NaN", text)
            self.assertNotIn("Infinity", text)
            self.assertEqual(
                json.loads(text), {"array": [1], "inf": None, "nan": None}
            )

    def test_legacy_output_marker_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "candidate"
            marker = output_dir / "tables" / "raw_results.json"
            marker.parent.mkdir(parents=True)
            marker.write_text("{}", encoding="utf-8")
            with self.assertRaises(V3OutputIsolationError):
                ensure_v3_output_directory(output_dir)


class EvaluationAndExportV3Tests(unittest.TestCase):
    def test_schema_v2_evaluator_remains_the_default(self):
        fit_calls = []
        classifier = FakeCompleteClassifier(fit_calls)
        classifier.fit(np.zeros((2, 1)), np.array([[1, 0], [0, 1]]))
        result = _evaluate_model(
            "BR_Logistic",
            classifier,
            np.zeros((2, 1)),
            np.array([[1, 0], [0, 1]]),
            [0.25],
        )
        self.assertEqual(set(result), {"full", "costs"})
        self.assertEqual(
            set(result["full"]),
            {"Macro-F1", "Micro-F1", "Hamming Loss", "Subset Accuracy", "Example-F1"},
        )

    def test_tiny_selective_evaluation_contains_every_scope(self):
        y_true = np.array(
            [[1, 0], [1, 0], [0, 1], [1, 0]], dtype=np.int32
        )
        result = _evaluate_model(
            "GSI_MLC_PA",
            FakeSelectiveClassifier(),
            np.zeros((4, 1)),
            y_true,
            [0.25, 0.50],
            metric_schema=3,
            label_names=("first", "second"),
            critical_labels=("first",),
        )
        self.assertEqual(result["Schema Version"], 3)
        self.assertEqual(tuple(result["Full"]), COMPLETE_METRIC_NAMES)
        self.assertEqual(set(result["Costs"]), {"0.25", "0.50"})
        cost_result = result["Costs"]["0.25"]
        self.assertEqual(tuple(cost_result["Selective"]), SELECTIVE_METRIC_NAMES)
        self.assertEqual(tuple(cost_result["Rejected"]), REJECTED_METRIC_NAMES)
        self.assertIn("Optimistic Macro-F1", cost_result["Optimistic"])
        self.assertIn("Full Error Count", cost_result["Diagnostics"])
        self.assertEqual(len(cost_result["Per Label"]), 2)
        self.assertEqual(set(cost_result["Groups"]), {"IL", "DL"})
        self.assertEqual(cost_result["Critical Labels"]["Status"], "Available")
        self.assertTrue(math.isfinite(cost_result["Selective"]["AURC"]))

    def test_summary_and_scope_exports_are_strict_and_nonempty(self):
        y_true = np.array(
            [[1, 0], [1, 0], [0, 1], [1, 0]], dtype=np.int32
        )
        metrics = _evaluate_model(
            "GSI_MLC_PA",
            FakeSelectiveClassifier(),
            np.zeros((4, 1)),
            y_true,
            [0.25],
            metric_schema=3,
            label_names=("first", "second"),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = {"test": "export", "n_splits": 1}
            path, checkpoint = load_or_create_fold_checkpoint(
                Path(temporary_directory) / "checkpoints",
                "GSI_MLC_PA",
                "tiny",
                config,
            )
            save_completed_fold(path, checkpoint, 1, metrics)
            mark_checkpoint_complete(path, checkpoint, 1)
            document = {
                "Schema Version": 3,
                "Metric Contract Version": metrics["Metric Contract Version"],
                "Config Hash": compute_config_hash(config),
                "Status": "complete",
                "Settings": config,
                "Results": {
                    "tiny": {"GSI_MLC_PA": summarize_v3_checkpoint(checkpoint)}
                },
            }
            artifacts = export_v3_artifacts(
                document, Path(temporary_directory) / "tables"
            )
            for artifact in artifacts.values():
                self.assertTrue(Path(artifact).exists(), msg=artifact)
            exported = json.loads(Path(artifacts["json"]).read_text(encoding="utf-8"))
            self.assertEqual(exported["Schema Version"], 3)
            self.assertEqual(exported["Artifacts"], artifacts)
            self.assertGreater(
                len(Path(artifacts["selective_csv"]).read_text(encoding="utf-8").splitlines()),
                1,
            )
            self.assertGreater(
                len(Path(artifacts["per_label_csv"]).read_text(encoding="utf-8").splitlines()),
                1,
            )
            self.assertGreater(
                len(Path(artifacts["group_csv"]).read_text(encoding="utf-8").splitlines()),
                1,
            )


class ResumeV3Tests(unittest.TestCase):
    @staticmethod
    def _run(output_dir, fit_calls, max_new_folds=None):
        def model_factory(model_name, **kwargs):
            return FakeCompleteClassifier(fit_calls)

        return run_experiment_v3(
            datasets=["tiny"],
            models=["BR_Logistic"],
            n_splits=3,
            random_state=42,
            output_dir=output_dir,
            abstention_costs=[0.25],
            report_cost=0.25,
            abstention_penalty="linear",
            mlc_pa_base="logistic",
            gsi_validation_size=0.2,
            dataset_loader=tiny_dataset_loader,
            cv_factory=fixed_cv_factory,
            model_factory=model_factory,
            evaluator=_evaluate_model,
            max_new_folds=max_new_folds,
        )

    def test_interrupted_run_resumes_without_duplicate_folds(self):
        with tempfile.TemporaryDirectory() as resumed_directory, tempfile.TemporaryDirectory() as uninterrupted_directory:
            resumed_calls = []
            partial = self._run(resumed_directory, resumed_calls, max_new_folds=1)
            self.assertEqual(partial["Status"], "partial")
            self.assertEqual(len(resumed_calls), 1)

            resumed = self._run(resumed_directory, resumed_calls)
            self.assertEqual(resumed["Status"], "complete")
            self.assertEqual(len(resumed_calls), 3)
            checkpoint_path = next(Path(resumed_directory).glob("checkpoints/*/*.json"))
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(completed_fold_indices(checkpoint), (1, 2, 3))

            self._run(resumed_directory, resumed_calls)
            self.assertEqual(len(resumed_calls), 3, "completed folds were rerun")

            uninterrupted_calls = []
            uninterrupted = self._run(uninterrupted_directory, uninterrupted_calls)
            self.assertEqual(len(uninterrupted_calls), 3)
            resumed_summary = resumed["Results"]["tiny"]["BR_Logistic"]
            uninterrupted_summary = uninterrupted["Results"]["tiny"]["BR_Logistic"]
            for key in (
                "Status",
                "Config Hash",
                "Completed Folds",
                "Full",
                "Per Label",
                "Groups",
                "Costs",
            ):
                self.assertEqual(resumed_summary[key], uninterrupted_summary[key], msg=key)

    def test_main_run_experiment_dispatches_to_isolated_v3_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fit_calls = []

            def model_factory(model_name, **kwargs):
                return FakeCompleteClassifier(fit_calls)

            with mock.patch.dict(
                benchmark_main.DATASET_CONFIG, {"tiny": {}}, clear=True
            ), mock.patch.object(
                benchmark_main, "load_dataset", side_effect=tiny_dataset_loader
            ), mock.patch.object(
                benchmark_main, "get_multilabel_cv", side_effect=fixed_cv_factory
            ), mock.patch.object(
                benchmark_main, "_create_model", side_effect=model_factory
            ):
                result = benchmark_main.run_experiment(
                    datasets=["tiny"],
                    models=["BR_Logistic"],
                    n_splits=3,
                    output_dir=temporary_directory,
                    abstention_costs=[0.25],
                    report_cost=0.25,
                    result_schema=3,
                )
            self.assertEqual(result["Schema Version"], 3)
            self.assertEqual(result["Status"], "complete")
            self.assertEqual(len(fit_calls), 3)
            self.assertTrue(Path(result["Artifacts"]["json"]).exists())

    def test_v3_only_options_are_rejected_by_schema_v2(self):
        with self.assertRaises(ValueError):
            benchmark_main.run_experiment(result_schema=2, max_new_folds=1)
        with self.assertRaises(ValueError):
            benchmark_main.run_experiment(result_schema=2, critical_labels=["x"])


if __name__ == "__main__":
    unittest.main()
