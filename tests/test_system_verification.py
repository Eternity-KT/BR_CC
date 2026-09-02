"""Phase-Q11 tests for frozen config and machine-readable result audits."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from main import _evaluate_model
from scripts.audit_v3_results import audit_results
from scripts.run_frozen_experiment import verify_frozen_config
from src.evaluation.pipeline_v3 import run_experiment_v3
from src.models.registry import MATCHED_MODEL_IDS


class TwoFoldCV:
    def split(self, x_data, y_data):
        midpoint = len(x_data) // 2
        indices = np.arange(len(x_data))
        yield indices[midpoint:], indices[:midpoint]
        yield indices[:midpoint], indices[midpoint:]


class TinyProbabilityModel:
    abstain_value = -1
    penalty = "linear"
    decision_policy = "hamming"
    cost = 0.3

    def fit(self, x_data, y_data):
        self.n_labels_ = y_data.shape[1]
        return self

    def predict_proba(self, x_data):
        base = 1.0 / (1.0 + np.exp(-np.asarray(x_data)[:, 0]))
        columns = [base if index % 2 == 0 else 1.0 - base for index in range(self.n_labels_)]
        return np.column_stack(columns)

    @staticmethod
    def predict_full_from_proba(probabilities):
        return (probabilities >= 0.5).astype(np.int32)

    def predict(self, x_data):
        return self.predict_full_from_proba(self.predict_proba(x_data))


def tiny_loader(dataset_name):
    if dataset_name != "tiny":
        raise ValueError(dataset_name)
    features = np.array(
        [[-2.0, 0.0], [-1.0, 1.0], [0.5, 0.0], [1.5, 1.0],
         [-1.5, 0.0], [-0.2, 1.0], [0.8, 0.0], [2.0, 1.0]],
        dtype=np.float64,
    )
    truth = np.column_stack((
        features[:, 0] >= 0.0,
        features[:, 0] < 0.0,
    )).astype(np.int32)
    return features, truth, ["x0", "x1"], ["positive", "negative"]


class FrozenConfigTests(unittest.TestCase):
    def test_frozen_primary_config_checksum_and_registry_are_locked(self):
        payload, digest = verify_frozen_config("configs/full_run.json")
        self.assertEqual(len(digest), 64)
        self.assertEqual(tuple(payload["settings"]["models"]), MATCHED_MODEL_IDS)
        self.assertEqual(payload["settings"]["result_schema"], 3)
        self.assertIn(0.5, payload["settings"]["abstention_costs"])
        self.assertEqual(
            payload["settings"]["operating_point_rule"],
            "min_generalized_loss",
        )


class ResultAuditTests(unittest.TestCase):
    def test_tiny_complete_and_selective_run_passes_machine_audit(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "schema_v3"
            document = run_experiment_v3(
                datasets=["tiny"],
                models=["BR_Logistic", "MLC_PA_Logistic"],
                n_splits=2,
                random_state=42,
                output_dir=output_dir,
                abstention_costs=[0.25, 0.5],
                report_cost=0.25,
                abstention_penalty="linear",
                mlc_pa_base="logistic",
                gsi_validation_size=0.2,
                dataset_loader=tiny_loader,
                cv_factory=lambda **kwargs: TwoFoldCV(),
                model_factory=lambda *args, **kwargs: TinyProbabilityModel(),
                evaluator=_evaluate_model,
            )
            summary = audit_results(
                output_dir,
                config_hash=document["Config Hash"],
                expected_datasets=["tiny"],
                expected_models=["BR_Logistic", "MLC_PA_Logistic"],
                expected_folds=2,
                expected_costs=[0.25, 0.5],
            )
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(summary["model_dataset_pairs"], 2)
            self.assertEqual(summary["selective_pairs"], 1)
            self.assertEqual(summary["csv_artifacts"], 9)
            self.assertEqual(summary["figure_artifacts"], 7)


if __name__ == "__main__":
    unittest.main()
