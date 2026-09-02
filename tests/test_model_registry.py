"""Q8 tests for the shared learner factory and matched model registry."""

import json
import unittest
from unittest import mock

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from main import (
    _create_model,
    _evaluate_model,
    _standardize_model_name,
)
from src.evaluation.cache_v3 import compute_config_hash
from src.evaluation.pipeline_v3 import _model_signature
from src.models.base_learners import (
    BackendUnavailableError,
    EXPERIMENT_CONFIG_PATH,
    base_learner_manifest,
    create_binary_estimator,
    create_multilabel_estimator,
    load_experiment_config,
)
from src.models.registry import (
    MATCHED_MODEL_IDS,
    canonical_registered_model_id,
    create_registered_model,
    get_model_spec,
    model_family,
)


class SharedBaseLearnerFactoryTests(unittest.TestCase):
    def test_experiment_config_and_shared_hyperparameters_are_explicit(self):
        config = load_experiment_config()
        self.assertTrue(EXPERIMENT_CONFIG_PATH.exists())
        self.assertEqual(config["schema_version"], 1)
        self.assertEqual(tuple(config["defaults"]["matched_model_ids"]), MATCHED_MODEL_IDS)
        logistic = create_binary_estimator("logistic", random_state=19)
        self.assertIsInstance(logistic, LogisticRegression)
        self.assertEqual(logistic.C, 1.0)
        self.assertEqual(logistic.solver, "liblinear")
        self.assertEqual(logistic.tol, 0.001)
        self.assertEqual(logistic.max_iter, 1000)
        self.assertEqual(logistic.random_state, 19)
        self.assertEqual(
            base_learner_manifest("mlp")["backend"],
            "pytorch",
        )

    def test_mlp_backend_never_silently_falls_back(self):
        explicit_sklearn = create_binary_estimator("mlp_sklearn", random_state=3)
        self.assertIsInstance(explicit_sklearn, MLPClassifier)
        with mock.patch(
            "src.models.base_learners._pytorch_classes",
            side_effect=BackendUnavailableError("backend unavailable fixture"),
        ):
            with self.assertRaises(BackendUnavailableError):
                create_binary_estimator("mlp", random_state=3)
            with self.assertRaises(BackendUnavailableError):
                create_multilabel_estimator("mlp", random_state=3)


class MatchedModelRegistryTests(unittest.TestCase):
    @staticmethod
    def _tiny_data():
        generator = np.random.default_rng(91)
        features = generator.normal(size=(24, 4)).astype(np.float32)
        truth = np.column_stack((
            features[:, 0] + 0.3 * features[:, 1] > 0.0,
            features[:, 2] - 0.2 * features[:, 0] > -0.1,
            features[:, 3] + features[:, 1] > 0.15,
        )).astype(np.int32)
        return features, truth

    def test_registry_contains_exactly_eight_matched_ids_and_aliases(self):
        self.assertEqual(
            MATCHED_MODEL_IDS,
            (
                "BR_Logistic",
                "BR_MLP",
                "CC_Logistic",
                "CC_MLP",
                "MLC_PA_Logistic",
                "MLC_PA_MLP",
                "GSI_MLC_PA_Logistic",
                "GSI_MLC_PA_MLP",
            ),
        )
        expected_families = {
            "BR": 2,
            "CC": 2,
            "MLC_PA": 2,
            "GSI_MLC_PA": 2,
        }
        actual_families = {
            family: sum(model_family(model_id) == family for model_id in MATCHED_MODEL_IDS)
            for family in expected_families
        }
        self.assertEqual(actual_families, expected_families)
        self.assertEqual(canonical_registered_model_id("br_lr"), "BR_Logistic")
        self.assertEqual(
            canonical_registered_model_id("gsimlcpa_mlp"),
            "GSI_MLC_PA_MLP",
        )
        self.assertEqual(_standardize_model_name("cc_nn"), "CC_MLP")

    def test_every_registered_model_has_json_safe_backend_manifest(self):
        for model_id in MATCHED_MODEL_IDS:
            with self.subTest(model_id=model_id):
                model = _create_model(model_id, random_state=13)
                manifest = model.experiment_manifest_
                spec = get_model_spec(model_id)
                self.assertEqual(manifest["model_id"], model_id)
                self.assertEqual(manifest["family"], spec.family)
                self.assertEqual(
                    manifest["base_learner"]["name"], spec.base_learner
                )
                self.assertIn("backend", manifest["base_learner"])
                self.assertTrue(manifest["base_learner"].get("binary_parameters"))
                json.dumps(manifest, sort_keys=True)
                signature = _model_signature(model)
                self.assertEqual(signature["experiment_manifest"], manifest)
                self.assertTrue(compute_config_hash(signature))

    def test_br_and_mlc_pa_share_exact_full_probabilities_for_each_base(self):
        features, truth = self._tiny_data()
        train_x, test_x = features[:18], features[18:]
        train_y = truth[:18]
        for base_name in ("Logistic", "MLP"):
            with self.subTest(base=base_name):
                br = create_registered_model(
                    f"BR_{base_name}", random_state=23
                ).fit(train_x, train_y)
                mlc = create_registered_model(
                    f"MLC_PA_{base_name}", random_state=23
                ).fit(train_x, train_y)
                br_probabilities = br.predict_proba(test_x)
                mlc_probabilities = mlc.predict_proba(test_x)
                np.testing.assert_allclose(
                    br_probabilities,
                    mlc_probabilities,
                    rtol=0.0,
                    atol=0.0,
                )
                np.testing.assert_array_equal(
                    br.predict(test_x),
                    mlc.predict_full(test_x),
                )

    def test_all_eight_ids_fit_predict_and_evaluate_on_tiny_data(self):
        features, truth = self._tiny_data()
        train_x, test_x = features[:18], features[18:]
        train_y, test_y = truth[:18], truth[18:]
        for model_id in MATCHED_MODEL_IDS:
            with self.subTest(model_id=model_id):
                model = _create_model(
                    model_id,
                    random_state=31,
                    gsi_validation_size=0.25,
                    gsi_partition_mode="all_il",
                    gsi_final_order="natural",
                )
                model.fit(train_x, train_y)
                family = model_family(model_id)
                if family in ("MLC_PA", "GSI_MLC_PA"):
                    prediction = model.predict_full(test_x)
                else:
                    prediction = model.predict(test_x)
                self.assertEqual(prediction.shape, test_y.shape)
                result = _evaluate_model(
                    model_id,
                    model,
                    test_x,
                    test_y,
                    [0.3],
                    metric_schema=3,
                    label_names=("a", "b", "c"),
                )
                self.assertIn("Macro-F1", result["Full"])
                self.assertEqual(
                    result["Model Metadata"]["Experiment Manifest"]["model_id"],
                    model_id,
                )
                if family in ("MLC_PA", "GSI_MLC_PA"):
                    self.assertEqual(set(result["Costs"]), {"0.30"})
                else:
                    self.assertEqual(result["Costs"], {})


if __name__ == "__main__":
    unittest.main()
