"""Unit tests for automated Platt Scaling calibration of PyTorch MLP."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unittest
import numpy as np

from src.models.pytorch_mlp import MultiLabelMLPClassifier, FastPyTorchBinaryMLP
from src.models.base_learners import create_multilabel_estimator, create_binary_estimator
from src.models.registry import create_registered_model


class MLPCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.n_samples = 60
        cls.n_features = 8
        cls.n_labels = 4
        cls.X = np.random.randn(cls.n_samples, cls.n_features).astype(np.float32)
        # 4 labels:
        # col 0: balanced (~50% 1s)
        # col 1: imbalanced (~10% 1s)
        # col 2: rare (1 single 1)
        # col 3: constant (all 0s)
        cls.Y = np.zeros((cls.n_samples, cls.n_labels), dtype=np.int32)
        cls.Y[:, 0] = (cls.X[:, 0] > 0).astype(np.int32)
        cls.Y[np.random.choice(cls.n_samples, 6, replace=False), 1] = 1
        cls.Y[0, 2] = 1  # only 1 positive sample
        # col 3 is all 0s

    def test_multilabel_mlp_calibrated_probabilities(self):
        mlp = MultiLabelMLPClassifier(
            hidden_layer_sizes=(32,),
            epochs=10,
            calibration="sigmoid",
            random_state=42,
        )
        mlp.fit(self.X, self.Y)

        probs = mlp.predict_proba(self.X)
        self.assertEqual(probs.shape, (self.n_samples, self.n_labels))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

        # Col 3 is constant 0
        np.testing.assert_allclose(probs[:, 3], 0.0)

        # Col 2 is rare (1 sample), should use smoothed Laplace prior: (1+1)/(60+2) = 2/62 ~ 0.03225
        np.testing.assert_allclose(probs[:, 2], 2.0 / 62.0, atol=1e-3)

        # Non-constant labels have calibrators
        self.assertIn(0, mlp.calibrators_)
        self.assertIn(1, mlp.calibrators_)
        self.assertIn(2, mlp.calibrators_)
        self.assertNotIn(3, mlp.calibrators_)

    def test_fast_binary_mlp_calibrated_probabilities(self):
        y_binary = self.Y[:, 0]
        bmlp = FastPyTorchBinaryMLP(
            hidden_layer_sizes=(32,),
            epochs=10,
            calibration="sigmoid",
            random_state=42,
        )
        bmlp.fit(self.X, y_binary)

        probs = bmlp.predict_proba(self.X)
        self.assertEqual(probs.shape, (self.n_samples, 2))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        self.assertIsNotNone(bmlp.calibrator_)

    def test_fast_binary_mlp_constant_label_fallback(self):
        y_const = np.zeros(self.n_samples, dtype=np.int32)
        bmlp = FastPyTorchBinaryMLP(epochs=5, calibration="sigmoid", random_state=42)
        bmlp.fit(self.X, y_const)
        probs = bmlp.predict_proba(self.X)
        np.testing.assert_allclose(probs[:, 1], 0.0)
        np.testing.assert_allclose(probs[:, 0], 1.0)

    def test_registry_integration(self):
        mlc_pa = create_registered_model("MLC_PA_MLP", random_state=42, abstention_cost=0.3)
        self.assertEqual(mlc_pa.base_estimator, "mlp")
        mlc_pa.fit(self.X, self.Y)
        self.assertEqual(mlc_pa.base_estimator_.model_.calibration, "sigmoid")

        gsi_pa = create_registered_model("GSI_MLC_PA_MLP", random_state=42, abstention_cost=0.3)
        self.assertEqual(gsi_pa.base_learner, "mlp")


if __name__ == "__main__":
    unittest.main()
