import json
import unittest

import numpy as np

from src.decision.hamming import HammingBOPPolicy
from src.models.bss_ug_spcc_pa import BSSUGSPCCPartialAbstentionClassifier


class BSSUGSPCCPATests(unittest.TestCase):
    @staticmethod
    def _fixture():
        rng = np.random.RandomState(19)
        X = rng.normal(size=(36, 5))
        y0 = (X[:, 0] + 0.25 * X[:, 1] > 0).astype(np.int32)
        y1 = (y0 + (X[:, 2] > 0).astype(np.int32) >= 1).astype(np.int32)
        y2 = ((y0 + y1 + (X[:, 3] > 0)) >= 2).astype(np.int32)
        return X, np.column_stack((y0, y1, y2))

    def _model(self):
        return BSSUGSPCCPartialAbstentionClassifier(
            inner_oof_splits=3,
            alpha_grid=(0.1, 0.3),
            q_max=2,
            calibration_splits=2,
            random_state=23,
        )

    def test_fit_public_api_structure_and_determinism(self):
        X, Y = self._fixture()
        first = self._model().fit(X, Y)
        second = self._model().fit(X, Y)

        raw = first.predict_raw_proba(X[:7])
        calibrated = first.predict_proba(X[:7])
        self.assertEqual(raw.shape, (7, 3))
        self.assertEqual(calibrated.shape, (7, 3))
        self.assertTrue(np.isfinite(calibrated).all())
        self.assertTrue(((calibrated >= 0.0) & (calibrated <= 1.0)).all())
        np.testing.assert_allclose(calibrated, second.predict_proba(X[:7]), atol=1e-12)
        self.assertEqual(first.global_order_, second.global_order_)
        self.assertEqual(first.parent_map_, second.parent_map_)

        positions = {label: index for index, label in enumerate(first.global_order_)}
        for child, parents in first.parent_map_.items():
            self.assertLessEqual(len(parents), 2)
            for parent in parents:
                self.assertLess(positions[parent], positions[child])
        json.dumps(first.get_structure_audit(), allow_nan=False)

    def test_fold_local_bss_reference_and_alpha_score(self):
        model = self._model()
        Y = np.asarray([[0], [0], [1], [1]], dtype=np.int32)
        p = np.asarray([[0.1], [0.2], [0.8], [0.9]])
        fold_local_reference = np.asarray([[1.0], [1.0], [0.0], [0.0]])
        bs_br, bs_ref, bss, degenerate = model._compute_bss(
            Y, p, fold_local_reference
        )
        self.assertAlmostEqual(bs_br[0], 0.025)
        self.assertAlmostEqual(bs_ref[0], 1.0)
        self.assertAlmostEqual(bss[0], 0.975)
        self.assertFalse(degenerate[0])

    def test_strict_hamming_boundaries_abstain(self):
        policy = HammingBOPPolicy(
            cost=0.3,
            penalty="linear",
            linear_boundary="strict_symmetric_thresholds",
        )
        probabilities = np.asarray([[0.3, 0.7, 0.299, 0.701]])
        np.testing.assert_array_equal(
            policy.predict(probabilities), np.asarray([[-1, -1, 0, 1]])
        )


if __name__ == "__main__":
    unittest.main()
