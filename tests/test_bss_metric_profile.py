import unittest

import numpy as np

from src.evaluation.abstention_metrics import compute_selective_instance_f1
from src.evaluation.metric_contract import (
    BSS_SPCC_FULL_METRIC_NAMES,
    BSS_SPCC_SELECTIVE_METRIC_NAMES,
)
from src.evaluation.metric_facade import compute_bss_spcc_metric_bundle


class BSSSPCCMetricProfileTests(unittest.TestCase):
    def test_exact_allowlists_and_identity(self):
        truth = np.asarray([[1, 0, 1], [0, 0, 0]], dtype=np.int32)
        full = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.int32)
        partial = np.asarray([[1, -1, 0], [0, 0, -1]], dtype=np.int32)
        bundle = compute_bss_spcc_metric_bundle(
            truth, full, y_partial=partial, cost=0.3
        )
        self.assertEqual(tuple(bundle["Full"]), BSS_SPCC_FULL_METRIC_NAMES)
        self.assertEqual(
            tuple(bundle["Selective"]), BSS_SPCC_SELECTIVE_METRIC_NAMES
        )
        self.assertAlmostEqual(
            bundle["Selective"]["AABS"],
            1.0 - bundle["Selective"]["Coverage"],
        )

    def test_selective_instance_f1_edge_conventions(self):
        truth = np.asarray(
            [[1, 0], [0, 0], [1, 1], [0, 1]], dtype=np.int32
        )
        partial = np.asarray(
            [[1, -1], [0, -1], [-1, -1], [1, 1]], dtype=np.int32
        )
        # Per-row scores: 1 (TP), 1 (decided both-empty), 0 (no decision), 2/3.
        self.assertAlmostEqual(
            compute_selective_instance_f1(truth, partial), (1 + 1 + 0 + 2 / 3) / 4
        )
        all_abstain = np.full_like(truth, -1)
        self.assertEqual(
            compute_selective_instance_f1(truth, all_abstain), 0.0
        )


if __name__ == "__main__":
    unittest.main()
