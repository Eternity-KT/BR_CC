"""Q7 tests for GSI partition providers, order controls and audit output."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from main import _cache_settings, _create_model, _evaluate_model
from src.evaluation.export_v3 import export_v3_artifacts
from src.evaluation.pipeline_v3 import summarize_v3_checkpoint
from src.models.gsi_mlc_pa import GSIMLCPartialAbstentionClassifier
from src.selection import (
    FINAL_ORDER_STRATEGIES,
    PARTITION_MODES,
    canonical_final_order_strategy,
    canonical_partition_mode,
    provide_partition,
)
from tests.test_selection_objectives import (
    _FixedProbabilityEstimator,
    _MeanFieldChainEstimator,
)


class PartitionProviderTests(unittest.TestCase):
    def test_public_modes_and_aliases_are_stable(self):
        self.assertEqual(
            PARTITION_MODES,
            (
                "learned",
                "learned_no_correlation_order",
                "all_il",
                "all_dl",
                "fixed",
                "random_matched",
            ),
        )
        self.assertEqual(FINAL_ORDER_STRATEGIES, ("correlation", "selection", "natural"))
        self.assertEqual(canonical_partition_mode("ALL-IL"), "all_il")
        self.assertEqual(canonical_partition_mode("random"), "random_matched")
        self.assertEqual(canonical_final_order_strategy("natural"), "natural")
        with self.assertRaises(ValueError):
            canonical_partition_mode("outer_test_selected")
        with self.assertRaises(ValueError):
            canonical_final_order_strategy("label_frequency")

    def test_all_il_all_dl_and_fixed_cover_empty_group_edges(self):
        all_il = provide_partition("all_il", 4)
        all_dl = provide_partition("all_dl", 4)
        fixed = provide_partition(
            "fixed", 4, fixed_independent_labels=[3, 1]
        )
        fixed_empty = provide_partition(
            "fixed", 4, fixed_independent_labels=[]
        )
        self.assertEqual(all_il.independent_labels, (0, 1, 2, 3))
        self.assertEqual(all_il.dependent_labels, ())
        self.assertEqual(all_dl.independent_labels, ())
        self.assertEqual(all_dl.dependent_labels, (0, 1, 2, 3))
        self.assertEqual(fixed.independent_labels, (1, 3))
        self.assertEqual(fixed.dependent_labels, (0, 2))
        self.assertEqual(fixed_empty.independent_labels, ())
        with self.assertRaises(ValueError):
            provide_partition("fixed", 4)
        with self.assertRaises(ValueError):
            provide_partition("fixed", 4, fixed_independent_labels=[4])
        with self.assertRaises(ValueError):
            provide_partition("fixed", 4, fixed_independent_labels=[1, 1])

    def test_random_matched_is_seeded_and_preserves_learned_size(self):
        first = provide_partition(
            "random_matched",
            12,
            learned_independent_labels=[0, 2, 4, 6],
            random_state=17,
        )
        repeated = provide_partition(
            "random_matched",
            12,
            learned_independent_labels=[0, 2, 4, 6],
            random_state=17,
        )
        different_seed = provide_partition(
            "random_matched",
            12,
            learned_independent_labels=[0, 2, 4, 6],
            random_state=18,
        )
        self.assertEqual(first, repeated)
        self.assertEqual(len(first.independent_labels), 4)
        self.assertEqual(first.reference_independent_count, 4)
        self.assertNotEqual(first.independent_labels, different_seed.independent_labels)
        self.assertEqual(
            sorted(first.independent_labels + first.dependent_labels),
            list(range(12)),
        )


class GSIPartitionModeTests(unittest.TestCase):
    @staticmethod
    def _data():
        features = np.linspace(0.02, 0.98, 120, dtype=np.float32).reshape(40, 3)
        truth = (
            features > np.array([0.45, 0.50, 0.55], dtype=np.float32)
        ).astype(np.int32)
        return features, truth

    def _fit(self, mode, *, fixed=None, final_order="natural", seed=7):
        features, truth = self._data()
        classifier = GSIMLCPartialAbstentionClassifier(
            validation_size=0.25,
            br_estimator=_FixedProbabilityEstimator(),
            cc_estimator=_MeanFieldChainEstimator(),
            random_state=seed,
            partition_mode=mode,
            fixed_independent_labels=fixed,
            final_order=final_order,
        ).fit(features, truth)
        return classifier, features, truth

    def test_every_mode_runs_through_the_same_fit_and_evaluation_api(self):
        expected = {
            "all_il": ([0, 1, 2], []),
            "all_dl": ([], [0, 1, 2]),
            "fixed": ([0, 2], [1]),
        }
        for mode in PARTITION_MODES:
            with self.subTest(mode=mode):
                fixed = [0, 2] if mode == "fixed" else None
                classifier, features, truth = self._fit(mode, fixed=fixed)
                probabilities = classifier.predict_proba(features[:4])
                partial = classifier.predict(features[:4])
                result = _evaluate_model(
                    "GSI_MLC_PA",
                    classifier,
                    features[:4],
                    truth[:4],
                    [0.3],
                    metric_schema=3,
                    label_names=("a", "b", "c"),
                )
                self.assertEqual(probabilities.shape, (4, 3))
                self.assertTrue(np.all(np.isin(partial, (-1, 0, 1))))
                self.assertEqual(set(result["Groups"]), {"IL", "DL"})
                self.assertEqual(result["Model Metadata"]["Partition Mode"], mode)
                self.assertGreaterEqual(classifier.selection_time_seconds_, 0.0)
                self.assertIn("Probability Inference Seconds", result["Model Metadata"])
                self.assertEqual(
                    sorted(classifier.independent_labels_ + classifier.dependent_labels_),
                    [0, 1, 2],
                )
                if mode in expected:
                    independent, dependent = expected[mode]
                    self.assertEqual(classifier.independent_labels_, independent)
                    self.assertEqual(classifier.dependent_labels_, dependent)

    def test_random_matched_uses_learned_count_and_is_reproducible(self):
        first, _, _ = self._fit("random_matched", seed=11)
        repeated, _, _ = self._fit("random_matched", seed=11)
        self.assertEqual(first.independent_labels_, repeated.independent_labels_)
        self.assertEqual(
            len(first.independent_labels_),
            len(first.reference_independent_labels_),
        )
        self.assertEqual(
            first.partition_audit_["reference_independent_count"],
            len(first.reference_independent_labels_),
        )
        self.assertTrue(first.reference_selection_history_)
        self.assertGreater(first.evaluated_configurations_, 1)

    def test_no_correlation_mode_and_final_order_controls_do_not_change_partition(self):
        with mock.patch.object(
            GSIMLCPartialAbstentionClassifier,
            "_correlation_order",
            return_value=[2, 1, 0],
        ):
            no_reorder, _, _ = self._fit(
                "learned_no_correlation_order", final_order="correlation"
            )
        learned, _, _ = self._fit("learned", final_order="selection")
        self.assertEqual(no_reorder.independent_labels_, learned.independent_labels_)
        self.assertEqual(no_reorder.dependent_labels_, learned.dependent_labels_)
        self.assertEqual(no_reorder.correlation_order_, [2, 1, 0])
        self.assertEqual(no_reorder.order_, no_reorder.selection_order_)
        self.assertEqual(no_reorder.final_order_strategy_, "selection")

        for mode, fixed in (("all_il", None), ("all_dl", None), ("fixed", [1])):
            with self.subTest(mode=mode):
                classifier, _, _ = self._fit(
                    mode, fixed=fixed, final_order="natural"
                )
                self.assertEqual(classifier.order_, [0, 1, 2])
                self.assertEqual(classifier.selection_order_, [0, 1, 2])

    def test_partition_audit_summary_stability_and_csv(self):
        classifier, features, truth = self._fit("fixed", fixed=[0, 2])
        fold_metrics = []
        for test_slice in (slice(0, 4), slice(4, 8)):
            fold_metrics.append(
                _evaluate_model(
                    "GSI_MLC_PA",
                    classifier,
                    features[test_slice],
                    truth[test_slice],
                    [0.3],
                    metric_schema=3,
                    label_names=("a", "b", "c"),
                )
            )
        checkpoint = {
            "status": "complete",
            "config_hash": "q7-fixture",
            "config": {"n_splits": 2},
            "folds": {
                str(index): {"metrics": metrics, "metadata": {}}
                for index, metrics in enumerate(fold_metrics, 1)
            },
        }
        summary = summarize_v3_checkpoint(checkpoint)
        audit = summary["Partition Audit"]
        self.assertEqual(audit["Pairwise IL Jaccard Stability"], 1.0)
        self.assertEqual(len(audit["Raw Records"]), 2)
        self.assertEqual(
            audit["Summary"]["Mean"]["Independent Label Count"], 2.0
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            document = {
                "Schema Version": 3,
                "Results": {"tiny": {"GSI_MLC_PA": summary}},
            }
            artifacts = export_v3_artifacts(document, Path(temporary_directory))
            partition_csv = Path(artifacts["partition_csv"])
            self.assertTrue(partition_csv.exists())
            self.assertEqual(len(partition_csv.read_text(encoding="utf-8").splitlines()), 3)

    def test_factory_and_cache_hash_inputs_include_ablation_controls(self):
        classifier = _create_model(
            "GSI_MLC_PA",
            gsi_partition_mode="fixed",
            gsi_fixed_independent_labels=[0, 2],
            gsi_final_order="natural",
        )
        self.assertEqual(classifier.partition_mode, "fixed")
        self.assertEqual(classifier.fixed_independent_labels, [0, 2])
        self.assertEqual(classifier.final_order, "natural")
        settings = _cache_settings(
            "GSI_MLC_PA",
            2,
            42,
            [0.3],
            0.3,
            "linear",
            "mlp",
            0.2,
            gsi_partition_mode="fixed",
            gsi_fixed_independent_labels=[0, 2],
            gsi_final_order="natural",
        )
        self.assertEqual(settings["partition_mode"], "fixed")
        self.assertEqual(settings["fixed_independent_labels"], [0, 2])
        self.assertEqual(settings["final_order_strategy"], "natural")


if __name__ == "__main__":
    unittest.main()
