import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.datasets import make_multilabel_classification

from src.evaluation.layer_metrics import compute_selective_subset_accuracy
from src.evaluation.result_store_v6 import V63ResultStore
from src.models.gsi_v6_3 import GSIMLCPAv6_3Classifier
from src.models.probabilistic_classifier_chain import ProbabilisticClassifierChain
from src.selection.single_stage_peeling import SingleStagePeelingSelector
from src.selection.cv_peeling import CVPeelingResult


class V63CoreTests(unittest.TestCase):
    def setUp(self):
        X, Y = make_multilabel_classification(
            n_samples=72,
            n_features=10,
            n_classes=4,
            random_state=17,
        )
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.int32)

    def test_single_stage_keeps_singleton_dl(self):
        selector = SingleStagePeelingSelector(
            threshold=2.0,
            n_folds=2,
            random_state=3,
        )
        result = selector.fit_partition(self.X, self.Y[:, [0]])
        self.assertEqual(result.num_stages_executed, 1)
        self.assertEqual(result.all_independent_labels, ())
        self.assertEqual(result.dependent_residual_labels, (0,))

    def test_pcc_uses_exact_sample_count_and_is_reproducible(self):
        chain = ProbabilisticClassifierChain(
            base_estimator="logistic",
            num_samples=1000,
            sampling_batch_size=31,
            random_state=19,
        ).fit(self.X[:55], self.Y[:55])
        first = chain.predict_proba(self.X[55:])
        second = chain.predict_proba(self.X[55:])
        self.assertEqual(first.shape, (17, 4))
        np.testing.assert_array_equal(first, second)
        # Empirical marginals from exactly 1,000 binary samples lie on this grid.
        np.testing.assert_allclose(first * 1000, np.round(first * 1000), atol=1e-4)

    def test_cc_and_pcc_share_correlation_order(self):
        partition = SingleStagePeelingSelector(
            threshold=2.0,
            n_folds=2,
            random_state=5,
        ).fit_partition(self.X[:55], self.Y[:55])
        cc = GSIMLCPAv6_3Classifier(
            base_learner="logistic",
            dl_inference="cc",
            feature_space="x",
            cv_folds=2,
            random_state=5,
        ).fit(self.X[:55], self.Y[:55], partition_result=partition)
        pcc = GSIMLCPAv6_3Classifier(
            base_learner="logistic",
            dl_inference="pcc",
            feature_space="x",
            cv_folds=2,
            pcc_num_samples=50,
            random_state=5,
        ).fit(
            self.X[:55],
            self.Y[:55],
            partition_result=partition,
            shared_il_models=cc.il_models_,
        )
        self.assertEqual(cc.dependent_labels_, pcc.dependent_labels_)
        self.assertEqual(cc.dl_model_.order_, pcc.dl_model_.order_)
        self.assertEqual(cc.dl_model_.order_, list(range(len(cc.dependent_labels_))))

    def _fixed_partition(self, il, dl):
        return CVPeelingResult(
            independent_layers=(tuple(il),) if il else (),
            all_independent_labels=tuple(il),
            dependent_residual_labels=tuple(dl),
            final_execution_order=tuple(il) + tuple(dl),
            stage_diagnostics=({"stage": 1, "stage_scores": {}},),
            br_independent_metrics={},
            stopping_reason="test_fixture",
            num_stages_executed=1,
            selection_time_seconds=0.0,
            oof_probabilities={
                label: self.Y[:55, label].astype(np.float64) for label in il
            },
        )

    def test_x_il_adds_only_predicted_il_feature_columns(self):
        partition = self._fixed_partition([0], [1, 2, 3])
        x_model = GSIMLCPAv6_3Classifier(
            base_learner="logistic", dl_inference="cc", feature_space="x", cv_folds=2
        ).fit(self.X[:55], self.Y[:55], partition_result=partition)
        xil_model = GSIMLCPAv6_3Classifier(
            base_learner="logistic", dl_inference="cc", feature_space="x_il", cv_folds=2
        ).fit(self.X[:55], self.Y[:55], partition_result=partition)
        self.assertEqual(x_model.dl_model_.classifiers_[0].n_features_in_, 10)
        self.assertEqual(xil_model.dl_model_.classifiers_[0].n_features_in_, 11)
        self.assertEqual(xil_model.predict_proba(self.X[55:]).shape, (17, 4))

    def test_empty_dl_is_supported(self):
        partition = self._fixed_partition([0, 1, 2, 3], [])
        model = GSIMLCPAv6_3Classifier(
            base_learner="logistic", dl_inference="pcc", feature_space="x_il", cv_folds=2
        ).fit(self.X[:55], self.Y[:55], partition_result=partition)
        probabilities = model.predict_proba(self.X[55:])
        self.assertEqual(probabilities.shape, (17, 4))
        self.assertTrue(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))

    def test_selective_subset_excludes_all_abstain_samples(self):
        truth = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int32)
        partial = np.array([[-1, -1], [0, -1], [1, 0]], dtype=np.int32)
        self.assertEqual(compute_selective_subset_accuracy(truth, partial), 0.5)

    def test_result_store_resume_requires_matching_prediction_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            store = V63ResultStore(
                Path(directory), ["logistic"], ["GSI_v6_3_CC_X"]
            )
            base = {
                "version": "v6_3",
                "source_version": "v6_3",
                "provenance_status": "computed",
                "base_learner": "logistic",
                "model": "GSI_v6_3_CC_X",
                "dataset": "toy",
                "mean_ir": 1.0,
                "expected_fold_count": 1,
                "config_hash": "abc",
            }
            fold = {
                "fold": 1,
                "provenance_status": "computed",
                "split_hash": "split",
                "complete_metrics": {},
                "selective_metrics": [],
                "calibration_metrics": {},
                "layer_metrics": [],
                "timing": {},
            }
            arrays = {
                "test_indices": np.array([0]),
                "y_true": np.array([[1]], dtype=np.int8),
                "probabilities": np.array([[0.8]], dtype=np.float32),
                "y_full": np.array([[1]], dtype=np.int8),
                "y_partial_c_0.30": np.array([[1]], dtype=np.int8),
                "fold_ids": np.array([1], dtype=np.int8),
            }
            store.save_computed_fold(
                learner="logistic",
                model="GSI_v6_3_CC_X",
                dataset="toy",
                checkpoint_base=base,
                fold_record=fold,
                fold_arrays=arrays,
            )
            self.assertTrue(
                store.is_fold_complete("logistic", "GSI_v6_3_CC_X", "toy", 1)
            )
            self.assertFalse(
                store.is_fold_complete(
                    "logistic",
                    "GSI_v6_3_CC_X",
                    "toy",
                    1,
                    expected_config_hash="different",
                )
            )
            replacement = dict(base)
            replacement["config_hash"] = "new"
            second_fold = dict(fold)
            second_fold["fold"] = 2
            second_arrays = dict(arrays)
            second_arrays["fold_ids"] = np.array([2], dtype=np.int8)
            store.save_computed_fold(
                learner="logistic",
                model="GSI_v6_3_CC_X",
                dataset="toy",
                checkpoint_base=replacement,
                fold_record=second_fold,
                fold_arrays=second_arrays,
            )
            checkpoint = store.load_checkpoint(
                "logistic", "GSI_v6_3_CC_X", "toy"
            )
            self.assertEqual([row["fold"] for row in checkpoint["folds"]], [2])


if __name__ == "__main__":
    unittest.main()
