"""Unit tests for Bipartite GSI dependency structure."""

import unittest
import numpy as np

from src.models.gsi_mlc_pa import GSIMLCPartialAbstentionClassifier
from src.models.classifier_chain import ClassifierChainClassifier


class BipartiteGSITests(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Synthetic multilabel dataset with known dependency structure
        self.n_samples = 60
        self.n_features = 4
        self.X = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        # y0, y1 are independent
        y0 = (self.X[:, 0] > 0).astype(np.int32)
        y1 = (self.X[:, 1] > 0).astype(np.int32)
        # y2 is dependent on y0
        y2 = (y0 ^ (self.X[:, 2] > 0)).astype(np.int32)
        # y3 is dependent on y1
        y3 = (y1 ^ (self.X[:, 3] > 0)).astype(np.int32)
        self.Y = np.column_stack([y0, y1, y2, y3])

    def test_bipartite_gsi_fit_and_predict(self):
        clf = GSIMLCPartialAbstentionClassifier(
            dependency_structure="bipartite",
            base_learner="logistic",
            validation_size=0.25,
            cost=0.3,
            random_state=42,
        )
        clf.fit(self.X, self.Y)

        self.assertTrue(hasattr(clf, "independent_labels_"))
        self.assertTrue(hasattr(clf, "dependent_labels_"))
        self.assertEqual(len(clf.independent_labels_) + len(clf.dependent_labels_), 4)
        self.assertEqual(
            sorted(clf.independent_labels_ + clf.dependent_labels_), [0, 1, 2, 3]
        )

        # Check that CC model has predecessor_map
        self.assertIsNotNone(clf.cc_model_.predecessor_map)
        for l in clf.independent_labels_:
            self.assertEqual(clf.cc_model_.predecessor_map[l], [])
        for d in clf.dependent_labels_:
            self.assertEqual(
                clf.cc_model_.predecessor_map[d], sorted(clf.independent_labels_)
            )

        # In order_, all IL must precede DL
        il_positions = [clf.order_.index(l) for l in clf.independent_labels_]
        dl_positions = [clf.order_.index(d) for d in clf.dependent_labels_]
        if il_positions and dl_positions:
            self.assertLess(max(il_positions), min(dl_positions))

        # Predict proba
        probs = clf.predict_proba(self.X)
        self.assertEqual(probs.shape, (self.n_samples, 4))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

        # Full predictions
        preds_full = clf.predict_full(self.X)
        self.assertEqual(preds_full.shape, (self.n_samples, 4))
        self.assertTrue(np.all(np.isin(preds_full, [0, 1])))

        # Partial abstention predictions
        preds_pa = clf.predict(self.X)
        self.assertEqual(preds_pa.shape, (self.n_samples, 4))
        self.assertTrue(np.all(np.isin(preds_pa, [-1, 0, 1])))

    def test_bipartite_parent_map_points_to_il(self):
        clf = GSIMLCPartialAbstentionClassifier(
            dependency_structure="bipartite",
            base_learner="logistic",
            validation_size=0.25,
            random_state=42,
        )
        clf.fit(self.X, self.Y)

        # Every dependent label must point to an independent label (or None if IL is empty)
        for d in clf.dependent_labels_:
            parent = clf.dependent_parent_map_[d]
            if clf.independent_labels_:
                self.assertIn(parent, clf.independent_labels_)
            else:
                self.assertIsNone(parent)


if __name__ == "__main__":
    unittest.main()
