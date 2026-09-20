import unittest

import numpy as np

from src.models.local_parent_pcc import LocalParentPCC, _binary_states


class _ProbabilityStub:
    def __init__(self, function):
        self.function = function

    def predict_proba(self, X):
        positive = np.asarray(self.function(np.asarray(X)), dtype=np.float64)
        return np.column_stack((1.0 - positive, positive))


class LocalParentPCCTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(7)
        self.X = rng.normal(size=(48, 4))
        p0 = (self.X[:, 0] > 0).astype(np.int32)
        p1 = (self.X[:, 1] + 0.8 * p0 > 0).astype(np.int32)
        child = ((p0 + p1 + (self.X[:, 2] > 0)) >= 2).astype(np.int32)
        self.Y = np.column_stack((p0, p1, child))

    def test_parent_joint_is_exactly_enumerated_and_normalized(self):
        model = LocalParentPCC(
            base_learner="logistic",
            parent_indices=(0, 1),
            child_index=2,
            random_state=11,
        ).fit(self.X, self.Y)

        joint = model.predict_parent_joint(self.X[:9])
        self.assertEqual(joint.shape, (9, 4))
        np.testing.assert_allclose(joint.sum(axis=1), 1.0, atol=1e-12)
        self.assertTrue(np.isfinite(joint).all())
        self.assertTrue(((joint >= 0.0) & (joint <= 1.0)).all())

        child_probability = model.predict_child_proba(self.X[:9])
        self.assertEqual(child_probability.shape, (9,))
        self.assertTrue(((child_probability >= 0.0) & (child_probability <= 1.0)).all())

    def test_zero_parent_and_constant_targets_are_supported(self):
        constant = np.column_stack(
            (
                np.zeros(self.X.shape[0], dtype=np.int32),
                np.ones(self.X.shape[0], dtype=np.int32),
            )
        )
        model = LocalParentPCC(
            parent_indices=(), child_index=1, random_state=3
        ).fit(self.X, constant)
        np.testing.assert_array_equal(
            model.predict_parent_joint(self.X[:4]), np.ones((4, 1))
        )
        np.testing.assert_array_equal(
            model.predict_child_proba(self.X[:4]), np.ones(4)
        )

    def test_rejects_invalid_parent_sets(self):
        with self.assertRaises(ValueError):
            LocalParentPCC(
                parent_indices=(0, 0), child_index=2
            ).fit(self.X, self.Y)
        with self.assertRaises(ValueError):
            LocalParentPCC(
                parent_indices=(0, 1, 2, 0, 1, 2), child_index=2
            ).fit(self.X, self.Y)

    def test_one_parent_marginal_matches_two_state_formula(self):
        X = np.asarray([[0.25], [0.50], [0.75]])
        model = LocalParentPCC(parent_indices=(0,), child_index=1)
        model.parent_indices_ = (0,)
        model.child_index_ = 1
        model.n_features_in_ = 1
        model.parent_states_ = _binary_states(1)
        model.parent_models_ = [_ProbabilityStub(lambda values: values[:, 0])]
        model.child_model_ = _ProbabilityStub(
            lambda values: 0.2 + 0.6 * values[:, -1]
        )
        expected = (1.0 - X[:, 0]) * 0.2 + X[:, 0] * 0.8
        np.testing.assert_allclose(model.predict_child_proba(X), expected)

    def test_two_parent_joint_matches_manual_enumeration(self):
        X = np.zeros((3, 1), dtype=np.float64)
        model = LocalParentPCC(parent_indices=(0, 1), child_index=2)
        model.parent_indices_ = (0, 1)
        model.child_index_ = 2
        model.n_features_in_ = 1
        model.parent_states_ = _binary_states(2)
        model.parent_models_ = [
            _ProbabilityStub(lambda values: np.full(values.shape[0], 0.4)),
            _ProbabilityStub(lambda values: 0.3 + 0.4 * values[:, -1]),
        ]
        model.child_model_ = _ProbabilityStub(
            lambda values: np.full(values.shape[0], 0.5)
        )
        expected = np.asarray([0.42, 0.18, 0.12, 0.28])
        np.testing.assert_allclose(
            model.predict_parent_joint(X), np.broadcast_to(expected, (3, 4))
        )


if __name__ == "__main__":
    unittest.main()
