"""Exact local Parent-PCC probability model.

The model represents a small parent-label joint distribution with a classifier
chain and marginalizes that distribution exactly when predicting one child
label.  It is intentionally bounded to at most five parents by the caller so
that inference remains deterministic and inexpensive (at most 32 states).
"""

from itertools import product

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .base_learners import create_binary_estimator


class ConstantBinaryProbabilityClassifier:
    """Binary probability estimator used when a training target is constant."""

    def __init__(self, value):
        self.value = int(value)

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        probabilities = np.zeros((np.asarray(X).shape[0], 2), dtype=np.float64)
        probabilities[:, self.value] = 1.0
        return probabilities

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], self.value, dtype=np.int32)


def _binary_states(parent_count):
    """Return lexicographically ordered binary states for zero to five parents."""

    count = int(parent_count)
    if count < 0 or count > 5:
        raise ValueError("parent_count must be between zero and five.")
    if count == 0:
        return np.zeros((1, 0), dtype=np.int32)
    return np.asarray(list(product((0, 1), repeat=count)), dtype=np.int32)


def positive_probability(estimator, X):
    """Return a clipped P(y=1|X) vector for a fitted binary estimator."""

    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(X), dtype=np.float64)
        if probabilities.ndim == 1:
            positive = probabilities
        elif probabilities.shape[1] == 1:
            classes = np.asarray(getattr(estimator, "classes_", [0]))
            positive = (
                probabilities[:, 0]
                if classes.size and int(classes[0]) == 1
                else np.zeros(probabilities.shape[0], dtype=np.float64)
            )
        else:
            classes = np.asarray(getattr(estimator, "classes_", [0, 1]))
            matches = np.flatnonzero(classes == 1)
            positive = probabilities[:, int(matches[0])] if matches.size else probabilities[:, -1]
    elif hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(X), dtype=np.float64)
        positive = 1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0)))
    else:
        positive = np.asarray(estimator.predict(X), dtype=np.float64)
    return np.clip(positive.reshape(-1), 0.0, 1.0)


def _fit_binary_with_constant_fallback(X, y, base_learner, random_state):
    """Fit one configured binary learner, preserving single-class folds."""

    target = np.asarray(y, dtype=np.int32).reshape(-1)
    unique = np.unique(target)
    if unique.size <= 1:
        value = int(unique[0]) if unique.size else 0
        return ConstantBinaryProbabilityClassifier(value)
    estimator = create_binary_estimator(base_learner, random_state=random_state)
    estimator.fit(np.asarray(X), target)
    return estimator


# Public-within-package alias retained for the model core's readable imports.
fit_binary_with_constant_fallback = _fit_binary_with_constant_fallback


class LocalParentPCC(BaseEstimator, ClassifierMixin):
    """Classifier-chain joint over local parents plus one child label."""

    def __init__(
        self,
        base_learner="logistic",
        parent_indices=(),
        child_index=None,
        random_state=42,
        epsilon=1e-12,
        normalization_tolerance=1e-10,
    ):
        self.base_learner = base_learner
        self.parent_indices = parent_indices
        self.child_index = child_index
        self.random_state = random_state
        self.epsilon = epsilon
        self.normalization_tolerance = normalization_tolerance

    @staticmethod
    def _validate_xy(X, Y):
        features = np.asarray(X, dtype=np.float64)
        labels = np.asarray(Y, dtype=np.int32)
        if features.ndim != 2 or labels.ndim != 2:
            raise ValueError("X and Y must both be two-dimensional arrays.")
        if features.shape[0] != labels.shape[0] or features.shape[0] == 0:
            raise ValueError("X and Y must contain the same non-zero sample count.")
        if labels.shape[1] == 0 or not np.isin(labels, (0, 1)).all():
            raise ValueError("Y must be a non-empty binary label matrix.")
        if not np.isfinite(features).all():
            raise ValueError("X must contain only finite values.")
        return features, labels

    def fit(self, X, Y, parent_indices=None, child_index=None):
        features, labels = self._validate_xy(X, Y)
        parents = tuple(
            int(index)
            for index in (
                self.parent_indices if parent_indices is None else parent_indices
            )
        )
        child = self.child_index if child_index is None else child_index
        if child is None:
            raise ValueError("child_index must be specified.")
        child = int(child)
        if len(parents) > 5:
            raise ValueError("LocalParentPCC supports at most five parents.")
        if len(set(parents)) != len(parents) or child in parents:
            raise ValueError("Parent indices must be unique and exclude the child.")
        if any(index < 0 or index >= labels.shape[1] for index in parents + (child,)):
            raise ValueError("Parent/child label index is out of range.")

        self.parent_indices_ = parents
        self.child_index_ = child
        self.n_features_in_ = features.shape[1]
        self.parent_models_ = []
        for offset, label_index in enumerate(parents):
            augmented = (
                features
                if offset == 0
                else np.hstack((features, labels[:, parents[:offset]]))
            )
            model = fit_binary_with_constant_fallback(
                augmented,
                labels[:, label_index],
                self.base_learner,
                self.random_state + offset,
            )
            self.parent_models_.append(model)

        child_features = (
            features if not parents else np.hstack((features, labels[:, parents]))
        )
        self.child_model_ = fit_binary_with_constant_fallback(
            child_features,
            labels[:, child],
            self.base_learner,
            self.random_state + len(parents),
        )
        self.parent_states_ = _binary_states(len(parents))
        return self

    def _validate_features(self, X):
        if not hasattr(self, "child_model_"):
            raise ValueError("LocalParentPCC must be fitted before prediction.")
        features = np.asarray(X, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.n_features_in_:
            raise ValueError("X has an incompatible feature shape.")
        if not np.isfinite(features).all():
            raise ValueError("X must contain only finite values.")
        return features

    def predict_parent_joint(self, X):
        """Return exact P(parent state | X) in lexicographic state order."""

        features = self._validate_features(X)
        n_samples = features.shape[0]
        if not self.parent_indices_:
            return np.ones((n_samples, 1), dtype=np.float64)

        joint = np.ones((n_samples, self.parent_states_.shape[0]), dtype=np.float64)
        for state_index, state in enumerate(self.parent_states_):
            for offset, model in enumerate(self.parent_models_):
                augmented = (
                    features
                    if offset == 0
                    else np.hstack(
                        (
                            features,
                            np.broadcast_to(state[:offset], (n_samples, offset)),
                        )
                    )
                )
                probability_one = positive_probability(model, augmented)
                joint[:, state_index] *= (
                    probability_one if state[offset] else 1.0 - probability_one
                )

        sums = joint.sum(axis=1, keepdims=True)
        tolerance = float(self.normalization_tolerance)
        if (
            not np.isfinite(joint).all()
            or np.any(joint < -tolerance)
            or np.any(np.abs(sums - 1.0) > tolerance)
        ):
            raise FloatingPointError("Invalid LocalParentPCC parent-joint distribution.")
        return np.clip(joint, 0.0, 1.0) / sums

    def predict_child_proba(self, X):
        """Marginalize parent states and return P(child=1|X)."""

        features = self._validate_features(X)
        joint = self.predict_parent_joint(features)
        conditionals = np.empty(joint.shape, dtype=np.float64)
        for state_index, state in enumerate(self.parent_states_):
            augmented = (
                features
                if not self.parent_indices_
                else np.hstack(
                    (features, np.broadcast_to(state, (features.shape[0], len(state))))
                )
            )
            conditionals[:, state_index] = positive_probability(
                self.child_model_, augmented
            )
        marginal = np.sum(joint * conditionals, axis=1)
        if not np.isfinite(marginal).all():
            raise FloatingPointError("Invalid LocalParentPCC child probability.")
        return np.clip(marginal, 0.0, 1.0)

    def predict_proba(self, X):
        positive = self.predict_child_proba(X)
        return np.column_stack((1.0 - positive, positive))

    def predict(self, X):
        return (self.predict_child_proba(X) >= 0.5).astype(np.int32)
