"""Probabilistic classifier chain with Monte-Carlo ancestral inference."""

from typing import Any, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from .base_learners import create_binary_estimator
from .binary_relevance import _ConstantClassifier
from .probability_adapter import ProbabilityAdapter, aggregate_calibration_audit


def _positive_probability(estimator: Any, X: np.ndarray) -> np.ndarray:
    """Return a clipped one-dimensional positive-class probability vector."""
    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(X), dtype=np.float64)
        if probabilities.ndim == 2:
            if probabilities.shape[1] == 1:
                probabilities = probabilities[:, 0]
            else:
                probabilities = probabilities[:, 1]
        else:
            probabilities = probabilities.ravel()
    elif hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(X), dtype=np.float64)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0)))
    else:
        probabilities = np.asarray(estimator.predict(X), dtype=np.float64).ravel()
    return np.clip(probabilities, 0.0, 1.0)


class ProbabilisticClassifierChain(BaseEstimator, ClassifierMixin):
    """Classifier-chain factors with sampled marginal inference.

    The factors are fitted using ground-truth predecessor labels, exactly as
    in a standard classifier chain.  At inference, ``num_samples`` complete
    label vectors are generated ancestrally for every input row.  The returned
    marginal probabilities are the empirical positive frequencies.
    """

    def __init__(
        self,
        base_estimator=None,
        order=None,
        num_samples: int = 1000,
        sampling_batch_size: int = 32,
        random_state: Optional[int] = 42,
    ):
        self.base_estimator = base_estimator
        self.order = order
        self.num_samples = int(num_samples)
        self.sampling_batch_size = int(sampling_batch_size)
        self.random_state = random_state

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int32)
        if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must be compatible two-dimensional matrices.")
        if self.num_samples < 1:
            raise ValueError("num_samples must be at least one.")
        if self.sampling_batch_size < 1:
            raise ValueError("sampling_batch_size must be at least one.")

        self.n_labels_ = int(Y.shape[1])
        self.order_ = (
            list(range(self.n_labels_)) if self.order is None else list(self.order)
        )
        if sorted(self.order_) != list(range(self.n_labels_)):
            raise ValueError("order must be a permutation of local label indices.")

        template = create_binary_estimator(
            self.base_estimator, random_state=self.random_state
        )
        self.classifiers_: List[Any] = []
        for position, label_index in enumerate(self.order_):
            predecessors = self.order_[:position]
            X_extended = (
                X
                if not predecessors
                else np.hstack([X, Y[:, predecessors].astype(np.float32)])
            )
            target = Y[:, label_index]
            unique = np.unique(target)
            if len(unique) <= 1 and not isinstance(template, ProbabilityAdapter):
                classifier = _ConstantClassifier(int(unique[0]) if len(unique) else 0)
            else:
                classifier = clone(template)
                classifier.fit(X_extended, target)
            self.classifiers_.append(classifier)

        calibration_audit = aggregate_calibration_audit(
            zip(self.order_, self.classifiers_)
        )
        if calibration_audit is not None:
            self.calibration_audit_ = calibration_audit
        self.is_fitted_ = True
        return self

    def predict_proba(self, X, *, random_state: Optional[int] = None):
        if not getattr(self, "is_fitted_", False):
            raise ValueError("ProbabilisticClassifierChain is not fitted.")
        X = np.asarray(X, dtype=np.float32)
        n_rows = X.shape[0]
        if self.n_labels_ == 0:
            return np.empty((n_rows, 0), dtype=np.float32)

        seed = self.random_state if random_state is None else random_state
        rng = np.random.default_rng(seed)
        positive_counts = np.zeros((n_rows, self.n_labels_), dtype=np.float64)
        completed = 0

        while completed < self.num_samples:
            batch_size = min(self.sampling_batch_size, self.num_samples - completed)
            # Row-major layout keeps all ancestral replicas of one x adjacent.
            X_repeated = np.repeat(X, batch_size, axis=0)
            sampled = np.zeros(
                (n_rows * batch_size, self.n_labels_), dtype=np.float32
            )
            for position, (label_index, classifier) in enumerate(
                zip(self.order_, self.classifiers_)
            ):
                predecessors = self.order_[:position]
                X_extended = (
                    X_repeated
                    if not predecessors
                    else np.hstack([X_repeated, sampled[:, predecessors]])
                )
                probabilities = _positive_probability(classifier, X_extended)
                draws = (rng.random(probabilities.shape[0]) < probabilities).astype(
                    np.float32
                )
                sampled[:, label_index] = draws

            positive_counts += sampled.reshape(
                n_rows, batch_size, self.n_labels_
            ).sum(axis=1)
            completed += batch_size

        return (positive_counts / float(self.num_samples)).astype(np.float32)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(np.int32)


__all__ = ["ProbabilisticClassifierChain"]
