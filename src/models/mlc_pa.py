"""
Multilabel classification with partial abstention (MLC-PA).

Reference
---------
Nguyen, V.-L. and Huellermeier, E. (2021). Multilabel Classification
with Partial Abstention: Bayes-Optimal Prediction under Label Independence.
Journal of Artificial Intelligence Research, 72, 613-665.

This module implements the Bayes-optimal prediction rule for the generalized
Hamming loss from Corollaries 1 and 2.  MLC-PA is a decision layer on top of a
probabilistic multilabel classifier; it is not a new probability estimator.
"""

from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted

from .binary_relevance import (
    BinaryRelevanceClassifier,
    BinaryRelevanceLogisticRegression,
    BinaryRelevanceMLP,
)


def _make_base_estimator(base_estimator, random_state):
    """Resolve the marginal-probability estimator used by MLC-PA."""
    if base_estimator is None or (
        isinstance(base_estimator, str)
        and base_estimator.lower() in ("mlp", "br_mlp", "pytorch_mlp")
    ):
        return BinaryRelevanceMLP(random_state=random_state)
    if isinstance(base_estimator, str) and base_estimator.lower() in (
        "logistic",
        "lr",
        "br_logistic",
    ):
        return BinaryRelevanceLogisticRegression(random_state=random_state)
    if isinstance(base_estimator, str) and base_estimator.lower() in (
        "svm",
        "linearsvc",
        "br_svm",
    ):
        return BinaryRelevanceClassifier(
            base_estimator="svm", random_state=random_state
        )
    if hasattr(base_estimator, "fit") and hasattr(base_estimator, "predict_proba"):
        try:
            return clone(base_estimator)
        except (TypeError, RuntimeError):
            return deepcopy(base_estimator)
    raise ValueError(
        "base_estimator must be 'mlp', 'logistic', 'svm', or an estimator "
        "implementing fit() and predict_proba()."
    )


class MLCPartialAbstentionClassifier(BaseEstimator, ClassifierMixin):
    """Bayes-optimal partial predictions for generalized Hamming loss.

    Parameters
    ----------
    base_estimator : str or estimator, default="mlp"
        Probabilistic multilabel estimator providing marginal probabilities
        ``P(Y_k=1 | x)``.  The string presets are ``"mlp"``, ``"logistic"``,
        and ``"svm"``.
    cost : float, default=0.3
        Abstention cost ``c``.  For the linear (SEP) penalty, a label is
        decided iff ``min(p_k, 1-p_k) <= c``.  Values at least 0.5 therefore
        produce a complete prediction.
    penalty : {"linear", "concave"}, default="linear"
        ``"linear"`` implements SEP, ``g(a)=a*c`` (paper Eq. 30).
        ``"concave"`` implements PAR, ``g(a)=a*K*c/(K+a)`` (paper Eq. 31).
    abstain_value : int, default=-1
        Sentinel used for the abstention symbol in returned arrays.
    random_state : int, default=42
        Random seed passed to string-based base estimators.

    Notes
    -----
    ``predict`` returns values in ``{0, abstain_value, 1}``.  Use
    ``predict_full`` for the conventional complete prediction obtained by
    thresholding the same marginal probabilities at 0.5.
    """

    def __init__(
        self,
        base_estimator="mlp",
        cost=0.3,
        penalty="linear",
        abstain_value=-1,
        random_state=42,
    ):
        self.base_estimator = base_estimator
        self.cost = cost
        self.penalty = penalty
        self.abstain_value = abstain_value
        self.random_state = random_state

    def _validate_parameters(self):
        if not 0.0 <= float(self.cost) <= 1.0:
            raise ValueError("cost must lie in [0, 1].")
        if self.penalty not in ("linear", "concave"):
            raise ValueError("penalty must be either 'linear' or 'concave'.")
        if self.abstain_value in (0, 1):
            raise ValueError("abstain_value must be different from 0 and 1.")

    def fit(self, X, Y):
        """Fit the marginal-probability estimator."""
        self._validate_parameters()
        Y_arr = np.asarray(Y, dtype=np.int32)
        if Y_arr.ndim != 2:
            raise ValueError("Y must be a two-dimensional binary label matrix.")
        if Y_arr.shape[1] == 0:
            raise ValueError("Y must contain at least one label column.")
        if not np.all(np.isin(Y_arr, (0, 1))):
            raise ValueError("Y must contain only binary values 0 and 1.")

        self.n_labels_ = Y_arr.shape[1]
        self.base_estimator_ = _make_base_estimator(
            self.base_estimator, self.random_state
        )
        self.base_estimator_.fit(X, Y_arr)
        return self

    def predict_proba(self, X):
        """Return clipped marginal relevance probabilities of shape (n, K)."""
        check_is_fitted(self, ("base_estimator_", "n_labels_"))
        probabilities = np.asarray(
            self.base_estimator_.predict_proba(X), dtype=np.float64
        )
        if probabilities.ndim != 2 or probabilities.shape[1] != self.n_labels_:
            raise ValueError(
                "base_estimator.predict_proba() must return an (n_samples, "
                "n_labels) matrix of positive-label probabilities."
            )
        return np.clip(probabilities, 0.0, 1.0)

    def predict_full(self, X):
        """Return the conventional complete prediction in ``{0, 1}^K``."""
        return self.predict_full_from_proba(self.predict_proba(X))

    def predict_full_from_proba(self, probabilities):
        """Threshold an already-computed probability matrix at 0.5."""
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != self.n_labels_:
            raise ValueError("probabilities must have shape (n_samples, n_labels).")
        return (np.clip(probabilities, 0.0, 1.0) >= 0.5).astype(np.int32)

    def _linear_decision_mask(self, probabilities, cost=None):
        cost = float(self.cost if cost is None else cost)
        expected_label_losses = np.minimum(probabilities, 1.0 - probabilities)
        return expected_label_losses <= cost

    def _concave_decision_mask(self, probabilities, cost=None):
        """Find the optimal decided-set size from Corollary 1 for each row."""
        cost = float(self.cost if cost is None else cost)
        n_samples, n_labels = probabilities.shape
        expected_label_losses = np.minimum(probabilities, 1.0 - probabilities)
        masks = np.zeros((n_samples, n_labels), dtype=bool)
        abstention_counts = n_labels - np.arange(n_labels + 1, dtype=np.float64)
        penalties = (
            abstention_counts
            * float(n_labels)
            * cost
            / (float(n_labels) + abstention_counts)
        )

        for row_idx in range(n_samples):
            order = np.argsort(expected_label_losses[row_idx], kind="stable")
            sorted_losses = expected_label_losses[row_idx, order]
            cumulative_losses = np.concatenate(
                ([0.0], np.cumsum(sorted_losses, dtype=np.float64))
            )
            risks = cumulative_losses + penalties
            minimum = np.min(risks)
            # Prefer more decided labels when several actions have equal risk.
            best_d = int(np.flatnonzero(np.isclose(risks, minimum))[-1])
            masks[row_idx, order[:best_d]] = True
        return masks

    def decision_mask(self, X):
        """Return True at label positions on which the classifier decides."""
        probabilities = self.predict_proba(X)
        if self.penalty == "linear":
            return self._linear_decision_mask(probabilities)
        return self._concave_decision_mask(probabilities)

    def predict(self, X):
        """Return Bayes-optimal partial predictions in ``{0, abstain, 1}^K``."""
        return self.predict_from_proba(self.predict_proba(X), cost=self.cost)

    def predict_from_proba(self, probabilities, cost=None):
        """Apply partial abstention after probability estimation.

        ``fit`` never uses the rejection cost.  Passing ``cost`` here allows
        several operating costs to reuse one fitted probabilistic model.
        """
        cost = float(self.cost if cost is None else cost)
        if not 0.0 <= cost <= 1.0:
            raise ValueError("cost must lie in [0, 1].")
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != self.n_labels_:
            raise ValueError("probabilities must have shape (n_samples, n_labels).")
        probabilities = np.clip(probabilities, 0.0, 1.0)
        full_predictions = self.predict_full_from_proba(probabilities)
        if self.penalty == "linear":
            decided = self._linear_decision_mask(probabilities, cost=cost)
        else:
            decided = self._concave_decision_mask(probabilities, cost=cost)
        partial = np.full(
            full_predictions.shape, self.abstain_value, dtype=np.int32
        )
        partial[decided] = full_predictions[decided]
        return partial

    def decision_function(self, X):
        """Return base scores when available, otherwise probability logits."""
        check_is_fitted(self, ("base_estimator_", "n_labels_"))
        if hasattr(self.base_estimator_, "decision_function"):
            return self.base_estimator_.decision_function(X)
        probabilities = np.clip(self.predict_proba(X), 1e-7, 1.0 - 1e-7)
        return np.log(probabilities / (1.0 - probabilities))


# Short alias used by the CLI and external callers.
MLCPAClassifier = MLCPartialAbstentionClassifier
