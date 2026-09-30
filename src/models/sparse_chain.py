"""
Sparse Classifier Chain (Sparse CC) Multi-Label Classifier with Correlation Thresholding.

References:
- Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2011). Classifier chains for multi-label classification.
- GSI-MLC-PA v5.1.1 Specification: Threshold-Filtered Sparse Classifier Chains & Static IL Context.

Core Principles:
1. Static Context Augmentation:
   Classifiers in DL receive the base features X and the normalized soft probabilities
   of the independent labels (IL) as static context features:
   X_context_DL = [X, Normalize(P_hat(Y_IL))].
2. Threshold-Filtered Active Conditioning:
   Instead of a dense chain where each classifier receives all preceding DL labels,
   label d_k only conditions on preceding DL labels with absolute correlation >= theta_corr (default 0.75).
   Predecessors with low correlation are filtered out to prevent noise pollution.
3. Adaptive Inference Marginalization:
   - 0 parents: Direct prediction from X_context_DL.
   - 1 parent: Exact two-state marginalization.
   - >= 2 parents: Continuous mean-field plug-in using soft predicted probabilities.
"""

from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from .base_learners import create_binary_estimator
from .probability_adapter import ProbabilityAdapter, aggregate_calibration_audit


class _ConstantClassifier:
    """Fallback classifier when a label has only one unique class in training set."""
    def __init__(self, constant_value: int):
        self.constant_value = int(constant_value)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.constant_value, dtype=np.int32)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        val = 1.0 if self.constant_value == 1 else -1.0
        return np.full(X.shape[0], val, dtype=np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, 2), dtype=np.float32)
        if self.constant_value == 1:
            probs[:, 1] = 1.0
        else:
            probs[:, 0] = 1.0
        return probs


def _positive_probability(classifier: Any, X: np.ndarray) -> np.ndarray:
    """Extract P(Y = 1 | X) safely across different classifier interfaces."""
    if hasattr(classifier, "predict_proba"):
        probs = np.asarray(classifier.predict_proba(X), dtype=np.float32)
        if probs.ndim == 2 and probs.shape[1] > 1:
            return np.clip(probs[:, 1], 0.0, 1.0)
        return np.clip(probs.ravel(), 0.0, 1.0)
    elif hasattr(classifier, "decision_function"):
        scores = np.asarray(classifier.decision_function(X), dtype=np.float32)
        return np.clip(1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0))), 0.0, 1.0)
    else:
        preds = np.asarray(classifier.predict(X), dtype=np.float32)
        return np.clip(preds.ravel(), 0.0, 1.0)


class SparseClassifierChainClassifier(BaseEstimator, ClassifierMixin):
    """Classifier Chain with correlation-threshold filtered active predecessor conditioning.

    Parameters:
        base_estimator: Base binary estimator instance or name ('logistic', 'svm', 'mlp', etc.).
        order: List of label indices defining the execution sequence for the chain.
        correlation_matrix: K x K absolute pairwise correlation matrix (e.g. Phi coefficients).
        correlation_threshold: Float in [0.0, 1.0]. Only predecessors with |corr| >= threshold
                               are included as inputs to subsequent classifiers. (Default: 0.75).
        random_state: Random seed for reproducibility.
        label_noise: Optional noise factor for training feature augmentation.
    """

    def __init__(
        self,
        base_estimator: Any = None,
        order: Optional[Sequence[int]] = None,
        correlation_matrix: Optional[np.ndarray] = None,
        correlation_threshold: float = 0.75,
        random_state: int = 42,
        label_noise: float = 0.0,
    ):
        self.base_estimator = base_estimator
        self.order = order
        self.correlation_matrix = correlation_matrix
        self.correlation_threshold = float(correlation_threshold)
        self.random_state = random_state
        self.label_noise = float(label_noise)

        self.classifiers_: List[Any] = []
        self.order_: List[int] = []
        self.active_parents_map_: Dict[int, List[int]] = {}
        self.calibration_audit_: Optional[Dict[str, Any]] = None

    def fit(self, X_context: np.ndarray, Y: np.ndarray) -> "SparseClassifierChainClassifier":
        """Fit sparse CC where each classifier only conditions on predecessors with corr >= threshold.

        Parameters:
            X_context: Context feature matrix [X, Normalize(P_IL)] of shape (n_samples, n_features).
            Y: Binary label matrix of shape (n_samples, n_labels).
        """
        X_context = np.asarray(X_context, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int32)
        n_samples, n_labels = Y.shape

        if self.order is None:
            self.order_ = list(range(n_labels))
        else:
            self.order_ = [int(l) for l in self.order]

        if self.correlation_matrix is None:
            corr_mat = np.ones((n_labels, n_labels), dtype=np.float32)
        else:
            corr_mat = np.asarray(self.correlation_matrix, dtype=np.float32)

        template_estimator = create_binary_estimator(self.base_estimator, self.random_state)
        self.classifiers_ = []
        self.active_parents_map_ = {}

        for i, target_label in enumerate(self.order_):
            y_target = Y[:, target_label]
            unique_classes = np.unique(y_target)

            # Find active parents among preceding labels in chain order
            candidate_preds = self.order_[:i]
            active_parents = [
                p for p in candidate_preds
                if float(corr_mat[p, target_label]) >= self.correlation_threshold
            ]
            self.active_parents_map_[target_label] = active_parents

            # Build extended feature vector for target_label
            if not active_parents:
                X_extended = X_context
            else:
                parents_features = Y[:, active_parents].astype(np.float32)
                if self.label_noise > 0.0:
                    rng = np.random.default_rng(
                        self.random_state + i if self.random_state is not None else None
                    )
                    eps = rng.uniform(0.0, self.label_noise, size=parents_features.shape).astype(np.float32)
                    parents_features = parents_features * (1.0 - eps) + 0.5 * eps
                X_extended = np.hstack([X_context, parents_features])

            # Handle trivial constant labels
            if len(unique_classes) <= 1 and not isinstance(template_estimator, ProbabilityAdapter):
                const_val = int(unique_classes[0]) if len(unique_classes) == 1 else 0
                clf = _ConstantClassifier(const_val)
            else:
                clf = clone(template_estimator)
                clf.fit(X_extended, y_target)

            self.classifiers_.append(clf)

        audit = aggregate_calibration_audit(zip(self.order_, self.classifiers_))
        if audit is not None:
            self.calibration_audit_ = audit

        return self

    def predict_proba(self, X_context: np.ndarray) -> np.ndarray:
        """Infer probabilities for labels in order_ using adaptive marginalization.

        Parameters:
            X_context: Context feature matrix of shape (n_samples, n_features).

        Returns:
            Y_proba: Probability matrix for the labels managed by this chain,
                     indexed by the true label indices in order_.
        """
        X_context = np.asarray(X_context, dtype=np.float32)
        n_samples = X_context.shape[0]

        # Allocate dictionary or matrix for computed probabilities
        computed_probs: Dict[int, np.ndarray] = {}

        for clf, target_label in zip(self.classifiers_, self.order_):
            parents = self.active_parents_map_.get(target_label, [])

            if len(parents) == 0:
                # 0 parents: Direct unconditional prediction from context
                p_target = _positive_probability(clf, X_context)

            elif len(parents) == 1:
                # 1 parent: Exact two-state marginalization
                parent_idx = parents[0]
                p_parent = computed_probs[parent_idx]

                zeros = np.zeros((n_samples, 1), dtype=np.float32)
                ones = np.ones((n_samples, 1), dtype=np.float32)

                prob_given_zero = _positive_probability(clf, np.hstack([X_context, zeros]))
                prob_given_one = _positive_probability(clf, np.hstack([X_context, ones]))

                p_target = (1.0 - p_parent) * prob_given_zero + p_parent * prob_given_one

            else:
                # >= 2 parents: Continuous mean-field plug-in with soft probabilities
                parent_probs_list = [computed_probs[p][:, np.newaxis] for p in parents]
                mf_features = np.hstack([X_context] + parent_probs_list)
                p_target = _positive_probability(clf, mf_features)

            computed_probs[target_label] = np.clip(p_target, 0.0, 1.0)

        # Return full probability matrix for these managed labels
        max_label = max(self.order_) + 1 if self.order_ else 0
        Y_proba = np.zeros((n_samples, max_label), dtype=np.float32)
        for label, prob in computed_probs.items():
            Y_proba[:, label] = prob

        return Y_proba

    def predict(self, X_context: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels using thresholding on predicted probabilities."""
        probs = self.predict_proba(X_context)
        return (probs >= threshold).astype(np.int32)
