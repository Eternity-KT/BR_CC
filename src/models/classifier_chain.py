"""
Classifier Chains (CC) Multi-Label Classifier
Reference: Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2011).
           Classifier chains for multi-label classification.
           Machine Learning, 85(3), 333–359.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from .base_learners import create_binary_estimator


class _ConstantClassifier:
    """Fallback classifier when a label has only one unique class in training set."""
    def __init__(self, constant_value):
        self.constant_value = int(constant_value)

    def predict(self, X):
        n_samples = X.shape[0]
        return np.full(n_samples, self.constant_value, dtype=np.int32)

    def decision_function(self, X):
        n_samples = X.shape[0]
        return np.full(n_samples, 1.0 if self.constant_value == 1 else -1.0, dtype=np.float32)

    def predict_proba(self, X):
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, 2), dtype=np.float32)
        if self.constant_value == 1:
            probs[:, 1] = 1.0
        else:
            probs[:, 0] = 1.0
        return probs


def _get_base_estimator(base_estimator, random_state=42):
    """Compatibility wrapper around the shared base-learner factory."""

    return create_binary_estimator(base_estimator, random_state)


class ClassifierChainClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier Chains (CC) Multi-Label Classifier.

    Chains q binary classifiers along a specified or default label order.
    Each classifier in the chain uses the true labels (during training)
    or predicted labels (during inference) of preceding classifiers as
    additional input features.

    Parameters:
        base_estimator (estimator object or str, default=None):
            The base binary classifier. Options:
            - None or 'svm': LinearSVC(C=1.0, dual='auto', max_iter=10000, random_state=42)
            - 'logistic' or 'lr': LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
            - Custom estimator instance with fit() and predict() methods.
        order (list or np.ndarray, default=None):
            The chain order of labels. If None, uses default natural order [0, 1, ..., q-1]
            as in Read et al. (2011).
        random_state (int, default=42):
            Random seed for reproducibility.
    """
    def __init__(self, base_estimator=None, order=None, random_state=42):
        self.base_estimator = base_estimator
        self.order = order
        self.random_state = random_state
        self.classifiers_ = []
        self.order_ = None
        self.n_labels_ = 0

    def fit(self, X, Y):
        """
        Fit the chain of binary classifiers using ground truth label augmentation.

        Parameters:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)
            Y (np.ndarray): Binary label matrix of shape (n_samples, n_labels)

        Returns:
            self: returns an instance of self.
        """
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int32)
        n_samples, n_labels = Y.shape
        self.n_labels_ = n_labels

        if self.order is None:
            self.order_ = list(range(n_labels))
        else:
            self.order_ = list(self.order)

        self.classifiers_ = []
        template_estimator = _get_base_estimator(self.base_estimator, self.random_state)

        for i, label_idx in enumerate(self.order_):
            y_j = Y[:, label_idx]
            unique_classes = np.unique(y_j)

            # Build extended feature vector
            if i == 0:
                X_extended = X
            else:
                prev_indices = self.order_[:i]
                prev_ground_truth = Y[:, prev_indices].astype(np.float32)
                X_extended = np.hstack([X, prev_ground_truth])

            if len(unique_classes) <= 1:
                const_val = unique_classes[0] if len(unique_classes) == 1 else 0
                clf = _ConstantClassifier(const_val)
            else:
                clf = clone(template_estimator)
                clf.fit(X_extended, y_j)

            self.classifiers_.append(clf)

        return self

    def predict(self, X):
        """
        Predict binary labels for samples in X using greedy chain inference.

        Parameters:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)

        Returns:
            Y_pred (np.ndarray): Binary label matrix of shape (n_samples, n_labels)
        """
        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]
        Y_pred = np.zeros((n_samples, self.n_labels_), dtype=np.int32)

        for i, (label_idx, clf) in enumerate(zip(self.order_, self.classifiers_)):
            if i == 0:
                X_extended = X
            else:
                prev_indices = self.order_[:i]
                prev_preds = Y_pred[:, prev_indices].astype(np.float32)
                X_extended = np.hstack([X, prev_preds])

            pred_j = clf.predict(X_extended)
            Y_pred[:, label_idx] = np.asarray(pred_j, dtype=np.int32)

        return Y_pred

    def predict_proba(self, X):
        """
        Predict probability P(Y_j = 1 | X) for samples in X using greedy chain inference.

        Parameters:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)

        Returns:
            Y_proba (np.ndarray): Probability matrix of shape (n_samples, n_labels)
        """
        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]
        Y_pred = np.zeros((n_samples, self.n_labels_), dtype=np.int32)
        Y_proba = np.zeros((n_samples, self.n_labels_), dtype=np.float32)

        for i, (label_idx, clf) in enumerate(zip(self.order_, self.classifiers_)):
            if i == 0:
                X_extended = X
            else:
                prev_indices = self.order_[:i]
                prev_preds = Y_pred[:, prev_indices].astype(np.float32)
                X_extended = np.hstack([X, prev_preds])

            pred_j = clf.predict(X_extended)
            Y_pred[:, label_idx] = np.asarray(pred_j, dtype=np.int32)

            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X_extended)
                Y_proba[:, label_idx] = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            elif hasattr(clf, "decision_function"):
                scores = clf.decision_function(X_extended)
                Y_proba[:, label_idx] = 1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0)))
            else:
                Y_proba[:, label_idx] = Y_pred[:, label_idx].astype(np.float32)

        return Y_proba

    def decision_function(self, X):
        """
        Predict decision function scores for samples in X.

        Parameters:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)

        Returns:
            Y_scores (np.ndarray): Decision function scores of shape (n_samples, n_labels)
        """
        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]
        Y_pred = np.zeros((n_samples, self.n_labels_), dtype=np.int32)
        Y_scores = np.zeros((n_samples, self.n_labels_), dtype=np.float32)

        for i, (label_idx, clf) in enumerate(zip(self.order_, self.classifiers_)):
            if i == 0:
                X_extended = X
            else:
                prev_indices = self.order_[:i]
                prev_preds = Y_pred[:, prev_indices].astype(np.float32)
                X_extended = np.hstack([X, prev_preds])

            pred_j = clf.predict(X_extended)
            Y_pred[:, label_idx] = np.asarray(pred_j, dtype=np.int32)

            if hasattr(clf, "decision_function"):
                Y_scores[:, label_idx] = clf.decision_function(X_extended)
            elif hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X_extended)
                Y_scores[:, label_idx] = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            else:
                Y_scores[:, label_idx] = Y_pred[:, label_idx].astype(np.float32)

        return Y_scores
