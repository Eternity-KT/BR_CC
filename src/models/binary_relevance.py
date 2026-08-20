"""
Binary Relevance (BR) Multi-Label Classifier
Reference: Zhang, M.-L., Li, Y.-K., Liu, X.-Y., & Geng, X. (2018).
           Binary relevance for multi-label learning: an overview.
           Frontiers of Computer Science, 12(2), 191–202.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


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
    """
    Resolve base classifier from string or estimator instance.

    Supported string presets:
    - 'svm', 'linearsvc', None: LinearSVC(C=1.0, dual='auto', max_iter=10000, random_state=random_state)
    - 'logistic', 'lr', 'logistic_regression': LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=random_state)
    """
    if base_estimator is None or (isinstance(base_estimator, str) and base_estimator.lower() in ("svm", "linearsvc", "linear_svc")):
        return LinearSVC(C=1.0, dual="auto", max_iter=10000, random_state=random_state)
    elif isinstance(base_estimator, str) and base_estimator.lower() in ("logistic", "lr", "logistic_regression", "logreg"):
        return LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=random_state)
    elif hasattr(base_estimator, "fit"):
        return clone(base_estimator)
    else:
        raise ValueError(
            f"Unsupported base_estimator: {base_estimator}. "
            f"Must be None, 'svm', 'logistic', or an estimator instance implementing fit/predict."
        )


class BinaryRelevanceClassifier(BaseEstimator, ClassifierMixin):
    """
    Binary Relevance (BR) Multi-Label Classifier.
    
    Decomposes the multi-label learning problem into q independent binary
    classification tasks, one for each label.

    Parameters:
        base_estimator (estimator object or str, default=None):
            The base binary classifier. Options:
            - None or 'svm': LinearSVC(C=1.0, dual='auto', max_iter=10000, random_state=42)
            - 'logistic' or 'lr': LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
            - Custom estimator instance with fit() and predict() methods.
        random_state (int, default=42):
            Random seed for reproducibility.
    """
    def __init__(self, base_estimator=None, random_state=42):
        self.base_estimator = base_estimator
        self.random_state = random_state
        self.classifiers_ = []
        self.n_labels_ = 0

    def fit(self, X, Y):
        """
        Fit q binary classifiers independently.

        Parameters:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)
            Y (np.ndarray): Binary label matrix of shape (n_samples, n_labels)

        Returns:
            self: returns an instance of self.
        """
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int32)
        self.n_labels_ = Y.shape[1]
        self.classifiers_ = []

        template_estimator = _get_base_estimator(self.base_estimator, self.random_state)

        for j in range(self.n_labels_):
            y_j = Y[:, j]
            unique_classes = np.unique(y_j)

            if len(unique_classes) <= 1:
                # Handle single-class degenerate case
                const_val = unique_classes[0] if len(unique_classes) == 1 else 0
                clf = _ConstantClassifier(const_val)
            else:
                clf = clone(template_estimator)
                clf.fit(X, y_j)

            self.classifiers_.append(clf)

        return self

    def predict(self, X):
        """
        Predict binary labels for samples in X.

        Parameters:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)

        Returns:
            Y_pred (np.ndarray): Binary label matrix of shape (n_samples, n_labels)
        """
        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]
        Y_pred = np.zeros((n_samples, self.n_labels_), dtype=np.int32)

        for j, clf in enumerate(self.classifiers_):
            pred_j = clf.predict(X)
            Y_pred[:, j] = np.asarray(pred_j, dtype=np.int32)

        return Y_pred

    def predict_proba(self, X):
        """
        Predict probability P(Y_j = 1 | X) for samples in X.

        Parameters:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)

        Returns:
            Y_proba (np.ndarray): Probability matrix of shape (n_samples, n_labels)
        """
        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]
        Y_proba = np.zeros((n_samples, self.n_labels_), dtype=np.float32)

        for j, clf in enumerate(self.classifiers_):
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X)
                Y_proba[:, j] = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            elif hasattr(clf, "decision_function"):
                scores = clf.decision_function(X)
                # Sigmoid transform for margin-based classifiers
                Y_proba[:, j] = 1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0)))
            else:
                Y_proba[:, j] = clf.predict(X).astype(np.float32)

        return Y_proba

    def decision_function(self, X):
        """
        Predict confidence scores for samples in X.

        Parameters:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)

        Returns:
            Y_scores (np.ndarray): Decision function scores of shape (n_samples, n_labels)
        """
        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]
        Y_scores = np.zeros((n_samples, self.n_labels_), dtype=np.float32)

        for j, clf in enumerate(self.classifiers_):
            if hasattr(clf, "decision_function"):
                Y_scores[:, j] = clf.decision_function(X)
            elif hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X)
                Y_scores[:, j] = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            else:
                Y_scores[:, j] = clf.predict(X).astype(np.float32)

        return Y_scores


class BinaryRelevanceLogisticRegression(BinaryRelevanceClassifier):
    """
    Binary Relevance (BR) Multi-Label Classifier using Logistic Regression.

    Decomposes multi-label learning into q independent Logistic Regression
    binary classification problems with calibrated probabilistic outputs.

    Parameters:
        C (float, default=1.0):
            Inverse of regularization strength.
        solver (str, default='liblinear'):
            Algorithm for optimization problem ('liblinear', 'lbfgs', 'saga').
        max_iter (int, default=1000):
            Maximum number of iterations for solver convergence.
        random_state (int, default=42):
            Random seed for reproducibility.
        **kwargs:
            Additional parameters passed to sklearn.linear_model.LogisticRegression.
    """
    def __init__(self, C=1.0, solver="liblinear", max_iter=1000, random_state=42, **kwargs):
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.lr_kwargs = kwargs
        base_lr = LogisticRegression(
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=random_state,
            **kwargs
        )
        super().__init__(base_estimator=base_lr, random_state=random_state)
