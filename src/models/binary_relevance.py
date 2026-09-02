"""
Binary Relevance (BR) Multi-Label Classifier
Reference: Zhang, M.-L., Li, Y.-K., Liu, X.-Y., & Geng, X. (2018).
           Binary relevance for multi-label learning: an overview.
           Frontiers of Computer Science, 12(2), 191–202.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression

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


class BinaryRelevanceMLP(BinaryRelevanceClassifier):
    """
    Binary Relevance (BR) Multi-Label Classifier using Multi-Layer Perceptron (MLP).

    Decomposes multi-label learning into q independent Multi-Layer Perceptron (MLP)
    binary classification problems. Supports non-linear decision boundaries through
    hidden layers and backpropagation learning.

    Parameters:
        hidden_layer_sizes (tuple, default=(100,)):
            The ith element represents the number of neurons in the ith hidden layer.
        activation (str, default='relu'):
            Activation function for the hidden layer ('identity', 'logistic', 'tanh', 'relu').
        solver (str, default='adam'):
            The solver for weight optimization ('lbfgs', 'sgd', 'adam').
        alpha (float, default=1e-4):
            Strength of the L2 regularization term.
        learning_rate_init (float, default=0.001):
            The initial learning rate used.
        max_iter (int, default=500):
            Maximum number of iterations.
        early_stopping (bool, default=True):
            Whether to use early stopping to terminate training when validation score is not improving.
        validation_fraction (float, default=0.1):
            The proportion of training data to set aside as validation set for early stopping.
        n_iter_no_change (int, default=10):
            Maximum number of epochs to not meet tol improvement.
        random_state (int, default=42):
            Random seed for reproducibility.
        **kwargs:
            Additional parameters passed to sklearn.neural_network.MLPClassifier.
    """
class BinaryRelevanceMLP(BaseEstimator, ClassifierMixin):
    """
    Binary Relevance (BR) Multi-Label Classifier using GPU-Accelerated Multi-Label MLP.

    Predicts all q labels simultaneously on GPU via PyTorch with BCEWithLogitsLoss
    and positive class imbalance weighting.

    Parameters:
        hidden_layer_sizes (tuple, default=(128, 64)):
            Neurons in hidden layers.
        lr (float, default=3e-3):
            Learning rate for AdamW optimizer.
        weight_decay (float, default=1e-3):
            L2 regularization strength.
        epochs (int, default=150):
            Number of training epochs.
        dropout (float, default=0.15):
            Dropout rate for regularization.
        device (str, default=None):
            'cuda', 'cpu', or None (auto-detects).
        random_state (int, default=42):
            Random seed for reproducibility.
    """
    def __init__(
        self,
        hidden_layer_sizes=(128, 64),
        lr=1e-3,
        weight_decay=1e-3,
        epochs=30,
        dropout=0.15,
        device=None,
        random_state=42,
        **kwargs
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dropout = dropout
        self.device = device
        self.random_state = random_state
        self.kwargs = kwargs
        from .base_learners import _pytorch_classes

        _, multilabel_class = _pytorch_classes()
        self.model_ = multilabel_class(
            hidden_layer_sizes=self.hidden_layer_sizes,
            lr=self.lr,
            weight_decay=self.weight_decay,
            epochs=self.epochs,
            dropout=self.dropout,
            device=self.device,
            random_state=self.random_state,
            **kwargs
        )
        self.backend_ = "pytorch"

    def fit(self, X, Y):
        self.model_.fit(X, Y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def decision_function(self, X):
        if hasattr(self.model_, "decision_function"):
            return self.model_.decision_function(X)
        probabilities = np.clip(self.predict_proba(X), 1e-7, 1.0 - 1e-7)
        return np.log(probabilities / (1.0 - probabilities))

