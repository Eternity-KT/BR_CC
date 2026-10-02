"""
Ensemble of Classifier Chains (ECC) Multi-Label Classifier.

Reference:
    Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2011).
    Classifier chains for multi-label classification.
    Machine Learning, 85(3), 333–359.
"""

from typing import Any, List, Optional, Sequence, Union
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from .base_learners import create_binary_estimator
from .classifier_chain import ClassifierChainClassifier, _ConstantClassifier


class EnsembleClassifierChainClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble of Classifier Chains (ECC) Multi-Label Classifier.

    Trains an ensemble of M Classifier Chains with diverse label orderings
    (random permutations) and optional bootstrap/subsampling. Predictions
    are pooled by averaging the predicted probability distributions across
    all member chains:
        P_hat_ECC(Y_j = 1 | X) = (1 / M) * sum_{m=1}^M P_hat_{C_m}(Y_j = 1 | X)

    This mitigates the fixed error propagation problem of a single CC without
    requiring explicit correlation-based ordering.

    Parameters:
        base_estimator: Estimator object or str (default=None)
            Base binary estimator (e.g. 'logistic', 'svm', 'mlp').
        n_chains: int (default=10)
            Number of Classifier Chain models in the ensemble.
        subsample: float (default=1.0)
            Fraction of samples used to train each chain (in (0.0, 1.0]).
            If 1.0 and bootstrap=False, uses the entire training set.
        bootstrap: bool (default=False)
            Whether to sample instances with replacement.
        random_state: int (default=42)
            Random seed for reproducibility.
        orders: Optional sequence of orders (default=None)
            Predefined chain orders. If None, generated via random permutations.
    """

    def __init__(
        self,
        base_estimator=None,
        n_chains: int = 10,
        subsample: float = 1.0,
        bootstrap: bool = False,
        random_state: int = 42,
        orders: Optional[Sequence[Sequence[int]]] = None,
    ):
        self.base_estimator = base_estimator
        self.n_chains = int(n_chains)
        self.subsample = float(subsample)
        self.bootstrap = bool(bootstrap)
        self.random_state = random_state
        self.orders = orders

        self.chains_: List[ClassifierChainClassifier] = []
        self.orders_: List[List[int]] = []
        self.n_labels_: int = 0
        self.n_features_in_: int = 0
        self.is_empty_: bool = False

    def fit(self, X, Y):
        """
        Fit M Classifier Chains with diverse label permutations.

        Parameters:
            X: np.ndarray of shape (n_samples, n_features)
            Y: np.ndarray of shape (n_samples, n_labels)

        Returns:
            self: fitted estimator
        """
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int32)

        if Y.ndim != 2:
            raise ValueError(f"Y must be 2D, got shape {Y.shape}")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Sample count mismatch: X has {X.shape[0]}, Y has {Y.shape[0]}"
            )

        n_samples, n_labels = Y.shape
        self.n_labels_ = n_labels
        self.n_features_in_ = X.shape[1]
        self.chains_ = []
        self.orders_ = []

        if n_labels == 0:
            self.is_empty_ = True
            return self

        self.is_empty_ = False
        rng = np.random.RandomState(self.random_state)

        for m in range(self.n_chains):
            chain_seed = (
                self.random_state + m * 31
                if self.random_state is not None
                else None
            )

            # Determine label order for this chain
            if self.orders is not None and m < len(self.orders):
                order_m = [int(lbl) for lbl in self.orders[m]]
            else:
                order_m = rng.permutation(n_labels).tolist()

            # Subsampling / Bootstrapping
            if self.bootstrap or self.subsample < 1.0:
                sub_size = max(4, int(np.round(self.subsample * n_samples)))
                sample_idx = rng.choice(n_samples, size=sub_size, replace=self.bootstrap)
                X_m = X[sample_idx]
                Y_m = Y[sample_idx]
            else:
                X_m = X
                Y_m = Y

            chain = ClassifierChainClassifier(
                base_estimator=self.base_estimator,
                order=order_m,
                random_state=chain_seed,
            )
            chain.fit(X_m, Y_m)
            self.chains_.append(chain)
            self.orders_.append(order_m)

        return self

    def predict_proba(self, X):
        """
        Predict posterior probabilities P(Y_j = 1 | X) by averaging across all M chains.

        Parameters:
            X: np.ndarray of shape (n_samples, n_features)

        Returns:
            Y_proba: np.ndarray of shape (n_samples, n_labels)
        """
        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]

        if self.is_empty_ or self.n_labels_ == 0:
            return np.zeros((n_samples, 0), dtype=np.float32)

        if not self.chains_:
            raise ValueError("EnsembleClassifierChainClassifier is not fitted.")

        probs_accum = np.zeros((n_samples, self.n_labels_), dtype=np.float64)
        for chain in self.chains_:
            probs_accum += chain.predict_proba(X)

        avg_probs = probs_accum / float(len(self.chains_))
        return np.clip(avg_probs, 0.0, 1.0).astype(np.float32)

    def predict(self, X):
        """
        Predict binary labels by thresholding ensemble probabilities at 0.5.

        Parameters:
            X: np.ndarray of shape (n_samples, n_features)

        Returns:
            Y_pred: np.ndarray of shape (n_samples, n_labels)
        """
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(np.int32)

    def decision_function(self, X):
        """
        Predict decision function scores (log-odds of ensemble probabilities).

        Parameters:
            X: np.ndarray of shape (n_samples, n_features)

        Returns:
            scores: np.ndarray of shape (n_samples, n_labels)
        """
        probs = np.clip(self.predict_proba(X), 1e-6, 1.0 - 1e-6)
        return np.log(probs / (1.0 - probs)).astype(np.float32)
