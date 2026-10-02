"""
GSI-MLC-PA v6.1.1: Stratified Peeling (v5.1 partition) + ECC on Residual DL.

Combines:
1. v5.1 Data-Driven Stratified Peeling Selector (StratifiedPeelingSelector with tau=0.70 on internal train/validation split).
2. Binary Relevance on Independent Labels (IL).
3. Ensemble of Classifier Chains (ECC, M=10 chains with diverse label permutations)
   on Residual Dependent Labels (DL) with augmented features X_aug = [X, Normalize(P_IL)].
4. Bayes-Optimal Prediction (BOP) with partial abstention at cost c = 0.30.

This model allows an exact, controlled comparison between:
- v5.1: Peeling + Dense CC (single ascending chain)
- v5.1.1: Peeling + Sparse CC (single thresholded chain theta=0.75)
- v6.1.1: Peeling + ECC (10 random chains with probability pooling)
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import train_test_split

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
except ImportError:
    MultilabelStratifiedShuffleSplit = None

from .base_learners import create_binary_estimator
from .binary_relevance import BinaryRelevanceClassifier
from .ensemble_classifier_chain import EnsembleClassifierChainClassifier
from ..selection.stratified_peeling import (
    StratifiedPeelingConfig,
    StratifiedPeelingSelector,
    StratifiedPeelingResult,
    normalize_augmented_probabilities,
)


class GSIMLCPAv6_1_1Classifier(BaseEstimator, ClassifierMixin):
    """
    GSI-MLC-PA v6.1.1: v5.1 Stratified Peeling + ECC on Dependent Labels.

    Parameters:
        base_learner: str or estimator (default='logistic')
            Base classifier: 'logistic', 'svm_calibrated', 'mlp'.
        stratified_threshold: float (default=0.70)
            Peeling threshold tau for independent label promotion.
        validation_size: float (default=0.20)
            Fraction of training data used for internal validation in peeling.
        max_peeling_depth: int (default=3)
            Maximum peeling stages.
        aug_normalization: str (default='matching')
            Normalization for augmented probability features: 'matching', 'centered', 'none'.
        n_chains: int (default=10)
            Number of random chains in ECC.
        cost: float (default=0.30)
            Abstention cost c for Bayes-Optimal Prediction.
        abstain_value: int (default=-1)
            Sentinel value for abstained predictions.
        random_state: int (default=42)
            Random seed for reproducibility.
    """

    def __init__(
        self,
        base_learner="logistic",
        stratified_threshold: float = 0.70,
        validation_size: float = 0.20,
        max_peeling_depth: int = 3,
        aug_normalization: str = "matching",
        n_chains: int = 10,
        cost: float = 0.30,
        abstain_value: int = -1,
        random_state: int = 42,
    ):
        self.base_learner = base_learner
        self.stratified_threshold = float(stratified_threshold)
        self.validation_size = float(validation_size)
        self.max_peeling_depth = int(max_peeling_depth)
        self.aug_normalization = str(aug_normalization)
        self.n_chains = int(n_chains)
        self.cost = float(cost)
        self.abstain_value = int(abstain_value)
        self.random_state = int(random_state)

        # Fitted attributes
        self.n_labels_: int = 0
        self.n_features_in_: int = 0
        self.independent_labels_: List[int] = []
        self.dependent_labels_: List[int] = []
        self.peeling_result_: Optional[StratifiedPeelingResult] = None
        self.br_model_: Optional[BinaryRelevanceClassifier] = None
        self.ecc_model_: Optional[EnsembleClassifierChainClassifier] = None
        self.single_dl_model_: Any = None
        self.is_fitted_: bool = False

    def _split_train_validation(self, X: np.ndarray, Y: np.ndarray):
        """Perform stratified train/validation split matching v5.1."""
        indices = np.arange(X.shape[0])
        if MultilabelStratifiedShuffleSplit is not None:
            try:
                splitter = MultilabelStratifiedShuffleSplit(
                    n_splits=1,
                    test_size=self.validation_size,
                    random_state=self.random_state,
                )
                train_idx, val_idx = next(splitter.split(X, Y))
                return train_idx, val_idx
            except (TypeError, ValueError):
                pass
        return train_test_split(
            indices,
            test_size=self.validation_size,
            random_state=self.random_state,
            shuffle=True,
        )

    def fit(self, X, Y):
        """
        Fit GSI-MLC-PA v6.1.1:
        1. Run v5.1 Stratified Peeling Selector on internal train/validation split.
        2. Fit BR on IL labels using full X.
        3. Fit ECC on DL labels using X_aug = [X, Normalize(P_IL)].
        """
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int32)

        if Y.ndim != 2:
            raise ValueError(f"Y must be 2D, got shape {Y.shape}")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Sample count mismatch: X={X.shape[0]}, Y={Y.shape[0]}")

        n_samples, n_labels = Y.shape
        self.n_samples_ = n_samples
        self.n_labels_ = n_labels
        self.n_features_in_ = X.shape[1]

        # Step 1: Stratified Peeling (v5.1 mechanism)
        train_idx, val_idx = self._split_train_validation(X, Y)
        X_subtrain, Y_subtrain = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]

        peeling_cfg = StratifiedPeelingConfig(
            threshold=self.stratified_threshold,
            max_depth=self.max_peeling_depth,
            dl_order_direction="ascending",
            aug_normalization=self.aug_normalization,
        )
        peeling_selector = StratifiedPeelingSelector(
            config=peeling_cfg,
            base_estimator_factory=lambda: create_binary_estimator(
                self.base_learner, random_state=self.random_state
            ),
        )
        peeling_res = peeling_selector.fit_partition(
            X_subtrain, Y_subtrain, X_val, Y_val
        )
        self.peeling_result_ = peeling_res
        self.independent_labels_ = sorted(list(peeling_res.all_independent_labels))
        self.dependent_labels_ = sorted(list(peeling_res.dependent_residual_labels))

        # Step 2: Fit BR on full X for Independent Labels (IL)
        if len(self.independent_labels_) > 0:
            self.br_model_ = BinaryRelevanceClassifier(
                base_estimator=self.base_learner,
                random_state=self.random_state,
            )
            self.br_model_.fit(X, Y)
            # Probability predictions for IL to augment features for DL
            P_il_train = self.br_model_.predict_proba(X)[:, self.independent_labels_]
            P_il_norm = normalize_augmented_probabilities(
                P_il_train, reference_X=X, strategy=self.aug_normalization
            )
            X_aug_train = np.hstack([X, P_il_norm])
        else:
            self.br_model_ = None
            X_aug_train = X

        # Step 3: Fit ECC on Dependent Labels (DL)
        n_dl = len(self.dependent_labels_)
        if n_dl >= 2:
            Y_dl_train = Y[:, self.dependent_labels_]
            self.ecc_model_ = EnsembleClassifierChainClassifier(
                base_estimator=self.base_learner,
                n_chains=self.n_chains,
                random_state=self.random_state,
            )
            self.ecc_model_.fit(X_aug_train, Y_dl_train)
            self.single_dl_model_ = None
        elif n_dl == 1:
            self.ecc_model_ = None
            dl_lbl = self.dependent_labels_[0]
            single_clf = create_binary_estimator(
                self.base_learner, random_state=self.random_state
            )
            single_clf.fit(X_aug_train, Y[:, dl_lbl])
            self.single_dl_model_ = single_clf
        else:
            self.ecc_model_ = None
            self.single_dl_model_ = None

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        """
        Predict probability matrix (N, K):
        - For labels in IL: P from BR model
        - For labels in DL: P from ECC model averaged across M chains
        """
        if not self.is_fitted_:
            raise ValueError("GSIMLCPAv6_1_1Classifier is not fitted.")

        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]
        P_final = np.zeros((n_samples, self.n_labels_), dtype=np.float32)

        # 1. IL probabilities from BR
        if self.br_model_ is not None and len(self.independent_labels_) > 0:
            P_br = self.br_model_.predict_proba(X)
            for lbl in self.independent_labels_:
                P_final[:, lbl] = P_br[:, lbl]

        # 2. Build augmented features for DL
        if len(self.independent_labels_) > 0:
            P_il_test = P_final[:, self.independent_labels_]
            P_il_norm = normalize_augmented_probabilities(
                P_il_test, reference_X=X, strategy=self.aug_normalization
            )
            X_aug_test = np.hstack([X, P_il_norm])
        else:
            X_aug_test = X

        # 3. DL probabilities from ECC
        n_dl = len(self.dependent_labels_)
        if n_dl >= 2 and self.ecc_model_ is not None:
            P_dl = self.ecc_model_.predict_proba(X_aug_test)
            for idx, lbl in enumerate(self.dependent_labels_):
                P_final[:, lbl] = P_dl[:, idx]
        elif n_dl == 1 and self.single_dl_model_ is not None:
            dl_lbl = self.dependent_labels_[0]
            if hasattr(self.single_dl_model_, "predict_proba"):
                probs = self.single_dl_model_.predict_proba(X_aug_test)
                P_final[:, dl_lbl] = probs[:, 1] if probs.ndim == 2 else probs
            elif hasattr(self.single_dl_model_, "decision_function"):
                scores = self.single_dl_model_.decision_function(X_aug_test)
                P_final[:, dl_lbl] = 1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0)))
            else:
                P_final[:, dl_lbl] = self.single_dl_model_.predict(X_aug_test)

        return np.clip(P_final, 0.0, 1.0)

    def predict(self, X, cost: Optional[float] = None):
        """Bayes-Optimal Prediction with partial abstention at cost c."""
        eff_cost = self.cost if cost is None else float(cost)
        probs = self.predict_proba(X)

        Y_pred = np.full(probs.shape, self.abstain_value, dtype=np.int32)
        decided_0 = probs <= eff_cost
        decided_1 = probs >= (1.0 - eff_cost)

        Y_pred[decided_0] = 0
        Y_pred[decided_1] = 1
        return Y_pred

    def predict_full(self, X):
        """Standard complete prediction thresholded at 0.5 without abstention."""
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(np.int32)

    def decision_function(self, X):
        """Log-odds decision scores."""
        probs = np.clip(self.predict_proba(X), 1e-6, 1.0 - 1e-6)
        return np.log(probs / (1.0 - probs)).astype(np.float32)
