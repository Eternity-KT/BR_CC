"""
GSI-MLC-PA v6.1: Adaptive Stratification & Ensemble of Classifier Chains (ECC) with Partial Abstention.

Reference:
    - meeting_summary.md (Core Section, Direction 1: v6.1)
    - spec/spec_v6_1.md

Core Innovations:
1. 5-Fold Cross-Validation Peeling Out-Of-Fold for finding Independent Labels (IL).
2. Singleton DL Promotion Rule: When len(DL) == 1, that single label is promoted directly into IL.
3. Training DL model:
   - For DL with len(DL) >= 2, uses Ensemble of Classifier Chains (ECC) with M random chain orders.
   - Data for DL: X_aug = [X, Normalize(P_IL_OOF)] without data leakage.
   - If len(DL) == 0 (all labels in IL), whole prediction is handled cleanly by BR.
4. Inference with Partial Abstention:
   - P_IL predicted independently via BR.
   - P_DL predicted via ECC with averaged probabilities across M chains.
   - Bayes-Optimal Decision Rule at rejection cost c (default 0.30):
     y_hat_j = 1 if p_j >= 1 - c, 0 if p_j <= c, -1 (abstain) if c < p_j < 1 - c.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from ..selection.cv_peeling import (
    CVPeelingConfig,
    CVPeelingResult,
    CVStratifiedPeelingSelector,
)
from ..selection.stratified_peeling import normalize_augmented_probabilities
from .base_learners import create_binary_estimator, create_multilabel_estimator
from .ensemble_classifier_chain import EnsembleClassifierChainClassifier


class GSIMLCPAv6_1Classifier(BaseEstimator, ClassifierMixin):
    """
    GSI-MLC-PA Version 6.1 Classifier.

    Parameters:
        base_learner: str or callable (default='logistic')
            Base classification model: 'logistic', 'svm_calibrated', 'mlp'.
        stratified_threshold: float (default=0.75)
            Selective-F1 threshold required to promote a label to IL.
        cv_folds: int (default=5)
            Number of cross-validation folds for out-of-fold peeling evaluation.
        max_peeling_depth: int (default=3)
            Maximum peeling stages.
        peeling_metric: str (default='selective_f1')
            Metric used for promotion comparison: 'selective_f1', 'optimal_f1', 'standard_f1'.
        decaying_threshold: bool (default=False)
            Whether to decay threshold across peeling stages.
        threshold_decay_step: float (default=0.05)
            Decay step when decaying_threshold is True.
        aug_normalization: str (default='matching')
            Normalization for augmented soft probabilities: 'matching', 'centered', etc.
        n_chains: int (default=10)
            Number of random Classifier Chains in the ECC model on DL.
        ecc_subsample: float (default=1.0)
            Subsample fraction for each chain in ECC.
        ecc_bootstrap: bool (default=False)
            Whether ECC samples instances with replacement.
        cost: float (default=0.30)
            Rejection cost c for Bayes-optimal partial abstention.
        abstain_value: int (default=-1)
            Integer sentinel representing abstention.
        random_state: int (default=42)
            Random seed for reproducibility.
    """

    def __init__(
        self,
        base_learner: str = "logistic",
        stratified_threshold: float = 0.75,
        cv_folds: int = 5,
        max_peeling_depth: int = 3,
        peeling_metric: str = "selective_f1",
        decaying_threshold: bool = False,
        threshold_decay_step: float = 0.05,
        aug_normalization: str = "matching",
        n_chains: int = 10,
        ecc_subsample: float = 1.0,
        ecc_bootstrap: bool = False,
        cost: float = 0.30,
        abstain_value: int = -1,
        random_state: int = 42,
    ):
        self.base_learner = base_learner
        self.stratified_threshold = float(stratified_threshold)
        self.cv_folds = int(cv_folds)
        self.max_peeling_depth = int(max_peeling_depth)
        self.peeling_metric = str(peeling_metric)
        self.decaying_threshold = bool(decaying_threshold)
        self.threshold_decay_step = float(threshold_decay_step)
        self.aug_normalization = str(aug_normalization)
        self.n_chains = int(n_chains)
        self.ecc_subsample = float(ecc_subsample)
        self.ecc_bootstrap = bool(ecc_bootstrap)
        self.cost = float(cost)
        self.abstain_value = int(abstain_value)
        self.random_state = int(random_state)

        # Fitted attributes
        self.n_labels_: int = 0
        self.n_features_in_: int = 0
        self.independent_layers_: Tuple[Tuple[int, ...], ...] = ()
        self.independent_labels_: List[int] = []
        self.dependent_labels_: List[int] = []
        self.peeling_result_: Optional[CVPeelingResult] = None
        self.br_model_: Any = None
        self.ecc_model_: Optional[EnsembleClassifierChainClassifier] = None
        self.is_fitted_: bool = False

    def fit(self, X, Y):
        """
        Fit GSI-MLC-PA v6.1 end-to-end:
        1. Run 5-fold CV Peeling with singleton DL promotion.
        2. Fit BR on IL labels.
        3. Fit ECC on DL labels with augmented features X_aug = [X, P_IL_OOF].
        """
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int32)

        if Y.ndim != 2:
            raise ValueError(f"Y must be 2D binary matrix, got shape {Y.shape}")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Sample count mismatch: X has {X.shape[0]}, Y has {Y.shape[0]}"
            )

        n_samples, n_labels = Y.shape
        self.n_samples_ = n_samples
        self.n_labels_ = n_labels
        self.n_features_in_ = X.shape[1]

        # Step 1: 5-Fold CV Peeling with Singleton DL Promotion
        peeling_cfg = CVPeelingConfig(
            threshold=self.stratified_threshold,
            n_folds=self.cv_folds,
            max_depth=self.max_peeling_depth,
            cost=self.cost,
            metric=self.peeling_metric,
            decaying_threshold=self.decaying_threshold,
            threshold_decay_step=self.threshold_decay_step,
            aug_normalization=self.aug_normalization,
            promote_singleton_dl=True,  # Critical v6.1 rule: len(DL) == 1 => IL
            random_state=self.random_state,
        )

        selector = CVStratifiedPeelingSelector(
            config=peeling_cfg,
            base_estimator_factory=lambda: create_binary_estimator(
                self.base_learner, random_state=self.random_state
            ),
        )

        self.peeling_result_ = selector.fit_partition(X, Y)
        self.independent_layers_ = self.peeling_result_.independent_layers
        self.independent_labels_ = list(self.peeling_result_.all_independent_labels)
        self.dependent_labels_ = list(self.peeling_result_.dependent_residual_labels)

        # Step 2: Fit BR on all labels (or specifically independent labels)
        if len(self.independent_labels_) > 0:
            self.br_model_ = create_multilabel_estimator(
                self.base_learner, random_state=self.random_state
            )
            self.br_model_.fit(X, Y)
        else:
            self.br_model_ = None

        # Step 3: Fit ECC on DL if at least 2 dependent labels remain
        if len(self.dependent_labels_) >= 2:
            # Build augmented training features using OOF probabilities from peeling
            if len(self.independent_labels_) > 0 and self.peeling_result_.oof_probabilities:
                oof_cols = []
                for lbl in self.independent_labels_:
                    if lbl in self.peeling_result_.oof_probabilities:
                        oof_cols.append(self.peeling_result_.oof_probabilities[lbl])
                    else:
                        # Fallback to in-sample BR probability if missing
                        oof_cols.append(self.br_model_.predict_proba(X)[:, lbl])
                oof_raw = np.column_stack(oof_cols)
                oof_norm = normalize_augmented_probabilities(
                    oof_raw, reference_X=X, strategy=self.aug_normalization
                )
                X_aug_train = np.hstack([X, oof_norm])
            else:
                X_aug_train = X

            Y_dl_train = Y[:, self.dependent_labels_]

            self.ecc_model_ = EnsembleClassifierChainClassifier(
                base_estimator=create_binary_estimator(
                    self.base_learner, random_state=self.random_state
                ),
                n_chains=self.n_chains,
                subsample=self.ecc_subsample,
                bootstrap=self.ecc_bootstrap,
                random_state=self.random_state,
            )
            self.ecc_model_.fit(X_aug_train, Y_dl_train)
        else:
            self.ecc_model_ = None

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        """
        Predict full label probability matrix (N, K):
        - For labels in IL: P from BR model
        - For labels in DL: P from ECC model averaged across M chains
        """
        if not self.is_fitted_:
            raise ValueError("GSIMLCPAv6_1Classifier is not fitted yet.")

        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]
        P_final = np.zeros((n_samples, self.n_labels_), dtype=np.float32)

        # 1. Predictions for Independent Labels (IL) from BR
        if self.br_model_ is not None and len(self.independent_labels_) > 0:
            P_br = self.br_model_.predict_proba(X)
            for lbl in self.independent_labels_:
                P_final[:, lbl] = P_br[:, lbl]

        # 2. Predictions for Dependent Labels (DL) from ECC
        if self.ecc_model_ is not None and len(self.dependent_labels_) >= 2:
            # Build augmented test features: [X_test, Normalize(P_IL_test)]
            if len(self.independent_labels_) > 0:
                P_il_test = P_final[:, self.independent_labels_]
                P_il_norm = normalize_augmented_probabilities(
                    P_il_test, reference_X=X, strategy=self.aug_normalization
                )
                X_aug_test = np.hstack([X, P_il_norm])
            else:
                X_aug_test = X

            P_dl_test = self.ecc_model_.predict_proba(X_aug_test)
            for idx, lbl in enumerate(self.dependent_labels_):
                P_final[:, lbl] = P_dl_test[:, idx]

        return np.clip(P_final, 0.0, 1.0)

    def predict(self, X, cost: Optional[float] = None):
        """
        Predict labels with partial abstention using Bayes-Optimal Prediction:
        y_hat_j = 1 if p_j >= 1 - c,
        y_hat_j = 0 if p_j <= c,
        y_hat_j = abstain_value (-1) otherwise.
        """
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
