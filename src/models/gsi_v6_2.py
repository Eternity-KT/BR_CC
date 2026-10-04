"""
GSI-MLC-PA v6.2: Adaptive Stratification & BR Residual Error Conditional Dependency with Partial Abstention.

Reference:
    - meeting_summary.md (Core Section, v6.2)
    - spec/spec_v6_2.md

Core Architecture:
1. Multi-Stage 5-Fold Cross-Validation Peeling:
   - Evaluates candidate labels sequentially via 5-fold CV Out-Of-Fold (OOF).
   - Labels achieving Selective-F1 >= threshold (default 0.75) are promoted to IL[i].
   - Promoted IL labels expand the feature space: FS = FS U IL[i].
   - Singleton DL Rule: If |DL| == 1, the single remaining label is trained on current FS
     and promoted immediately to IL, terminating phase 1.
2. Residual Error Correlation & Conditional BR Coupling for DL:
   - For remaining dependent labels (len(DL) >= 2), base BR models f_l are evaluated on FS.
   - Prediction error residuals e_l = Y[:, l] - P_OOF[:, l] are computed.
   - Pairwise Pearson Correlation Coefficients (PCC) measure conditional dependencies:
     Corr(l, p) = PCC((l - f(l)), (p - f(p))).
   - Local dependency graph DL_temp[l] couples labels with Corr(l, p) >= tau_corr.
   - Individual conditioned BR models g_l are trained on FS[l] = FS(X, IL, DL_temp[l]).
3. Two-Stage Leakage-Free Inference with Partial Abstention:
   - Stage 1: P_IL predicted across layers; base P_DL^(base) predicted from f_p.
   - Stage 2: For l in DL, refined probabilities predicted from g_l([FS, P_DL_temp^(base)]).
   - Partial Abstention: Bayes-Optimal Decision Rule at rejection cost c (default 0.30):
     y_hat_j = 1 if p_j >= 1 - c, 0 if p_j <= c, -1 (abstain) if c < p_j < 1 - c.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from ..selection.cv_peeling import (
    CVPeelingConfig,
    CVPeelingResult,
    CVStratifiedPeelingSelector,
    _fit_predict_prob_binary,
    evaluate_label_5fold_cv,
)
from ..selection.stratified_peeling import normalize_augmented_probabilities
from ..selection.br_residual_correlation import (
    build_residual_dependency_graph,
    compute_br_residual_matrix,
    compute_residual_pcc_matrix,
    export_residual_correlation_audit_table,
)
from .base_learners import create_binary_estimator, create_multilabel_estimator
from .binary_relevance import _ConstantClassifier


def _fit_binary_model_safe(estimator_factory, X: np.ndarray, y: np.ndarray) -> Any:
    """Safely fit binary estimator, handling single-class degenerate edge cases."""
    unique = np.unique(y)
    if len(unique) <= 1:
        const_val = int(unique[0]) if len(unique) == 1 else 0
        return _ConstantClassifier(const_val)
    est = estimator_factory()
    est.fit(X, y)
    return est


class GSIMLCPAv6_2Classifier(BaseEstimator, ClassifierMixin):
    """
    GSI-MLC-PA Version 6.2 Classifier.

    Parameters:
        base_learner: str or callable (default='logistic')
            Base classification model: 'logistic', 'svm_calibrated', 'mlp'.
        stratified_threshold: float (default=0.75)
            Selective-F1 threshold required to promote a label to IL.
        residual_corr_threshold: float (default=0.25)
            PCC threshold to couple dependent labels in DL_temp[l].
        cv_folds: int (default=5)
            Number of cross-validation folds for out-of-fold peeling and DL evaluation.
        max_peeling_depth: int (default=3)
            Maximum peeling stages.
        peeling_metric: str (default='selective_f1')
            Metric used for promotion comparison: 'selective_f1', 'optimal_f1', 'standard_f1'.
        decaying_threshold: bool (default=False)
            Whether to decay threshold across peeling stages.
        threshold_decay_step: float (default=0.05)
            Decay step when decaying_threshold is True.
        aug_normalization: str (default='matching')
            Normalization for augmented soft probabilities: 'matching', 'centered', 'none'.
        error_format: str (default='residual')
            Format for residual calculation: 'residual', 'abs_residual', 'binary'.
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
        residual_corr_threshold: float = 0.25,
        cv_folds: int = 5,
        max_peeling_depth: int = 3,
        peeling_metric: str = "selective_f1",
        decaying_threshold: bool = False,
        threshold_decay_step: float = 0.05,
        aug_normalization: str = "matching",
        error_format: str = "residual",
        cost: float = 0.40,
        abstain_value: int = -1,
        random_state: int = 42,
    ):
        self.base_learner = base_learner
        self.stratified_threshold = float(stratified_threshold)
        self.residual_corr_threshold = float(residual_corr_threshold)
        self.cv_folds = int(cv_folds)
        self.max_peeling_depth = int(max_peeling_depth)
        self.peeling_metric = str(peeling_metric)
        self.decaying_threshold = bool(decaying_threshold)
        self.threshold_decay_step = float(threshold_decay_step)
        self.aug_normalization = str(aug_normalization)
        self.error_format = str(error_format)
        self.cost = float(cost)
        self.abstain_value = int(abstain_value)
        self.random_state = int(random_state)

        # Fitted attributes
        self.n_samples_: int = 0
        self.n_labels_: int = 0
        self.n_features_in_: int = 0
        self.independent_layers_: Tuple[Tuple[int, ...], ...] = ()
        self.independent_labels_: List[int] = []
        self.dependent_labels_: List[int] = []
        self.peeling_result_: Optional[CVPeelingResult] = None

        # Models storage
        self.br_il_layer_models_: List[Dict[int, Any]] = []
        self.br_base_dl_models_: Dict[int, Any] = {}
        self.br_cond_dl_models_: Dict[int, Any] = {}

        # Correlation and graph structures
        self.residual_corr_matrix_: Optional[np.ndarray] = None
        self.dependency_graph_: Dict[int, List[int]] = {}
        self.dl_error_rates_: Dict[int, float] = {}
        self.audit_table_: Optional[Any] = None
        self.is_fitted_: bool = False

    def _create_binary_model(self) -> Any:
        return create_binary_estimator(self.base_learner, random_state=self.random_state)

    def fit(self, X, Y):
        """
        Fit GSI-MLC-PA v6.2 according to meeting_summary.md:
        Phase 1: Do-While Layered Peeling for IL discovery (with singleton DL rule).
        Phase 2: BR Residual Error Correlation & Conditional BR Coupling for DL.
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

        # -------------------------------------------------------------------------
        # PHASE 1: Layered Peeling to discover Independent Labels (IL)
        # -------------------------------------------------------------------------
        peeling_cfg = CVPeelingConfig(
            threshold=self.stratified_threshold,
            n_folds=self.cv_folds,
            max_depth=self.max_peeling_depth,
            cost=self.cost,
            metric=self.peeling_metric,
            decaying_threshold=self.decaying_threshold,
            threshold_decay_step=self.threshold_decay_step,
            aug_normalization=self.aug_normalization,
            promote_singleton_dl=True,  # meeting_summary.md rule: |DL| == 1 => IL
            random_state=self.random_state,
        )

        selector = CVStratifiedPeelingSelector(
            config=peeling_cfg,
            base_estimator_factory=self._create_binary_model,
        )

        self.peeling_result_ = selector.fit_partition(X, Y)
        self.independent_layers_ = self.peeling_result_.independent_layers
        self.independent_labels_ = list(self.peeling_result_.all_independent_labels)
        self.dependent_labels_ = list(self.peeling_result_.dependent_residual_labels)

        # Train dedicated BR models for each layer in IL
        self.br_il_layer_models_ = []
        current_train_X = X
        accumulated_il_labels: List[int] = []

        for layer_idx, layer in enumerate(self.independent_layers_):
            layer_dict: Dict[int, Any] = {}
            for lbl in layer:
                est = _fit_binary_model_safe(self._create_binary_model, current_train_X, Y[:, lbl])
                layer_dict[lbl] = est
            self.br_il_layer_models_.append(layer_dict)

            # Update feature space for next layer using leakage-free OOF probabilities
            accumulated_il_labels.extend(layer)
            if accumulated_il_labels and self.peeling_result_.oof_probabilities:
                oof_cols = [
                    self.peeling_result_.oof_probabilities[l]
                    for l in accumulated_il_labels
                    if l in self.peeling_result_.oof_probabilities
                ]
                if oof_cols:
                    oof_raw = np.column_stack(oof_cols)
                    oof_norm = normalize_augmented_probabilities(
                        oof_raw, reference_X=X, strategy=self.aug_normalization
                    )
                    current_train_X = np.hstack([X, oof_norm])

        # Context features FS_context = [X, Normalize(P_IL_OOF)]
        FS_context = current_train_X

        # -------------------------------------------------------------------------
        # PHASE 2: Handle Residual Dependent Labels (DL) via Residual Correlation
        # -------------------------------------------------------------------------
        self.br_base_dl_models_ = {}
        self.br_cond_dl_models_ = {}
        self.dependency_graph_ = {lbl: [] for lbl in self.dependent_labels_}
        self.dl_error_rates_ = {}
        self.residual_corr_matrix_ = None

        n_dl = len(self.dependent_labels_)
        if n_dl >= 2:
            # Step 2.1: Train base BR models f_l on FS_context and collect OOF probabilities
            P_OOF_DL = np.zeros((n_samples, n_dl), dtype=np.float64)

            for idx_dl, lbl in enumerate(self.dependent_labels_):
                # 5-fold CV to get leakage-free OOF probabilities
                _, oof_p, diag = evaluate_label_5fold_cv(
                    X=FS_context,
                    y=Y[:, lbl],
                    estimator_factory=self._create_binary_model,
                    n_folds=self.cv_folds,
                    cost=self.cost,
                    metric=self.peeling_metric,
                    random_state=self.random_state + lbl,
                )
                P_OOF_DL[:, idx_dl] = oof_p

                # Train full base estimator on complete FS_context for test inference
                base_est = _fit_binary_model_safe(self._create_binary_model, FS_context, Y[:, lbl])
                self.br_base_dl_models_[lbl] = base_est

                # Compute mean absolute error
                self.dl_error_rates_[lbl] = float(np.mean(np.abs(Y[:, lbl] - oof_p)))

            # Step 2.2: Compute residual error matrix and Pearson Correlation Matrix
            residuals_dl = compute_br_residual_matrix(
                Y_true=Y[:, self.dependent_labels_],
                P_prob=P_OOF_DL,
                error_format=self.error_format,
            )

            self.residual_corr_matrix_ = compute_residual_pcc_matrix(
                residual_matrix=residuals_dl,
                use_absolute=True,
            )

            # Step 2.3: Build conditional dependency graph DL_temp[l]
            self.dependency_graph_ = build_residual_dependency_graph(
                corr_matrix=self.residual_corr_matrix_,
                dl_labels=self.dependent_labels_,
                threshold=self.residual_corr_threshold,
            )

            # Step 2.4: Train conditioned BR models g_l on FS[l] = FS(X, IL, DL_temp[l])
            dl_label_to_col = {lbl: i for i, lbl in enumerate(self.dependent_labels_)}

            for lbl in self.dependent_labels_:
                coupled_partners = self.dependency_graph_.get(lbl, [])

                if coupled_partners:
                    # Collect OOF probabilities of coupled dependent partners
                    partner_cols = [dl_label_to_col[p] for p in coupled_partners]
                    P_coupled_oof = P_OOF_DL[:, partner_cols]
                    P_coupled_norm = normalize_augmented_probabilities(
                        P_coupled_oof, reference_X=X, strategy=self.aug_normalization
                    )
                    # FS[l] = [FS_context, Normalize(P_coupled)]
                    FS_l_train = np.hstack([FS_context, P_coupled_norm])
                else:
                    # If DL_temp[l] is empty, FS[l] = FS_context
                    FS_l_train = FS_context

                # Train g_l
                cond_est = _fit_binary_model_safe(self._create_binary_model, FS_l_train, Y[:, lbl])
                self.br_cond_dl_models_[lbl] = cond_est

            # Export audit table
            self.audit_table_ = export_residual_correlation_audit_table(
                corr_matrix=self.residual_corr_matrix_,
                dl_labels=self.dependent_labels_,
                dependency_graph=self.dependency_graph_,
                error_rates=self.dl_error_rates_,
            )

        elif n_dl == 1:
            # Fallback for len(DL) == 1 if promote_singleton_dl was not triggered
            single_lbl = self.dependent_labels_[0]
            est = _fit_binary_model_safe(self._create_binary_model, FS_context, Y[:, single_lbl])
            self.br_base_dl_models_[single_lbl] = est
            self.br_cond_dl_models_[single_lbl] = est
            self.dependency_graph_[single_lbl] = []

        self.is_fitted_ = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        """
        Two-stage inference for test samples:
        1. Predict independent layers sequentially: P_IL.
        2. Construct FS_context_test = [X_test, Normalize(P_IL_test)].
        3. Predict base probabilities P_DL^(base) from f_p.
        4. Predict refined probabilities for l in DL from g_l([FS_context_test, P_DL_temp^(base)]).
        """
        if not self.is_fitted_:
            raise ValueError("GSIMLCPAv6_2Classifier is not fitted yet.")

        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]
        P_final = np.zeros((n_samples, self.n_labels_), dtype=np.float32)

        # -------------------------------------------------------------------------
        # STAGE 1: Predict Independent Labels (IL) across layers
        # -------------------------------------------------------------------------
        current_test_X = X
        accumulated_il_test_probs: List[np.ndarray] = []

        for layer_idx, layer in enumerate(self.independent_layers_):
            layer_models = self.br_il_layer_models_[layer_idx]
            for lbl in layer:
                model = layer_models[lbl]
                p_lbl = self._predict_prob_single(model, current_test_X)
                P_final[:, lbl] = p_lbl
                accumulated_il_test_probs.append(p_lbl)

            # Update feature space for next layer
            if accumulated_il_test_probs:
                p_raw = np.column_stack(accumulated_il_test_probs)
                p_norm = normalize_augmented_probabilities(
                    p_raw, reference_X=X, strategy=self.aug_normalization
                )
                current_test_X = np.hstack([X, p_norm])

        FS_context_test = current_test_X

        # -------------------------------------------------------------------------
        # STAGE 2: Predict Dependent Labels (DL)
        # -------------------------------------------------------------------------
        n_dl = len(self.dependent_labels_)
        if n_dl > 0:
            # Step 2.1: Base predictions f_p(FS_context_test)
            P_base_DL = {}
            for lbl in self.dependent_labels_:
                base_model = self.br_base_dl_models_[lbl]
                P_base_DL[lbl] = self._predict_prob_single(base_model, FS_context_test)

            # Step 2.2: Refined predictions g_l(FS[l]_test)
            for lbl in self.dependent_labels_:
                coupled_partners = self.dependency_graph_.get(lbl, [])
                cond_model = self.br_cond_dl_models_.get(lbl, self.br_base_dl_models_[lbl])

                if coupled_partners:
                    # Form FS[l]_test = [FS_context_test, Normalize(P_coupled_base)]
                    coupled_cols = [P_base_DL[p] for p in coupled_partners]
                    P_coupled_raw = np.column_stack(coupled_cols)
                    P_coupled_norm = normalize_augmented_probabilities(
                        P_coupled_raw, reference_X=X, strategy=self.aug_normalization
                    )
                    FS_l_test = np.hstack([FS_context_test, P_coupled_norm])
                    p_final_l = self._predict_prob_single(cond_model, FS_l_test)
                else:
                    # No coupled partners, use base prediction directly
                    p_final_l = P_base_DL[lbl]

                P_final[:, lbl] = p_final_l

        return np.clip(P_final, 0.0, 1.0)

    @staticmethod
    def _predict_prob_single(estimator: Any, X_eval: np.ndarray) -> np.ndarray:
        """Helper to safely extract 1D positive class probabilities from estimator."""
        if hasattr(estimator, "predict_proba"):
            probs = np.asarray(estimator.predict_proba(X_eval), dtype=np.float64)
            if probs.ndim == 2:
                return np.clip(probs[:, 1], 0.0, 1.0).astype(np.float32)
            return np.clip(probs.ravel(), 0.0, 1.0).astype(np.float32)
        elif hasattr(estimator, "decision_function"):
            scores = np.asarray(estimator.decision_function(X_eval), dtype=np.float64)
            return np.clip(1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0))), 0.0, 1.0).astype(np.float32)
        else:
            preds = np.asarray(estimator.predict(X_eval), dtype=np.float64)
            return np.clip(preds.ravel(), 0.0, 1.0).astype(np.float32)

    def predict(self, X, cost: Optional[float] = None) -> np.ndarray:
        """
        Predict multilabel decisions with Bayes-Optimal Partial Abstention:
        y_hat_j = 1 if p_j >= 1 - c,
        y_hat_j = 0 if p_j <= c,
        y_hat_j = abstain_value (-1) otherwise.

        Parameters:
            X: Feature matrix of shape (n_samples, n_features).
            cost: Optional override for rejection cost c. If None, uses self.cost.

        Returns:
            Y_pred: Matrix of shape (n_samples, n_labels) with values in {0, 1, abstain_value}.
        """
        c = float(self.cost if cost is None else cost)
        probs = self.predict_proba(X)

        predictions = np.full(probs.shape, self.abstain_value, dtype=np.int32)
        predictions[probs >= (1.0 - c)] = 1
        predictions[probs <= c] = 0
        return predictions

    def predict_full(self, X, threshold: float = 0.5) -> np.ndarray:
        """Predict standard binary multilabel decisions without abstention."""
        probs = self.predict_proba(X)
        return (probs >= float(threshold)).astype(np.int32)
