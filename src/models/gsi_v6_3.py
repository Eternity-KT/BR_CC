"""
GSI-MLC-PA v6.3: Prior-Calibrated Asymmetric Negative Verification & Adaptive Confidence Abstention for Extremely Imbalanced Dependent Labels.

Reference:
    - spec/spec_v6_3.md
    - meeting_summary.md

Architectural Enhancements over v6.2.1:
1. Standardized Residual Coupling Threshold (tau_corr = 0.25):
   Couples conditional dependencies between remaining DL labels using Pearson Correlation of residuals.
2. Tail Probability Calibration:
   Applies weighted Platt/Beta calibration on Out-Of-Fold probabilities to rectify tail miscalibration on rare labels.
3. Prior-Calibrated Asymmetric Decision Rule (Bayes Likelihood Ratio):
   Calculates label-specific acceptance thresholds [tau_0(l), tau_1(l)] driven by prior density pi_l.
   - High confidence in negative class -> predict 0.
   - Significant likelihood ratio support -> predict 1.
   - Equivocal / rejection zone -> abstain (-1).
4. Coverage Guard:
   Guarantees minimum empirical decision coverage (gamma_min >= 0.70) to prevent rare-class positive samples
   from being completely swallowed by symmetric rejection bands.
5. Strict Feature Purity Invariance:
   Only continuous soft probabilities are propagated across stages; discrete labels and abstentions are strictly isolated.
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
from ..selection.asymmetric_decision import (
    apply_asymmetric_abstention,
    compute_adaptive_thresholds_table,
    compute_bayes_lr_thresholds,
)
from ..calibration.tail_calibrator import MultiLabelTailCalibrator
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


class GSIMLCPAv6_3Classifier(BaseEstimator, ClassifierMixin):
    """
    GSI-MLC-PA Version 6.3 Classifier with Prior-Calibrated Asymmetric Negative Verification.

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
        decaying_threshold: bool (default=True)
            Whether to decay threshold across peeling stages.
        threshold_decay_step: float (default=0.05)
            Decay step when decaying_threshold is True.
        aug_normalization: str (default='matching')
            Normalization for augmented soft probabilities: 'matching', 'centered', 'none'.
        error_format: str (default='residual')
            Format for residual calculation: 'residual', 'abs_residual', 'binary'.
        cost: float (default=0.30)
            Rejection cost c for Bayes decision rule.
        gamma_min: float (default=0.70)
            Minimum coverage guard floor for each label.
        use_prior_adaptive: bool (default=True)
            Whether to use prior-calibrated asymmetric thresholds instead of static Chow rule.
        calibrate_tail: bool (default=True)
            Whether to apply tail probability calibration on OOF predictions.
        abstain_value: int (default=-1)
            Integer sentinel representing abstention.
        random_state: int (default=42)
    """

    def __init__(
        self,
        base_learner: str = "logistic",
        stratified_threshold: float = 0.75,
        residual_corr_threshold: float = 0.25,
        cv_folds: int = 5,
        max_peeling_depth: int = 3,
        peeling_metric: str = "selective_f1",
        decaying_threshold: bool = True,
        threshold_decay_step: float = 0.05,
        aug_normalization: str = "matching",
        error_format: str = "residual",
        cost: float = 0.30,
        gamma_min: float = 0.70,
        use_prior_adaptive: bool = True,
        calibrate_tail: bool = True,
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
        self.gamma_min = float(gamma_min)
        self.use_prior_adaptive = bool(use_prior_adaptive)
        self.calibrate_tail = bool(calibrate_tail)
        self.abstain_value = int(abstain_value)
        self.random_state = int(random_state)
        self.is_fitted_ = False

    def _create_binary_model(self):
        """Helper to create binary estimator matching base_learner."""
        return create_binary_estimator(self.base_learner, random_state=self.random_state)

    def fit(self, X, Y) -> "GSIMLCPAv6_3Classifier":
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
            promote_singleton_dl=True,
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

        P_OOF_all = np.zeros((n_samples, n_labels), dtype=np.float64)

        for layer_idx, layer in enumerate(self.independent_layers_):
            layer_dict: Dict[int, Any] = {}
            for lbl in layer:
                est = _fit_binary_model_safe(self._create_binary_model, current_train_X, Y[:, lbl])
                layer_dict[lbl] = est
                if self.peeling_result_.oof_probabilities and lbl in self.peeling_result_.oof_probabilities:
                    P_OOF_all[:, lbl] = self.peeling_result_.oof_probabilities[lbl]
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
                P_OOF_all[:, lbl] = oof_p

                base_est = _fit_binary_model_safe(self._create_binary_model, FS_context, Y[:, lbl])
                self.br_base_dl_models_[lbl] = base_est
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

            # Step 2.4: Train conditioned BR models g_l on FS[l]
            dl_label_to_col = {lbl: i for i, lbl in enumerate(self.dependent_labels_)}

            for lbl in self.dependent_labels_:
                coupled_partners = self.dependency_graph_.get(lbl, [])

                if coupled_partners:
                    partner_cols = [dl_label_to_col[p] for p in coupled_partners]
                    P_coupled_oof = P_OOF_DL[:, partner_cols]
                    P_coupled_norm = normalize_augmented_probabilities(
                        P_coupled_oof, reference_X=X, strategy=self.aug_normalization
                    )
                    FS_l_train = np.hstack([FS_context, P_coupled_norm])
                else:
                    FS_l_train = FS_context

                cond_est = _fit_binary_model_safe(self._create_binary_model, FS_l_train, Y[:, lbl])
                self.br_cond_dl_models_[lbl] = cond_est

                # Re-evaluate OOF conditioned probability if coupled
                if coupled_partners:
                    _, oof_p_cond, _ = evaluate_label_5fold_cv(
                        X=FS_l_train,
                        y=Y[:, lbl],
                        estimator_factory=self._create_binary_model,
                        n_folds=self.cv_folds,
                        cost=self.cost,
                        metric=self.peeling_metric,
                        random_state=self.random_state + lbl * 11,
                    )
                    P_OOF_all[:, lbl] = oof_p_cond

            self.audit_table_ = export_residual_correlation_audit_table(
                corr_matrix=self.residual_corr_matrix_,
                dl_labels=self.dependent_labels_,
                dependency_graph=self.dependency_graph_,
                error_rates=self.dl_error_rates_,
            )

        elif n_dl == 1:
            single_lbl = self.dependent_labels_[0]
            est = _fit_binary_model_safe(self._create_binary_model, FS_context, Y[:, single_lbl])
            self.br_base_dl_models_[single_lbl] = est
            self.br_cond_dl_models_[single_lbl] = est
            self.dependency_graph_[single_lbl] = []

            _, oof_p_single, _ = evaluate_label_5fold_cv(
                X=FS_context,
                y=Y[:, single_lbl],
                estimator_factory=self._create_binary_model,
                n_folds=self.cv_folds,
                cost=self.cost,
                metric=self.peeling_metric,
                random_state=self.random_state + single_lbl,
            )
            P_OOF_all[:, single_lbl] = oof_p_single

        # -------------------------------------------------------------------------
        # PHASE 2.5: Tail Probability Calibration
        # -------------------------------------------------------------------------
        self.calibrator_: Optional[MultiLabelTailCalibrator] = None
        if self.calibrate_tail:
            self.calibrator_ = MultiLabelTailCalibrator(
                method="weighted_platt",
                random_state=self.random_state,
            )
            self.calibrator_.fit(P_OOF_all, Y)
            P_OOF_all = self.calibrator_.transform(P_OOF_all)

        self.oof_probabilities_all_ = P_OOF_all

        # -------------------------------------------------------------------------
        # PHASE 3: Prior-Calibrated Asymmetric Thresholds Table
        # -------------------------------------------------------------------------
        if self.use_prior_adaptive:
            self.adaptive_thresholds_ = compute_adaptive_thresholds_table(
                Y_train=Y,
                OOF_probs=P_OOF_all,
                cost=self.cost,
                gamma_min=self.gamma_min,
            )
        else:
            self.adaptive_thresholds_ = {
                j: {
                    "tau_0": float(self.cost),
                    "tau_1": float(1.0 - self.cost),
                    "prior": float(np.mean(Y[:, j])),
                    "coverage_guard": False,
                    "empirical_coverage": 1.0,
                }
                for j in range(n_labels)
            }

        self.is_fitted_ = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        """
        Two-stage inference for test samples:
        1. Predict independent layers sequentially: P_IL.
        2. Construct FS_context_test = [X_test, Normalize(P_IL_test)].
        3. Predict base probabilities P_DL^(base) from f_p.
        4. Predict refined probabilities for l in DL from g_l([FS_context_test, P_DL_temp^(base)]).
        5. Apply tail probability calibration if configured.
        """
        if not self.is_fitted_:
            raise ValueError("GSIMLCPAv6_3Classifier is not fitted yet.")

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
                p_l = self._predict_prob_single(model, current_test_X)
                P_final[:, lbl] = p_l
                accumulated_il_test_probs.append(p_l)

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
                    coupled_cols = [P_base_DL[p] for p in coupled_partners]
                    P_coupled_raw = np.column_stack(coupled_cols)
                    P_coupled_norm = normalize_augmented_probabilities(
                        P_coupled_raw, reference_X=X, strategy=self.aug_normalization
                    )
                    FS_l_test = np.hstack([FS_context_test, P_coupled_norm])
                    p_final_l = self._predict_prob_single(cond_model, FS_l_test)
                else:
                    p_final_l = P_base_DL[lbl]

                P_final[:, lbl] = p_final_l

        # Apply tail calibration
        if self.calibrate_tail and self.calibrator_ is not None:
            P_final = self.calibrator_.transform(P_final)

        return np.clip(P_final, 0.0, 1.0).astype(np.float32)

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
        Predict multilabel decisions with Prior-Calibrated Asymmetric Partial Abstention:
        y_hat_j = 0 if p_j <= tau_0(j) (high negative confidence)
        y_hat_j = 1 if p_j >= tau_1(j) (sufficient positive likelihood ratio)
        y_hat_j = abstain_value (-1) otherwise.
        """
        probs = self.predict_proba(X)

        if not self.use_prior_adaptive:
            c = float(self.cost if cost is None else cost)
            predictions = np.full(probs.shape, self.abstain_value, dtype=np.int32)
            predictions[probs >= (1.0 - c)] = 1
            predictions[probs <= c] = 0
            return predictions

        if cost is not None and abs(cost - self.cost) > 1e-5:
            temp_thresholds = {}
            for j in range(self.n_labels_):
                prior_j = self.adaptive_thresholds_[j]["prior"]
                t0, t1 = compute_bayes_lr_thresholds(prior=prior_j, cost=cost)
                temp_thresholds[j] = {"tau_0": t0, "tau_1": t1}
            return apply_asymmetric_abstention(probs, temp_thresholds, self.abstain_value)

        return apply_asymmetric_abstention(probs, self.adaptive_thresholds_, self.abstain_value)

    def predict_full(self, X, threshold: float = 0.5) -> np.ndarray:
        """Predict standard binary multilabel decisions without abstention."""
        probs = self.predict_proba(X)
        return (probs >= float(threshold)).astype(np.int32)
