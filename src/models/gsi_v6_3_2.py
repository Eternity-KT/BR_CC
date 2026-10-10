"""
GSI-MLC-PA Version 6.3.2: One-Step Normalized Mean-Field Coupling for Dependent Labels (DL).

Enhancements over v6.3.1:
1. One-Step Normalized Mean-Field (OS-NMF) for DL Coupling:
   Replaces complex secondary conditioned models g_l with an algebraic one-step belief update
   driven by signed Pearson correlation of prediction residuals.
2. Degree Normalization & Bounded Shifts:
   Normalizes coupling weights by max(1.0, row_sums) and clips logit shifts in [-z_max, +z_max]
   to completely eliminate probability saturation, oscillation, and double counting.
3. Symmetric Spin-Centering:
   Employs 2P - 1 spin centering to preserve balance under extreme label imbalance.
4. Preserved Calibrated Asymmetry:
   Seamlessly integrates with Balanced-Root Platt Calibration and Precision Guard (tau_1 >= 0.50).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from .gsi_v6_3 import _fit_binary_model_safe, normalize_augmented_probabilities
from .gsi_v6_3_1 import GSIMLCPAv6_3_1Classifier
from ..selection.cv_peeling import (
    CVPeelingConfig,
    CVStratifiedPeelingSelector,
    evaluate_label_5fold_cv,
)
from ..selection.br_residual_correlation import (
    build_residual_dependency_graph,
    compute_br_residual_matrix,
    compute_residual_pcc_matrix,
    export_residual_correlation_audit_table,
    one_step_normalized_mean_field,
)
from ..selection.asymmetric_decision import compute_adaptive_thresholds_table
from ..calibration.tail_calibrator import MultiLabelTailCalibrator


class GSIMLCPAv6_3_2Classifier(GSIMLCPAv6_3_1Classifier):
    """
    GSI-MLC-PA Version 6.3.2 Classifier with One-Step Normalized Mean-Field (OS-NMF) Coupling,
    Precision Guard, and Balanced-Root Platt Calibration.
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
        calibration_method: str = "sqrt_platt",
        precision_guard: bool = True,
        min_tau_1: float = 0.50,
        abstain_value: int = -1,
        random_state: int = 42,
        mf_alpha: float = 0.25,
        mf_z_max: float = 0.5,
    ):
        super().__init__(
            base_learner=base_learner,
            stratified_threshold=stratified_threshold,
            residual_corr_threshold=residual_corr_threshold,
            cv_folds=cv_folds,
            max_peeling_depth=max_peeling_depth,
            peeling_metric=peeling_metric,
            decaying_threshold=decaying_threshold,
            threshold_decay_step=threshold_decay_step,
            aug_normalization=aug_normalization,
            error_format=error_format,
            cost=cost,
            gamma_min=gamma_min,
            use_prior_adaptive=use_prior_adaptive,
            calibrate_tail=calibrate_tail,
            calibration_method=calibration_method,
            precision_guard=precision_guard,
            min_tau_1=min_tau_1,
            abstain_value=abstain_value,
            random_state=random_state,
        )
        self.mf_alpha = float(mf_alpha)
        self.mf_z_max = float(mf_z_max)
        self.signed_residual_corr_matrix_ = None

    def fit(self, X, Y):
        """Fit GSI-MLC-PA v6.3.2 using CV peeling and One-Step Mean-Field coupling."""
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

        FS_context = current_train_X

        # -------------------------------------------------------------------------
        # PHASE 2: Handle Residual Dependent Labels (DL) via One-Step Mean-Field
        # -------------------------------------------------------------------------
        self.br_base_dl_models_ = {}
        self.br_cond_dl_models_ = {}
        self.dependency_graph_ = {lbl: [] for lbl in self.dependent_labels_}
        self.dl_error_rates_ = {}
        self.residual_corr_matrix_ = None
        self.signed_residual_corr_matrix_ = None

        n_dl = len(self.dependent_labels_)
        if n_dl >= 2:
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
                self.br_cond_dl_models_[lbl] = base_est
                self.dl_error_rates_[lbl] = float(np.mean(np.abs(Y[:, lbl] - oof_p)))

            residuals_dl = compute_br_residual_matrix(
                Y_true=Y[:, self.dependent_labels_],
                P_prob=P_OOF_DL,
                error_format=self.error_format,
            )

            self.signed_residual_corr_matrix_ = compute_residual_pcc_matrix(
                residual_matrix=residuals_dl,
                use_absolute=False,
            )
            self.residual_corr_matrix_ = np.abs(self.signed_residual_corr_matrix_)

            self.dependency_graph_ = build_residual_dependency_graph(
                corr_matrix=self.residual_corr_matrix_,
                dl_labels=self.dependent_labels_,
                threshold=self.residual_corr_threshold,
            )

            # Refine OOF predictions directly via One-Step Normalized Mean-Field
            refined_oof_dl = one_step_normalized_mean_field(
                P_base=P_OOF_DL,
                corr_matrix=self.signed_residual_corr_matrix_,
                alpha=self.mf_alpha,
                threshold=self.residual_corr_threshold,
                z_max=self.mf_z_max,
            )
            P_OOF_all[:, self.dependent_labels_] = refined_oof_dl

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
                method=self.calibration_method,
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
                precision_guard=self.precision_guard,
                min_tau_1=self.min_tau_1,
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
        """Predict probabilities on test samples using One-Step Mean-Field on DL."""
        if not self.is_fitted_:
            raise ValueError("GSIMLCPAv6_3_2Classifier is not fitted yet.")

        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]
        P_final = np.zeros((n_samples, self.n_labels_), dtype=np.float32)

        # STAGE 1: Predict Independent Labels across layers
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

        # STAGE 2: Predict Dependent Labels via Base BR + Mean-Field
        dl = self.dependent_labels_
        if len(dl) > 0:
            P_base_DL = np.column_stack([
                self._predict_prob_single(self.br_base_dl_models_[lbl], FS_context_test)
                for lbl in dl
            ])

            if len(dl) >= 2 and self.signed_residual_corr_matrix_ is not None:
                P_refined_DL = one_step_normalized_mean_field(
                    P_base=P_base_DL,
                    corr_matrix=self.signed_residual_corr_matrix_,
                    alpha=self.mf_alpha,
                    threshold=self.residual_corr_threshold,
                    z_max=self.mf_z_max,
                )
                P_final[:, dl] = P_refined_DL
            else:
                P_final[:, dl] = P_base_DL

        # Apply tail calibration
        if self.calibrate_tail and self.calibrator_ is not None:
            P_final = self.calibrator_.transform(P_final)

        return np.clip(P_final, 0.0, 1.0).astype(np.float32)
