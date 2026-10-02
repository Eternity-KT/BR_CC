"""Multi-Stage 5-Fold Cross-Validation Peeling Partition Provider for GSI-MLC-PA v6.

This module implements the core architecture specified in `meeting_summary.md`:
1. Sequential Label-by-Label Evaluation:
   Tests candidate labels individually for independent predictability.
2. 5-Fold Cross Validation:
   Evaluates base classification models (Logistic Regression, SVM, MLP) via 5-fold CV.
3. Out-Of-Fold (OOF) Selective-F1 Evaluation:
   Computes Selective-F1 under partial abstention (cost c = 0.30 by default).
   Labels with Selective-F1 >= threshold (default 0.75) are promoted to IL.
4. Leakage-Free OOF Feature Augmentation:
   Promoted IL labels pass their out-of-fold predicted probabilities to stage t+1:
   X^(t+1) = [X, Normalize(P_hat_OOF_{IL_{1:t}})].
5. Iterative Multi-Stage Peeling:
   Remaining candidate DL labels are re-evaluated on augmented features.
6. Residual DL Ascending Correlation Ordering:
   Residual DL labels are ordered by ascending total correlation.
7. Diagnostics and BR IL Performance Logging:
   Monitors BR performance on discovered independent labels and logs full audit trails.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score
from sklearn.model_selection import KFold, StratifiedKFold

from .stratified_peeling import (
    _safe_f1_score,
    compute_label_correlation_matrix,
    find_optimal_binary_threshold,
    normalize_augmented_probabilities,
    order_dl_by_correlation,
)


def _fit_predict_prob_binary(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
) -> np.ndarray:
    """Fit a binary estimator and return predicted probabilities on X_eval."""
    unique = np.unique(y_train)
    if len(unique) <= 1:
        const_val = float(unique[0]) if len(unique) == 1 else 0.0
        return np.full(X_eval.shape[0], const_val, dtype=np.float64)

    model = clone(estimator)
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(X_eval), dtype=np.float64)
        if probs.ndim == 2:
            return np.clip(probs[:, 1], 0.0, 1.0)
        return np.clip(probs.ravel(), 0.0, 1.0)
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X_eval), dtype=np.float64)
        return np.clip(1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0))), 0.0, 1.0)
    else:
        preds = np.asarray(model.predict(X_eval), dtype=np.float64)
        return np.clip(preds.ravel(), 0.0, 1.0)


def evaluate_label_5fold_cv(
    X: np.ndarray,
    y: np.ndarray,
    estimator_factory: Callable[[], Any],
    n_folds: int = 5,
    cost: float = 0.30,
    metric: str = "selective_f1",
    random_state: int = 42,
) -> Tuple[float, np.ndarray, Dict[str, Any]]:
    """Perform 5-fold cross-validation for a single label and compute out-of-fold metrics.

    Parameters:
        X: Feature matrix of shape (n_samples, n_features).
        y: Binary target vector of shape (n_samples,).
        estimator_factory: Callable returning an unfitted binary scikit-learn estimator.
        n_folds: Number of CV folds (default 5).
        cost: Rejection cost c for partial abstention (default 0.30).
        metric: Evaluation metric to determine promotion:
                "selective_f1" (default), "optimal_f1", or "standard_f1".
        random_state: Random state for fold splitting.

    Returns:
        (primary_score, oof_probabilities, diagnostic_dict)
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    n_samples = len(y)

    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_count = int(np.min(class_counts)) if len(unique_classes) > 1 else 0

    # Ensure robust fold creation even under extreme class rarity
    if len(unique_classes) <= 1 or min_count < n_folds:
        cv_splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    else:
        cv_splitter = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=random_state
        )

    oof_probs = np.zeros(n_samples, dtype=np.float64)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        estimator = estimator_factory()
        probs_val = _fit_predict_prob_binary(estimator, X_tr, y_tr, X_val)
        oof_probs[val_idx] = probs_val

        # Fold level selective check
        decided_fold = np.minimum(probs_val, 1.0 - probs_val) <= cost
        if np.any(decided_fold):
            pred_fold = (probs_val[decided_fold] >= 0.5).astype(np.int32)
            f1_fold = _safe_f1_score(y_val[decided_fold], pred_fold)
        else:
            f1_fold = 0.0

        fold_metrics.append({
            "fold": fold_idx,
            "val_size": len(val_idx),
            "selective_f1": float(f1_fold),
            "coverage": float(np.mean(decided_fold)),
        })

    # Overall Out-Of-Fold Evaluation
    # 1. Selective F1 under linear rejection cost c:
    decided_all = np.minimum(oof_probs, 1.0 - oof_probs) <= cost
    coverage_all = float(np.mean(decided_all))

    if np.any(decided_all):
        y_pred_decided = (oof_probs[decided_all] >= 0.5).astype(np.int32)
        y_true_decided = y[decided_all]
        selective_f1 = _safe_f1_score(y_true_decided, y_pred_decided)
        sel_tp = np.sum((y_true_decided == 1) & (y_pred_decided == 1))
        sel_fp = np.sum((y_true_decided == 0) & (y_pred_decided == 1))
        selective_prec = float(sel_tp / (sel_tp + sel_fp)) if (sel_tp + sel_fp) > 0 else 0.0
        selective_acc = float(np.mean(y_true_decided == y_pred_decided))
    else:
        selective_f1 = 0.0
        selective_prec = 0.0
        selective_acc = 0.0

    # 2. Optimal-threshold binary F1 and Standard 0.5 binary F1
    optimal_th, optimal_f1 = find_optimal_binary_threshold(y, oof_probs)
    standard_f1 = _safe_f1_score(y, (oof_probs >= 0.5).astype(np.int32))

    # Primary score selection
    if metric == "selective_f1":
        primary_score = selective_f1
    elif metric == "optimal_f1":
        primary_score = optimal_f1
    elif metric == "standard_f1":
        primary_score = standard_f1
    else:
        raise ValueError(
            f"Unknown metric '{metric}'. Available: ['selective_f1', 'optimal_f1', 'standard_f1']."
        )

    diagnostic = {
        "primary_score": float(primary_score),
        "selective_f1": float(selective_f1),
        "selective_precision": float(selective_prec),
        "selective_accuracy": float(selective_acc),
        "coverage": float(coverage_all),
        "optimal_threshold": float(optimal_th),
        "optimal_f1": float(optimal_f1),
        "standard_f1": float(standard_f1),
        "positive_rate": float(np.mean(y == 1)),
        "n_samples": n_samples,
        "fold_metrics": fold_metrics,
    }

    return float(primary_score), oof_probs, diagnostic


@dataclass(frozen=True)
class CVPeelingConfig:
    """Configuration for Multi-Stage 5-Fold CV Peeling Selector (v6).

    Attributes:
        threshold: Performance threshold (Selective-F1) required to promote label to IL (default 0.75).
        n_folds: Number of cross-validation folds (default 5).
        max_depth: Maximum number of peeling stages (t = 1 .. max_depth).
        cost: Rejection cost c for partial abstention (default 0.30).
        metric: Evaluation metric used for threshold comparison:
                "selective_f1" (default), "optimal_f1", or "standard_f1".
        decaying_threshold: Whether to decay threshold across peeling stages (default False).
        threshold_decay_step: Amount to decrease threshold at each stage (default 0.05).
        min_threshold: Lower bound when decaying threshold (default 0.50).
        dl_order_direction: "ascending" (default v5.1/v6) or "descending".
        aug_normalization: Strategy to normalize augmented soft probability features:
                           "matching" (default), "centered", "standard", "logit", "none".
        min_labels_residual: Minimum number of remaining DL labels before stopping peeling (default 1).
        promote_singleton_dl: Whether to promote remaining singleton DL (len(DL)==1) into IL (default False).
        random_state: Seed for fold shuffling and base estimators.
    """
    threshold: float = 0.75
    n_folds: int = 5
    max_depth: int = 3
    cost: float = 0.30
    metric: str = "selective_f1"
    decaying_threshold: bool = False
    threshold_decay_step: float = 0.05
    min_threshold: float = 0.50
    dl_order_direction: str = "ascending"
    aug_normalization: str = "matching"
    min_labels_residual: int = 1
    promote_singleton_dl: bool = False
    random_state: int = 42

    def __post_init__(self):
        if not 0.0 <= float(self.threshold) <= 2.0:
            raise ValueError(f"threshold must lie in [0.0, 2.0], got {self.threshold}.")
        if int(self.n_folds) < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds}.")
        if int(self.max_depth) < 1:
            raise ValueError(f"max_depth must be >= 1, got {self.max_depth}.")
        if not 0.0 <= float(self.cost) <= 1.0:
            raise ValueError(f"cost must lie in [0.0, 1.0], got {self.cost}.")
        if self.metric not in ("selective_f1", "optimal_f1", "standard_f1"):
            raise ValueError(
                f"Unknown metric '{self.metric}'. Available: ['selective_f1', 'optimal_f1', 'standard_f1']."
            )
        if self.dl_order_direction not in ("ascending", "descending"):
            raise ValueError(
                f"Unknown dl_order_direction '{self.dl_order_direction}'. Available: ['ascending', 'descending']."
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "threshold": float(self.threshold),
            "n_folds": int(self.n_folds),
            "max_depth": int(self.max_depth),
            "cost": float(self.cost),
            "metric": self.metric,
            "decaying_threshold": bool(self.decaying_threshold),
            "threshold_decay_step": float(self.threshold_decay_step),
            "min_threshold": float(self.min_threshold),
            "dl_order_direction": self.dl_order_direction,
            "aug_normalization": self.aug_normalization,
            "min_labels_residual": int(self.min_labels_residual),
            "promote_singleton_dl": bool(self.promote_singleton_dl),
            "random_state": int(self.random_state),
        }


@dataclass(frozen=True)
class CVPeelingResult:
    """Immutable audit record and partition result produced by CVStratifiedPeelingSelector."""
    independent_layers: Tuple[Tuple[int, ...], ...]
    all_independent_labels: Tuple[int, ...]
    dependent_residual_labels: Tuple[int, ...]
    final_execution_order: Tuple[int, ...]
    stage_diagnostics: Tuple[Dict[str, Any], ...]
    br_independent_metrics: Dict[str, Any]
    stopping_reason: str
    num_stages_executed: int
    selection_time_seconds: float
    oof_probabilities: Dict[int, Any] = field(default_factory=dict)

    @property
    def n_il_stage_1(self) -> int:
        return len(self.independent_layers[0]) if len(self.independent_layers) > 0 else 0

    @property
    def n_il_stage_2(self) -> int:
        return len(self.independent_layers[1]) if len(self.independent_layers) > 1 else 0

    @property
    def n_il_stage_3(self) -> int:
        return len(self.independent_layers[2]) if len(self.independent_layers) > 2 else 0

    @property
    def labels_il_stage_1(self) -> Tuple[int, ...]:
        return self.independent_layers[0] if len(self.independent_layers) > 0 else ()

    @property
    def labels_il_stage_2(self) -> Tuple[int, ...]:
        return self.independent_layers[1] if len(self.independent_layers) > 1 else ()

    @property
    def labels_il_stage_3(self) -> Tuple[int, ...]:
        return self.independent_layers[2] if len(self.independent_layers) > 2 else ()

    @property
    def n_total_il(self) -> int:
        return len(self.all_independent_labels)

    @property
    def n_residual_dl(self) -> int:
        return len(self.dependent_residual_labels)

    @property
    def total_labels(self) -> int:
        return self.n_total_il + self.n_residual_dl

    @property
    def pct_total_il(self) -> float:
        total = self.total_labels
        return float(self.n_total_il / total * 100.0) if total > 0 else 0.0

    @property
    def pct_residual_dl(self) -> float:
        total = self.total_labels
        return float(self.n_residual_dl / total * 100.0) if total > 0 else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "independent_layers": [list(layer) for layer in self.independent_layers],
            "all_independent_labels": [int(l) for l in self.all_independent_labels],
            "dependent_residual_labels": [
                int(l) for l in self.dependent_residual_labels
            ],
            "final_execution_order": [int(l) for l in self.final_execution_order],
            "stage_diagnostics": list(self.stage_diagnostics),
            "br_independent_metrics": dict(self.br_independent_metrics),
            "stopping_reason": self.stopping_reason,
            "num_stages_executed": int(self.num_stages_executed),
            "selection_time_seconds": float(self.selection_time_seconds),
            "n_il_stage_1": self.n_il_stage_1,
            "labels_il_stage_1": list(self.labels_il_stage_1),
            "n_il_stage_2": self.n_il_stage_2,
            "labels_il_stage_2": list(self.labels_il_stage_2),
            "n_il_stage_3": self.n_il_stage_3,
            "labels_il_stage_3": list(self.labels_il_stage_3),
            "n_total_il": self.n_total_il,
            "pct_total_il": self.pct_total_il,
            "n_residual_dl": self.n_residual_dl,
            "pct_residual_dl": self.pct_residual_dl,
            "labels_residual_dl": list(self.dependent_residual_labels),
        }


class CVStratifiedPeelingSelector:
    """Multi-Stage 5-Fold Cross-Validation Peeling Selector for MLC (v6 Core).

    Evaluates each candidate label via 5-fold CV to determine independent
    predictability (Selective-F1 >= threshold).
    Promoted IL labels pass their out-of-fold predicted probabilities as augmented
    features to subsequent stages without data leakage.
    """

    def __init__(
        self,
        config: Optional[CVPeelingConfig] = None,
        base_estimator_factory: Optional[Callable[[], Any]] = None,
    ):
        self.config = config or CVPeelingConfig()
        self.base_estimator_factory = base_estimator_factory

    def fit_partition(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        base_estimator_factory: Optional[Callable[[], Any]] = None,
    ) -> CVPeelingResult:
        """Execute the 5-fold CV multi-stage peeling algorithm.

        Parameters:
            X: Training feature matrix of shape (n_samples, n_features).
            Y: Training binary label matrix of shape (n_samples, n_labels).
            base_estimator_factory: Optional callable returning a fresh binary estimator.
        """
        start_time = perf_counter()
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int32)

        # Defensive matrix dimensions and bounds validation
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got ndim={X.ndim}.")
        if Y.ndim != 2:
            raise ValueError(f"Y must be a 2D array, got ndim={Y.ndim}.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Sample count mismatch: X has {X.shape[0]} samples, Y has {Y.shape[0]} samples."
            )
        if X.shape[0] == 0:
            raise ValueError("Dataset cannot be empty (0 samples).")
        if Y.shape[1] == 0:
            raise ValueError("Y must contain at least one label.")

        # Guard against NaN/Inf in features
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        n_samples, n_labels = Y.shape

        factory = base_estimator_factory or self.base_estimator_factory
        if factory is None:
            from ..models.base_learners import create_binary_estimator
            factory = lambda: create_binary_estimator("logistic", random_state=self.config.random_state)

        correlation_matrix = compute_label_correlation_matrix(Y)

        independent_layers: List[List[int]] = []
        accumulated_il: List[int] = []
        candidate_dl: List[int] = list(range(n_labels))
        stage_diagnostics: List[Dict[str, Any]] = []
        stopping_reason = "max_depth_reached"

        # Out-of-fold probability cache for all promoted labels
        all_oof_probs: Dict[int, np.ndarray] = {}
        all_label_diagnostics: Dict[int, Dict[str, Any]] = {}

        current_X = X

        for stage in range(1, self.config.max_depth + 1):
            # Calculate effective threshold for this stage
            if self.config.decaying_threshold:
                effective_threshold = max(
                    self.config.min_threshold,
                    float(self.config.threshold - (stage - 1) * self.config.threshold_decay_step),
                )
            else:
                effective_threshold = float(self.config.threshold)

            stage_scores: Dict[int, float] = {}
            stage_diagnostics_per_label: Dict[int, Dict[str, Any]] = {}
            new_promoted_labels: List[int] = []
            stage_oof_probs: Dict[int, np.ndarray] = {}

            # Step 1: Evaluate each remaining candidate DL label individually via 5-fold CV
            for label in candidate_dl:
                y_label = Y[:, label]
                score, oof_p, diag = evaluate_label_5fold_cv(
                    X=current_X,
                    y=y_label,
                    estimator_factory=factory,
                    n_folds=self.config.n_folds,
                    cost=self.config.cost,
                    metric=self.config.metric,
                    random_state=self.config.random_state + stage,
                )
                stage_scores[label] = score
                stage_diagnostics_per_label[label] = diag
                stage_oof_probs[label] = oof_p

                # Promotion condition: Selective-F1 >= effective threshold
                if score >= effective_threshold:
                    new_promoted_labels.append(label)

            # Record stage diagnostic record
            stage_diag = {
                "stage": int(stage),
                "effective_threshold": float(effective_threshold),
                "num_candidate_dl": int(len(candidate_dl)),
                "num_promoted": int(len(new_promoted_labels)),
                "promoted_labels": list(new_promoted_labels),
                "stage_scores": {int(k): float(v) for k, v in stage_scores.items()},
                "label_diagnostics": {
                    int(k): v for k, v in stage_diagnostics_per_label.items()
                },
            }
            stage_diagnostics.append(stage_diag)

            # Check termination: no new labels promoted
            if not new_promoted_labels:
                stopping_reason = "no_promotion_in_stage"
                break

            # Cache OOF probabilities for promoted labels
            for l in new_promoted_labels:
                all_oof_probs[l] = stage_oof_probs[l]
                all_label_diagnostics[l] = stage_diagnostics_per_label[l]

            independent_layers.append(list(new_promoted_labels))
            accumulated_il.extend(new_promoted_labels)
            candidate_dl = [l for l in candidate_dl if l not in new_promoted_labels]

            # Check termination: DL exhausted or reached minimal residual
            if len(candidate_dl) <= self.config.min_labels_residual:
                stopping_reason = (
                    "all_labels_independent"
                    if len(candidate_dl) == 0
                    else "minimal_dl_residual_reached"
                )
                break

            if stage >= self.config.max_depth:
                stopping_reason = "max_depth_reached"
                break

            # Prepare augmented features for next stage using leakage-free OOF probabilities:
            # X^(stage+1) = [X, Normalize(P_hat_OOF_{IL})]
            aug_raw = np.column_stack([all_oof_probs[l] for l in accumulated_il])
            aug_norm = normalize_augmented_probabilities(
                aug_raw, reference_X=X, strategy=self.config.aug_normalization
            )
            current_X = np.hstack([X, aug_norm])

        # Step 2: Singleton DL promotion check (meeting_summary.md rule: len(DL) == 1 => IL)
        if getattr(self.config, "promote_singleton_dl", False) and len(candidate_dl) == 1:
            singleton_label = candidate_dl[0]
            independent_layers.append([singleton_label])
            accumulated_il.append(singleton_label)
            if singleton_label in stage_oof_probs:
                all_oof_probs[singleton_label] = stage_oof_probs[singleton_label]
            if singleton_label in stage_diagnostics_per_label:
                all_label_diagnostics[singleton_label] = stage_diagnostics_per_label[singleton_label]
            candidate_dl = []
            stopping_reason = "singleton_dl_promoted_to_il"

        # Step 3: Post-peeling ordering
        ordered_il = [l for layer in independent_layers for l in layer]

        # Step 3: Order residual DL labels by correlation (ascending by default)
        ordered_dl = order_dl_by_correlation(
            correlation_matrix,
            candidate_dl,
            direction=self.config.dl_order_direction,
        )

        final_execution_order = ordered_il + ordered_dl

        # Step 4: Compute BR performance summary on discovered independent labels
        br_independent_metrics: Dict[str, Any] = {}
        if ordered_il:
            il_selective_f1s = [
                all_label_diagnostics[l]["selective_f1"] for l in ordered_il if l in all_label_diagnostics
            ]
            il_coverages = [
                all_label_diagnostics[l]["coverage"] for l in ordered_il if l in all_label_diagnostics
            ]
            il_precisions = [
                all_label_diagnostics[l]["selective_precision"] for l in ordered_il if l in all_label_diagnostics
            ]
            br_independent_metrics = {
                "num_il_labels": len(ordered_il),
                "mean_selective_f1": float(np.mean(il_selective_f1s)) if il_selective_f1s else 0.0,
                "mean_coverage": float(np.mean(il_coverages)) if il_coverages else 0.0,
                "mean_selective_precision": float(np.mean(il_precisions)) if il_precisions else 0.0,
                "per_label_selective_f1": {
                    int(l): float(all_label_diagnostics[l]["selective_f1"])
                    for l in ordered_il if l in all_label_diagnostics
                },
            }

        total_time = float(perf_counter() - start_time)

        return CVPeelingResult(
            independent_layers=tuple(tuple(layer) for layer in independent_layers),
            all_independent_labels=tuple(ordered_il),
            dependent_residual_labels=tuple(ordered_dl),
            final_execution_order=tuple(final_execution_order),
            stage_diagnostics=tuple(stage_diagnostics),
            br_independent_metrics=br_independent_metrics,
            stopping_reason=stopping_reason,
            num_stages_executed=len(stage_diagnostics),
            selection_time_seconds=total_time,
            oof_probabilities=dict(all_oof_probs),
        )
