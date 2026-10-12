"""GSI-MLC-PA v6.3 single-stage IL and DL-inference ablations."""

from time import perf_counter
from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from ..decision.hamming import HammingBOPPolicy
from ..selection.br_residual_correlation import (
    build_residual_dependency_graph,
    compute_br_residual_matrix,
    compute_residual_pcc_matrix,
    export_residual_correlation_audit_table,
)
from ..selection.cv_peeling import CVPeelingResult, evaluate_label_5fold_cv
from ..selection.single_stage_peeling import SingleStagePeelingSelector
from ..selection.stratified_peeling import normalize_augmented_probabilities
from .base_learners import create_binary_estimator
from .classifier_chain import ClassifierChainClassifier
from .gsi_v6_2 import _fit_binary_model_safe
from .probabilistic_classifier_chain import ProbabilisticClassifierChain, _positive_probability


VALID_DL_INFERENCE = ("residual_br", "cc", "pcc")
VALID_FEATURE_SPACES = ("x", "x_il")


class GSIMLCPAv6_3Classifier(BaseEstimator, ClassifierMixin):
    """Fit the v6.3 architecture under one explicitly named ablation.

    ``dl_inference`` chooses residual-BR, greedy CC, or Monte-Carlo PCC.
    ``feature_space`` applies to DL only.  Independent labels always use X.
    CC and PCC always consume the exact same correlation-ascending DL order.
    """

    def __init__(
        self,
        base_learner: str = "logistic",
        dl_inference: str = "residual_br",
        feature_space: str = "x_il",
        stratified_threshold: float = 0.75,
        residual_corr_threshold: float = 0.25,
        cv_folds: int = 5,
        peeling_metric: str = "selective_f1",
        aug_normalization: str = "matching",
        pcc_num_samples: int = 1000,
        pcc_sampling_batch_size: int = 32,
        cost: float = 0.30,
        abstention_penalty: str = "linear",
        linear_boundary: str = "minimum_loss",
        abstain_value: int = -1,
        random_state: int = 42,
    ):
        self.base_learner = base_learner
        self.dl_inference = str(dl_inference)
        self.feature_space = str(feature_space)
        self.stratified_threshold = float(stratified_threshold)
        self.residual_corr_threshold = float(residual_corr_threshold)
        self.cv_folds = int(cv_folds)
        self.peeling_metric = str(peeling_metric)
        self.aug_normalization = str(aug_normalization)
        self.pcc_num_samples = int(pcc_num_samples)
        self.pcc_sampling_batch_size = int(pcc_sampling_batch_size)
        self.cost = float(cost)
        self.abstention_penalty = str(abstention_penalty)
        self.linear_boundary = str(linear_boundary)
        self.abstain_value = int(abstain_value)
        self.random_state = int(random_state)

    def _create_binary_model(self):
        return create_binary_estimator(
            self.base_learner, random_state=self.random_state
        )

    @staticmethod
    def _validate_xy(X, Y):
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int32)
        if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must be compatible two-dimensional matrices.")
        if not np.all(np.isin(Y, (0, 1))):
            raise ValueError("Y must contain only 0 and 1.")
        return X, Y

    def _fit_il_models(self, X, Y):
        models = {}
        for label in self.independent_labels_:
            models[label] = _fit_binary_model_safe(
                self._create_binary_model, X, Y[:, label]
            )
        return models

    def _training_dl_features(self, X):
        if self.feature_space == "x" or not self.independent_labels_:
            return X
        oof_raw = np.column_stack(
            [self.peeling_result_.oof_probabilities[label] for label in self.independent_labels_]
        )
        oof_norm = normalize_augmented_probabilities(
            oof_raw, reference_X=X, strategy=self.aug_normalization
        )
        return np.hstack([X, oof_norm])

    def fit(
        self,
        X,
        Y,
        *,
        partition_result: Optional[CVPeelingResult] = None,
        shared_il_models: Optional[Dict[int, Any]] = None,
    ):
        X, Y = self._validate_xy(X, Y)
        if self.dl_inference not in VALID_DL_INFERENCE:
            raise ValueError(f"dl_inference must be one of {VALID_DL_INFERENCE}.")
        if self.feature_space not in VALID_FEATURE_SPACES:
            raise ValueError(f"feature_space must be one of {VALID_FEATURE_SPACES}.")
        if self.dl_inference == "residual_br" and self.feature_space != "x_il":
            raise ValueError("The residual-BR control is defined on x_il only.")

        self.n_features_in_ = int(X.shape[1])
        self.n_labels_ = int(Y.shape[1])
        partition_start = perf_counter()
        if partition_result is None:
            selector = SingleStagePeelingSelector(
                threshold=self.stratified_threshold,
                n_folds=self.cv_folds,
                cost=self.cost,
                metric=self.peeling_metric,
                aug_normalization=self.aug_normalization,
                random_state=self.random_state,
                base_estimator_factory=self._create_binary_model,
            )
            self.peeling_result_ = selector.fit_partition(X, Y)
            self.partition_wall_time_seconds_ = float(perf_counter() - partition_start)
            self.partition_reused_ = False
        else:
            self.peeling_result_ = partition_result
            self.partition_wall_time_seconds_ = 0.0
            self.partition_reused_ = True

        self.independent_layers_ = self.peeling_result_.independent_layers
        self.independent_labels_ = list(self.peeling_result_.all_independent_labels)
        self.dependent_labels_ = list(self.peeling_result_.dependent_residual_labels)
        if len(self.independent_layers_) > 1:
            raise RuntimeError("v6.3 requires no more than one IL layer.")
        if sorted(self.independent_labels_ + self.dependent_labels_) != list(
            range(self.n_labels_)
        ):
            raise RuntimeError("The v6.3 IL/DL partition is not exhaustive.")

        il_start = perf_counter()
        if shared_il_models is None:
            self.il_models_ = self._fit_il_models(X, Y)
            self.il_models_reused_ = False
            self.il_wall_time_seconds_ = float(perf_counter() - il_start)
        else:
            missing = set(self.independent_labels_) - set(shared_il_models)
            if missing:
                raise ValueError(f"shared_il_models misses labels: {sorted(missing)}")
            self.il_models_ = dict(shared_il_models)
            self.il_models_reused_ = True
            self.il_wall_time_seconds_ = 0.0

        FS_dl = self._training_dl_features(X)
        Y_dl = Y[:, self.dependent_labels_]
        dl_start = perf_counter()
        self.dl_model_ = None
        self.br_base_dl_models_: Dict[int, Any] = {}
        self.br_cond_dl_models_: Dict[int, Any] = {}
        self.dependency_graph_: Dict[int, list] = {
            label: [] for label in self.dependent_labels_
        }
        self.residual_corr_matrix_ = None
        self.dl_error_rates_: Dict[int, float] = {}
        self.audit_table_ = None

        if self.dependent_labels_ and self.dl_inference in ("cc", "pcc"):
            # Y_dl columns are already in the locked global DL order.  The
            # natural local order therefore gives CC and PCC the same chain.
            if self.dl_inference == "cc":
                self.dl_model_ = ClassifierChainClassifier(
                    base_estimator=self._create_binary_model(),
                    order=list(range(len(self.dependent_labels_))),
                    random_state=self.random_state,
                )
            else:
                self.dl_model_ = ProbabilisticClassifierChain(
                    base_estimator=self._create_binary_model(),
                    order=list(range(len(self.dependent_labels_))),
                    num_samples=self.pcc_num_samples,
                    sampling_batch_size=self.pcc_sampling_batch_size,
                    random_state=self.random_state,
                )
            self.dl_model_.fit(FS_dl, Y_dl)
        elif self.dependent_labels_:
            self._fit_residual_br(FS_dl, X, Y)

        self.dl_wall_time_seconds_ = float(perf_counter() - dl_start)
        self.is_fitted_ = True
        return self

    def _fit_residual_br(self, FS_dl, reference_X, Y):
        n_dl = len(self.dependent_labels_)
        P_oof = np.zeros((Y.shape[0], n_dl), dtype=np.float64)
        for local_index, label in enumerate(self.dependent_labels_):
            _, probabilities, _ = evaluate_label_5fold_cv(
                X=FS_dl,
                y=Y[:, label],
                estimator_factory=self._create_binary_model,
                n_folds=self.cv_folds,
                cost=self.cost,
                metric=self.peeling_metric,
                random_state=self.random_state + label,
            )
            P_oof[:, local_index] = probabilities
            self.br_base_dl_models_[label] = _fit_binary_model_safe(
                self._create_binary_model, FS_dl, Y[:, label]
            )
            self.dl_error_rates_[label] = float(
                np.mean(np.abs(Y[:, label] - probabilities))
            )

        if n_dl >= 2:
            residuals = compute_br_residual_matrix(
                Y_true=Y[:, self.dependent_labels_],
                P_prob=P_oof,
                error_format="residual",
            )
            self.residual_corr_matrix_ = compute_residual_pcc_matrix(
                residual_matrix=residuals, use_absolute=True
            )
            self.dependency_graph_ = build_residual_dependency_graph(
                corr_matrix=self.residual_corr_matrix_,
                dl_labels=self.dependent_labels_,
                threshold=self.residual_corr_threshold,
            )

        local_column = {
            label: index for index, label in enumerate(self.dependent_labels_)
        }
        for label in self.dependent_labels_:
            partners = self.dependency_graph_.get(label, [])
            if partners:
                raw = P_oof[:, [local_column[partner] for partner in partners]]
                normalized = normalize_augmented_probabilities(
                    raw, reference_X=reference_X, strategy=self.aug_normalization
                )
                FS_label = np.hstack([FS_dl, normalized])
            else:
                FS_label = FS_dl
            self.br_cond_dl_models_[label] = _fit_binary_model_safe(
                self._create_binary_model, FS_label, Y[:, label]
            )

        if self.residual_corr_matrix_ is not None:
            self.audit_table_ = export_residual_correlation_audit_table(
                corr_matrix=self.residual_corr_matrix_,
                dl_labels=self.dependent_labels_,
                dependency_graph=self.dependency_graph_,
                error_rates=self.dl_error_rates_,
            )

    def _predict_il_probabilities(self, X):
        probabilities = np.empty((X.shape[0], len(self.independent_labels_)), dtype=np.float32)
        for column, label in enumerate(self.independent_labels_):
            probabilities[:, column] = _positive_probability(
                self.il_models_[label], X
            ).astype(np.float32)
        return probabilities

    def _test_dl_features(self, X, P_il):
        if self.feature_space == "x" or P_il.shape[1] == 0:
            return X
        normalized = normalize_augmented_probabilities(
            P_il, reference_X=X, strategy=self.aug_normalization
        )
        return np.hstack([X, normalized])

    def predict_proba(self, X):
        if not getattr(self, "is_fitted_", False):
            raise ValueError("GSIMLCPAv6_3Classifier is not fitted.")
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        P_final = np.zeros((X.shape[0], self.n_labels_), dtype=np.float32)
        P_il = self._predict_il_probabilities(X)
        if self.independent_labels_:
            P_final[:, self.independent_labels_] = P_il
        if not self.dependent_labels_:
            return P_final

        FS_dl = self._test_dl_features(X, P_il)
        if self.dl_inference in ("cc", "pcc"):
            P_final[:, self.dependent_labels_] = self.dl_model_.predict_proba(FS_dl)
        else:
            P_base = {
                label: _positive_probability(self.br_base_dl_models_[label], FS_dl)
                for label in self.dependent_labels_
            }
            for label in self.dependent_labels_:
                partners = self.dependency_graph_.get(label, [])
                if partners:
                    raw = np.column_stack([P_base[partner] for partner in partners])
                    normalized = normalize_augmented_probabilities(
                        raw, reference_X=X, strategy=self.aug_normalization
                    )
                    FS_label = np.hstack([FS_dl, normalized])
                else:
                    FS_label = FS_dl
                model = self.br_cond_dl_models_.get(
                    label, self.br_base_dl_models_[label]
                )
                P_final[:, label] = _positive_probability(model, FS_label)
        return np.clip(P_final, 0.0, 1.0)

    def predict_full(self, X, threshold: float = 0.5):
        return (self.predict_proba(X) >= float(threshold)).astype(np.int32)

    def predict(self, X, cost: Optional[float] = None):
        policy = HammingBOPPolicy(
            cost=self.cost if cost is None else float(cost),
            penalty=self.abstention_penalty,
            abstain_value=self.abstain_value,
            linear_boundary=self.linear_boundary,
        )
        return policy.predict(self.predict_proba(X))

    def audit_dict(self):
        return {
            "single_stage": True,
            "num_stages_executed": int(self.peeling_result_.num_stages_executed),
            "independent_labels": list(self.independent_labels_),
            "dependent_labels": list(self.dependent_labels_),
            "shared_dl_order": list(self.dependent_labels_),
            "dl_inference": self.dl_inference,
            "feature_space": self.feature_space,
            "pcc_num_samples": (
                self.pcc_num_samples if self.dl_inference == "pcc" else None
            ),
            "partition_reused": bool(self.partition_reused_),
            "il_models_reused": bool(self.il_models_reused_),
            "partition_wall_time_seconds": self.partition_wall_time_seconds_,
            "il_wall_time_seconds": self.il_wall_time_seconds_,
            "dl_wall_time_seconds": self.dl_wall_time_seconds_,
            "peeling": self.peeling_result_.as_dict(),
        }


__all__ = [
    "GSIMLCPAv6_3Classifier",
    "VALID_DL_INFERENCE",
    "VALID_FEATURE_SPACES",
]
