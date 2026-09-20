"""BSS-guided sparse probabilistic classifier chains with abstention.

All structural choices are learned from out-of-fold predictions created inside
``fit``.  The caller is responsible for passing only the outer-training split;
no prediction method accepts labels, which keeps the outer-test boundary
explicit.
"""

from time import perf_counter

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold

from src.decision.hamming import HammingBOPPolicy
from src.evaluation.cv import get_multilabel_cv

from .base_learners import canonical_base_learner_name, create_multilabel_estimator
from .local_parent_pcc import (
    LocalParentPCC,
    fit_binary_with_constant_fallback,
    positive_probability,
)


class _IdentityCalibrator:
    def predict(self, probabilities):
        return np.asarray(probabilities, dtype=np.float64)


class _PlattCalibrator:
    def __init__(self, estimator, epsilon):
        self.estimator = estimator
        self.epsilon = float(epsilon)

    def predict(self, probabilities):
        values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        clipped = np.clip(values, self.epsilon, 1.0 - self.epsilon)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        return self.estimator.predict_proba(logits)[:, 1]


class BSSUGSPCCPartialAbstentionClassifier(BaseEstimator, ClassifierMixin):
    """BSS-UG sparse local-PCC model using OOF-only structure selection."""

    def __init__(
        self,
        base_learner="logistic",
        inner_oof_splits=5,
        alpha_grid=(0.1, 0.2, 0.3, 0.4, 0.5),
        ug_threshold=0.0005,
        parent_gain_threshold=0.0005,
        q_max=5,
        epsilon=1e-12,
        alpha_tie_tolerance=1e-12,
        calibration="platt_oof_gated",
        calibration_splits=3,
        calibration_tolerance=1e-12,
        cost=0.3,
        penalty="linear",
        abstain_value=-1,
        random_state=42,
    ):
        self.base_learner = base_learner
        self.inner_oof_splits = inner_oof_splits
        self.alpha_grid = alpha_grid
        self.ug_threshold = ug_threshold
        self.parent_gain_threshold = parent_gain_threshold
        self.q_max = q_max
        self.epsilon = epsilon
        self.alpha_tie_tolerance = alpha_tie_tolerance
        self.calibration = calibration
        self.calibration_splits = calibration_splits
        self.calibration_tolerance = calibration_tolerance
        self.cost = cost
        self.penalty = penalty
        self.abstain_value = abstain_value
        self.random_state = random_state

    @staticmethod
    def _validate_xy(X, Y):
        features = np.asarray(X, dtype=np.float64)
        labels = np.asarray(Y, dtype=np.int32)
        if features.ndim != 2 or labels.ndim != 2:
            raise ValueError("X and Y must both be two-dimensional arrays.")
        if features.shape[0] != labels.shape[0] or features.shape[0] < 2:
            raise ValueError("X and Y must share at least two samples.")
        if features.shape[1] == 0 or labels.shape[1] == 0:
            raise ValueError("X and Y must have non-zero feature/label dimensions.")
        if not np.isfinite(features).all():
            raise ValueError("X must contain only finite values.")
        if not np.isin(labels, (0, 1)).all():
            raise ValueError("Y must be binary.")
        return features, labels

    def _validate_parameters(self):
        learner = canonical_base_learner_name(self.base_learner, default="logistic")
        if learner not in ("logistic", "mlp", "svm_calibrated"):
            raise ValueError(
                "BSS-UG-SPCC-PA requires logistic, mlp or svm_calibrated; "
                "uncalibrated svm_raw is not supported."
            )
        alphas = tuple(sorted({float(value) for value in self.alpha_grid}))
        if not alphas or any(value < 0.0 or value > 1.0 for value in alphas):
            raise ValueError("alpha_grid must contain unique values in [0, 1].")
        if int(self.inner_oof_splits) < 2:
            raise ValueError("inner_oof_splits must be at least 2.")
        if int(self.q_max) < 1 or int(self.q_max) > 5:
            raise ValueError("q_max must be between 1 and 5.")
        if float(self.ug_threshold) < 0 or float(self.parent_gain_threshold) < 0:
            raise ValueError("UG and parent-gain thresholds must be non-negative.")
        if float(self.epsilon) <= 0 or float(self.epsilon) >= 0.5:
            raise ValueError("epsilon must be in (0, 0.5).")
        if self.calibration != "platt_oof_gated":
            raise ValueError("Only calibration='platt_oof_gated' is supported.")
        if int(self.calibration_splits) < 2:
            raise ValueError("calibration_splits must be at least 2.")
        return learner, alphas

    def _make_inner_splits(self, X, Y):
        requested = int(self.inner_oof_splits)
        actual = min(requested, X.shape[0])
        if actual < 2:
            raise ValueError("At least two outer-training samples are required for OOF.")
        try:
            splitter = get_multilabel_cv(
                n_splits=actual, random_state=int(self.random_state), shuffle=True
            )
            splits = list(splitter.split(X, Y))
            strategy = type(splitter).__name__
        except (TypeError, ValueError):
            splitter = KFold(
                n_splits=actual, shuffle=True, random_state=int(self.random_state)
            )
            splits = list(splitter.split(X))
            strategy = type(splitter).__name__
        validation_counts = np.zeros(X.shape[0], dtype=np.int32)
        for train_indices, validation_indices in splits:
            if len(train_indices) == 0 or len(validation_indices) == 0:
                raise ValueError("Inner OOF produced an empty train/validation fold.")
            if np.intersect1d(train_indices, validation_indices).size:
                raise ValueError("Inner OOF train and validation indices overlap.")
            validation_counts[np.asarray(validation_indices, dtype=np.int32)] += 1
        if not np.all(validation_counts == 1):
            raise ValueError("Every training sample must be OOF validation exactly once.")
        self.inner_split_audit_ = {
            "requested_splits": requested,
            "actual_splits": len(splits),
            "strategy": strategy,
            "validation_once": True,
        }
        return [(np.asarray(a), np.asarray(b)) for a, b in splits]

    def _fit_br_oof(self, X, Y):
        n_samples, n_labels = Y.shape
        probabilities = np.full((n_samples, n_labels), np.nan, dtype=np.float64)
        references = np.full((n_samples, n_labels), np.nan, dtype=np.float64)
        fold_diagnostics = []
        for fold_index, (train_indices, validation_indices) in enumerate(self.inner_splits_):
            X_train, X_validation = X[train_indices], X[validation_indices]
            Y_train = Y[train_indices]
            prevalence = np.mean(Y_train, axis=0, dtype=np.float64)
            references[validation_indices] = prevalence
            constants = []
            for label_index in range(n_labels):
                estimator = fit_binary_with_constant_fallback(
                    X_train,
                    Y_train[:, label_index],
                    self.base_learner_,
                    int(self.random_state) + fold_index * n_labels + label_index,
                )
                probabilities[validation_indices, label_index] = positive_probability(
                    estimator, X_validation
                )
                unique = np.unique(Y_train[:, label_index])
                if unique.size <= 1:
                    constants.append(
                        {"label": label_index, "value": int(unique[0]) if unique.size else 0}
                    )
            fold_diagnostics.append(
                {
                    "fold": fold_index + 1,
                    "train_count": int(len(train_indices)),
                    "validation_count": int(len(validation_indices)),
                    "constant_targets": constants,
                }
            )
        if not np.isfinite(probabilities).all() or not np.isfinite(references).all():
            raise FloatingPointError("BR OOF generation left invalid probabilities.")
        return probabilities, references, fold_diagnostics

    def _compute_bss(self, Y, br_probabilities, reference_probabilities):
        bs_br = np.mean((br_probabilities - Y) ** 2, axis=0)
        bs_reference = np.mean((reference_probabilities - Y) ** 2, axis=0)
        degenerate = bs_reference <= float(self.epsilon)
        bss = np.full(Y.shape[1], -np.inf, dtype=np.float64)
        bss[~degenerate] = 1.0 - bs_br[~degenerate] / bs_reference[~degenerate]
        return bs_br, bs_reference, bss, degenerate

    def _conditional_log_likelihood(self, y, probabilities):
        values = np.clip(
            np.asarray(probabilities, dtype=np.float64),
            float(self.epsilon),
            1.0 - float(self.epsilon),
        )
        target = np.asarray(y, dtype=np.float64)
        return float(np.mean(target * np.log(values) + (1.0 - target) * np.log(1.0 - values)))

    def _compute_pairwise_ug(self, X, Y):
        n_samples, n_labels = Y.shape
        ug = np.full((n_labels, n_labels), -np.inf, dtype=np.float64)
        base_cll = np.asarray(
            [
                self._conditional_log_likelihood(Y[:, child], self.br_oof_[:, child])
                for child in range(n_labels)
            ],
            dtype=np.float64,
        )
        for parent in range(n_labels):
            if self.degenerate_labels_[parent]:
                continue
            for child in range(n_labels):
                if child == parent or self.degenerate_labels_[child]:
                    continue
                edge_probabilities = np.full(n_samples, np.nan, dtype=np.float64)
                for fold_index, (train_indices, validation_indices) in enumerate(
                    self.inner_splits_
                ):
                    augmented_train = np.column_stack(
                        (X[train_indices], Y[train_indices, parent])
                    )
                    estimator = fit_binary_with_constant_fallback(
                        augmented_train,
                        Y[train_indices, child],
                        self.base_learner_,
                        int(self.random_state)
                        + 100000
                        + fold_index * n_labels * n_labels
                        + parent * n_labels
                        + child,
                    )
                    X_validation = X[validation_indices]
                    q0 = positive_probability(
                        estimator,
                        np.column_stack(
                            (X_validation, np.zeros(len(validation_indices)))
                        ),
                    )
                    q1 = positive_probability(
                        estimator,
                        np.column_stack(
                            (X_validation, np.ones(len(validation_indices)))
                        ),
                    )
                    parent_probability = self.br_oof_[validation_indices, parent]
                    edge_probabilities[validation_indices] = (
                        parent_probability * q1 + (1.0 - parent_probability) * q0
                    )
                if not np.isfinite(edge_probabilities).all():
                    raise FloatingPointError("Pairwise UG OOF prediction is invalid.")
                ug[parent, child] = (
                    self._conditional_log_likelihood(Y[:, child], edge_probabilities)
                    - base_cll[child]
                )
        return ug, base_cll

    def _order_dependent_labels(self, dependent_labels):
        remaining = set(int(label) for label in dependent_labels)
        ordered = []
        history = []
        threshold = float(self.ug_threshold)
        while remaining:
            records = []
            for label in sorted(remaining):
                out_score = sum(
                    max(float(self.ug_matrix_[label, other]), 0.0)
                    for other in remaining
                    if other != label and self.ug_matrix_[label, other] > threshold
                )
                in_score = sum(
                    max(float(self.ug_matrix_[other, label]), 0.0)
                    for other in remaining
                    if other != label and self.ug_matrix_[other, label] > threshold
                )
                records.append(
                    {
                        "label": label,
                        "score": out_score - in_score,
                        "out": out_score,
                        "bss": float(self.bss_[label]),
                    }
                )
            chosen = max(
                records,
                key=lambda record: (
                    record["score"],
                    record["out"],
                    record["bss"],
                    -record["label"],
                ),
            )
            ordered.append(chosen["label"])
            history.append(chosen)
            remaining.remove(chosen["label"])
        return tuple(ordered), history

    def _candidate_parents(self, child, global_order):
        position = {label: offset for offset, label in enumerate(global_order)}
        return tuple(
            parent
            for parent in global_order[: position[child]]
            if not self.degenerate_labels_[parent]
            and self.ug_matrix_[parent, child] > float(self.ug_threshold)
        )

    def _local_pcc_oof(self, X, Y, child, parents):
        ordered_parents = tuple(int(parent) for parent in parents)
        key = (int(child), ordered_parents)
        if key in self._pcc_oof_cache:
            return self._pcc_oof_cache[key]
        probabilities = np.full(X.shape[0], np.nan, dtype=np.float64)
        for fold_index, (train_indices, validation_indices) in enumerate(self.inner_splits_):
            model = LocalParentPCC(
                base_learner=self.base_learner_,
                parent_indices=ordered_parents,
                child_index=int(child),
                random_state=int(self.random_state)
                + 200000
                + fold_index * Y.shape[1]
                + int(child),
                epsilon=float(self.epsilon),
            ).fit(X[train_indices], Y[train_indices])
            probabilities[validation_indices] = model.predict_child_proba(
                X[validation_indices]
            )
        if not np.isfinite(probabilities).all():
            raise FloatingPointError("Local Parent-PCC OOF prediction is invalid.")
        self._pcc_oof_cache[key] = probabilities
        return probabilities

    def _select_parents(self, X, Y, child, candidates, global_order):
        selected = []
        current_score = self.br_cll_[child]
        history = []
        positions = {label: offset for offset, label in enumerate(global_order)}
        while len(selected) < int(self.q_max):
            trials = []
            for candidate in candidates:
                if candidate in selected:
                    continue
                parents = tuple(
                    sorted(selected + [candidate], key=lambda label: positions[label])
                )
                probabilities = self._local_pcc_oof(X, Y, child, parents)
                score = self._conditional_log_likelihood(Y[:, child], probabilities)
                trials.append(
                    {
                        "candidate": int(candidate),
                        "parents": parents,
                        "score": score,
                        "delta": score - current_score,
                        "pairwise_ug": float(self.ug_matrix_[candidate, child]),
                        "global_position": int(positions[candidate]),
                    }
                )
            if not trials:
                break
            best = max(
                trials,
                key=lambda record: (
                    record["delta"],
                    record["pairwise_ug"],
                    -record["global_position"],
                    -record["candidate"],
                ),
            )
            accepted = best["delta"] > float(self.parent_gain_threshold)
            history.append(
                {
                    **best,
                    "parents": list(best["parents"]),
                    "accepted": bool(accepted),
                }
            )
            if not accepted:
                break
            selected = list(best["parents"])
            current_score = float(best["score"])
        return tuple(selected), history

    def _evaluate_alpha(self, X, Y, alpha):
        anchors = tuple(
            sorted(
                (
                    label
                    for label in range(Y.shape[1])
                    if self.degenerate_labels_[label] or self.bss_[label] >= alpha
                ),
                key=lambda label: (-float(self.bss_[label]), label),
            )
        )
        anchor_set = set(anchors)
        dependent = tuple(label for label in range(Y.shape[1]) if label not in anchor_set)
        dependent_order, order_history = self._order_dependent_labels(dependent)
        global_order = anchors + dependent_order
        parent_map = {}
        parent_history = {}
        raw_oof = np.asarray(self.br_oof_, dtype=np.float64).copy()
        for child in dependent_order:
            candidates = self._candidate_parents(child, global_order)
            parents, history = self._select_parents(
                X, Y, child, candidates, global_order
            )
            parent_map[int(child)] = parents
            parent_history[int(child)] = {
                "candidates": list(candidates),
                "steps": history,
            }
            if parents:
                raw_oof[:, child] = self._local_pcc_oof(
                    X, Y, child, parents
                )
        macro_brier = float(np.mean(np.mean((raw_oof - Y) ** 2, axis=0)))
        return {
            "alpha": float(alpha),
            "anchors": anchors,
            "dependent": dependent_order,
            "global_order": global_order,
            "order_history": order_history,
            "parent_map": parent_map,
            "parent_history": parent_history,
            "raw_oof": raw_oof,
            "macro_brier": macro_brier,
        }

    @staticmethod
    def _fit_platt(logits, target, random_state):
        estimator = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=int(random_state),
        )
        estimator.fit(np.asarray(logits).reshape(-1, 1), np.asarray(target))
        return estimator

    def _fit_calibrators(self, raw_oof, Y):
        calibrators = []
        audit = []
        for label in range(Y.shape[1]):
            probabilities = np.asarray(raw_oof[:, label], dtype=np.float64)
            target = np.asarray(Y[:, label], dtype=np.int32)
            raw_brier = float(np.mean((probabilities - target) ** 2))
            counts = np.bincount(target, minlength=2)
            n_splits = min(int(self.calibration_splits), int(counts.min()))
            record = {
                "label": label,
                "raw_brier": raw_brier,
                "calibrated_brier": None,
                "accepted": False,
                "reason": None,
                "crossfit_splits": int(max(n_splits, 0)),
            }
            if n_splits < 2:
                calibrators.append(_IdentityCalibrator())
                record["reason"] = "insufficient_class_count"
                audit.append(record)
                continue
            clipped = np.clip(probabilities, self.epsilon, 1.0 - self.epsilon)
            logits = np.log(clipped / (1.0 - clipped))
            calibrated_oof = np.full(probabilities.shape, np.nan, dtype=np.float64)
            try:
                splitter = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=int(self.random_state) + 300000 + label,
                )
                for fold_index, (train_indices, validation_indices) in enumerate(
                    splitter.split(logits.reshape(-1, 1), target)
                ):
                    estimator = self._fit_platt(
                        logits[train_indices],
                        target[train_indices],
                        int(self.random_state) + 310000 + label * n_splits + fold_index,
                    )
                    calibrated_oof[validation_indices] = estimator.predict_proba(
                        logits[validation_indices].reshape(-1, 1)
                    )[:, 1]
                calibrated_brier = float(
                    np.mean((calibrated_oof - target) ** 2)
                )
                record["calibrated_brier"] = calibrated_brier
                if (
                    calibrated_brier + float(self.calibration_tolerance)
                    < raw_brier
                ):
                    estimator = self._fit_platt(
                        logits,
                        target,
                        int(self.random_state) + 320000 + label,
                    )
                    calibrators.append(_PlattCalibrator(estimator, self.epsilon))
                    record["accepted"] = True
                    record["reason"] = "cross_fitted_brier_improved"
                else:
                    calibrators.append(_IdentityCalibrator())
                    record["reason"] = "no_cross_fitted_brier_improvement"
            except (FloatingPointError, RuntimeError, ValueError) as exc:
                calibrators.append(_IdentityCalibrator())
                record["reason"] = f"fit_failed:{type(exc).__name__}"
            audit.append(record)
        return calibrators, audit

    def fit(self, X, Y):
        started = perf_counter()
        features, labels = self._validate_xy(X, Y)
        self.base_learner_, self.alpha_grid_ = self._validate_parameters()
        self.n_features_in_ = features.shape[1]
        self.n_labels_ = labels.shape[1]
        self.inner_splits_ = self._make_inner_splits(features, labels)
        self._pcc_oof_cache = {}
        timings = {}

        stage = perf_counter()
        self.br_oof_, self.reference_oof_, self.br_fold_diagnostics_ = (
            self._fit_br_oof(features, labels)
        )
        (
            self.bs_br_,
            self.bs_reference_,
            self.bss_,
            self.degenerate_labels_,
        ) = self._compute_bss(labels, self.br_oof_, self.reference_oof_)
        timings["br_bss_seconds"] = perf_counter() - stage

        stage = perf_counter()
        self.ug_matrix_, self.br_cll_ = self._compute_pairwise_ug(features, labels)
        timings["ug_seconds"] = perf_counter() - stage

        stage = perf_counter()
        self.alpha_records_ = [
            self._evaluate_alpha(features, labels, alpha)
            for alpha in self.alpha_grid_
        ]
        best = self.alpha_records_[0]
        for record in self.alpha_records_[1:]:
            improvement = best["macro_brier"] - record["macro_brier"]
            if improvement > float(self.alpha_tie_tolerance) or (
                abs(improvement) <= float(self.alpha_tie_tolerance)
                and record["alpha"] < best["alpha"]
            ):
                best = record
        self.selected_alpha_ = float(best["alpha"])
        self.anchor_labels_ = tuple(best["anchors"])
        self.dependent_labels_ = tuple(best["dependent"])
        self.global_order_ = tuple(best["global_order"])
        self.parent_map_ = {
            int(child): tuple(parents)
            for child, parents in best["parent_map"].items()
        }
        self.selected_raw_oof_ = np.asarray(best["raw_oof"], dtype=np.float64)
        timings["alpha_parent_selection_seconds"] = perf_counter() - stage

        stage = perf_counter()
        self.br_model_ = create_multilabel_estimator(
            self.base_learner_, random_state=int(self.random_state)
        )
        self.br_model_.fit(features, labels)
        self.local_pcc_models_ = {}
        for child, parents in self.parent_map_.items():
            if parents:
                self.local_pcc_models_[child] = LocalParentPCC(
                    base_learner=self.base_learner_,
                    parent_indices=parents,
                    child_index=child,
                    random_state=int(self.random_state) + 400000 + child,
                    epsilon=float(self.epsilon),
                ).fit(features, labels)
        timings["refit_seconds"] = perf_counter() - stage

        stage = perf_counter()
        self.calibrators_, self.calibration_audit_ = self._fit_calibrators(
            self.selected_raw_oof_, labels
        )
        timings["calibration_seconds"] = perf_counter() - stage
        timings["fit_total_seconds"] = perf_counter() - started
        self.timing_audit_ = timings
        return self

    def _validate_prediction_features(self, X):
        if not hasattr(self, "br_model_"):
            raise ValueError("Model must be fitted before prediction.")
        features = np.asarray(X, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.n_features_in_:
            raise ValueError("X has an incompatible feature shape.")
        if not np.isfinite(features).all():
            raise ValueError("X must contain only finite values.")
        return features

    def predict_raw_proba(self, X):
        features = self._validate_prediction_features(X)
        probabilities = np.asarray(
            self.br_model_.predict_proba(features), dtype=np.float64
        )
        if probabilities.shape != (features.shape[0], self.n_labels_):
            raise ValueError("BR model returned an incompatible probability shape.")
        for child, model in self.local_pcc_models_.items():
            probabilities[:, child] = model.predict_child_proba(features)
        if not np.isfinite(probabilities).all():
            raise FloatingPointError("Raw probability inference returned NaN/Inf.")
        return np.clip(probabilities, 0.0, 1.0)

    def predict_proba(self, X):
        raw = self.predict_raw_proba(X)
        calibrated = np.column_stack(
            [
                calibrator.predict(raw[:, label])
                for label, calibrator in enumerate(self.calibrators_)
            ]
        )
        if not np.isfinite(calibrated).all():
            raise FloatingPointError("Calibrated probability inference returned NaN/Inf.")
        return np.clip(calibrated, 0.0, 1.0)

    @staticmethod
    def predict_full_from_proba(probabilities):
        matrix = np.asarray(probabilities, dtype=np.float64)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError("probabilities must be a finite two-dimensional matrix.")
        if np.any((matrix < 0.0) | (matrix > 1.0)):
            raise ValueError("probabilities must lie in [0, 1].")
        return (matrix >= 0.5).astype(np.int32)

    def predict_full(self, X):
        return self.predict_full_from_proba(self.predict_proba(X))

    def predict_from_proba(self, probabilities, cost=None):
        policy = HammingBOPPolicy(
            cost=self.cost if cost is None else cost,
            penalty=self.penalty,
            abstain_value=self.abstain_value,
            linear_boundary="strict_symmetric_thresholds",
        )
        return policy.predict(probabilities)

    def predict(self, X):
        return self.predict_from_proba(self.predict_proba(X))

    def decision_function(self, X):
        probabilities = np.clip(
            self.predict_proba(X), float(self.epsilon), 1.0 - float(self.epsilon)
        )
        return np.log(probabilities / (1.0 - probabilities))

    @staticmethod
    def _audit_alpha(record):
        return {
            "alpha": float(record["alpha"]),
            "macro_brier": float(record["macro_brier"]),
            "anchors": list(record["anchors"]),
            "dependent_order": list(record["dependent"]),
            "global_order": list(record["global_order"]),
            "order_history": record["order_history"],
            "parent_map": {
                str(child): list(parents)
                for child, parents in record["parent_map"].items()
            },
            "parent_history": {
                str(child): history
                for child, history in record["parent_history"].items()
            },
        }

    def get_structure_audit(self):
        if not hasattr(self, "selected_alpha_"):
            raise ValueError("Model must be fitted before requesting its audit.")
        bss_records = []
        for label in range(self.n_labels_):
            bss_records.append(
                {
                    "label": label,
                    "bs_br": float(self.bs_br_[label]),
                    "bs_reference": float(self.bs_reference_[label]),
                    "bss": None
                    if self.degenerate_labels_[label]
                    else float(self.bss_[label]),
                    "degenerate": bool(self.degenerate_labels_[label]),
                    "degenerate_reason": "reference_brier_at_or_below_epsilon"
                    if self.degenerate_labels_[label]
                    else None,
                }
            )
        ug_edges = []
        for parent in range(self.n_labels_):
            for child in range(self.n_labels_):
                if parent != child and np.isfinite(self.ug_matrix_[parent, child]):
                    ug_edges.append(
                        {
                            "parent": parent,
                            "child": child,
                            "ug": float(self.ug_matrix_[parent, child]),
                            "passes_threshold": bool(
                                self.ug_matrix_[parent, child]
                                > float(self.ug_threshold)
                            ),
                        }
                    )
        return {
            "model": "BSS_UG_SPCC_PA",
            "base_learner": self.base_learner_,
            "inner_split": self.inner_split_audit_,
            "bss": bss_records,
            "br_fold_diagnostics": self.br_fold_diagnostics_,
            "ug_edges": ug_edges,
            "alpha_records": [self._audit_alpha(record) for record in self.alpha_records_],
            "selected_alpha": float(self.selected_alpha_),
            "anchor_labels": list(self.anchor_labels_),
            "dependent_order": list(self.dependent_labels_),
            "global_order": list(self.global_order_),
            "parent_map": {
                str(child): list(parents)
                for child, parents in self.parent_map_.items()
            },
            "calibration": self.calibration_audit_,
            "timing": {key: float(value) for key, value in self.timing_audit_.items()},
            "settings": {
                "alpha_grid": list(self.alpha_grid_),
                "ug_threshold": float(self.ug_threshold),
                "parent_gain_threshold": float(self.parent_gain_threshold),
                "q_max": int(self.q_max),
                "epsilon": float(self.epsilon),
                "alpha_tie_tolerance": float(self.alpha_tie_tolerance),
                "calibration": self.calibration,
                "calibration_splits": int(self.calibration_splits),
                "calibration_tolerance": float(self.calibration_tolerance),
                "decision_boundary": "strict_symmetric_thresholds",
                "random_state": int(self.random_state),
            },
        }


BSSUGSPCCPAClassifier = BSSUGSPCCPartialAbstentionClassifier

