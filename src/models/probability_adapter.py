"""Leakage-safe probability adapters for calibrated binary SVM learners."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted


class _PriorProbabilityClassifier:
    def __init__(self, positive_probability):
        self.positive_probability = float(positive_probability)

    def predict_proba(self, X):
        positive = np.full(len(X), self.positive_probability, dtype=np.float64)
        return np.column_stack((1.0 - positive, positive))

    def predict(self, X):
        return (
            self.predict_proba(X)[:, 1] >= 0.5
        ).astype(np.int32)


class ProbabilityAdapter(BaseEstimator, ClassifierMixin):
    """Calibrate a binary margin estimator using training data only.

    The adapter is fitted only when its owning BR/CC/GSI component is fitted.
    Calibration folds are therefore nested inside the caller's training fold.
    Constant and one-minority-sample labels use deterministic fallbacks whose
    provenance is exposed through ``calibration_audit_``.
    """

    def __init__(
        self,
        estimator=None,
        method="sigmoid",
        cv=3,
        rare_label_strategy="smoothed_prior",
        random_state=42,
    ):
        self.estimator = estimator
        self.method = method
        self.cv = cv
        self.rare_label_strategy = rare_label_strategy
        self.random_state = random_state

    def _validate_parameters(self):
        if self.method != "sigmoid":
            raise ValueError("Only sigmoid/Platt calibration is registered.")
        if isinstance(self.cv, (bool, np.bool_)) or not isinstance(
            self.cv, (int, np.integer)
        ) or int(self.cv) < 2:
            raise ValueError("cv must be an integer of at least 2.")
        if self.rare_label_strategy != "smoothed_prior":
            raise ValueError(
                "rare_label_strategy must be 'smoothed_prior'."
            )

    def _raw_estimator(self):
        if self.estimator is None:
            return LinearSVC(
                C=1.0,
                dual="auto",
                tol=1e-3,
                max_iter=5000,
                random_state=self.random_state,
            )
        return clone(self.estimator)

    @staticmethod
    def _calibrator(estimator, method, cv):
        try:
            return CalibratedClassifierCV(
                estimator=estimator,
                method=method,
                cv=cv,
                ensemble=False,
            )
        except TypeError:  # scikit-learn < 1.2 compatibility
            return CalibratedClassifierCV(
                base_estimator=estimator,
                method=method,
                cv=cv,
                ensemble=False,
            )

    def fit(self, X, y):
        self._validate_parameters()
        features = np.asarray(X, dtype=np.float32)
        target = np.asarray(y, dtype=np.int32).reshape(-1)
        if features.ndim != 2 or target.shape[0] != features.shape[0]:
            raise ValueError("X/y shapes are incompatible for calibration.")
        if not np.all(np.isin(target, (0, 1))):
            raise ValueError("Calibration targets must contain only 0 and 1.")
        if len(target) == 0:
            raise ValueError("Calibration requires at least one sample.")

        counts = np.bincount(target, minlength=2).astype(int)
        minority_count = int(np.min(counts))
        fallback = None
        effective_cv = None
        if np.count_nonzero(counts) == 1:
            positive_probability = float(target[0])
            self.model_ = _PriorProbabilityClassifier(positive_probability)
            strategy = "constant_label"
            fallback = "single_observed_class"
        elif minority_count < 2:
            positive_probability = float((counts[1] + 1.0) / (len(target) + 2.0))
            self.model_ = _PriorProbabilityClassifier(positive_probability)
            strategy = "smoothed_prior"
            fallback = "minority_count_below_two"
        else:
            effective_cv = min(int(self.cv), minority_count)
            self.model_ = self._calibrator(
                self._raw_estimator(), self.method, effective_cv
            )
            self.model_.fit(features, target)
            positive_probability = None
            strategy = "calibrated_sigmoid_cv"

        self.classes_ = np.array([0, 1], dtype=np.int32)
        self.calibration_audit_ = {
            "strategy": strategy,
            "method": self.method,
            "requested_cv": int(self.cv),
            "effective_cv": effective_cv,
            "fit_sample_count": int(len(target)),
            "class_counts": {"0": int(counts[0]), "1": int(counts[1])},
            "fallback": fallback,
            "fallback_probability": positive_probability,
            "rare_label_strategy": self.rare_label_strategy,
            "outer_test_access": False,
        }
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ("model_", "calibration_audit_"))
        probabilities = np.asarray(self.model_.predict_proba(X), dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError("Calibrated binary model must return two probabilities.")
        return np.clip(probabilities, 0.0, 1.0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int32)

    def decision_function(self, X):
        probabilities = np.clip(self.predict_proba(X)[:, 1], 1e-7, 1.0 - 1e-7)
        return np.log(probabilities / (1.0 - probabilities))


def aggregate_calibration_audit(label_classifiers):
    """Collect fitted adapter provenance without retaining training data."""

    records = []
    for label_index, classifier in label_classifiers:
        audit = getattr(classifier, "calibration_audit_", None)
        if audit is not None:
            records.append({"label_index": int(label_index), **dict(audit)})
    if not records:
        return None
    return {
        "labels": records,
        "calibrated_label_count": int(
            sum(row["strategy"] == "calibrated_sigmoid_cv" for row in records)
        ),
        "fallback_label_count": int(
            sum(row["fallback"] is not None for row in records)
        ),
        "fit_sample_counts": sorted(
            {int(row["fit_sample_count"]) for row in records}
        ),
        "outer_test_access": False,
    }
