"""
Adaptive Tri-Regime Calibrator for GSI-MLC-PA Version 6.3.3.

Reference:
    - spec/spec_v6_3_3.md
    - Only applies tail Platt scaling when a label is in an extreme imbalance regime (beta_l < tau_balance).
    - Preserves uncorrupted raw base probabilities for balanced/mild labels (beta_l >= tau_balance).
    - Provides symmetric inverted root weighting when positive class is majority (pi_l > 0.80).
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from sklearn.linear_model import LogisticRegression
from .tail_calibrator import TailCalibrator


class IdentityCalibrator:
    """Pass-through calibrator that preserves raw probabilities unchanged."""

    def __init__(self):
        self.is_fitted_ = True

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "IdentityCalibrator":
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        p = np.asarray(probs, dtype=np.float64).ravel()
        return np.clip(p, 0.0, 1.0)


class InvertedTailCalibrator:
    """
    Tail Calibrator for minority negative class (pi_l > 0.50).
    Applies square-root class weighting favoring label 0.
    """

    def __init__(
        self,
        method: str = "sqrt_platt_inv",
        clip_eps: float = 1e-5,
        random_state: int = 42,
    ):
        self.method = method
        self.clip_eps = clip_eps
        self.random_state = random_state
        self.is_fitted_ = False
        self.is_degenerate_ = False
        self.degenerate_value_ = 1.0
        self.model_: Optional[LogisticRegression] = None

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "InvertedTailCalibrator":
        p = np.asarray(probs, dtype=np.float64).ravel()
        labels = np.asarray(y, dtype=np.int32).ravel()

        if len(p) != len(labels):
            raise ValueError(f"Shape mismatch: {len(p)} vs {len(labels)}")

        unique = np.unique(labels)
        if len(unique) <= 1:
            self.is_fitted_ = True
            self.is_degenerate_ = True
            self.degenerate_value_ = float(unique[0]) if len(unique) == 1 else 1.0
            return self

        p_clipped = np.clip(p, self.clip_eps, 1.0 - self.clip_eps)
        logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)

        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)

        # Weight favoring negative class
        neg_weight = float(n_pos) / float(max(n_neg, 1))
        sample_weight = np.where(labels == 0, np.sqrt(neg_weight), 1.0).astype(np.float64)

        try:
            lr = LogisticRegression(
                C=1.0,
                solver="liblinear",
                max_iter=1000,
                random_state=self.random_state,
            )
            lr.fit(logits, labels, sample_weight=sample_weight)

            if lr.coef_[0, 0] <= 0.0:
                self.model_ = None
            else:
                self.model_ = lr
        except Exception:
            self.model_ = None

        self.is_fitted_ = True
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        p = np.asarray(probs, dtype=np.float64).ravel()
        if not self.is_fitted_:
            return p

        if self.is_degenerate_:
            return np.full_like(p, self.degenerate_value_)

        if self.model_ is None:
            return np.clip(p, 0.0, 1.0)

        p_clipped = np.clip(p, self.clip_eps, 1.0 - self.clip_eps)
        logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)

        cal_probs = self.model_.predict_proba(logits)[:, 1]
        return np.clip(cal_probs, 0.0, 1.0)


class MultiLabelAdaptiveCalibrator:
    """
    Multilabel Tri-Regime Calibrator:
    - Balanced regime (beta_l >= tau_balance): Identity (no calibration).
    - Rare negative regime (pi_l < 0.20): Balanced-Root Platt.
    - Rare positive regime (pi_l > 0.80): Inverted Balanced-Root Platt.
    """

    def __init__(
        self,
        tau_balance: float = 0.75,
        method: str = "sqrt_platt",
        clip_eps: float = 1e-5,
        random_state: int = 42,
    ):
        self.tau_balance = float(tau_balance)
        self.method = method
        self.clip_eps = clip_eps
        self.random_state = random_state
        self.calibrators_: Dict[int, Any] = {}
        self.regimes_: Dict[int, str] = {}
        self.n_labels_: int = 0

    def fit(self, probs: np.ndarray, Y: np.ndarray) -> "MultiLabelAdaptiveCalibrator":
        P = np.asarray(probs, dtype=np.float64)
        Y_arr = np.asarray(Y, dtype=np.int32)
        n_samples, n_labels = P.shape

        self.n_labels_ = n_labels
        self.calibrators_ = {}
        self.regimes_ = {}

        for j in range(n_labels):
            prior_j = float(np.mean(Y_arr[:, j]))
            beta_j = float(1.0 - 2.0 * abs(prior_j - 0.50))

            if beta_j >= self.tau_balance:
                # Regime 2: Symmetric / Balanced -> Preserve raw probabilities
                cal = IdentityCalibrator()
                regime = "Symmetric"
            elif prior_j <= 0.50:
                # Regime 1: Rare-Negative -> Balanced-Root Platt
                cal = TailCalibrator(
                    method=self.method,
                    clip_eps=self.clip_eps,
                    random_state=self.random_state,
                )
                cal.fit(P[:, j], Y_arr[:, j])
                regime = "Rare_Negative"
            else:
                # Regime 3: Rare-Positive -> Inverted Balanced-Root Platt
                cal = InvertedTailCalibrator(
                    method="sqrt_platt_inv",
                    clip_eps=self.clip_eps,
                    random_state=self.random_state,
                )
                cal.fit(P[:, j], Y_arr[:, j])
                regime = "Rare_Positive"

            self.calibrators_[j] = cal
            self.regimes_[j] = regime

        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        P = np.asarray(probs, dtype=np.float64)
        n_samples, n_labels = P.shape
        out = np.zeros_like(P)

        for j in range(n_labels):
            if j in self.calibrators_:
                out[:, j] = self.calibrators_[j].predict_proba(P[:, j])
            else:
                out[:, j] = np.clip(P[:, j], 0.0, 1.0)

        return np.clip(out, 0.0, 1.0)
