"""
Unit and Integration Tests for GSI-MLC-PA v6.3:
Prior-Calibrated Asymmetric Negative Verification & Adaptive Confidence Abstention.

Reference:
    - spec/spec_v6_3.md
"""

import numpy as np
import pytest

from src.models.gsi_v6_3 import GSIMLCPAv6_3Classifier
from src.selection.asymmetric_decision import (
    apply_asymmetric_abstention,
    apply_coverage_guard,
    compute_adaptive_thresholds_table,
    compute_bayes_lr_thresholds,
)
from src.calibration.tail_calibrator import MultiLabelTailCalibrator, TailCalibrator


def test_bayes_lr_thresholds_balanced():
    """Verify that under balanced prior pi=0.5, thresholds match Chow's symmetric rule exactly."""
    c = 0.30
    t0, t1 = compute_bayes_lr_thresholds(prior=0.5, cost=c)
    assert abs(t0 - c) < 1e-4
    assert abs(t1 - (1.0 - c)) < 1e-4


def test_bayes_lr_thresholds_extreme_imbalance():
    """Verify that under rare prior (pi=0.03), thresholds adapt asymmetrically."""
    c = 0.30
    t0, t1 = compute_bayes_lr_thresholds(prior=0.03, cost=c)
    assert t0 < 0.10, f"Expected tau_0 to shift near 0, got {t0}"
    assert t0 < t1, f"tau_0 ({t0}) must be less than tau_1 ({t1})"
    assert t1 <= 0.995


def test_coverage_guard():
    """Verify coverage guard expands acceptance when coverage falls below gamma_min."""
    # Simulated probabilities where many fall in the ambiguous middle zone
    probs = np.array([0.45, 0.48, 0.50, 0.52, 0.55, 0.05, 0.95], dtype=np.float64)
    tau_0_orig = 0.20
    tau_1_orig = 0.80
    # Originally only 2 out of 7 samples are decided -> coverage = 28.5% < 70%
    t0_adj, t1_adj, adjusted = apply_coverage_guard(probs, tau_0_orig, tau_1_orig, gamma_min=0.70)
    assert adjusted is True
    # Now coverage should be >= 70%
    cov_new = np.mean((probs <= t0_adj) | (probs >= t1_adj))
    assert cov_new >= 0.70 - 1e-5


def test_tail_calibrator():
    """Verify TailCalibrator handles binary and degenerate labels safely."""
    cal = TailCalibrator(random_state=42)
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    y = np.array([0, 0, 0, 1, 1])
    cal.fit(p, y)
    cal_p = cal.predict_proba(p)
    assert cal_p.shape == (5,)
    assert np.all(cal_p >= 0.0) and np.all(cal_p <= 1.0)

    # Degenerate label test (all zeros)
    cal_deg = TailCalibrator(random_state=42)
    cal_deg.fit(p, np.zeros(5, dtype=int))
    cal_p_deg = cal_deg.predict_proba(p)
    assert np.all(cal_p_deg == 0.0)


def _make_synthetic_imbalanced(n_samples: int = 200, random_state: int = 42):
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, 6).astype(np.float32)

    # Label 0: Independent balanced (~50%)
    p0 = 1.0 / (1.0 + np.exp(-(2.0 * X[:, 0])))
    y0 = (p0 > 0.5).astype(np.int32)

    # Label 1: Extreme imbalance (~5% positives)
    p1 = 1.0 / (1.0 + np.exp(-(X[:, 1] + X[:, 2] - 3.0)))
    y1 = (p1 > 0.5).astype(np.int32)

    # Label 2: Dependent on label 1 with noise (~8% positives)
    p2 = 1.0 / (1.0 + np.exp(-(1.5 * y1 + X[:, 3] - 2.5)))
    y2 = (p2 > 0.5).astype(np.int32)

    # Label 3: Correlated residual dependent label
    p3 = 1.0 / (1.0 + np.exp(-(0.5 * y2 + X[:, 4] - 2.0)))
    y3 = (p3 > 0.5).astype(np.int32)

    Y = np.column_stack([y0, y1, y2, y3])
    return X, Y


def test_gsi_v6_3_classifier_logistic():
    """Verify GSIMLCPAv6_3Classifier end-to-end execution with logistic base learner."""
    X, Y = _make_synthetic_imbalanced(n_samples=150, random_state=42)
    clf = GSIMLCPAv6_3Classifier(
        base_learner="logistic",
        stratified_threshold=0.75,
        residual_corr_threshold=0.25,
        cost=0.30,
        gamma_min=0.70,
        use_prior_adaptive=True,
        calibrate_tail=True,
        random_state=42,
    )
    clf.fit(X, Y)

    # Check fitted attributes
    assert hasattr(clf, "adaptive_thresholds_")
    assert len(clf.adaptive_thresholds_) == 4
    for j in range(4):
        t0 = clf.adaptive_thresholds_[j]["tau_0"]
        t1 = clf.adaptive_thresholds_[j]["tau_1"]
        assert t0 < t1

    # Check predict_proba
    probs = clf.predict_proba(X)
    assert probs.shape == (150, 4)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    # Check predict (selective decisions)
    preds = clf.predict(X)
    assert preds.shape == (150, 4)
    unique_vals = set(np.unique(preds))
    assert unique_vals.issubset({-1, 0, 1})

    # Check predict_full
    full_preds = clf.predict_full(X)
    assert full_preds.shape == (150, 4)
    assert set(np.unique(full_preds)).issubset({0, 1})


def test_gsi_v6_3_classifier_svm_and_mlp():
    """Verify GSIMLCPAv6_3Classifier runs with svm_calibrated and mlp base learners."""
    X, Y = _make_synthetic_imbalanced(n_samples=100, random_state=42)

    for bl in ["svm_calibrated", "mlp"]:
        clf = GSIMLCPAv6_3Classifier(
            base_learner=bl,
            stratified_threshold=0.75,
            residual_corr_threshold=0.25,
            cost=0.30,
            cv_folds=3,
            max_peeling_depth=2,
            random_state=42,
        )
        clf.fit(X, Y)
        preds = clf.predict(X)
        assert preds.shape == (100, 4)
        assert set(np.unique(preds)).issubset({-1, 0, 1})
