"""
Unit Tests for GSI-MLC-PA v6.3.3 Core Components.
Tests:
1. Label balance factor beta_l computation.
2. Tri-Regime threshold computation & smooth boundary blending.
3. Directional precision guards (tau_1 >= 0.50 for rare-pos, tau_0 <= 0.50 for rare-neg).
4. MultiLabelAdaptiveCalibrator regime assignment.
5. GSIMLCPAv6_3_3Classifier fit, predict_proba, and predict on synthetic multi-label data.
"""

import numpy as np
import pytest

from src.selection.tri_regime_decision import (
    compute_label_balance_factor,
    compute_tri_regime_raw_thresholds,
    apply_dual_coverage_guard,
    compute_tri_regime_adaptive_thresholds_table,
    apply_tri_regime_abstention,
)
from src.calibration.adaptive_calibrator import (
    IdentityCalibrator,
    InvertedTailCalibrator,
    MultiLabelAdaptiveCalibrator,
)
from src.models.gsi_v6_3_3 import GSIMLCPAv6_3_3Classifier


def test_label_balance_factor():
    """Verify beta_l equals 1.0 at balance and 0.0 at extremes."""
    assert compute_label_balance_factor(0.50) == pytest.approx(1.0, abs=1e-5)
    assert compute_label_balance_factor(0.0) == pytest.approx(0.0, abs=1e-5)
    assert compute_label_balance_factor(1.0) == pytest.approx(0.0, abs=1e-5)
    assert compute_label_balance_factor(0.30) == pytest.approx(0.60, abs=1e-5)
    assert compute_label_balance_factor(0.70) == pytest.approx(0.60, abs=1e-5)
    assert compute_label_balance_factor(0.10) == pytest.approx(0.20, abs=1e-5)
    assert compute_label_balance_factor(0.90) == pytest.approx(0.20, abs=1e-5)


def test_tri_regime_thresholds_regimes():
    """Verify threshold behavior across the 3 regimes."""
    cost = 0.30

    # 1. Perfectly balanced (pi = 0.50) -> Must recover Chow rule [0.30, 0.70]
    t0_mid, t1_mid, regime_mid, blend_mid = compute_tri_regime_raw_thresholds(
        prior=0.50, cost=cost, tau_balance=0.40
    )
    assert regime_mid == "Symmetric"
    assert blend_mid > 0.99
    assert t0_mid == pytest.approx(0.30, abs=0.01)
    assert t1_mid == pytest.approx(0.70, abs=0.01)

    # 2. Extreme rare positive (pi = 0.02) -> Regime 1 (Rare_Negative)
    t0_low, t1_low, regime_low, blend_low = compute_tri_regime_raw_thresholds(
        prior=0.02, cost=cost, tau_balance=0.40
    )
    assert regime_low == "Rare_Negative"
    assert blend_low < 0.05
    assert t0_low < 0.15
    assert t1_low >= 0.50  # Precision guard invariant

    # 3. Extreme rare negative (pi = 0.95) -> Regime 3 (Rare_Positive)
    t0_high, t1_high, regime_high, blend_high = compute_tri_regime_raw_thresholds(
        prior=0.95, cost=cost, tau_balance=0.40
    )
    assert regime_high == "Rare_Positive"
    assert blend_high < 0.05
    assert t0_high <= 0.50  # Negative precision guard invariant
    assert t1_high > 0.85


def test_smooth_blending_continuity():
    """Verify smooth sigmoid blending has no discontinuous jumps across prior sweep."""
    priors = np.linspace(0.01, 0.99, 100)
    t0_list = []
    t1_list = []

    for p in priors:
        t0, t1, _, _ = compute_tri_regime_raw_thresholds(prior=p, cost=0.30, tau_balance=0.40)
        t0_list.append(t0)
        t1_list.append(t1)

    # Max step difference between adjacent evaluation points should be small (continuous)
    max_step_t0 = np.max(np.abs(np.diff(t0_list)))
    max_step_t1 = np.max(np.abs(np.diff(t1_list)))
    assert max_step_t0 < 0.08
    assert max_step_t1 < 0.08


def test_multilabel_adaptive_calibrator():
    """Verify adaptive calibrator selects Identity for balanced labels and Platt for tails."""
    np.random.seed(42)
    N = 200
    probs = np.random.uniform(0.1, 0.9, size=(N, 3))

    # Column 0: Rare positive (~5%)
    y0 = (np.random.rand(N) < 0.05).astype(int)
    # Column 1: Balanced (~50%)
    y1 = (np.random.rand(N) < 0.50).astype(int)
    # Column 2: Rare negative (~90% positive)
    y2 = (np.random.rand(N) < 0.90).astype(int)

    Y = np.column_stack([y0, y1, y2])

    calibrator = MultiLabelAdaptiveCalibrator(tau_balance=0.40)
    calibrator.fit(probs, Y)

    # Check regime mapping
    assert calibrator.regimes_[0] == "Rare_Negative"
    assert calibrator.regimes_[1] == "Symmetric"
    assert calibrator.regimes_[2] == "Rare_Positive"

    # Verify column 1 uses IdentityCalibrator
    assert isinstance(calibrator.calibrators_[1], IdentityCalibrator)

    # Transform test
    cal_probs = calibrator.transform(probs)
    assert cal_probs.shape == probs.shape
    assert np.all(cal_probs >= 0.0) and np.all(cal_probs <= 1.0)
    # Column 1 should match raw probs exactly
    assert np.allclose(cal_probs[:, 1], probs[:, 1])


def test_gsi_v6_3_3_classifier_end_to_end():
    """Test full GSIMLCPAv6_3_3Classifier training and inference on synthetic dataset."""
    np.random.seed(42)
    N, d, K = 120, 8, 4
    X = np.random.randn(N, d).astype(np.float32)

    # Create synthetic Y with varied imbalance
    # Label 0: Rare positive (5%)
    # Label 1: Balanced (50%)
    # Label 2: Mild (35%)
    # Label 3: Rare negative (85%)
    y0 = (np.random.rand(N) < 0.08).astype(int)
    y1 = (np.random.rand(N) < 0.50).astype(int)
    y2 = (np.random.rand(N) < 0.35).astype(int)
    y3 = (np.random.rand(N) < 0.85).astype(int)
    Y = np.column_stack([y0, y1, y2, y3])

    clf = GSIMLCPAv6_3_3Classifier(
        base_learner="logistic",
        stratified_threshold=0.70,
        residual_corr_threshold=0.25,
        cv_folds=3,
        cost=0.30,
        tau_balance=0.40,
        random_state=42,
    )

    clf.fit(X, Y)

    assert clf.is_fitted_
    assert hasattr(clf, "adaptive_thresholds_")
    assert len(clf.adaptive_thresholds_) == K

    # Inspect regimes
    for j in range(K):
        th = clf.adaptive_thresholds_[j]
        assert "regime" in th
        assert "beta" in th
        assert "blend_weight" in th
        assert th["tau_0"] < th["tau_1"]

    # Test predict_proba
    probs = clf.predict_proba(X)
    assert probs.shape == (N, K)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    # Test predict (partial abstention in {-1, 0, 1})
    preds = clf.predict(X)
    assert preds.shape == (N, K)
    unique_vals = set(np.unique(preds))
    assert unique_vals.issubset({-1, 0, 1})

    # Test predict_full (0 or 1 only)
    full_preds = clf.predict_full(X)
    assert full_preds.shape == (N, K)
    assert set(np.unique(full_preds)).issubset({0, 1})
