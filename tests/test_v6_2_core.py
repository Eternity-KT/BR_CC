"""
Unit and Integration Tests for GSI-MLC-PA v6.2 (Residual Error Correlation & Conditional Dependency).

Reference:
    - meeting_summary.md (Core Section, v6.2)
    - spec/spec_v6_2.md

Verifies:
1. Residual Calculation & PCC Matrix:
   - compute_br_residual_matrix computes signed, absolute, and binary errors accurately.
   - compute_residual_pcc_matrix produces symmetric, bounded [0, 1] matrices with diagonal 1.0.
   - Handles zero-variance (constant prediction) columns safely without NaN or Inf.
2. Conditional Dependency Graph (DL_temp):
   - build_residual_dependency_graph couples labels exceeding tau_corr.
   - No self-loops (l not in DL_temp[l]).
   - Partners sorted descending by correlation strength.
3. GSIMLCPAv6_2Classifier End-to-End:
   - Full fit, predict_proba, and predict with partial abstention (cost c = 0.30).
   - Singleton DL promotion rule (|DL| == 1 => IL).
   - Robustness under boundary conditions (all-IL, all-DL).
   - Base learner support: Logistic Regression and Calibrated SVM.
"""

import numpy as np
import pytest

from src.models.gsi_v6_2 import GSIMLCPAv6_2Classifier
from src.selection.br_residual_correlation import (
    build_residual_dependency_graph,
    compute_br_residual_matrix,
    compute_residual_pcc_matrix,
    export_residual_correlation_audit_table,
)


def _make_synthetic_multilabel(n_samples: int = 250, random_state: int = 42):
    """
    Generate synthetic multilabel dataset with 5 labels:
    - y0: highly predictable from X[:, 0:2] -> IL_1
    - y1: highly predictable from y0 and X[:, 2] -> IL_2
    - y2: dependent label (noisy function of y1 and X[:, 3]) -> DL
    - y3: correlated with y2 in noise/errors -> DL (should couple with y2!)
    - y4: independent background noise label -> DL
    """
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, 8).astype(np.float32)

    # Label 0: Independent
    p0 = 1.0 / (1.0 + np.exp(-(3.0 * X[:, 0] + 2.5 * X[:, 1])))
    y0 = (p0 > 0.5).astype(np.int32)

    # Label 1: Predictable once y0 is known
    p1 = 1.0 / (1.0 + np.exp(-(3.5 * y0 + 2.0 * X[:, 2] - 2.0)))
    y1 = (p1 > 0.5).astype(np.int32)

    # Labels 2 and 3: Hard labels with shared latent noise causing correlated BR errors
    latent_noise = rng.randn(n_samples) * 1.5
    p2 = 1.0 / (1.0 + np.exp(-(0.8 * X[:, 3] + latent_noise)))
    y2 = (p2 > 0.5).astype(np.int32)

    p3 = 1.0 / (1.0 + np.exp(-(0.7 * X[:, 4] + latent_noise)))
    y3 = (p3 > 0.5).astype(np.int32)

    # Label 4: Noisy uncoupled
    p4 = 1.0 / (1.0 + np.exp(-(0.5 * X[:, 5] + rng.randn(n_samples) * 2.0)))
    y4 = (p4 > 0.5).astype(np.int32)

    Y = np.column_stack([y0, y1, y2, y3, y4])
    return X, Y


def test_br_residual_matrix_computation():
    """Verify calculation of signed, absolute, and binary residual matrices."""
    Y_true = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0]
    ], dtype=np.int32)

    P_prob = np.array([
        [0.8, 0.2, 0.3],
        [0.1, 0.9, 0.7],
        [0.6, 0.4, 0.1]
    ], dtype=np.float64)

    # Signed residual: Y - P
    res_signed = compute_br_residual_matrix(Y_true, P_prob, error_format="residual")
    np.testing.assert_allclose(
        res_signed,
        [[0.2, -0.2, 0.7], [-0.1, 0.1, -0.7], [0.4, 0.6, -0.1]],
        atol=1e-5
    )

    # Absolute residual: |Y - P|
    res_abs = compute_br_residual_matrix(Y_true, P_prob, error_format="abs_residual")
    np.testing.assert_allclose(
        res_abs,
        np.abs(res_signed),
        atol=1e-5
    )

    # Binary error: |Y - (P >= 0.5)|
    res_bin = compute_br_residual_matrix(Y_true, P_prob, error_format="binary")
    expected_bin = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0]
    ], dtype=np.float64)
    np.testing.assert_array_equal(res_bin, expected_bin)


def test_residual_pcc_matrix_properties():
    """Verify PCC matrix symmetry, bounding [0, 1], diagonal 1.0, and constant column safety."""
    rng = np.random.RandomState(42)
    # 4 columns: col 0 and 1 are correlated; col 2 is random; col 3 is zero variance
    e0 = rng.randn(100)
    e1 = 0.8 * e0 + 0.2 * rng.randn(100)
    e2 = rng.randn(100)
    e3 = np.zeros(100)  # Constant prediction (0 variance)

    residuals = np.column_stack([e0, e1, e2, e3])
    corr = compute_residual_pcc_matrix(residuals, use_absolute=True)

    assert corr.shape == (4, 4)
    # Check diagonal
    np.testing.assert_allclose(np.diag(corr), [1.0, 1.0, 1.0, 1.0])
    # Check symmetry
    np.testing.assert_allclose(corr, corr.T, atol=1e-6)
    # High correlation between col 0 and 1
    assert corr[0, 1] > 0.75
    # Constant column should not yield NaN or Inf
    assert not np.isnan(corr).any()
    assert not np.isinf(corr).any()
    assert corr[0, 3] == 0.0
    assert corr[3, 0] == 0.0


def test_build_residual_dependency_graph():
    """Verify graph extraction based on thresholding and no self-loops."""
    # Corr matrix for labels [2, 3, 4]
    corr = np.array([
        [1.0, 0.45, 0.10],
        [0.45, 1.0, 0.30],
        [0.10, 0.30, 1.0]
    ], dtype=np.float64)

    dl_labels = [2, 3, 4]
    graph = build_residual_dependency_graph(corr, dl_labels, threshold=0.25)

    # For label 2: partner 3 has 0.45 >= 0.25; partner 4 has 0.10 < 0.25
    assert graph[2] == [3]
    # For label 3: partner 2 (0.45) and partner 4 (0.30), sorted descending
    assert graph[3] == [2, 4]
    # For label 4: partner 3 (0.30)
    assert graph[4] == [3]

    # Verify audit table export
    audit_df = export_residual_correlation_audit_table(corr, dl_labels, graph)
    assert len(audit_df) == 3
    assert "top_correlated_partner" in audit_df.columns
    assert "num_coupled" in audit_df.columns


def test_gsi_v6_2_end_to_end_synthetic():
    """Verify full end-to-end fit, predict_proba, and predict with partial abstention."""
    X, Y = _make_synthetic_multilabel(n_samples=200, random_state=42)

    model = GSIMLCPAv6_2Classifier(
        base_learner="logistic",
        stratified_threshold=0.70,
        residual_corr_threshold=0.20,
        cv_folds=3,
        max_peeling_depth=2,
        cost=0.30,
        random_state=42,
    )
    model.fit(X, Y)

    assert model.is_fitted_
    assert model.n_labels_ == 5
    assert model.n_features_in_ == 8

    # Predictions
    probs = model.predict_proba(X)
    assert probs.shape == (200, 5)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    # Predict with partial abstention (cost = 0.30)
    preds = model.predict(X, cost=0.30)
    assert preds.shape == (200, 5)
    assert np.all(np.isin(preds, [0, 1, -1]))

    # Predict full without abstention
    preds_full = model.predict_full(X, threshold=0.5)
    assert preds_full.shape == (200, 5)
    assert np.all(np.isin(preds_full, [0, 1]))

    # If DL has labels, audit table must exist
    if len(model.dependent_labels_) >= 2:
        assert model.residual_corr_matrix_ is not None
        assert model.audit_table_ is not None


def test_gsi_v6_2_singleton_dl_promotion():
    """Verify meeting_summary.md rule: when only 1 label is in DL, promote directly to IL."""
    rng = np.random.RandomState(42)
    X = rng.randn(150, 4).astype(np.float32)

    # 2 easily separable labels, 1 noisy label
    y0 = (X[:, 0] > 0).astype(np.int32)
    y1 = (X[:, 1] > 0).astype(np.int32)
    y2 = rng.binomial(1, 0.1, size=150).astype(np.int32)  # Hard/rare label

    Y = np.column_stack([y0, y1, y2])

    model = GSIMLCPAv6_2Classifier(
        base_learner="logistic",
        stratified_threshold=0.60,
        cv_folds=3,
        random_state=42,
    )
    model.fit(X, Y)

    # Candidate DL should have had singleton y2 promoted, leaving DL empty
    assert len(model.dependent_labels_) == 0
    assert len(model.independent_labels_) == 3

    probs = model.predict_proba(X)
    assert probs.shape == (150, 3)


def test_gsi_v6_2_all_independent_and_all_dependent():
    """Verify boundary conditions: all labels independent vs all labels dependent."""
    rng = np.random.RandomState(99)
    X = rng.randn(120, 4).astype(np.float32)
    Y = (rng.rand(120, 3) > 0.5).astype(np.int32)

    # 1. Very high threshold -> All labels remain in DL
    model_dl = GSIMLCPAv6_2Classifier(
        base_learner="logistic",
        stratified_threshold=0.99,  # Unreachable threshold
        cv_folds=3,
        random_state=42,
    )
    model_dl.fit(X, Y)
    assert len(model_dl.independent_labels_) == 0
    assert len(model_dl.dependent_labels_) == 3
    probs_dl = model_dl.predict_proba(X)
    assert probs_dl.shape == (120, 3)

    # 2. Very low threshold -> All labels promoted to IL
    model_il = GSIMLCPAv6_2Classifier(
        base_learner="logistic",
        stratified_threshold=0.01,  # Easily reachable
        cv_folds=3,
        random_state=42,
    )
    model_il.fit(X, Y)
    assert len(model_il.independent_labels_) == 3
    assert len(model_il.dependent_labels_) == 0
    probs_il = model_il.predict_proba(X)
    assert probs_il.shape == (120, 3)


def test_gsi_v6_2_calibrated_svm():
    """Verify base learner compatibility with calibrated SVM."""
    X, Y = _make_synthetic_multilabel(n_samples=120, random_state=42)

    model = GSIMLCPAv6_2Classifier(
        base_learner="svm_calibrated",
        stratified_threshold=0.70,
        residual_corr_threshold=0.25,
        cv_folds=3,
        random_state=42,
    )
    model.fit(X, Y)
    probs = model.predict_proba(X)
    assert probs.shape == (120, 5)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
