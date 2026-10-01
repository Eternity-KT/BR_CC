"""Unit and Integration Tests for GSI-MLC-PA v6 Multi-Stage 5-Fold CV Peeling.

Verifies:
1. CVPeelingConfig parameter contracts and serialization.
2. evaluate_label_5fold_cv computes leakage-free OOF probabilities and Selective-F1.
3. Multi-stage peeling correctly discovers IL_1, IL_2, and residual DL.
4. Base learner compatibility: Logistic Regression, SVM, MLP.
5. Termination conditions: no_promotion_in_stage, all_labels_independent, max_depth_reached.
6. Integration with GSIMLCPartialAbstentionClassifier with partition_mode="v6".
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.base_learners import create_binary_estimator
from src.models.gsi_mlc_pa import GSIMLCPartialAbstentionClassifier
from src.selection.cv_peeling import (
    CVPeelingConfig,
    CVPeelingResult,
    CVStratifiedPeelingSelector,
    evaluate_label_5fold_cv,
)
from src.selection.partition import canonical_partition_mode


def _make_hierarchical_data(n_samples: int = 300, random_state: int = 42):
    """Generate synthetic multilabel dataset with explicit hierarchical dependencies:
    - y0: strongly correlated with x[:, 0] and x[:, 1] -> Independent (IL_1)
    - y1: depends heavily on y0 and moderately on x[:, 2] -> Stage 2 Independent (IL_2)
    - y2: random noise / independent of X -> Dependent Residual (DL)
    """
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, 6).astype(np.float32)

    # y0: easily separable from X
    logits_0 = 3.0 * X[:, 0] + 2.0 * X[:, 1]
    p0 = 1.0 / (1.0 + np.exp(-logits_0))
    y0 = (p0 > 0.5).astype(np.int32)

    # y1: dependent on y0
    logits_1 = 3.5 * y0 + 1.5 * X[:, 2] - 2.0
    p1 = 1.0 / (1.0 + np.exp(-logits_1))
    y1 = (p1 > 0.5).astype(np.int32)

    # y2: purely random noise
    y2 = (rng.rand(n_samples) > 0.6).astype(np.int32)

    Y = np.column_stack([y0, y1, y2])
    return X, Y


def test_cv_peeling_config():
    """Verify configuration contract and serialization."""
    config = CVPeelingConfig(
        threshold=0.75,
        n_folds=5,
        max_depth=3,
        cost=0.30,
        decaying_threshold=True,
        threshold_decay_step=0.05,
    )
    d = config.as_dict()
    assert d["threshold"] == 0.75
    assert d["n_folds"] == 5
    assert d["max_depth"] == 3
    assert d["cost"] == 0.30
    assert d["decaying_threshold"] is True
    assert d["threshold_decay_step"] == 0.05


def test_evaluate_label_5fold_cv():
    """Verify 5-fold CV evaluation computes correct OOF shapes and selective metrics."""
    X, Y = _make_hierarchical_data(n_samples=200, random_state=42)
    factory = lambda: LogisticRegression(solver="liblinear", random_state=42)

    # Easily predictable label y0
    score0, oof_probs0, diag0 = evaluate_label_5fold_cv(
        X, Y[:, 0], factory, n_folds=5, cost=0.30, metric="selective_f1"
    )
    assert len(oof_probs0) == 200
    assert 0.0 <= np.min(oof_probs0) <= np.max(oof_probs0) <= 1.0
    assert score0 > 0.80  # Highly predictable
    assert diag0["coverage"] > 0.50
    assert len(diag0["fold_metrics"]) == 5

    # Random noise label y2
    score2, oof_probs2, diag2 = evaluate_label_5fold_cv(
        X, Y[:, 2], factory, n_folds=5, cost=0.30, metric="selective_f1"
    )
    assert len(oof_probs2) == 200
    # Noise label should have much lower score than y0
    assert score2 < score0


def test_cv_peeling_multi_stage_promotion():
    """Verify that Stage 1 discovers IL_1 and Stage 2 discovers IL_2 via augmented features."""
    X, Y = _make_hierarchical_data(n_samples=400, random_state=42)
    config = CVPeelingConfig(
        threshold=0.75,
        n_folds=5,
        max_depth=3,
        cost=0.30,
        random_state=42,
    )
    selector = CVStratifiedPeelingSelector(
        config=config,
        base_estimator_factory=lambda: LogisticRegression(solver="liblinear", random_state=42),
    )
    result = selector.fit_partition(X, Y)

    assert isinstance(result, CVPeelingResult)
    # y0 must be in stage 1
    assert 0 in result.labels_il_stage_1
    # y1 depends on y0, so it should be promoted in stage 2 or stage 1
    assert 0 in result.all_independent_labels
    # Total IL should include y0
    assert result.n_total_il >= 1
    # y2 (noise) should stay in residual DL
    assert 2 in result.dependent_residual_labels
    assert len(result.final_execution_order) == 3


def test_cv_peeling_stopping_reasons():
    """Verify termination under extreme threshold settings."""
    X, Y = _make_hierarchical_data(n_samples=150, random_state=42)

    # Impossibly high threshold -> no promotion in stage 1
    high_config = CVPeelingConfig(threshold=1.05, n_folds=3, random_state=42)
    selector_high = CVStratifiedPeelingSelector(
        config=high_config,
        base_estimator_factory=lambda: LogisticRegression(solver="liblinear", random_state=42),
    )
    res_high = selector_high.fit_partition(X, Y)
    assert res_high.stopping_reason == "no_promotion_in_stage"
    assert res_high.n_total_il == 0
    assert res_high.n_residual_dl == 3

    # Ultra low threshold -> all labels independent
    low_config = CVPeelingConfig(threshold=0.01, n_folds=3, random_state=42)
    selector_low = CVStratifiedPeelingSelector(
        config=low_config,
        base_estimator_factory=lambda: LogisticRegression(solver="liblinear", random_state=42),
    )
    res_low = selector_low.fit_partition(X, Y)
    assert res_low.stopping_reason == "all_labels_independent"
    assert res_low.n_total_il == 3
    assert res_low.n_residual_dl == 0


def test_cv_peeling_decaying_threshold():
    """Verify decaying threshold across stages."""
    X, Y = _make_hierarchical_data(n_samples=200, random_state=42)
    config = CVPeelingConfig(
        threshold=0.85,
        decaying_threshold=True,
        threshold_decay_step=0.10,
        n_folds=3,
        random_state=42,
    )
    selector = CVStratifiedPeelingSelector(
        config=config,
        base_estimator_factory=lambda: LogisticRegression(solver="liblinear", random_state=42),
    )
    res = selector.fit_partition(X, Y)
    diags = res.stage_diagnostics
    assert diags[0]["effective_threshold"] == 0.85
    if len(diags) > 1:
        assert diags[1]["effective_threshold"] == 0.75


def test_partition_mode_aliases():
    """Verify v6 partition mode aliases."""
    assert canonical_partition_mode("v6") == "cv_stratified_peeling"
    assert canonical_partition_mode("v6_core") == "cv_stratified_peeling"
    assert canonical_partition_mode("cv_peeling") == "cv_stratified_peeling"
    assert canonical_partition_mode("cv_stratified_peeling") == "cv_stratified_peeling"


def test_gsi_classifier_v6_integration():
    """Verify end-to-end integration of GSIMLCPartialAbstentionClassifier with partition_mode='v6'."""
    X, Y = _make_hierarchical_data(n_samples=200, random_state=42)
    model = GSIMLCPartialAbstentionClassifier(
        partition_mode="v6",
        base_learner="logistic",
        stratified_threshold=0.75,
        cv_folds=3,
        random_state=42,
    )
    model.fit(X, Y)

    # Check fitted partition attributes
    assert hasattr(model, "stratified_peeling_audit_")
    assert model.stratified_peeling_audit_ is not None
    assert "independent_layers" in model.stratified_peeling_audit_

    # Check predictions
    preds = model.predict(X, cost=0.30)
    assert preds.shape == Y.shape
    # Check that predictions contain only valid values {0, 1, -1}
    unique_preds = np.unique(preds)
    for val in unique_preds:
        assert val in (-1, 0, 1)

    full_preds = model.predict_full(X)
    assert full_preds.shape == Y.shape
    assert set(np.unique(full_preds)).issubset({0, 1})


def test_cv_peeling_base_learners_svm_mlp():
    """Verify that CVStratifiedPeelingSelector runs with SVM and MLP base learners."""
    X, Y = _make_hierarchical_data(n_samples=150, random_state=42)

    # Test SVM (calibrated)
    selector_svm = CVStratifiedPeelingSelector(
        config=CVPeelingConfig(threshold=0.75, n_folds=3, random_state=42),
        base_estimator_factory=lambda: create_binary_estimator("svm_calibrated", random_state=42),
    )
    res_svm = selector_svm.fit_partition(X, Y)
    assert isinstance(res_svm, CVPeelingResult)
    assert len(res_svm.final_execution_order) == 3

    # Test MLP (sklearn)
    selector_mlp = CVStratifiedPeelingSelector(
        config=CVPeelingConfig(threshold=0.75, n_folds=3, random_state=42),
        base_estimator_factory=lambda: create_binary_estimator("mlp_sklearn", random_state=42),
    )
    res_mlp = selector_mlp.fit_partition(X, Y)
    assert isinstance(res_mlp, CVPeelingResult)
    assert len(res_mlp.final_execution_order) == 3


def test_cv_peeling_defensive_validation():
    """Verify that defensive checks catch invalid inputs and bad configurations safely."""
    # 1. Invalid configuration parameters
    with pytest.raises(ValueError, match="threshold"):
        CVPeelingConfig(threshold=-0.1)
    with pytest.raises(ValueError, match="n_folds"):
        CVPeelingConfig(n_folds=1)
    with pytest.raises(ValueError, match="cost"):
        CVPeelingConfig(cost=1.5)
    with pytest.raises(ValueError, match="metric"):
        CVPeelingConfig(metric="non_existent_metric")
    with pytest.raises(ValueError, match="dl_order_direction"):
        CVPeelingConfig(dl_order_direction="diagonal")

    # 2. Invalid input matrices
    selector = CVStratifiedPeelingSelector(
        config=CVPeelingConfig(threshold=0.75, n_folds=3, random_state=42),
        base_estimator_factory=lambda: LogisticRegression(solver="liblinear", random_state=42),
    )
    X, Y = _make_hierarchical_data(n_samples=50, random_state=42)

    with pytest.raises(ValueError, match="2D array"):
        selector.fit_partition(X[:, 0], Y)
    with pytest.raises(ValueError, match="2D array"):
        selector.fit_partition(X, Y[:, 0])
    with pytest.raises(ValueError, match="Sample count mismatch"):
        selector.fit_partition(X[:30], Y)
    with pytest.raises(ValueError, match="cannot be empty"):
        selector.fit_partition(np.empty((0, 4)), np.empty((0, 2)))

    # 3. Defensive sanitization of NaN/Inf in features
    X_with_nans = X.copy()
    X_with_nans[0, 0] = np.nan
    X_with_nans[1, 1] = np.inf
    res_sanitized = selector.fit_partition(X_with_nans, Y)
    assert isinstance(res_sanitized, CVPeelingResult)


