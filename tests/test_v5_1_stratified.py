"""Unit tests for GSI-MLC-PA v5.1 components:
1. Standalone Complexity Penalty Evaluator
2. Optimal binary threshold search and ascending correlation ordering
3. Multi-stage Stratified Peeling Selector
4. GSIMLCPartialAbstentionClassifier integration with stratified peeling
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.gsi_mlc_pa import GSIMLCPartialAbstentionClassifier
from src.selection.complexity_penalty import (
    ComplexityPenaltyConfig,
    ComplexityPenaltyEvaluator,
)
from src.selection.stratified_peeling import (
    StratifiedPeelingConfig,
    StratifiedPeelingSelector,
    compute_label_correlation_matrix,
    find_optimal_binary_threshold,
    order_dl_by_correlation,
)


def test_complexity_penalty_evaluator_standalone():
    """Verify that ComplexityPenaltyEvaluator correctly computes P(t, |D|) and stopping condition."""
    config = ComplexityPenaltyConfig(
        penalty_lambda=0.05,
        penalty_beta=0.5,
        enabled=True,
    )
    evaluator = ComplexityPenaltyEvaluator(config)

    # Stage 1: t=1, exp(beta*0) = 1.0. With 4 out of 10 labels: P = 0.4
    raw_p1 = evaluator.compute_raw_penalty(stage=1, remaining_dl_count=4, total_labels=10)
    assert np.isclose(raw_p1, 0.40, atol=1e-6)

    # When F1 gain is 0.05 > weighted penalty 0.05 * 0.4 = 0.02 -> should not stop
    res1 = evaluator.evaluate(stage=1, remaining_dl_count=4, total_labels=10, f1_gain=0.05)
    assert not res1.should_stop
    assert res1.penalized_utility > 0

    # Stage 3: t=3, exp(0.5 * 2) = exp(1.0) approx 2.718.
    # With 4 out of 10 labels: P = 0.4 * 2.718 = 1.087
    # Weighted penalty = 0.05 * 1.087 = 0.0543.
    # If F1 gain is only 0.01 -> utility < 0 -> should stop!
    res3 = evaluator.evaluate(stage=3, remaining_dl_count=4, total_labels=10, f1_gain=0.01)
    assert res3.should_stop
    assert res3.penalized_utility <= 0


def test_ascending_vs_descending_correlation_order():
    """Verify that order_dl_by_correlation puts smallest total correlation first in ascending mode."""
    # Synthetic correlation matrix for 4 labels:
    # Label 0: total corr with others = 0.1
    # Label 1: total corr with others = 0.8
    # Label 2: total corr with others = 0.4
    # Label 3: total corr with others = 0.6
    corr = np.array([
        [1.0, 0.05, 0.02, 0.03],  # sum = 0.10
        [0.05, 1.0, 0.35, 0.40],  # sum = 0.80
        [0.02, 0.35, 1.0, 0.03],  # sum = 0.40
        [0.03, 0.40, 0.03, 1.0],  # sum = 0.46
    ])
    dl_labels = [0, 1, 2, 3]

    asc_order = order_dl_by_correlation(corr, dl_labels, direction="ascending")
    assert asc_order == [0, 2, 3, 1], f"Expected [0, 2, 3, 1], got {asc_order}"

    desc_order = order_dl_by_correlation(corr, dl_labels, direction="descending")
    assert desc_order == [1, 3, 2, 0], f"Expected [1, 3, 2, 0], got {desc_order}"


def test_stratified_peeling_selector_synthetic_peeling():
    """Verify that StratifiedPeelingSelector correctly identifies IL_1, IL_2, and residual DL."""
    np.random.seed(42)
    n_train = 300
    n_val = 150
    n_features = 10

    X_train = np.random.randn(n_train, n_features)
    X_val = np.random.randn(n_val, n_features)

    # Label 0 is easily predicted directly from X (feature 0) -> should be IL_1
    y0_train = (X_train[:, 0] > 0.0).astype(int)
    y0_val = (X_val[:, 0] > 0.0).astype(int)

    # Label 1 is NOT easily predicted from X alone, but depends strongly on Label 0 -> should be IL_2
    y1_train = (y0_train ^ (np.random.rand(n_train) > 0.90).astype(int)).astype(int)
    y1_val = (y0_val ^ (np.random.rand(n_val) > 0.90).astype(int)).astype(int)

    # Label 2 is purely random noise -> should remain in DL_residual
    y2_train = (np.random.rand(n_train) > 0.5).astype(int)
    y2_val = (np.random.rand(n_val) > 0.5).astype(int)

    Y_train = np.column_stack([y0_train, y1_train, y2_train])
    Y_val = np.column_stack([y0_val, y1_val, y2_val])

    selector = StratifiedPeelingSelector(
        config=StratifiedPeelingConfig(
            threshold=0.75,
            max_depth=3,
            dl_order_direction="ascending",
            use_complexity_penalty=False,
        ),
        base_estimator_factory=lambda: LogisticRegression(solver="liblinear", random_state=42),
    )

    result = selector.fit_partition(X_train, Y_train, X_val, Y_val)

    # Verify Label 0 is in the first layer
    assert 0 in result.independent_layers[0], f"Label 0 should be in IL_1, got {result.independent_layers}"
    assert 2 in result.dependent_residual_labels, f"Label 2 should remain in DL, got {result.dependent_residual_labels}"
    assert len(result.final_execution_order) == 3


def test_gsi_classifier_with_stratified_peeling():
    """Verify that GSIMLCPartialAbstentionClassifier integrates smoothly with partition_mode='stratified_peeling'."""
    np.random.seed(42)
    n_samples = 200
    n_features = 8
    n_labels = 4

    X = np.random.randn(n_samples, n_features)
    Y = np.zeros((n_samples, n_labels), dtype=int)
    Y[:, 0] = (X[:, 0] > 0).astype(int)
    Y[:, 1] = (X[:, 1] + 0.5 * Y[:, 0] > 0).astype(int)
    Y[:, 2] = (X[:, 2] > 0.5).astype(int)
    Y[:, 3] = (Y[:, 1] & (X[:, 3] > 0)).astype(int)

    clf = GSIMLCPartialAbstentionClassifier(
        base_learner="logistic",
        partition_mode="stratified_peeling",
        stratified_threshold=0.70,
        dl_order_direction="ascending",
        final_order="ascending_correlation",
        decision_policy="macro_f1",
        cost=0.30,
        random_state=42,
    )

    clf.fit(X, Y)

    assert hasattr(clf, "stratified_peeling_audit_")
    assert clf.stratified_peeling_audit_ is not None
    assert "independent_layers" in clf.stratified_peeling_audit_

    preds = clf.predict(X)
    assert preds.shape == (n_samples, n_labels)
    # Ensure abstain values are valid (-1, 0, 1)
    assert np.all(np.isin(preds, (-1, 0, 1)))

    probs = clf.predict_proba(X)
    assert probs.shape == (n_samples, n_labels)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_stratified_peeling_with_complexity_penalty():
    """Verify that complexity penalty can be enabled and successfully restricts unnecessary stages."""
    np.random.seed(42)
    n_train = 200
    n_val = 100
    n_features = 8
    n_labels = 4

    X_train = np.random.randn(n_train, n_features)
    X_val = np.random.randn(n_val, n_features)
    Y_train = np.random.randint(0, 2, size=(n_train, n_labels))
    Y_val = np.random.randint(0, 2, size=(n_val, n_labels))

    # GSI model with penalty enabled
    clf = GSIMLCPartialAbstentionClassifier(
        base_learner="logistic",
        partition_mode="stratified_peeling",
        stratified_threshold=0.70,
        use_complexity_penalty=True,
        complexity_penalty_lambda=0.01,
        random_state=42,
    )
    clf.fit(X_train, Y_train)

    audit = clf.stratified_peeling_audit_
    assert audit is not None
    assert "stopping_reason" in audit
    assert "num_stages_executed" in audit
    preds = clf.predict(X_val)
    assert preds.shape == (n_val, n_labels)

