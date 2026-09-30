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


def test_normalize_augmented_probabilities_strategies():
    """Verify that normalize_augmented_probabilities correctly implements all normalization strategies."""
    from src.selection.stratified_peeling import normalize_augmented_probabilities

    probs = np.array([[0.0, 0.5, 1.0], [0.2, 0.8, 0.4]], dtype=np.float32)

    # Strategy: "none"
    p_none = normalize_augmented_probabilities(probs, reference_X=np.ones((2, 2)), strategy="none")
    assert np.allclose(p_none, probs)

    # Strategy: "centered" -> 2*p - 1.0 in [-1.0, 1.0]
    p_centered = normalize_augmented_probabilities(probs, reference_X=np.ones((2, 2)), strategy="centered")
    assert np.allclose(p_centered[0], [-1.0, 0.0, 1.0])
    assert np.all(p_centered >= -1.0) and np.all(p_centered <= 1.0)

    # Strategy: "matching" with negative values in reference_X -> centered
    p_matching_neg = normalize_augmented_probabilities(probs, reference_X=np.array([[-1.0, 0.5]]), strategy="matching")
    assert np.allclose(p_matching_neg, p_centered)

    # Strategy: "matching" with non-negative reference_X -> leaves as [0, 1]
    p_matching_pos = normalize_augmented_probabilities(probs, reference_X=np.array([[0.1, 0.5]]), strategy="matching")
    assert np.allclose(p_matching_pos, probs)

    # Strategy: "standard"
    p_std = normalize_augmented_probabilities(probs, reference_X=np.ones((2, 2)), strategy="standard")
    assert p_std.shape == probs.shape
    assert np.allclose(np.mean(p_std, axis=0), 0.0, atol=1e-5)


def test_stratified_peeling_audit_properties_and_decaying_threshold():
    """Verify that StratifiedPeelingResult exposes fine-grained stage counts and percentages."""
    from src.selection.stratified_peeling import (
        StratifiedPeelingConfig,
        StratifiedPeelingResult,
        StratifiedPeelingSelector,
    )

    # Create synthetic result
    res = StratifiedPeelingResult(
        independent_layers=((0, 1), (2,)),
        all_independent_labels=(0, 1, 2),
        dependent_residual_labels=(3, 4),
        final_execution_order=(0, 1, 2, 3, 4),
        stage_diagnostics=(),
        stopping_reason="no_promotion",
        num_stages_executed=2,
        selection_time_seconds=0.12,
    )

    assert res.n_il_stage_1 == 2
    assert res.labels_il_stage_1 == (0, 1)
    assert res.n_il_stage_2 == 1
    assert res.labels_il_stage_2 == (2,)
    assert res.n_il_stage_3 == 0
    assert res.n_total_il == 3
    assert res.n_residual_dl == 2
    assert res.total_labels == 5
    assert np.isclose(res.pct_total_il, 60.0)
    assert np.isclose(res.pct_residual_dl, 40.0)

    d = res.as_dict()
    assert d["n_il_stage_1"] == 2
    assert d["labels_il_stage_1"] == [0, 1]
    assert d["pct_total_il"] == 60.0


def test_sparse_classifier_chain_active_parents_filtering():
    """Verify that ClassifierChainClassifier correctly filters active predecessors by correlation threshold."""
    from src.models.classifier_chain import ClassifierChainClassifier

    np.random.seed(42)
    n_samples = 150
    n_features = 6
    X = np.random.randn(n_samples, n_features)

    # 4 labels:
    # 0 and 1 are correlated (corr ~ 0.95)
    # 2 is independent
    # 3 is correlated with 1 (corr ~ 0.85)
    y0 = (X[:, 0] > 0).astype(int)
    y1 = y0.copy()
    flip_idx = np.random.choice(n_samples, size=5, replace=False)
    y1[flip_idx] = 1 - y1[flip_idx]
    y2 = (np.random.rand(n_samples) > 0.5).astype(int)
    y3 = (X[:, 1] > 0).astype(int)
    Y = np.column_stack([y0, y1, y2, y3])

    corr_matrix = np.corrcoef(Y, rowvar=False)
    abs_corr = np.abs(np.nan_to_num(corr_matrix, nan=0.0))

    # Order: [0, 1, 2, 3] with threshold 0.75
    # For label 1: predecessor 0 has corr > 0.75 -> active parent: [0]
    # For label 2: predecessors 0, 1 have corr < 0.75 -> active parents: []
    chain = ClassifierChainClassifier(
        base_estimator="logistic",
        order=[0, 1, 2, 3],
        correlation_matrix=abs_corr,
        correlation_threshold=0.75,
        random_state=42,
    )
    chain.fit(X, Y)

    assert 0 in chain.active_parents_map_[1]
    assert len(chain.active_parents_map_[2]) == 0  # Label 2 has no correlated predecessors!

    probs = chain.predict_proba(X)
    assert probs.shape == (n_samples, 4)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    preds = chain.predict(X)
    assert preds.shape == (n_samples, 4)


def test_gsi_with_sparse_cc_threshold_integration():
    """Verify end-to-end integration of GSI-MLC-PA v5.1.1 with sparse_cc_threshold and normalization."""
    np.random.seed(42)
    n_train = 200
    n_val = 100
    n_features = 8
    n_labels = 4

    X_train = np.random.randn(n_train, n_features)
    X_val = np.random.randn(n_val, n_features)
    Y_train = np.random.randint(0, 2, size=(n_train, n_labels))
    Y_val = np.random.randint(0, 2, size=(n_val, n_labels))

    clf = GSIMLCPartialAbstentionClassifier(
        base_learner="logistic",
        partition_mode="stratified_peeling",
        stratified_threshold=0.65,
        sparse_cc_threshold=0.75,
        aug_normalization="matching",
        random_state=42,
    )
    clf.fit(X_train, Y_train)

    probs = clf.predict_proba(X_val)
    assert probs.shape == (n_val, n_labels)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    preds = clf.predict(X_val)
    assert preds.shape == (n_val, n_labels)
    assert np.all(np.isin(preds, (-1, 0, 1)))


def test_layer_ablation_evaluator():
    """Verify that LayerAblationEvaluator correctly configures the 6 regimes and calculates metrics."""
    from src.evaluation.layer_ablation import LayerAblationEvaluator, evaluate_label_subset

    n_samples = 50
    n_labels = 6
    y_true = np.random.randint(0, 2, size=(n_samples, n_labels))
    y_full = np.random.randint(0, 2, size=(n_samples, n_labels))
    y_partial = y_full.copy()
    y_partial[0, :] = -1  # One abstained instance

    # 3 layers: IL1 = [0, 1], IL2 = [2], IL3 = [] (empty), DL = [3, 4, 5]
    evaluator = LayerAblationEvaluator(
        independent_layers=[[0, 1], [2]],
        dependent_residual_labels=[3, 4, 5],
        total_labels=n_labels,
        abstain_value=-1,
    )

    regimes = evaluator.get_regime_subsets()
    assert regimes["ABL_1_ONLY_IL1"] == [0, 1]
    assert regimes["ABL_2_ONLY_IL2"] == [2]
    assert regimes["ABL_2_ACCUM_IL12"] == [0, 1, 2]
    assert regimes["ABL_3_ONLY_IL3"] == []
    assert regimes["ABL_3_ALL_IL"] == [0, 1, 2]
    assert regimes["ABL_3_ONLY_DL"] == [3, 4, 5]
    assert regimes["FULL_SYSTEM"] == [0, 1, 2, 3, 4, 5]

    res = evaluator.evaluate_all_regimes(y_true, y_full, y_partial)
    assert len(res) == 7

    # IL1 regime should have 2 labels
    assert res["ABL_1_ONLY_IL1"]["num_labels"] == 2
    assert 0.0 <= res["ABL_1_ONLY_IL1"]["coverage"] <= 1.0
    assert 0.0 <= res["ABL_1_ONLY_IL1"]["selective_macro_f1"] <= 1.0

    # Empty IL3 regime should have 0 labels and NaN coverage
    assert res["ABL_3_ONLY_IL3"]["num_labels"] == 0
    assert np.isnan(res["ABL_3_ONLY_IL3"]["coverage"])

    # Full system has 6 labels
    assert res["FULL_SYSTEM"]["num_labels"] == 6




