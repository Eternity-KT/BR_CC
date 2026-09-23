"""Unit tests for v5 improvements and fixes:
1. Anti-degeneracy in PerLabelMacroF1Policy (no all-ones collapse)
2. Cost sweeping dynamics and Chow boundary condition (c=0.5 -> Coverage=1.0)
3. Elimination of data leakage in GSIMLCPartialAbstentionClassifier
4. MLC-PA parity with Macro-F1 policy and min_coverage
"""

import numpy as np
import pytest

from src.decision.macro_f1 import PerLabelMacroF1Policy
from src.models.gsi_mlc_pa import GSIMLCPartialAbstentionClassifier
from src.models.mlc_pa import MLCPartialAbstentionClassifier
from src.models.registry import create_registered_model


def test_anti_degeneracy_guardrail():
    """Verify that PerLabelMacroF1Policy never collapses to all-ones (tau_low=tau_high=0.0) on imbalanced data."""
    np.random.seed(42)
    n_samples = 300
    n_labels = 4

    # Skewed ground truth: priors are 10%, 20%, 30%, 5%
    priors = [0.10, 0.20, 0.30, 0.05]
    y_true = np.column_stack([
        np.random.binomial(1, p, size=n_samples) for p in priors
    ])

    # Probabilities with noise and poor separation
    probs = np.column_stack([
        np.clip(np.where(y_true[:, k] == 1, np.random.beta(3, 2, n_samples), np.random.beta(1, 4, n_samples)), 0.01, 0.99)
        for k in range(n_labels)
    ])

    policy = PerLabelMacroF1Policy(cost=0.3, min_coverage=0.80)
    policy.fit(probs, y_true)

    tau_low = policy.tau_low
    tau_high = policy.tau_high

    # Crucial assertion: no label should have tau_low == 0.0 AND tau_high == 0.0
    for k in range(n_labels):
        assert not (tau_low[k] == 0.0 and tau_high[k] == 0.0), (
            f"Label {k} collapsed to all-ones: tau_low={tau_low[k]}, tau_high={tau_high[k]}"
        )

    # Test predictions: positive prediction rate must not be 1.0
    preds = policy.predict(probs)
    for k in range(n_labels):
        pos_rate = np.mean(preds[:, k] == 1)
        assert pos_rate < 0.80, (
            f"Label {k} has degenerate positive prediction rate: {pos_rate:.3f}"
        )


def test_chow_boundary_and_cost_monotonicity():
    """Verify Chow boundary (c >= 0.5 -> Coverage = 1.0) and non-decreasing coverage across costs."""
    np.random.seed(42)
    n_samples = 400
    y_true = np.random.binomial(1, 0.3, size=(n_samples, 3))
    probs = np.clip(
        np.where(y_true == 1, np.random.beta(4, 2, size=(n_samples, 3)), np.random.beta(2, 4, size=(n_samples, 3))),
        0.0, 1.0
    )

    policy = PerLabelMacroF1Policy(cost=0.3, min_coverage=0.80)
    policy.fit(probs, y_true)

    # At c = 0.50: Coverage must be exactly 1.0 (no abstentions, tau_low == tau_high)
    tl_50, th_50 = policy._get_resolved_thresholds(3, 0.50, "linear")
    np.testing.assert_allclose(tl_50, th_50, atol=1e-9)
    preds_50 = policy.predict(probs, cost=0.50)
    assert np.all(preds_50 != -1), "At c=0.50, model must not abstain on any sample."

    # Coverage should be non-decreasing as c increases
    costs = [0.20, 0.25, 0.30, 0.40, 0.50]
    coverages = []
    for c in costs:
        p = policy.predict(probs, cost=c)
        cov = np.mean(p != -1)
        coverages.append(cov)

    assert coverages[-1] == 1.0, f"Coverage at c=0.50 must be 1.0, got {coverages[-1]}"
    for i in range(len(coverages) - 1):
        assert coverages[i] <= coverages[i + 1] + 1e-6, (
            f"Coverage decreased from c={costs[i]} ({coverages[i]:.4f}) to c={costs[i+1]} ({coverages[i+1]:.4f})"
        )


def test_gsi_no_data_leakage():
    """Verify that GSI fits its decision policy on out-of-fold validation data before refitting."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    Y = np.random.binomial(1, 0.3, size=(100, 4))

    # Fit with refit=True
    model_refit = GSIMLCPartialAbstentionClassifier(
        base_learner="logistic",
        decision_policy="macro_f1",
        min_coverage=0.80,
        random_state=42,
        refit=True,
    )
    model_refit.fit(X, Y)

    # Fit with refit=False
    model_norefit = GSIMLCPartialAbstentionClassifier(
        base_learner="logistic",
        decision_policy="macro_f1",
        min_coverage=0.80,
        random_state=42,
        refit=False,
    )
    model_norefit.fit(X, Y)

    # Thresholds should be IDENTICAL because both fit on the selection models' validation probabilities!
    policy_refit = model_refit.decision_policy_
    policy_norefit = model_norefit.decision_policy_

    np.testing.assert_allclose(policy_refit.tau_low, policy_norefit.tau_low, atol=1e-9)
    np.testing.assert_allclose(policy_refit.tau_high, policy_norefit.tau_high, atol=1e-9)


def test_mlc_pa_macro_f1_parity():
    """Verify that MLCPartialAbstentionClassifier properly supports macro_f1 policy and min_coverage."""
    np.random.seed(42)
    X = np.random.randn(120, 8)
    Y = np.random.binomial(1, 0.3, size=(120, 3))

    model = MLCPartialAbstentionClassifier(
        base_estimator="logistic",
        decision_policy="macro_f1",
        min_coverage=0.80,
        cost=0.3,
        random_state=42,
    )
    model.fit(X, Y)

    assert hasattr(model, "decision_policy_")
    assert model.decision_policy_.policy_name == "per_label_macro_f1"
    assert model.decision_policy_.min_coverage == 0.80

    # Predictions
    preds = model.predict(X)
    assert preds.shape == Y.shape
    assert set(np.unique(preds)).issubset({0, 1, -1})

    cov = np.mean(preds != -1)
    assert cov >= 0.75, f"Coverage should satisfy min_coverage floor, got {cov}"


def test_registered_model_parity_v5():
    """Verify registered MLC_PA models accept and use macro_f1 decision policy."""
    model = create_registered_model(
        "MLC_PA_Logistic",
        random_state=42,
        gsi_decision_policy="macro_f1",
        min_coverage=0.80,
    )
    assert model.decision_policy == "macro_f1"
    assert model.min_coverage == 0.80
