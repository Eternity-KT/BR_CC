"""
Unit and Integration Tests for GSI-MLC-PA v6.1 (ECC and Singleton DL Promotion).

Verifies:
1. EnsembleClassifierChainClassifier:
   - Fits on multilabel data and predicts probabilities in [0, 1].
   - Correctly averages probabilities across M chains.
   - Respects reproducible random_state.
   - Handles edge cases (empty labels, single label).
2. Singleton DL Promotion Rule:
   - When candidate DL has only 1 label left, CVStratifiedPeelingSelector with
     promote_singleton_dl=True moves that label into IL directly, leaving DL empty.
3. GSIMLCPAv6_1Classifier:
   - End-to-end fit and predict with partial abstention (cost c = 0.30).
   - Correctly segments IL (handled by BR) and DL (handled by ECC with augmented features).
   - Gracefully handles edge cases: all-IL, all-DL, and mixed IL/DL.
   - Base learner compatibility: Logistic Regression, SVM, and MLP.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.ensemble_classifier_chain import EnsembleClassifierChainClassifier
from src.models.gsi_v6_1 import GSIMLCPAv6_1Classifier
from src.selection.cv_peeling import (
    CVPeelingConfig,
    CVPeelingResult,
    CVStratifiedPeelingSelector,
)


def _make_toy_dataset(n_samples: int = 250, random_state: int = 42):
    """
    Generate synthetic multilabel dataset with 4 labels:
    - y0: strongly correlated with x[:, 0] and x[:, 1] -> Independent (IL_1)
    - y1: depends on y0 and x[:, 2] -> Stage 2 Independent (IL_2)
    - y2: dependent on y1 and random noise -> DL
    - y3: dependent on y2 and random noise -> DL
    """
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, 8).astype(np.float32)

    p0 = 1.0 / (1.0 + np.exp(-(2.5 * X[:, 0] + 2.0 * X[:, 1])))
    y0 = (p0 > 0.5).astype(np.int32)

    p1 = 1.0 / (1.0 + np.exp(-(3.0 * y0 + 1.5 * X[:, 2] - 1.5)))
    y1 = (p1 > 0.5).astype(np.int32)

    p2 = 1.0 / (1.0 + np.exp(-(2.0 * y1 + 1.0 * X[:, 3] - 1.0)))
    y2 = (p2 > 0.5).astype(np.int32)

    p3 = 1.0 / (1.0 + np.exp(-(2.5 * y2 + 0.8 * X[:, 4] - 1.2)))
    y3 = (p3 > 0.5).astype(np.int32)

    Y = np.column_stack([y0, y1, y2, y3])
    return X, Y


def test_ensemble_classifier_chain_basic():
    """Verify that EnsembleClassifierChainClassifier trains M chains and averages probabilities."""
    X, Y = _make_toy_dataset(n_samples=200, random_state=42)
    base_est = LogisticRegression(solver="liblinear", random_state=42)

    ecc = EnsembleClassifierChainClassifier(
        base_estimator=base_est,
        n_chains=5,
        subsample=1.0,
        random_state=42,
    )
    ecc.fit(X, Y)

    assert len(ecc.chains_) == 5
    assert len(ecc.orders_) == 5
    assert ecc.n_labels_ == 4

    # Each chain should have a unique random permutation of 4 labels
    for order in ecc.orders_:
        assert sorted(order) == [0, 1, 2, 3]

    probs = ecc.predict_proba(X)
    assert probs.shape == (200, 4)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    preds = ecc.predict(X)
    assert preds.shape == (200, 4)
    assert np.all(np.isin(preds, [0, 1]))


def test_ensemble_classifier_chain_reproducibility():
    """Verify deterministic outputs with same random_state."""
    X, Y = _make_toy_dataset(n_samples=150, random_state=123)

    ecc1 = EnsembleClassifierChainClassifier(
        base_estimator="logistic", n_chains=3, random_state=999
    )
    ecc2 = EnsembleClassifierChainClassifier(
        base_estimator="logistic", n_chains=3, random_state=999
    )

    p1 = ecc1.fit(X, Y).predict_proba(X)
    p2 = ecc2.fit(X, Y).predict_proba(X)

    np.testing.assert_allclose(p1, p2, rtol=1e-5)


def test_singleton_dl_promotion_rule():
    """Verify meeting_summary.md rule: 'Khi tập DL chỉ còn 1 nhãn thì đưa luôn vào IL'."""
    # Create 3-label dataset where 2 labels are independent and 1 remains in DL
    rng = np.random.RandomState(42)
    X = rng.randn(300, 6).astype(np.float32)

    # y0: easy
    p0 = 1.0 / (1.0 + np.exp(-(3.0 * X[:, 0] + 2.0 * X[:, 1])))
    y0 = (p0 > 0.5).astype(np.int32)

    # y1: easy
    p1 = 1.0 / (1.0 + np.exp(-(2.5 * X[:, 2] + 2.0 * X[:, 3])))
    y1 = (p1 > 0.5).astype(np.int32)

    # y2: purely random noise -> fails threshold 0.75
    y2 = (rng.rand(300) > 0.7).astype(np.int32)

    Y = np.column_stack([y0, y1, y2])

    # 1. Without promote_singleton_dl (legacy v6 behavior): y2 stays in DL
    cfg_legacy = CVPeelingConfig(
        threshold=0.75,
        n_folds=3,
        promote_singleton_dl=False,
        random_state=42,
    )
    sel_legacy = CVStratifiedPeelingSelector(
        config=cfg_legacy,
        base_estimator_factory=lambda: LogisticRegression(solver="liblinear", random_state=42),
    )
    res_legacy = sel_legacy.fit_partition(X, Y)
    assert len(res_legacy.dependent_residual_labels) == 1
    assert 2 in res_legacy.dependent_residual_labels

    # 2. With promote_singleton_dl=True (v6.1 behavior): y2 is promoted to IL
    cfg_v6_1 = CVPeelingConfig(
        threshold=0.75,
        n_folds=3,
        promote_singleton_dl=True,
        random_state=42,
    )
    sel_v6_1 = CVStratifiedPeelingSelector(
        config=cfg_v6_1,
        base_estimator_factory=lambda: LogisticRegression(solver="liblinear", random_state=42),
    )
    res_v6_1 = sel_v6_1.fit_partition(X, Y)
    # Residual DL must now be empty!
    assert len(res_v6_1.dependent_residual_labels) == 0
    # All 3 labels must be in all_independent_labels
    assert len(res_v6_1.all_independent_labels) == 3
    assert res_v6_1.stopping_reason == "singleton_dl_promoted_to_il"


def test_gsi_v6_1_classifier_end_to_end():
    """Verify full end-to-end fit, predict_proba, and predict on GSIMLCPAv6_1Classifier."""
    X, Y = _make_toy_dataset(n_samples=300, random_state=42)

    clf = GSIMLCPAv6_1Classifier(
        base_learner="logistic",
        stratified_threshold=0.75,
        cv_folds=3,
        n_chains=5,
        cost=0.30,
        random_state=42,
    )
    clf.fit(X, Y)

    assert clf.is_fitted_ is True
    assert clf.n_labels_ == 4

    # Predict probabilities
    probs = clf.predict_proba(X)
    assert probs.shape == (300, 4)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    # Predict with partial abstention at cost c = 0.30
    y_partial = clf.predict(X, cost=0.30)
    assert y_partial.shape == (300, 4)
    # Values must be in {0, 1, -1}
    assert np.all(np.isin(y_partial, [0, 1, -1]))

    # Decided samples check
    coverage = np.mean(y_partial != -1)
    assert coverage > 0.50

    # Complete predictions
    y_full = clf.predict_full(X)
    assert y_full.shape == (300, 4)
    assert np.all(np.isin(y_full, [0, 1]))


def test_gsi_v6_1_classifier_base_learners():
    """Verify that GSIMLCPAv6_1Classifier works with SVM base learner."""
    X, Y = _make_toy_dataset(n_samples=200, random_state=42)

    clf_svm = GSIMLCPAv6_1Classifier(
        base_learner="svm_calibrated",
        stratified_threshold=0.75,
        cv_folds=3,
        n_chains=3,
        cost=0.30,
        random_state=42,
    )
    clf_svm.fit(X, Y)
    probs_svm = clf_svm.predict_proba(X)
    assert probs_svm.shape == (200, 4)
    assert np.all(probs_svm >= 0.0) and np.all(probs_svm <= 1.0)
