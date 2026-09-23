"""Unit tests for v3.1 decoupled DL architecture in GSI-MLC-PA."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.gsi_mlc_pa import GSIMLCPartialAbstentionClassifier
from src.models.registry import (
    create_registered_model,
    get_model_spec,
    is_registered_model_id,
)


def test_v3_1_registered_models():
    """Verify that v3.1 model IDs are properly registered with decoupled_dl=True."""
    assert is_registered_model_id("GSI_MLC_PA_v3_1_Logistic")
    assert is_registered_model_id("GSI_MLC_PA_v3_1_MLP")
    assert is_registered_model_id("GSI_MLC_PA_v3_1_SVM")

    spec_log = get_model_spec("GSI_MLC_PA_v3_1_Logistic")
    assert spec_log.family == "GSI_MLC_PA"
    assert spec_log.base_learner == "logistic"
    assert spec_log.decoupled_dl is True

    model_log = create_registered_model("GSI_MLC_PA_v3_1_Logistic")
    assert isinstance(model_log, GSIMLCPartialAbstentionClassifier)
    assert model_log.decoupled_dl is True
    assert model_log.experiment_manifest_["model_parameters"]["decoupled_dl"] is True

    # Standard v3 model should have decoupled_dl=False and omitted from manifest for hash stability
    model_v3 = create_registered_model("GSI_MLC_PA_Logistic")
    assert model_v3.decoupled_dl is False
    assert "decoupled_dl" not in model_v3.experiment_manifest_["model_parameters"]


def test_decoupled_il_probabilities_match_br():
    """Verify that for IL labels, decoupled GSI probabilities match BR exactly."""
    np.random.seed(42)
    n_samples = 120
    n_features = 8
    n_labels = 4

    X = np.random.randn(n_samples, n_features)
    # Label 0, 1 independent; Label 2, 3 correlated with 0 and each other
    y0 = (X[:, 0] > 0).astype(int)
    y1 = (X[:, 1] > 0).astype(int)
    y2 = (y0 ^ (X[:, 2] > 0)).astype(int)
    y3 = (y2 & (X[:, 3] > 0)).astype(int)
    Y = np.column_stack([y0, y1, y2, y3])

    # Force fixed partition: IL = [0, 1], DL = [2, 3]
    model = GSIMLCPartialAbstentionClassifier(
        base_learner="logistic",
        partition_mode="fixed",
        fixed_independent_labels=[0, 1],
        decoupled_dl=True,
        random_state=42,
    )
    model.fit(X, Y)

    assert set(model.independent_labels_) == {0, 1}
    assert set(model.dependent_labels_) == {2, 3}

    X_test = np.random.randn(30, n_features)
    probs = model.predict_proba(X_test)
    br_probs = model._direct_probabilities(model.br_model_, X_test)

    # IL probabilities must match BR directly and exactly
    np.testing.assert_allclose(probs[:, 0], br_probs[:, 0], atol=1e-6)
    np.testing.assert_allclose(probs[:, 1], br_probs[:, 1], atol=1e-6)


def test_sub_cc_never_receives_il_features():
    """Verify that Sub-CC classifiers only receive DL predecessor features, never IL."""
    np.random.seed(42)
    n_samples = 150
    n_features = 10
    n_labels = 5

    X = np.random.randn(n_samples, n_features)
    Y = np.random.binomial(1, 0.4, size=(n_samples, n_labels))

    # Fix IL=[0, 1, 2], DL=[3, 4] -> |DL|=2
    model = GSIMLCPartialAbstentionClassifier(
        base_learner="logistic",
        partition_mode="fixed",
        fixed_independent_labels=[0, 1, 2],
        decoupled_dl=True,
        random_state=42,
    )
    model.fit(X, Y)

    # Sub-CC has 2 classifiers (one for each DL label)
    assert len(model.cc_model_.classifiers_) == 2
    # First DL classifier has n_features (0 predecessors)
    # Second DL classifier has n_features + 1 (1 preceding DL label)
    # Neither should have n_features + 2 or n_features + 3 (which would include IL labels)
    for i, est in enumerate(model.cc_model_.classifiers_):
        assert est.n_features_in_ == n_features + i


def test_edge_case_pure_br_empty_dl():
    """Verify edge case: DL is empty (all labels IL)."""
    np.random.seed(42)
    X = np.random.randn(80, 6)
    Y = np.random.binomial(1, 0.3, size=(80, 3))

    model = GSIMLCPartialAbstentionClassifier(
        base_learner="logistic",
        partition_mode="all_il",
        decoupled_dl=True,
        random_state=42,
    )
    model.fit(X, Y)

    assert len(model.dependent_labels_) == 0
    assert model.cc_model_ is None

    preds = model.predict(X)
    probs = model.predict_proba(X)
    assert preds.shape == (80, 3)
    assert probs.shape == (80, 3)


def test_edge_case_pure_cc_empty_il():
    """Verify edge case: IL is empty (all labels DL)."""
    np.random.seed(42)
    X = np.random.randn(80, 6)
    Y = np.random.binomial(1, 0.3, size=(80, 3))

    model = GSIMLCPartialAbstentionClassifier(
        base_learner="logistic",
        partition_mode="all_dl",
        decoupled_dl=True,
        random_state=42,
    )
    model.fit(X, Y)

    assert len(model.independent_labels_) == 0
    assert len(model.dependent_labels_) == 3
    assert len(model.cc_model_.classifiers_) == 3

    preds = model.predict(X)
    probs = model.predict_proba(X)
    assert preds.shape == (80, 3)
    assert probs.shape == (80, 3)


def test_edge_case_single_dl_label():
    """Verify edge case: |DL| = 1."""
    np.random.seed(42)
    X = np.random.randn(80, 6)
    Y = np.random.binomial(1, 0.3, size=(80, 4))

    model = GSIMLCPartialAbstentionClassifier(
        base_learner="logistic",
        partition_mode="fixed",
        fixed_independent_labels=[0, 1, 2],
        decoupled_dl=True,
        random_state=42,
    )
    model.fit(X, Y)

    assert model.dependent_labels_ == [3]
    assert len(model.cc_model_.classifiers_) == 1
    # 0 predecessor labels, only X features
    assert model.cc_model_.classifiers_[0].n_features_in_ == 6

    probs = model.predict_proba(X)
    assert probs.shape == (80, 4)
