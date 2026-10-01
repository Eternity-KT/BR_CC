"""Unit tests for Model-Aware Label Correlation and Confidence Ordering."""

import numpy as np
import pytest

from src.selection.model_aware_correlation import (
    compute_predicted_correlation_matrix,
    compute_residual_correlation_matrix,
    order_dl_by_confidence,
    order_dl_by_model_correlation,
)


def test_compute_predicted_correlation_shape_and_diagonal():
    # 100 samples, 4 labels
    np.random.seed(42)
    P_prob = np.random.uniform(0.0, 1.0, size=(100, 4))
    # Make label 1 and label 2 strongly positively correlated
    P_prob[:, 1] = P_prob[:, 0] * 0.9 + np.random.normal(0, 0.05, size=100)
    P_prob = np.clip(P_prob, 0.0, 1.0)

    corr = compute_predicted_correlation_matrix(P_prob)
    assert corr.shape == (4, 4)
    # Check diagonal is 1.0
    for i in range(4):
        assert pytest.approx(corr[i, i], abs=1e-5) == 1.0
    # Symmetric
    assert np.allclose(corr, corr.T)
    # Strongly correlated
    assert corr[0, 1] > 0.8


def test_compute_residual_correlation():
    np.random.seed(42)
    Y_true = np.random.randint(0, 2, size=(100, 3))
    P_prob = np.random.uniform(0.1, 0.9, size=(100, 3))

    corr = compute_residual_correlation_matrix(Y_true, P_prob)
    assert corr.shape == (3, 3)
    for i in range(3):
        assert pytest.approx(corr[i, i], abs=1e-5) == 1.0
    assert np.all(corr >= 0.0) and np.all(corr <= 1.0)


def test_constant_prediction_handling():
    # When predictions are constant (zero variance)
    P_prob = np.zeros((50, 3))
    P_prob[:, 1] = 0.5  # constant

    corr = compute_predicted_correlation_matrix(P_prob)
    assert corr.shape == (3, 3)
    assert np.all(np.isfinite(corr))
    assert pytest.approx(corr[0, 0]) == 1.0
    assert pytest.approx(corr[1, 1]) == 1.0
    assert pytest.approx(corr[0, 1]) == 0.0


def test_order_dl_by_confidence():
    scores = {2: 0.85, 4: 0.65, 5: 0.92, 7: 0.85}
    dl_labels = [2, 4, 5, 7]

    # Descending: highest score first. Ties broken deterministically.
    ordered = order_dl_by_confidence(scores, dl_labels, direction="descending")
    assert ordered[0] == 5  # 0.92
    assert ordered[-1] == 4  # 0.65
    assert set(ordered) == set(dl_labels)

    # Ascending: lowest score first
    ordered_asc = order_dl_by_confidence(scores, dl_labels, direction="ascending")
    assert ordered_asc[0] == 4  # 0.65
    assert ordered_asc[-1] == 5  # 0.92


def test_order_dl_by_model_correlation():
    corr = np.array([
        [1.0, 0.2, 0.8],
        [0.2, 1.0, 0.1],
        [0.8, 0.1, 1.0],
    ])
    dl_labels = [0, 1, 2]

    # Label 1 has total corr = 0.2 + 0.1 = 0.3 (least correlated)
    # Label 0 has total corr = 0.2 + 0.8 = 1.0
    # Label 2 has total corr = 0.8 + 0.1 = 0.9
    ordered_asc = order_dl_by_model_correlation(corr, dl_labels, direction="ascending")
    assert ordered_asc[0] == 1  # 0.3 is smallest
    assert ordered_asc[1] == 2  # 0.9
    assert ordered_asc[2] == 0  # 1.0 is largest
