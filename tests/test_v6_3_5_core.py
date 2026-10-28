"""Tests for DL-only rejection in v6.3.5."""

import numpy as np

from src.models.gsi_v6_3_3 import GSIMLCPAv6_3_3Classifier
from src.models.gsi_v6_3_5 import GSIMLCPAv6_3_5Classifier


def _stub_model(model):
    probabilities = np.array([[0.50, 0.50, 0.50], [0.20, 0.80, 0.90]])
    model.predict_proba = lambda _x: probabilities
    model.independent_labels_ = [0, 2]
    model.dependent_labels_ = [1]
    model.adaptive_thresholds_ = {
        0: {"tau_0": 0.30, "tau_1": 0.70, "prior": 0.50},
        1: {"tau_0": 0.30, "tau_1": 0.70, "prior": 0.50},
        2: {"tau_0": 0.10, "tau_1": 0.80, "prior": 0.10},
    }
    return probabilities


def test_v6_3_5_rejects_only_dl_labels():
    model = GSIMLCPAv6_3_5Classifier()
    probabilities = _stub_model(model)

    predictions = model.predict(np.zeros((2, 1)))
    full_predictions = model._predict_full_from_probabilities(probabilities)

    assert model.rejection_scope == "dl_only"
    np.testing.assert_array_equal(
        predictions[:, model.independent_labels_],
        full_predictions[:, model.independent_labels_],
    )
    assert -1 not in predictions[:, model.independent_labels_]
    np.testing.assert_array_equal(predictions[:, 1], np.array([-1, 1]))


def test_v6_3_3_keeps_all_label_rejection_by_default():
    model = GSIMLCPAv6_3_3Classifier()
    _stub_model(model)

    predictions = model.predict(np.zeros((2, 1)))

    assert model.rejection_scope == "all_labels"
    assert predictions[0, 0] == -1
    assert predictions[0, 1] == -1
    assert predictions[0, 2] == -1


def test_rejection_scope_validation():
    try:
        GSIMLCPAv6_3_3Classifier(rejection_scope="unknown")
    except ValueError as error:
        assert "rejection_scope" in str(error)
    else:
        raise AssertionError("Invalid rejection scope should fail fast")
