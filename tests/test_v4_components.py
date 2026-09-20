"""Unit tests for v4 optimization components:
1. AsymmetricLoss
2. Vectorized GPU Platt Scaling
3. min_coverage constraint in PerLabelMacroF1Policy
4. ClassifierChain label_noise
5. Model registry instantiation
"""

import numpy as np
import pytest
import torch

from src.decision.macro_f1 import PerLabelMacroF1Policy
from src.decision.configuration import create_configured_policy
from src.models.classifier_chain import ClassifierChainClassifier
from src.models.pytorch_mlp import (
    AsymmetricLoss,
    FastPyTorchBinaryMLP,
    MultiLabelMLPClassifier,
)
from src.models.registry import create_registered_model


def test_asymmetric_loss():
    loss_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
    logits = torch.randn(10, 5, requires_grad=True)
    targets = (torch.rand(10, 5) > 0.7).float()

    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() > 0

    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_vectorized_platt_scaling():
    # Test that MultiLabelMLPClassifier with calibration="sigmoid" trains and scales correctly
    rng = np.random.RandomState(42)
    X = rng.randn(100, 10).astype(np.float32)
    Y = (rng.rand(100, 4) > 0.7).astype(np.int32)

    model = MultiLabelMLPClassifier(
        hidden_layer_sizes=(32,),
        epochs=10,
        calibration="sigmoid",
        loss="asymmetric",
        random_state=42,
    )
    model.fit(X, Y)

    probs = model.predict_proba(X)
    assert probs.shape == (100, 4)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert hasattr(model, "calibrators_")
    assert len(model.calibrators_) == 4
    for j, (a, b) in model.calibrators_.items():
        assert isinstance(a, float) and a > 0
        assert isinstance(b, float)


def test_min_coverage_constraint():
    # Test that setting min_coverage in PerLabelMacroF1Policy prevents degenerate threshold selection
    rng = np.random.RandomState(42)
    val_probs = rng.rand(100, 4)
    val_y = (rng.rand(100, 4) > 0.8).astype(np.int32)

    # Policy with high min_coverage
    policy = PerLabelMacroF1Policy(
        cost=0.3,
        allow_abstention=True,
        min_coverage=0.85,
    )
    policy.fit(val_probs, val_y)

    test_probs = rng.rand(50, 4)
    pred = policy.predict_from_proba(test_probs, cost=0.3)
    coverage = np.mean(pred != -1)
    # The coverage on test probs with uniform random should be high because thresholds
    # were constrained to avoid high abstention
    assert coverage > 0.5


def test_classifier_chain_label_noise():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.RandomState(42)
    X = rng.randn(60, 5).astype(np.float32)
    Y = (rng.rand(60, 3) > 0.5).astype(np.int32)

    cc = ClassifierChainClassifier(
        base_estimator=LogisticRegression(),
        random_state=42,
        label_noise=0.2,
    )
    cc.fit(X, Y)
    preds = cc.predict(X)
    assert preds.shape == (60, 3)
    probs = cc.predict_proba(X)
    assert probs.shape == (60, 3)


def test_create_registered_model_v4():
    # Test instantiation of GSI_MLC_PA_MLP and MLC_PA_MLP with v4 arguments
    gsi_model = create_registered_model(
        "GSI_MLC_PA_MLP",
        min_coverage=0.80,
        cc_label_noise=0.1,
        mlp_loss="asymmetric",
        random_state=42,
    )
    assert gsi_model.min_coverage == 0.80
    assert gsi_model.cc_label_noise == 0.1
    assert gsi_model.mlp_loss == "asymmetric"

    mlc_model = create_registered_model(
        "MLC_PA_MLP",
        mlp_loss="asymmetric",
        random_state=42,
    )
    assert mlc_model.base_kwargs.get("loss") == "asymmetric"
