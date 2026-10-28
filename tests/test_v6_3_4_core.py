"""Focused tests for the minimal OOF-constrained SAMG2 change."""

import numpy as np

from src.models.gsi_v6_3_4 import GSIMLCPAv6_3_4Classifier
from src.selection.tri_regime_decision import (
    compute_tri_regime_adaptive_thresholds_table,
    select_oof_constrained_tau_1,
)


def test_oof_constrained_samg_lowers_threshold_for_safe_separation():
    y = np.r_[np.zeros(90, dtype=int), np.ones(10, dtype=int)]
    p = np.r_[np.linspace(0.01, 0.20, 90), np.linspace(0.30, 0.40, 10)]

    tau_1, applied, reason, audit = select_oof_constrained_tau_1(
        y, p, tau_0=0.10, tau_1=0.50
    )

    assert applied
    assert reason == "ACTIVATED"
    assert 0.20 < tau_1 < 0.50
    assert audit["samg_selected_f1_positive"] > audit["samg_baseline_f1_positive"]
    assert audit["samg_selected_f1_negative"] >= audit["samg_baseline_f1_negative"] - 0.01


def test_oof_constrained_samg_rejects_overlap():
    y = np.r_[np.zeros(90, dtype=int), np.ones(10, dtype=int)]
    p = np.full(100, 0.20)

    tau_1, applied, reason, _ = select_oof_constrained_tau_1(
        y, p, tau_0=0.10, tau_1=0.50
    )

    assert not applied
    assert reason == "NO_SAFE_OOF_IMPROVEMENT"
    assert tau_1 == 0.50


def test_threshold_table_exposes_final_samg_audit():
    y = np.r_[np.zeros(90, dtype=int), np.ones(10, dtype=int)].reshape(-1, 1)
    p = np.r_[np.linspace(0.01, 0.20, 90), np.linspace(0.30, 0.40, 10)].reshape(-1, 1)

    table = compute_tri_regime_adaptive_thresholds_table(
        y,
        OOF_probs=p,
        gamma_min=0.0,
        samg_strategy="oof_constrained",
    )
    row = table[0]

    assert row["samg_probability_source"] == "oof"
    assert row["samg_applied"]
    expected_coverage = np.mean((p[:, 0] <= row["tau_0"]) | (p[:, 0] >= row["tau_1"]))
    assert row["empirical_coverage"] == expected_coverage


def test_oof_guarded_margin_accepts_candidate_that_is_safe_on_oof():
    y = np.r_[np.zeros(90, dtype=int), np.ones(10, dtype=int)].reshape(-1, 1)
    in_sample = np.r_[np.linspace(0.01, 0.15, 90), np.linspace(0.30, 0.40, 10)].reshape(-1, 1)
    oof = np.r_[np.linspace(0.01, 0.15, 90), np.linspace(0.30, 0.40, 10)].reshape(-1, 1)

    row = compute_tri_regime_adaptive_thresholds_table(
        y,
        OOF_probs=oof,
        in_sample_probs=in_sample,
        gamma_min=0.0,
        samg_strategy="oof_guarded_margin",
    )[0]

    assert row["samg_applied"]
    assert row["samg_reason"] == "ACTIVATED"
    assert row["samg_probability_source"] == "in_sample_candidate+oof_guard"


def test_oof_guarded_margin_rejects_candidate_that_hurts_oof_negative_f1():
    y = np.r_[np.zeros(90, dtype=int), np.ones(10, dtype=int)].reshape(-1, 1)
    in_sample = np.r_[np.linspace(0.01, 0.15, 90), np.linspace(0.30, 0.40, 10)].reshape(-1, 1)
    oof = np.r_[np.linspace(0.01, 0.35, 90), np.linspace(0.20, 0.25, 10)].reshape(-1, 1)

    row = compute_tri_regime_adaptive_thresholds_table(
        y,
        OOF_probs=oof,
        in_sample_probs=in_sample,
        gamma_min=0.0,
        samg_strategy="oof_guarded_margin",
    )[0]

    assert not row["samg_applied"]
    assert row["samg_reason"] == "OOF_GUARD_REJECTED"


def test_v6_3_4_selects_oof_strategy_by_default():
    model = GSIMLCPAv6_3_4Classifier()
    assert model.samg_strategy == "oof_guarded_margin"

