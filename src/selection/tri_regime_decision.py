"""
Adaptive Tri-Regime Decision & Smooth Boundary Blending for GSI-MLC-PA Version 6.3.3.

Reference:
    - spec/spec_v6_3_3.md
    - Chow (1970) Optimal Rejection Rule.
    - Extended to Tri-Regime Inference:
        * Regime 1 (Rare-Negative): pi_l < 0.20 -> Asymmetric Negative Verification (Bayes LR + Precision Guard).
        * Regime 2 (Symmetric / Balanced): 0.20 <= pi_l <= 0.80 -> Symmetric Chow Rule [c, 1-c].
        * Regime 3 (Rare-Positive): pi_l > 0.80 -> Inverted Asymmetric Positive Verification.
    - Continuous blending via sigmoid factor eliminates discontinuity cliffs.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np


def compute_label_balance_factor(prior: float) -> float:
    """
    Compute Label Balance Factor beta_l in [0, 1].
    beta_l = 1.0 - 2.0 * |pi_l - 0.50|.
    beta_l = 1.0 when perfectly balanced (pi = 0.50).
    beta_l -> 0.0 when extremely imbalanced (pi -> 0.0 or pi -> 1.0).
    """
    pi = float(np.clip(prior, 0.0, 1.0))
    return float(1.0 - 2.0 * abs(pi - 0.50))


def compute_tri_regime_raw_thresholds(
    prior: float,
    cost: float = 0.30,
    tau_balance: float = 0.75,
    blend_kappa: float = 20.0,
    clip_bounds: Tuple[float, float] = (0.005, 0.995),
    min_separation: float = 0.02,
) -> Tuple[float, float, str, float]:
    """
    Compute blended decision thresholds [tau_0, tau_1] and regime label.

    Parameters:
        prior: Label positive prior probability pi in [0, 1].
        cost: Base rejection cost c in (0, 0.5). Default 0.30.
        tau_balance: Threshold on beta_l to separate balanced from extreme regime. Default 0.40.
        blend_kappa: Slope for smooth sigmoid transition. Default 20.0.
        clip_bounds: Bound tuple for numerical stability.
        min_separation: Minimum gap between tau_0 and tau_1.

    Returns:
        tau_0: Blended negative decision threshold.
        tau_1: Blended positive decision threshold.
        regime: Name of primary regime ("Symmetric", "Rare_Negative", "Rare_Positive").
        blend_weight: Weight sigma_blend in [0, 1] allocated to symmetric Chow rule.
    """
    c = float(cost)
    if c <= 0.0 or c >= 0.5:
        raise ValueError(f"Rejection cost c must be in (0, 0.5), got {c}")

    pi = float(np.clip(prior, 1e-6, 1.0 - 1e-6))
    beta = compute_label_balance_factor(pi)
    min_t, max_t = clip_bounds

    # 1. Symmetric Chow Rule Thresholds (Regime 2)
    tau_0_chow = c
    tau_1_chow = 1.0 - c

    # 2. Asymmetric Bayes Likelihood Ratio Thresholds
    if pi <= 0.50:
        # Regime 1: Rare-Negative (0 is majority, 1 is minority)
        one_minus_pi = 1.0 - pi
        num_1 = (1.0 - c) * pi
        den_1 = c * one_minus_pi + num_1
        tau_1_raw = num_1 / (den_1 + 1e-12)

        num_0 = c * pi
        den_0 = (1.0 - c) * one_minus_pi + num_0
        tau_0_raw = num_0 / (den_0 + 1e-12)

        tau_0_asym = float(np.clip(tau_0_raw, min_t, 0.49))
        tau_1_asym = float(np.clip(tau_1_raw, 0.51, max_t))

        # Precision guard: positive threshold never below 0.50
        tau_1_asym = max(0.50, tau_1_asym)

        if beta >= tau_balance:
            regime = "Symmetric"
        else:
            regime = "Rare_Negative"
    else:
        # Regime 3: Rare-Positive (1 is majority, 0 is minority)
        # Symmetry inversion: swap roles of 0 and 1
        inv_pi = 1.0 - pi
        one_minus_inv = pi
        num_1 = (1.0 - c) * inv_pi
        den_1 = c * one_minus_inv + num_1
        inv_tau_1_raw = num_1 / (den_1 + 1e-12)

        num_0 = c * inv_pi
        den_0 = (1.0 - c) * one_minus_inv + num_0
        inv_tau_0_raw = num_0 / (den_0 + 1e-12)

        # Invert thresholds back to positive space
        tau_0_asym = 1.0 - float(np.clip(inv_tau_1_raw, 0.51, max_t))
        tau_1_asym = 1.0 - float(np.clip(inv_tau_0_raw, min_t, 0.49))

        # Negative precision guard: negative threshold never above 0.50
        tau_0_asym = min(0.50, tau_0_asym)

        if beta >= tau_balance:
            regime = "Symmetric"
        else:
            regime = "Rare_Positive"

    # 3. Smooth Boundary Blending via Sigmoid
    # sigma_blend -> 1.0 when beta >> tau_balance (Symmetric dominant)
    # sigma_blend -> 0.0 when beta << tau_balance (Asymmetric dominant)
    arg = np.clip(blend_kappa * (beta - tau_balance), -30.0, 30.0)
    sigma_blend = float(1.0 / (1.0 + np.exp(-arg)))

    tau_0_blended = sigma_blend * tau_0_chow + (1.0 - sigma_blend) * tau_0_asym
    tau_1_blended = sigma_blend * tau_1_chow + (1.0 - sigma_blend) * tau_1_asym

    # Ensure valid ordering and separation
    tau_0_blended = float(np.clip(tau_0_blended, min_t, 0.49))
    tau_1_blended = float(np.clip(tau_1_blended, 0.51, max_t))

    if tau_1_blended - tau_0_blended < min_separation:
        midpoint = 0.5 * (tau_0_blended + tau_1_blended)
        tau_0_blended = max(min_t, midpoint - 0.5 * min_separation)
        tau_1_blended = min(max_t, midpoint + 0.5 * min_separation)

    return tau_0_blended, tau_1_blended, regime, sigma_blend


def apply_dual_coverage_guard(
    probs_1d: np.ndarray,
    tau_0: float,
    tau_1: float,
    prior: float,
    gamma_min: float = 0.70,
    max_iterations: int = 50,
    precision_guard: bool = True,
    min_tau_1: float = 0.50,
    max_tau_0: float = 0.50,
) -> Tuple[float, float, bool]:
    """
    Adjust [tau_0, tau_1] rejection band if empirical decision coverage < gamma_min,
    respecting directional precision guards.
    """
    probs_clean = np.asarray(probs_1d, dtype=np.float64).ravel()
    if len(probs_clean) == 0:
        return tau_0, tau_1, False

    gamma = float(gamma_min)
    decided = (probs_clean <= tau_0) | (probs_clean >= tau_1)
    current_cov = float(np.mean(decided))

    if current_cov >= gamma:
        return tau_0, tau_1, False

    t0, t1 = float(tau_0), float(tau_1)
    step = (t1 - t0) / (2.0 * max_iterations)

    for _ in range(max_iterations):
        # Determine whether to contract t0 or t1 based on prior direction
        if prior <= 0.50:
            # For rare positive, raise t0 safely towards 0.50
            t0 += step
            if not precision_guard or t1 > min_tau_1:
                t1 -= step
                if precision_guard and t1 < min_tau_1:
                    t1 = min_tau_1
        else:
            # For rare negative (positive majority), lower t1 safely towards 0.50
            t1 -= step
            if not precision_guard or t0 < max_tau_0:
                t0 += step
                if precision_guard and t0 > max_tau_0:
                    t0 = max_tau_0

        if t0 >= t1:
            mid = 0.5 * (tau_0 + tau_1)
            t0 = mid - 1e-4
            t1 = mid + 1e-4
            break

        cov = float(np.mean((probs_clean <= t0) | (probs_clean >= t1)))
        if cov >= gamma:
            break

    return float(t0), float(t1), True


def compute_tri_regime_adaptive_thresholds_table(
    Y_train: np.ndarray,
    OOF_probs: Optional[np.ndarray] = None,
    cost: float = 0.30,
    tau_balance: float = 0.75,
    blend_kappa: float = 20.0,
    gamma_min: float = 0.70,
    clip_bounds: Tuple[float, float] = (0.005, 0.995),
    precision_guard: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute full Adaptive Tri-Regime decision table across all K labels.

    Parameters:
        Y_train: In-fold binary training targets (N, K).
        OOF_probs: Optional continuous Out-Of-Fold probabilities (N, K).
        cost: Rejection cost c. Default 0.30.
        tau_balance: Balance cutoff for beta. Default 0.40.
        blend_kappa: Sigmoid blend slope. Default 20.0.
        gamma_min: Coverage guard floor. Default 0.70.
        clip_bounds: Threshold bounding tuple.
        precision_guard: Whether to enforce directional precision floors.

    Returns:
        thresholds_dict: Mapping label_idx -> {
            'tau_0': float,
            'tau_1': float,
            'prior': float,
            'beta': float,
            'regime': str,
            'blend_weight': float,
            'coverage_guard': bool,
            'empirical_coverage': float,
        }
    """
    Y = np.asarray(Y_train, dtype=np.int32)
    n_samples, n_labels = Y.shape
    table: Dict[int, Dict[str, Any]] = {}

    for j in range(n_labels):
        col_y = Y[:, j]
        prior_j = float(np.mean(col_y))
        beta_j = compute_label_balance_factor(prior_j)

        t0, t1, regime_j, blend_w = compute_tri_regime_raw_thresholds(
            prior=prior_j,
            cost=cost,
            tau_balance=tau_balance,
            blend_kappa=blend_kappa,
            clip_bounds=clip_bounds,
        )

        guard_applied = False
        emp_cov = 1.0

        if OOF_probs is not None and OOF_probs.shape[1] > j:
            p_j = OOF_probs[:, j]
            t0, t1, guard_applied = apply_dual_coverage_guard(
                probs_1d=p_j,
                tau_0=t0,
                tau_1=t1,
                prior=prior_j,
                gamma_min=gamma_min,
                precision_guard=precision_guard,
            )
            emp_cov = float(np.mean((p_j <= t0) | (p_j >= t1)))

        table[j] = {
            "tau_0": float(t0),
            "tau_1": float(t1),
            "prior": float(prior_j),
            "beta": float(beta_j),
            "regime": regime_j,
            "blend_weight": float(blend_w),
            "coverage_guard": bool(guard_applied),
            "empirical_coverage": float(emp_cov),
        }

    return table


def apply_tri_regime_abstention(
    probs: np.ndarray,
    thresholds_table: Dict[int, Dict[str, Any]],
    abstain_value: int = -1,
) -> np.ndarray:
    """
    Apply label-specific Tri-Regime thresholds to predicted probability matrix.
    """
    P = np.asarray(probs, dtype=np.float64)
    M, K = P.shape
    preds = np.full((M, K), abstain_value, dtype=np.int32)

    for j in range(K):
        t0 = thresholds_table[j]["tau_0"]
        t1 = thresholds_table[j]["tau_1"]
        p_col = P[:, j]

        preds[p_col <= t0, j] = 0
        preds[p_col >= t1, j] = 1

    return preds
