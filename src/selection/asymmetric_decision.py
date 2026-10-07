"""
Asymmetric Prior-Calibrated Negative Verification & Adaptive Confidence Abstention.

Reference:
    - spec/spec_v6_3.md
    - Chow's Rule (1970) extended to asymmetric label imbalance via Bayes Likelihood Ratio.

Mathematical Formulation:
1. Label Prior:
   pi_j = P(Y_j = 1) = Mean(Y[:, j])
2. Bayes Likelihood Ratio (LR) Thresholds:
   Given rejection cost c in (0, 0.5):
   tau_1(j) = ((1 - c) * pi_j) / (c * (1 - pi_j) + (1 - c) * pi_j)
   tau_0(j) = (c * pi_j) / ((1 - c) * (1 - pi_j) + c * pi_j)

   Properties:
   - When pi_j = 0.5 (balanced), tau_0(j) = c and tau_1(j) = 1 - c (recovers Chow's rule exactly).
   - When pi_j << 0.5 (extreme imbalance, e.g. pi_j = 0.03):
     tau_0(j) shifts toward 0 (~ 0.013), requiring strong negative confidence (> 98.7%) to predict 0.
     tau_1(j) shifts downward (~ 0.067), enabling positive detection when likelihood ratio >= (1-c)/c.
3. Coverage Guard (gamma_min):
   Ensures that individual label decision coverage >= gamma_min (default 0.70) by shrinking
   the rejection band [tau_0(j), tau_1(j)] if validation coverage falls below gamma_min.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def compute_bayes_lr_thresholds(
    prior: float,
    cost: float = 0.30,
    clip_bounds: Tuple[float, float] = (0.005, 0.995),
    min_separation: float = 0.02,
) -> Tuple[float, float]:
    """
    Compute asymmetric decision thresholds [tau_0, tau_1] based on Bayes Likelihood Ratio.

    Parameters:
        prior: Label prior probability pi in [0, 1].
        cost: Rejection cost c in (0, 0.5). Default 0.30.
        clip_bounds: Tuple of (min_tau, max_tau) to prevent extreme boundary collapse.
        min_separation: Minimum gap between tau_0 and tau_1.

    Returns:
        tau_0: Threshold for negative decision (p <= tau_0 -> predict 0).
        tau_1: Threshold for positive decision (p >= tau_1 -> predict 1).
    """
    c = float(cost)
    if c <= 0.0 or c >= 0.5:
        raise ValueError(f"Rejection cost c must be in (0, 0.5), got {c}")

    pi = float(np.clip(prior, 1e-6, 1.0 - 1e-6))
    one_minus_pi = 1.0 - pi

    # Bayes Likelihood Ratio odds formulation
    num_1 = (1.0 - c) * pi
    den_1 = c * one_minus_pi + num_1
    tau_1_raw = num_1 / (den_1 + 1e-12)

    num_0 = c * pi
    den_0 = (1.0 - c) * one_minus_pi + num_0
    tau_0_raw = num_0 / (den_0 + 1e-12)

    # Clipping to safe bounds
    min_t, max_t = clip_bounds
    tau_0 = float(np.clip(tau_0_raw, min_t, 0.49))
    tau_1 = float(np.clip(tau_1_raw, 0.51, max_t))

    # For very low prior, ensure tau_0 is strictly below tau_1 with proper separation
    if tau_1 - tau_0 < min_separation:
        midpoint = 0.5 * (tau_0 + tau_1)
        tau_0 = max(min_t, midpoint - 0.5 * min_separation)
        tau_1 = min(max_t, midpoint + 0.5 * min_separation)

    return tau_0, tau_1


def apply_coverage_guard(
    probs_1d: np.ndarray,
    tau_0: float,
    tau_1: float,
    gamma_min: float = 0.70,
    max_iterations: int = 50,
) -> Tuple[float, float, bool]:
    """
    Adjust [tau_0, tau_1] rejection band if empirical decision coverage < gamma_min.

    Parameters:
        probs_1d: 1D array of validation/OOF probabilities for the label.
        tau_0: Initial negative decision threshold.
        tau_1: Initial positive decision threshold.
        gamma_min: Minimum required coverage in [0, 1]. Default 0.70.
        max_iterations: Maximum contraction steps.

    Returns:
        tau_0_adj: Adjusted tau_0.
        tau_1_adj: Adjusted tau_1.
        was_adjusted: Boolean flag indicating if coverage guard intervened.
    """
    probs_clean = np.asarray(probs_1d, dtype=np.float64).ravel()
    if len(probs_clean) == 0:
        return tau_0, tau_1, False

    gamma = float(gamma_min)
    decided = (probs_clean <= tau_0) | (probs_clean >= tau_1)
    current_cov = float(np.mean(decided))

    if current_cov >= gamma:
        return tau_0, tau_1, False

    # Shrink rejection band iteratively towards the median of rejection zone
    t0, t1 = float(tau_0), float(tau_1)
    step = (t1 - t0) / (2.0 * max_iterations)

    for _ in range(max_iterations):
        t0 += step
        t1 -= step
        if t0 >= t1:
            mid = 0.5 * (tau_0 + tau_1)
            t0 = mid - 1e-4
            t1 = mid + 1e-4
            break
        cov = float(np.mean((probs_clean <= t0) | (probs_clean >= t1)))
        if cov >= gamma:
            break

    return float(t0), float(t1), True


def compute_adaptive_thresholds_table(
    Y_train: np.ndarray,
    OOF_probs: Optional[np.ndarray] = None,
    cost: float = 0.30,
    gamma_min: float = 0.70,
    clip_bounds: Tuple[float, float] = (0.005, 0.995),
) -> Dict[int, Dict[str, Any]]:
    """
    Compute prior-calibrated adaptive thresholds for all labels.

    Parameters:
        Y_train: Binary training labels of shape (N, K).
        OOF_probs: Optional Out-Of-Fold probabilities of shape (N, K).
        cost: Rejection cost c. Default 0.30.
        gamma_min: Coverage guard floor. Default 0.70.
        clip_bounds: Bound tuple for thresholds.

    Returns:
        thresholds_dict: Mapping label_idx -> {
            'tau_0': float,
            'tau_1': float,
            'prior': float,
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

        # Compute Bayes LR thresholds
        t0, t1 = compute_bayes_lr_thresholds(
            prior=prior_j,
            cost=cost,
            clip_bounds=clip_bounds,
        )

        guard_applied = False
        emp_cov = 1.0

        if OOF_probs is not None and OOF_probs.shape[1] > j:
            p_j = OOF_probs[:, j]
            t0, t1, guard_applied = apply_coverage_guard(
                probs_1d=p_j,
                tau_0=t0,
                tau_1=t1,
                gamma_min=gamma_min,
            )
            emp_cov = float(np.mean((p_j <= t0) | (p_j >= t1)))

        table[j] = {
            "tau_0": float(t0),
            "tau_1": float(t1),
            "prior": float(prior_j),
            "coverage_guard": bool(guard_applied),
            "empirical_coverage": float(emp_cov),
        }

    return table


def apply_asymmetric_abstention(
    probs: np.ndarray,
    thresholds_table: Dict[int, Dict[str, Any]],
    abstain_value: int = -1,
) -> np.ndarray:
    """
    Apply label-specific asymmetric thresholds to predicted probability matrix.

    Parameters:
        probs: Probability matrix of shape (M, K) with values in [0, 1].
        thresholds_table: Dict returned by compute_adaptive_thresholds_table.
        abstain_value: Sentinel value for rejection. Default -1.

    Returns:
        Y_pred: Array of shape (M, K) with values in {0, 1, abstain_value}.
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
