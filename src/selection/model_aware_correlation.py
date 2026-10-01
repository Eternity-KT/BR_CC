"""Model-Aware Label Correlation and Confidence Ordering Module for GSI-MLC-PA.

This module provides feature-aware and model-driven alternatives to raw ground-truth
label correlation matrix estimation:
1. Predicted Probability Correlation (Phi_pred): Correlation between soft probability vectors.
2. Residual Error Correlation (Phi_resid): Correlation between prediction residuals (Y - P_hat),
   directly approximating conditional label dependency P(Y_j, Y_k | X, P_hat(IL)).
3. Confidence-Driven DL Ordering: Ordering labels in Classifier Chains based on base model
   performance/uncertainty (e.g. Validation F1 descending or entropy ascending).

Design constraint: Non-destructive, purely functional, backwards-compatible.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


def compute_predicted_correlation_matrix(P_prob: np.ndarray) -> np.ndarray:
    """Compute absolute Pearson correlation matrix across predicted soft probability vectors.

    Parameters:
        P_prob: Continuous prediction matrix of shape (n_samples, n_labels),
                values typically in [0.0, 1.0].

    Returns:
        K x K symmetric matrix of absolute correlation values in [0.0, 1.0],
        with diagonal entries set to 1.0.
    """
    P_prob = np.asarray(P_prob, dtype=np.float64)
    if P_prob.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {P_prob.shape}")

    n_samples, n_labels = P_prob.shape
    if n_labels == 0:
        return np.empty((0, 0), dtype=np.float32)
    if n_labels == 1:
        return np.ones((1, 1), dtype=np.float32)

    # Check for zero-variance columns (constant predictions)
    stds = np.std(P_prob, axis=0)
    valid_mask = stds > 1e-12

    if not np.any(valid_mask):
        corr = np.zeros((n_labels, n_labels), dtype=np.float64)
        np.fill_diagonal(corr, 1.0)
        return corr.astype(np.float32)

    with np.errstate(all="ignore"):
        corr = np.corrcoef(P_prob, rowvar=False)

    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    abs_corr = np.abs(np.clip(corr, -1.0, 1.0))
    np.fill_diagonal(abs_corr, 1.0)
    return abs_corr.astype(np.float32)


def compute_residual_correlation_matrix(
    Y_true: np.ndarray,
    P_prob: np.ndarray,
) -> np.ndarray:
    """Compute absolute Pearson correlation matrix between prediction residuals E = Y_true - P_prob.

    Residual correlation measures whether the base model makes correlated errors on pairs
    of labels after conditioning on context features [X, P_hat(IL)].
    High residual correlation indicates true conditional dependency requiring CC coupling.

    Parameters:
        Y_true: Ground-truth binary label matrix of shape (n_samples, n_labels).
        P_prob: Predicted soft probability matrix of shape (n_samples, n_labels).

    Returns:
        K x K symmetric matrix of absolute residual correlation in [0.0, 1.0],
        with diagonal entries set to 1.0.
    """
    Y_true = np.asarray(Y_true, dtype=np.float64)
    P_prob = np.asarray(P_prob, dtype=np.float64)

    if Y_true.shape != P_prob.shape:
        raise ValueError(
            f"Shape mismatch: Y_true {Y_true.shape} vs P_prob {P_prob.shape}"
        )

    residuals = Y_true - P_prob
    return compute_predicted_correlation_matrix(residuals)


def order_dl_by_confidence(
    confidence_scores: Dict[int, float],
    dl_labels: Sequence[int],
    direction: str = "descending",
) -> List[int]:
    """Order residual dependent labels by model confidence or validation metric.

    Parameters:
        confidence_scores: Mapping from label index to score (e.g. Validation F1, Precision, Certainty).
        dl_labels: List of label indices belonging to DL.
        direction: 'descending' (highest score / easiest label first) or
                   'ascending' (hardest label first).

    Returns:
        List of label indices ordered by score. Ties broken deterministically by label index.
    """
    dl_list = [int(lbl) for lbl in dl_labels]
    if len(dl_list) <= 1:
        return list(dl_list)

    reverse = (direction == "descending")
    # Sort key: primary = score, secondary = label index (inverted if reverse to maintain stable tie-break)
    return sorted(
        dl_list,
        key=lambda lbl: (confidence_scores.get(lbl, 0.0), -lbl if reverse else lbl),
        reverse=reverse,
    )


def order_dl_by_model_correlation(
    correlation_matrix: np.ndarray,
    dl_labels: Sequence[int],
    direction: str = "ascending",
) -> List[int]:
    """Order residual dependent labels by cumulative absolute model-aware correlation.

    Parameters:
        correlation_matrix: Full K x K correlation matrix (Phi_pred or Phi_resid).
        dl_labels: List of label indices belonging to DL.
        direction: 'ascending' (least correlated first) or 'descending' (most correlated first).

    Returns:
        List of label indices ordered by cumulative internal correlation.
    """
    dl_list = [int(lbl) for lbl in dl_labels]
    if len(dl_list) <= 1:
        return list(dl_list)

    scores: Dict[int, float] = {}
    for label in dl_list:
        other_labels = [l for l in dl_list if l != label]
        if other_labels:
            total_corr = float(np.sum(correlation_matrix[label, other_labels]))
        else:
            total_corr = 0.0
        scores[label] = total_corr

    reverse = (direction == "descending")
    return sorted(
        dl_list,
        key=lambda l: (scores[l], -l if reverse else l),
        reverse=reverse,
    )
