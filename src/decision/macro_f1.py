"""Per-label thresholding decision policy optimizing Macro-F1 with partial abstention."""

import numpy as np

from .base import (
    DecisionPolicy,
    abstention_penalty,
    validate_cost,
    validate_penalty,
    validate_probability_matrix,
)


class PerLabelMacroF1Policy(DecisionPolicy):
    """Per-label adaptive thresholding optimizing Selective Macro-F1 with partial abstention.

    Unlike global symmetric Hamming thresholds (c, 1-c), this policy optimizes
    per-label cutoff thresholds (tau_k_low, tau_k_high) to directly maximize
    Selective Macro-F1 with an abstention cost penalty on validation data.

    For each label k:
        - If p_k >= tau_k_high -> 1 (Positive)
        - If p_k <= tau_k_low  -> 0 (Negative)
        - If tau_k_low < p_k < tau_k_high -> -1 (Abstain)

    When tau_k_low == tau_k_high == tau_k (or allow_abstention=False):
        - If p_k >= tau_k -> 1
        - If p_k < tau_k  -> 0
    """

    policy_name = "per_label_macro_f1"

    def __init__(
        self,
        cost=0.3,
        penalty="linear",
        allow_abstention=True,
        abstain_value=-1,
        tau_low=None,
        tau_high=None,
        min_positive_support=1,
        min_coverage=None,
    ):
        if abstain_value in (0, 1):
            raise ValueError("abstain_value must be different from 0 and 1.")
        self.cost = validate_cost(cost)
        self.penalty = validate_penalty(penalty)
        self.allow_abstention = bool(allow_abstention)
        self.abstain_value = int(abstain_value)
        self.min_positive_support = int(min_positive_support)
        self.min_coverage = float(min_coverage) if min_coverage is not None else None

        self.tau_low = (
            None if tau_low is None else np.asarray(tau_low, dtype=np.float64)
        )
        self.tau_high = (
            None if tau_high is None else np.asarray(tau_high, dtype=np.float64)
        )
        if self.tau_low is not None and self.tau_high is not None:
            if self.tau_low.shape != self.tau_high.shape:
                raise ValueError("tau_low and tau_high must have the same shape.")
            if np.any(self.tau_low > self.tau_high):
                raise ValueError("tau_low must be less than or equal to tau_high elementwise.")

        self._fitted_cache = {}
        self._val_probs = None
        self._val_y_true = None

    def _settings(self, cost, penalty):
        resolved_cost = self.cost if cost is None else validate_cost(cost)
        resolved_penalty = (
            self.penalty if penalty is None else validate_penalty(penalty)
        )
        return resolved_cost, resolved_penalty

    def _solve_thresholds(self, probs, y_true, cost, penalty):
        n_samples, n_labels = probs.shape
        tau_low = np.zeros(n_labels, dtype=np.float64)
        tau_high = np.ones(n_labels, dtype=np.float64)

        for k in range(n_labels):
            p_col = probs[:, k]
            y_col = y_true[:, k]
            pos_count = int(np.sum(y_col == 1))

            # Degenerate all-negative: predict 0 everywhere
            if pos_count < self.min_positive_support:
                tau_low[k] = 1.0
                tau_high[k] = 1.0
                continue

            # Degenerate all-positive: predict 1 everywhere
            if pos_count == n_samples:
                tau_low[k] = 0.0
                tau_high[k] = 0.0
                continue

            # Candidate thresholds
            thresholds = np.unique(np.clip(p_col, 0.0, 1.0))
            if len(thresholds) > 101:
                thresholds = np.unique(np.percentile(p_col, np.linspace(0, 100, 101)))
            thresholds = np.sort(
                np.unique(np.concatenate(([0.0, 0.5, 1.0], thresholds)))
            )
            m_thresh = len(thresholds)

            y_pos = (y_col == 1).astype(np.float64)
            y_neg = (y_col == 0).astype(np.float64)
            is_pos = (p_col[:, None] >= thresholds[None, :]).astype(np.float64)
            is_neg = (p_col[:, None] <= thresholds[None, :]).astype(np.float64)

            tp_high = y_pos @ is_pos
            fp_high = y_neg @ is_pos
            fn_low = y_pos @ is_neg

            denom_1d = 2 * tp_high + fp_high + (pos_count - tp_high)
            f1_1d = np.where(denom_1d > 0, 2.0 * tp_high / np.maximum(denom_1d, 1e-12), 0.0)

            prior = float(pos_count) / float(n_samples)
            pos_rate = np.mean(is_pos, axis=0)

            # Step 1: 1D center threshold search with anti-degeneracy guardrail
            valid_1d = (thresholds > 0.0) & (thresholds < 1.0)
            if prior < 0.4:
                valid_1d = valid_1d & (pos_rate <= max(0.20, 2.5 * prior))

            if np.any(valid_1d):
                valid_indices = np.where(valid_1d)[0]
                best_1d_idx = valid_indices[int(np.argmax(f1_1d[valid_1d]))]
                tau_star = float(thresholds[best_1d_idx])
            else:
                tau_star = 0.5

            # If cost >= 0.5 or abstention not allowed: complete 1D decision, Coverage = 1.0
            if not self.allow_abstention or cost >= 0.5 - 1e-9:
                tau_low[k] = tau_star
                tau_high[k] = tau_star
                continue

            # Step 2: Vectorized 2D search for [tau_low, tau_high] bracketing tau_star
            low_idx = np.where(thresholds <= tau_star)[0]
            high_idx = np.where(thresholds >= tau_star)[0]

            neg_count = np.sum(is_neg[:, low_idx], axis=0)
            pos_count_arr = np.sum(is_pos[:, high_idx], axis=0)

            decided_mat = np.clip(neg_count[:, None] + pos_count_arr[None, :], 0, n_samples)
            overlap_mask = (thresholds[low_idx, None] == thresholds[None, high_idx])
            decided_mat[overlap_mask] = n_samples

            tp_mat = tp_high[high_idx][None, :]
            fp_mat = fp_high[high_idx][None, :]
            fn_mat = fn_low[low_idx][:, None]
            denom_mat = 2 * tp_mat + fp_mat + fn_mat
            f1_mat = np.where(denom_mat > 0, 2.0 * tp_mat / np.maximum(denom_mat, 1e-12), 0.0)

            a_mat = n_samples - decided_mat
            cost_factor = float(cost) / 0.5
            pen_mat = cost_factor * (a_mat / float(n_samples)) * 0.5
            util_mat = f1_mat - pen_mat

            if self.min_coverage is not None:
                cov_ratio_mat = decided_mat / float(n_samples)
                util_mat[cov_ratio_mat < self.min_coverage - 1e-6] = -np.inf

            # Tie-breaking: maximize utility, then maximize coverage
            max_util = np.max(util_mat)
            if not np.isfinite(max_util):
                # Fallback to complete decision at tau_star
                tau_low[k] = tau_star
                tau_high[k] = tau_star
                continue

            close_mask = np.isclose(util_mat, max_util, atol=1e-12)
            masked_decided = np.where(close_mask, decided_mat, -1)
            best_i, best_j = np.unravel_index(np.argmax(masked_decided), util_mat.shape)

            tau_low[k] = float(thresholds[low_idx[best_i]])
            tau_high[k] = float(thresholds[high_idx[best_j]])

        return tau_low, tau_high

    def fit(self, val_probabilities, val_y_true, *, cost=None, penalty=None):
        """Fit optimal per-label thresholds on a validation split."""
        probs = validate_probability_matrix(val_probabilities)
        y_true = np.asarray(val_y_true, dtype=np.int32)
        if y_true.shape != probs.shape:
            raise ValueError("val_probabilities and val_y_true must have identical shapes.")

        resolved_cost, resolved_penalty = self._settings(cost, penalty)
        self._val_probs = probs
        self._val_y_true = y_true

        tau_low, tau_high = self._solve_thresholds(
            probs, y_true, resolved_cost, resolved_penalty
        )
        self.tau_low = tau_low
        self.tau_high = tau_high
        self._fitted_cache[(resolved_cost, resolved_penalty)] = (tau_low, tau_high)
        return self

    def _get_resolved_thresholds(self, n_labels, cost, penalty):
        cache_key = (float(cost), str(penalty))
        if cache_key in self._fitted_cache:
            return self._fitted_cache[cache_key]

        if self._val_probs is not None and self._val_y_true is not None:
            tau_low, tau_high = self._solve_thresholds(
                self._val_probs, self._val_y_true, cost, penalty
            )
            self._fitted_cache[cache_key] = (tau_low, tau_high)
            return tau_low, tau_high

        if self.tau_low is not None and self.tau_high is not None:
            if self.tau_low.size == n_labels and self.tau_high.size == n_labels:
                return self.tau_low, self.tau_high
            if self.tau_low.size == 1 and self.tau_high.size == 1:
                return (
                    np.full(n_labels, float(self.tau_low[0])),
                    np.full(n_labels, float(self.tau_high[0])),
                )

        # Default fallback if not fitted
        if self.allow_abstention:
            c = float(cost)
            if c >= 0.5:
                return np.full(n_labels, 0.5), np.full(n_labels, 0.5)
            return np.full(n_labels, c), np.full(n_labels, 1.0 - c)
        return np.full(n_labels, 0.5), np.full(n_labels, 0.5)

    def predict(self, probabilities, *, cost=None, penalty=None):
        matrix = validate_probability_matrix(probabilities)
        resolved_cost, resolved_penalty = self._settings(cost, penalty)
        n_samples, n_labels = matrix.shape

        tau_low, tau_high = self._get_resolved_thresholds(
            n_labels, resolved_cost, resolved_penalty
        )

        output = np.full(matrix.shape, self.abstain_value, dtype=np.int32)
        pos_mask = matrix >= tau_high[None, :]
        neg_mask = matrix <= tau_low[None, :]

        # When tau_low == tau_high, boundary convention: >= tau is 1, < tau is 0
        degenerate = (tau_low == tau_high)[None, :]
        if np.any(degenerate):
            neg_mask = np.where(degenerate, ~pos_mask, neg_mask)

        output[neg_mask] = 0
        output[pos_mask] = 1
        return output

    def expected_utility(self, probabilities, *, cost=None, penalty=None):
        """Return the row-wise expected generalized accuracy minus abstention penalty."""
        matrix = validate_probability_matrix(probabilities)
        resolved_cost, resolved_penalty = self._settings(cost, penalty)
        predictions = self.predict(
            matrix, cost=resolved_cost, penalty=resolved_penalty
        )

        decided = predictions != self.abstain_value
        predicted_pos = predictions == 1
        expected_accuracy = np.where(predicted_pos, matrix, 1.0 - matrix)
        decided_utility = np.sum(expected_accuracy * decided, axis=1)

        abstentions = np.sum(~decided, axis=1)
        penalties = abstention_penalty(
            abstentions, matrix.shape[1], resolved_cost, resolved_penalty
        )
        row_utility = (decided_utility - penalties) / float(matrix.shape[1])
        return np.asarray(row_utility, dtype=np.float64)

    def get_config(self):
        return {
            "name": self.policy_name,
            "objective": "selective_macro_f1",
            "cost": float(self.cost),
            "penalty": self.penalty,
            "allow_abstention": bool(self.allow_abstention),
            "abstain_value": int(self.abstain_value),
            "tau_low": (
                None if self.tau_low is None else self.tau_low.tolist()
            ),
            "tau_high": (
                None if self.tau_high is None else self.tau_high.tolist()
            ),
            "min_coverage": self.min_coverage,
            "algorithm": "per_label_adaptive_thresholding_with_abstention",
            "inference_complexity": "O(K) time, O(K) space",
            "tie_breaking": "more_decisions_then_smaller_margin",
        }
