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
    ):
        if abstain_value in (0, 1):
            raise ValueError("abstain_value must be different from 0 and 1.")
        self.cost = validate_cost(cost)
        self.penalty = validate_penalty(penalty)
        self.allow_abstention = bool(allow_abstention)
        self.abstain_value = int(abstain_value)
        self.min_positive_support = int(min_positive_support)

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

            if not self.allow_abstention:
                # 1D search over complete thresholds
                y_pos = (y_col == 1)
                is_pos = p_col[:, None] >= thresholds[None, :]
                tp_arr = y_pos @ is_pos
                fp_arr = (~y_pos) @ is_pos
                fn_arr = pos_count - tp_arr
                denom = 2 * tp_arr + fp_arr + fn_arr
                f1_arr = np.where(denom > 0, 2.0 * tp_arr / np.maximum(denom, 1e-12), 0.0)
                best_idx = int(np.argmax(f1_arr))
                tau_low[k] = float(thresholds[best_idx])
                tau_high[k] = float(thresholds[best_idx])
            else:
                # 2D search with vectorized inner loop
                is_pos = p_col[:, None] >= thresholds[None, :]
                is_neg = p_col[:, None] <= thresholds[None, :]
                y_pos = (y_col == 1)

                tp_high = y_pos @ is_pos
                fp_high = (~y_pos) @ is_pos
                fn_low = y_pos @ is_neg

                best_utility = -np.inf
                best_decided = -1
                best_i = 0
                best_j = m_thresh - 1

                for i in range(m_thresh):
                    neg_col = is_neg[:, i:i+1]
                    pos_cols = is_pos[:, i:]
                    decided = np.sum(neg_col | pos_cols, axis=0)
                    a = n_samples - decided

                    tp = tp_high[i:]
                    fp = fp_high[i:]
                    fn = fn_low[i]
                    denom = 2 * tp + fp + fn
                    both_empty = (tp + fp + fn) == 0
                    f1 = np.where(
                        denom > 0,
                        2.0 * tp / np.maximum(denom, 1e-12),
                        np.where(both_empty, 1.0, 0.0),
                    )
                    f1[decided == 0] = 0.0
                    pen = np.asarray(
                        abstention_penalty(a, 1, cost, penalty),
                        dtype=np.float64,
                    ) / float(n_samples)
                    utils = f1 - pen

                    max_idx = int(np.argmax(utils))
                    candidate_util = utils[max_idx]
                    candidate_decided = int(decided[max_idx])

                    if (
                        candidate_util > best_utility + 1e-12
                        or (
                            np.isclose(candidate_util, best_utility, atol=1e-12)
                            and candidate_decided > best_decided
                        )
                    ):
                        best_utility = candidate_util
                        best_decided = candidate_decided
                        best_i = i
                        best_j = i + max_idx

                tau_low[k] = float(thresholds[best_i])
                tau_high[k] = float(thresholds[best_j])

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
            "algorithm": "per_label_adaptive_thresholding_with_abstention",
            "inference_complexity": "O(K) time, O(K) space",
            "tie_breaking": "more_decisions_then_smaller_margin",
        }
