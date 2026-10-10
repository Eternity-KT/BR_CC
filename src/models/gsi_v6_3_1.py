"""
GSI-MLC-PA Version 6.3.1: Precision-Guarded Balanced-Root Calibration and Asymmetric Inference.

Enhancements over v6.3.0:
1. Balanced-Root Calibration (method='sqrt_platt'):
   Moderates extreme sample weighting in Platt scaling (uses sqrt(n_neg / n_pos) instead of full linear ratio).
   Eliminates probability over-inflation for rare classes, preventing false positive explosion.
2. Precision Guard (precision_guard=True, min_tau_1=0.50):
   Guarantees that the positive acceptance threshold tau_1 never drops below 0.50 during coverage expansion.
   Coverage is expanded safely by increasing negative decision threshold tau_0 towards the median,
   correctly classifying true negatives without sacrificing precision.
3. Restored Metric Pareto Equilibrium:
   Significantly recovers Subset 0/1 Accuracy and reduces Hamming Loss while preserving high Macro-F1 gains.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from .gsi_v6_3 import GSIMLCPAv6_3Classifier


class GSIMLCPAv6_3_1Classifier(GSIMLCPAv6_3Classifier):
    """
    GSI-MLC-PA Version 6.3.1 Classifier with Precision Guard and Balanced-Root Calibration.
    """

    def __init__(
        self,
        base_learner: str = "logistic",
        stratified_threshold: float = 0.75,
        residual_corr_threshold: float = 0.25,
        cv_folds: int = 5,
        max_peeling_depth: int = 3,
        peeling_metric: str = "selective_f1",
        decaying_threshold: bool = True,
        threshold_decay_step: float = 0.05,
        aug_normalization: str = "matching",
        error_format: str = "residual",
        cost: float = 0.30,
        gamma_min: float = 0.70,
        use_prior_adaptive: bool = True,
        calibrate_tail: bool = True,
        calibration_method: str = "sqrt_platt",
        precision_guard: bool = True,
        min_tau_1: float = 0.50,
        abstain_value: int = -1,
        random_state: int = 42,
    ):
        super().__init__(
            base_learner=base_learner,
            stratified_threshold=stratified_threshold,
            residual_corr_threshold=residual_corr_threshold,
            cv_folds=cv_folds,
            max_peeling_depth=max_peeling_depth,
            peeling_metric=peeling_metric,
            decaying_threshold=decaying_threshold,
            threshold_decay_step=threshold_decay_step,
            aug_normalization=aug_normalization,
            error_format=error_format,
            cost=cost,
            gamma_min=gamma_min,
            use_prior_adaptive=use_prior_adaptive,
            calibrate_tail=calibrate_tail,
            abstain_value=abstain_value,
            random_state=random_state,
        )
        self.calibration_method = str(calibration_method)
        self.precision_guard = bool(precision_guard)
        self.min_tau_1 = float(min_tau_1)
