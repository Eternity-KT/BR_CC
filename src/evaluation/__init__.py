"""
Evaluation metrics and cross-validation utilities for multi-label classification.
"""

from .metrics import (
    compute_all_metrics,
    compute_f1_with_abstentions_as_zero,
    compute_partial_abstention_metrics,
    compute_selective_macro_f1,
    compute_selective_micro_f1,
    METRICS_INFO,
)
from .cv import get_multilabel_cv
from .cache import backup_model_cache

__all__ = [
    "compute_all_metrics",
    "compute_f1_with_abstentions_as_zero",
    "compute_partial_abstention_metrics",
    "compute_selective_macro_f1",
    "compute_selective_micro_f1",
    "METRICS_INFO",
    "get_multilabel_cv",
    "backup_model_cache",
]
