"""
Evaluation metrics and cross-validation utilities for multi-label classification.
"""

from .metrics import (
    compute_all_metrics,
    compute_f1_with_abstentions_as_zero,
    compute_partial_abstention_metrics,
    compute_selective_instance_f1,
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
    "compute_selective_instance_f1",
    "compute_selective_macro_f1",
    "compute_selective_micro_f1",
    "METRICS_INFO",
    "get_multilabel_cv",
    "backup_model_cache",
]
from .calibration_metrics import (
    CALIBRATION_METRIC_NAMES,
    compute_calibration_metrics,
)
from .deployment import (
    OPERATING_POINT_RULES,
    compute_deployment_metrics,
    operating_point_record,
    select_operating_point,
)

__all__ += [
    "CALIBRATION_METRIC_NAMES",
    "compute_calibration_metrics",
    "OPERATING_POINT_RULES",
    "compute_deployment_metrics",
    "operating_point_record",
    "select_operating_point",
]
