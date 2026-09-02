"""Selection objectives and partition infrastructure for GSI models."""

from .objectives import (
    SelectionObjectiveResult,
    available_selection_objectives,
    canonical_selection_objective,
    evaluate_selection_objective,
)
from .partition import (
    FINAL_ORDER_STRATEGIES,
    PARTITION_MODES,
    PartitionResult,
    canonical_final_order_strategy,
    canonical_partition_mode,
    provide_partition,
)


__all__ = [
    "SelectionObjectiveResult",
    "available_selection_objectives",
    "canonical_selection_objective",
    "evaluate_selection_objective",
    "FINAL_ORDER_STRATEGIES",
    "PARTITION_MODES",
    "PartitionResult",
    "canonical_final_order_strategy",
    "canonical_partition_mode",
    "provide_partition",
]
