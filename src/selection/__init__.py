"""Selection objectives and partition infrastructure for GSI models."""

from .objectives import (
    SelectionObjectiveResult,
    available_selection_objectives,
    canonical_selection_objective,
    evaluate_selection_objective,
)


__all__ = [
    "SelectionObjectiveResult",
    "available_selection_objectives",
    "canonical_selection_objective",
    "evaluate_selection_objective",
]
