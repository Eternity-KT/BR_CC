"""Deterministic IL/DL partition providers for GSI ablations.

This module deliberately contains no model fitting or validation scoring.  A
caller supplies the learned reference partition when a mode needs one, and the
provider returns a frozen partition that can be consumed by the unchanged GSI
probability inference path.
"""

from dataclasses import dataclass

import numpy as np


PARTITION_MODES = (
    "learned",
    "learned_no_correlation_order",
    "all_il",
    "all_dl",
    "fixed",
    "random_matched",
)
FINAL_ORDER_STRATEGIES = ("correlation", "selection", "natural")

_PARTITION_ALIASES = {
    "learned_no_reorder": "learned_no_correlation_order",
    "random": "random_matched",
}


def canonical_partition_mode(mode):
    """Return one public partition-mode name or raise a clear error."""

    key = str(mode).strip().lower().replace("-", "_")
    key = _PARTITION_ALIASES.get(key, key)
    if key not in PARTITION_MODES:
        raise ValueError(
            f"Unknown partition mode: {mode}. Available: {PARTITION_MODES}."
        )
    return key


def canonical_final_order_strategy(strategy):
    """Validate the strategy used after a partition has been frozen."""

    key = str(strategy).strip().lower().replace("-", "_")
    if key not in FINAL_ORDER_STRATEGIES:
        raise ValueError(
            "Unknown final order strategy: "
            f"{strategy}. Available: {FINAL_ORDER_STRATEGIES}."
        )
    return key


def _validated_label_set(labels, n_labels, field_name):
    if labels is None:
        raise ValueError(f"{field_name} is required for this partition mode.")
    try:
        raw_labels = tuple(labels)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be an iterable of label indices.") from exc
    normalized = []
    seen = set()
    for label in raw_labels:
        if isinstance(label, (bool, np.bool_)) or not isinstance(
            label, (int, np.integer)
        ):
            raise ValueError(f"{field_name} must contain only integer indices.")
        label = int(label)
        if not 0 <= label < n_labels:
            raise ValueError(
                f"{field_name} contains out-of-range label {label} for "
                f"{n_labels} labels."
            )
        if label in seen:
            raise ValueError(f"{field_name} contains duplicate label {label}.")
        normalized.append(label)
        seen.add(label)
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class PartitionResult:
    """One validated, frozen partition plus its ablation provenance."""

    mode: str
    independent_labels: tuple
    dependent_labels: tuple
    random_state: int
    reference_independent_count: int | None = None

    def as_dict(self):
        return {
            "mode": self.mode,
            "independent_labels": [int(label) for label in self.independent_labels],
            "dependent_labels": [int(label) for label in self.dependent_labels],
            "independent_count": int(len(self.independent_labels)),
            "dependent_count": int(len(self.dependent_labels)),
            "random_state": int(self.random_state),
            "reference_independent_count": (
                None
                if self.reference_independent_count is None
                else int(self.reference_independent_count)
            ),
        }


def provide_partition(
    mode,
    n_labels,
    *,
    learned_independent_labels=None,
    fixed_independent_labels=None,
    random_state=42,
):
    """Return a deterministic IL/DL partition for one ablation mode.

    ``random_matched`` samples uniformly without replacement while preserving
    the number of independent labels in ``learned_independent_labels``.
    """

    if isinstance(n_labels, (bool, np.bool_)) or not isinstance(
        n_labels, (int, np.integer)
    ):
        raise ValueError("n_labels must be a positive integer.")
    n_labels = int(n_labels)
    if n_labels < 1:
        raise ValueError("n_labels must be a positive integer.")
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(
        random_state, (int, np.integer)
    ):
        raise ValueError("random_state must be an integer.")
    random_state = int(random_state)
    canonical_mode = canonical_partition_mode(mode)
    learned = None
    reference_count = None
    if canonical_mode in (
        "learned",
        "learned_no_correlation_order",
        "random_matched",
    ):
        learned = _validated_label_set(
            learned_independent_labels,
            n_labels,
            "learned_independent_labels",
        )
        reference_count = len(learned)

    if canonical_mode in ("learned", "learned_no_correlation_order"):
        independent = learned
    elif canonical_mode == "all_il":
        independent = tuple(range(n_labels))
    elif canonical_mode == "all_dl":
        independent = ()
    elif canonical_mode == "fixed":
        independent = _validated_label_set(
            fixed_independent_labels,
            n_labels,
            "fixed_independent_labels",
        )
    else:
        generator = np.random.default_rng(random_state)
        independent = tuple(
            sorted(
                int(label)
                for label in generator.choice(
                    n_labels,
                    size=reference_count,
                    replace=False,
                )
            )
        )

    independent_set = set(independent)
    dependent = tuple(
        label for label in range(n_labels) if label not in independent_set
    )
    return PartitionResult(
        mode=canonical_mode,
        independent_labels=tuple(independent),
        dependent_labels=dependent,
        random_state=random_state,
        reference_independent_count=reference_count,
    )
