"""Importable schema-v3 facade reserved for the Q1/Q2 migration.

It remains disconnected from ``src.evaluation`` and the experiment runner in
Q0, so adding the contract cannot change existing schema-v2 results.
"""

from .metric_contract import DEFAULT_ABSTAIN_VALUE


def compute_metric_bundle(
    y_true,
    y_full,
    *,
    y_partial=None,
    label_groups=None,
    cost=None,
    penalty="linear",
    abstain_value=DEFAULT_ABSTAIN_VALUE,
):
    """Return all schema-v3 metric scopes once Q1 implements the modules."""

    raise NotImplementedError(
        "Schema-v3 metric facade is a Q0 contract skeleton; implement in Phase Q1."
    )
