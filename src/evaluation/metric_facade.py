"""Structured schema-v3 metric facade, kept outside production until Q2."""

from numbers import Integral

from .abstention_metrics import compute_abstention_metrics
from .complete_metrics import compute_complete_metrics
from .group_metrics import compute_group_metrics, compute_per_label_metrics
from .metric_contract import (
    DEFAULT_ABSTAIN_VALUE,
    LEGACY_ALIASES,
    METRIC_CONTRACT_VERSION,
)


class AliasAwareMetricDict(dict):
    """Resolve legacy names on lookup without serializing duplicate columns."""

    def __init__(self, values, aliases=None):
        super().__init__(values)
        self._aliases = dict(aliases or {})

    def __missing__(self, key):
        canonical = self._aliases.get(key)
        if canonical is None or canonical not in self:
            raise KeyError(key)
        return self[canonical]

    def get(self, key, default=None):
        """Resolve aliases consistently for both indexed and ``get`` lookup."""

        try:
            return self[key]
        except KeyError:
            return default


def _critical_label_payload(per_label_records, critical_labels):
    if critical_labels is None:
        return {
            "Status": "N/A",
            "Reason": "critical_labels is not configured by the domain",
            "Records": [],
        }
    requested = (
        (critical_labels,)
        if isinstance(critical_labels, str)
        else tuple(critical_labels)
    )
    if not requested:
        return {
            "Status": "N/A",
            "Reason": "critical_labels is configured but empty",
            "Records": [],
        }

    by_index = {record["Label Index"]: record for record in per_label_records}
    by_name = {record["Label Name"]: record for record in per_label_records}
    selected = []
    seen_indices = set()
    for reference in requested:
        if isinstance(reference, bool):
            raise ValueError("critical_labels cannot use boolean label references.")
        record = (
            by_index.get(int(reference))
            if isinstance(reference, Integral)
            else by_name.get(str(reference))
        )
        if record is None:
            raise ValueError(f"Unknown critical label reference: {reference}")
        if record["Label Index"] in seen_indices:
            raise ValueError(f"Duplicate critical label reference: {reference}")
        selected.append(record)
        seen_indices.add(record["Label Index"])
    return {"Status": "Available", "Reason": None, "Records": selected}


def compute_metric_bundle(
    y_true,
    y_full,
    *,
    y_partial=None,
    label_groups=None,
    cost=None,
    penalty="linear",
    abstain_value=DEFAULT_ABSTAIN_VALUE,
    acceptance_confidence=None,
    label_names=None,
    critical_labels=None,
):
    """Return schema-v3 metrics separated by evaluation scope.

    When ``y_partial`` is omitted the complete output is treated as a
    no-abstention policy.  Supplying a partial output requires an explicit cost.
    The returned ``Full`` mapping resolves ``Example-F1`` as a lookup alias for
    ``Instance-F1`` but serializes only the canonical field.
    """

    full_metrics = compute_complete_metrics(y_true, y_full)
    effective_partial = y_full if y_partial is None else y_partial
    if y_partial is not None and cost is None:
        raise ValueError("cost is required when y_partial is supplied.")
    effective_cost = 0.0 if cost is None else cost
    abstention = compute_abstention_metrics(
        y_true,
        effective_partial,
        y_full,
        cost=effective_cost,
        penalty=penalty,
        abstain_value=abstain_value,
        acceptance_confidence=acceptance_confidence,
    )
    per_label = compute_per_label_metrics(
        y_true,
        y_full,
        effective_partial,
        abstain_value=abstain_value,
        label_names=label_names,
    )
    groups = compute_group_metrics(
        y_true,
        y_full,
        effective_partial,
        {} if label_groups is None else label_groups,
        abstain_value=abstain_value,
    )

    full_aliases = {
        alias: canonical
        for alias, canonical in LEGACY_ALIASES.items()
        if canonical in full_metrics
    }
    selective_aliases = {
        alias: canonical
        for alias, canonical in LEGACY_ALIASES.items()
        if canonical in abstention["Selective"]
    }
    return {
        "Metric Contract Version": METRIC_CONTRACT_VERSION,
        "Full": AliasAwareMetricDict(full_metrics, full_aliases),
        "Selective": AliasAwareMetricDict(
            abstention["Selective"], selective_aliases
        ),
        "Rejected": abstention["Rejected"],
        "Optimistic": abstention["Optimistic"],
        "Diagnostics": abstention["Diagnostics"],
        "Per Label": per_label,
        "Groups": groups,
        "Critical Labels": _critical_label_payload(per_label, critical_labels),
    }
