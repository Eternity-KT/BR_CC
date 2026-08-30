"""Schema-v3 metric names, formulas, scopes and edge-case conventions.

This executable contract contains no metric computation and is deliberately
not exported by the schema-v2 production facade during Phase Q0.
"""

from dataclasses import dataclass


TARGET_RESULT_SCHEMA_VERSION = 3
METRIC_CONTRACT_VERSION = "3.0-contract"
DEFAULT_ABSTAIN_VALUE = -1
ZERO_DIVISION_VALUE = 0
EMPTY_INSTANCE_SCORE = 1.0
UNDEFINED_POLICY = "nan"

SCOPES = ("Full", "Selective", "Rejected", "Optimistic", "Group")

COMPLETE_METRIC_NAMES = (
    "Macro-F1",
    "Micro-F1",
    "Hamming Accuracy",
    "Hamming Loss",
    "Subset Accuracy",
    "Instance-F1",
    "Instance Jaccard",
    "Macro Precision",
    "Macro Recall",
)

SELECTIVE_METRIC_NAMES = (
    "Selective Hamming Accuracy",
    "Selective Macro-F1",
    "Selective Micro-F1",
    "Generalized Loss",
    "Coverage",
    "ABS",
    "AABS",
    "Risk at Coverage",
    "AURC",
)

REJECTED_METRIC_NAMES = (
    "Rejected Counterfactual Macro-F1",
    "Rejected Error Rate",
    "Error Capture Rate",
)

OPTIMISTIC_METRIC_NAMES = tuple(
    f"Optimistic {name}"
    for name in (
        "Macro-F1",
        "Micro-F1",
        "Hamming Accuracy",
        "Instance-F1",
        "Instance Jaccard",
    )
)

ORACLE_GAIN_METRIC_NAMES = tuple(
    name.replace("Optimistic ", "Oracle Gain ", 1)
    for name in OPTIMISTIC_METRIC_NAMES
)

# Templates: Q1 replaces ``Group`` with a configured name such as IL or DL.
GROUP_METRIC_NAMES = (
    "Group Label Count",
    "Group Full Macro-F1",
    "Group Full Macro Precision",
    "Group Full Macro Recall",
    "Group Full Instance Jaccard",
    "Group Coverage",
)

# Per-label records are exported separately from aggregate metric dictionaries.
PER_LABEL_FIELD_NAMES = (
    "Label Index",
    "Label Name",
    "Positive Support",
    "Negative Support",
    "Prevalence",
    "Full Precision",
    "Full Recall",
    "Full F1",
    "Full Hamming Accuracy",
    "Selective Precision",
    "Selective Recall",
    "Selective F1",
    "Coverage",
    "Rejected Position Count",
    "Rejected Error Count",
)

# Mandatory companions for metrics whose denominator may be zero.
DIAGNOSTIC_FIELD_NAMES = (
    "Total Position Count",
    "Decided Position Count",
    "Abstained Position Count",
    "Full Error Count",
    "Rejected Position Count",
    "Rejected Error Count",
    "Errors Avoided",
)

# Aliases only rename fields and never change their numeric value.
LEGACY_ALIASES = {
    "Example-F1": "Instance-F1",
    "Generalized Hamming Loss": "Generalized Loss",
    "Abstention Rate": "AABS",
}

# Compatibility fields needing a numeric transformation are kept separate.
LEGACY_DERIVED_FIELDS = {
    "Selective Hamming Loss": "1 - Selective Hamming Accuracy",
}


@dataclass(frozen=True)
class MetricDefinition:
    """A scalar metric definition used to verify the schema-v3 contract."""

    name: str
    scope: str
    formula: str
    denominator: str
    direction: str
    undefined_when: str = "never for a valid non-empty N x K input"


@dataclass(frozen=True)
class EdgeCaseConvention:
    """A machine-readable description of one metric edge-case rule."""

    case: str
    result: str
    rationale: str


METRIC_DEFINITIONS = (
    MetricDefinition(
        "Macro-F1", "Full",
        "mean_k(2*TP_k / (2*TP_k + FP_k + FN_k))",
        "K labels; each zero class denominator contributes 0", "higher_better",
    ),
    MetricDefinition(
        "Micro-F1", "Full",
        "2*sum_k(TP_k) / (2*sum_k(TP_k) + sum_k(FP_k) + sum_k(FN_k))",
        "all positive prediction/truth events across N*K positions", "higher_better",
    ),
    MetricDefinition(
        "Hamming Accuracy", "Full", "sum(Y_full == Y_true) / (N*K)",
        "N*K label-instance positions", "higher_better",
    ),
    MetricDefinition(
        "Hamming Loss", "Full", "sum(Y_full != Y_true) / (N*K)",
        "N*K label-instance positions", "lower_better",
    ),
    MetricDefinition(
        "Subset Accuracy", "Full",
        "sum_i(all_k(Y_full[i,k] == Y_true[i,k])) / N",
        "N instances", "higher_better",
    ),
    MetricDefinition(
        "Instance-F1", "Full", "mean_i(2*TP_i / (2*TP_i + FP_i + FN_i))",
        "N instances; a both-empty true/predicted set scores 1", "higher_better",
    ),
    MetricDefinition(
        "Instance Jaccard", "Full", "mean_i(TP_i / (TP_i + FP_i + FN_i))",
        "N instances; a both-empty true/predicted set scores 1", "higher_better",
    ),
    MetricDefinition(
        "Macro Precision", "Full", "mean_k(TP_k / (TP_k + FP_k))",
        "K labels; each zero class denominator contributes 0", "higher_better",
    ),
    MetricDefinition(
        "Macro Recall", "Full", "mean_k(TP_k / (TP_k + FN_k))",
        "K labels; each zero class denominator contributes 0", "higher_better",
    ),
    MetricDefinition(
        "Selective Hamming Accuracy", "Selective",
        "sum(D * (Y_partial == Y_true)) / sum(D)", "decided positions sum(D)",
        "higher_better", "sum(D) == 0",
    ),
    MetricDefinition(
        "Selective Macro-F1", "Selective",
        "mean_k(F1 on decided positions of label k)",
        "K labels; a label with zero decided positions contributes 0", "higher_better",
    ),
    MetricDefinition(
        "Selective Micro-F1", "Selective",
        "positive-class micro-F1 over all decided positions",
        "positive events across decided positions; all-abstain returns 0", "higher_better",
    ),
    MetricDefinition(
        "Generalized Loss", "Selective",
        "sum_i(decided_errors_i + psi_c(abstentions_i)) / (N*K)",
        "N*K positions; psi_c(a)=c*a (linear) or c*K*a/(K+a) (concave)",
        "lower_better",
    ),
    MetricDefinition(
        "Coverage", "Selective", "sum(D) / (N*K)",
        "N*K label-instance positions", "descriptive",
    ),
    MetricDefinition(
        "ABS", "Selective",
        "sum_i(any_k(Y_partial[i,k] == abstain_value)) / N",
        "N instances", "lower_better",
    ),
    MetricDefinition(
        "AABS", "Selective", "sum(1-D) / (N*K) = 1 - Coverage",
        "N*K label-instance positions", "lower_better",
    ),
    MetricDefinition(
        "Risk at Coverage", "Selective",
        "decided_errors / decided_positions at a declared coverage gamma",
        "positions accepted at the named operating point", "lower_better",
        "coverage == 0",
    ),
    MetricDefinition(
        "AURC", "Selective",
        "mean_m(error_count among top-m confidence positions / m), m=1..N*K",
        "N*K ranked prefixes; ties use stable row-major position order", "lower_better",
    ),
    MetricDefinition(
        "Rejected Counterfactual Macro-F1", "Rejected",
        "mean_k(F1 of Y_full on rejected positions for label k)",
        "K labels; labels without rejected positions are NaN and excluded",
        "descriptive", "there are no rejected positions for any label",
    ),
    MetricDefinition(
        "Rejected Error Rate", "Rejected",
        "Rejected Error Count / Rejected Position Count", "rejected positions",
        "higher_better_for_triage", "Rejected Position Count == 0",
    ),
    MetricDefinition(
        "Error Capture Rate", "Rejected",
        "Rejected Error Count / Full Error Count",
        "errors made by the complete prediction Y_full", "higher_better_for_triage",
        "Full Error Count == 0",
    ),
    MetricDefinition(
        "Optimistic Macro-F1", "Optimistic",
        "Macro-F1(Y_true, where(abstained, Y_true, Y_partial))",
        "same as Full Macro-F1", "upper_bound",
    ),
    MetricDefinition(
        "Optimistic Micro-F1", "Optimistic",
        "Micro-F1(Y_true, where(abstained, Y_true, Y_partial))",
        "same as Full Micro-F1", "upper_bound",
    ),
    MetricDefinition(
        "Optimistic Hamming Accuracy", "Optimistic",
        "Hamming Accuracy(Y_true, where(abstained, Y_true, Y_partial))",
        "N*K label-instance positions", "upper_bound",
    ),
    MetricDefinition(
        "Optimistic Instance-F1", "Optimistic",
        "Instance-F1(Y_true, where(abstained, Y_true, Y_partial))",
        "N instances", "upper_bound",
    ),
    MetricDefinition(
        "Optimistic Instance Jaccard", "Optimistic",
        "Instance Jaccard(Y_true, where(abstained, Y_true, Y_partial))",
        "N instances", "upper_bound",
    ),
    MetricDefinition(
        "Oracle Gain Macro-F1", "Optimistic", "Optimistic Macro-F1 - Macro-F1",
        "difference of metrics on the same full evaluation base", "descriptive",
    ),
    MetricDefinition(
        "Oracle Gain Micro-F1", "Optimistic", "Optimistic Micro-F1 - Micro-F1",
        "difference of metrics on the same full evaluation base", "descriptive",
    ),
    MetricDefinition(
        "Oracle Gain Hamming Accuracy", "Optimistic",
        "Optimistic Hamming Accuracy - Hamming Accuracy",
        "difference of metrics on the same full evaluation base", "descriptive",
    ),
    MetricDefinition(
        "Oracle Gain Instance-F1", "Optimistic",
        "Optimistic Instance-F1 - Instance-F1",
        "difference of metrics on the same N-instance base", "descriptive",
    ),
    MetricDefinition(
        "Oracle Gain Instance Jaccard", "Optimistic",
        "Optimistic Instance Jaccard - Instance Jaccard",
        "difference of metrics on the same N-instance base", "descriptive",
    ),
    MetricDefinition(
        "Group Label Count", "Group", "number of labels assigned to the named group",
        "configured group membership", "descriptive",
    ),
    MetricDefinition(
        "Group Full Macro-F1", "Group",
        "mean positive-class F1 over labels in the named group", "labels in the group",
        "higher_better", "Group Label Count == 0",
    ),
    MetricDefinition(
        "Group Full Macro Precision", "Group",
        "mean positive-class precision over labels in the named group",
        "labels in the group", "higher_better", "Group Label Count == 0",
    ),
    MetricDefinition(
        "Group Full Macro Recall", "Group",
        "mean positive-class recall over labels in the named group",
        "labels in the group", "higher_better", "Group Label Count == 0",
    ),
    MetricDefinition(
        "Group Full Instance Jaccard", "Group",
        "mean instance Jaccard after restricting Y_true and Y_full to the group",
        "N instances and labels in the group", "higher_better",
        "Group Label Count == 0",
    ),
    MetricDefinition(
        "Group Coverage", "Group",
        "decided positions in the group / (N * Group Label Count)",
        "label-instance positions in the group", "descriptive",
        "Group Label Count == 0",
    ),
)


EDGE_CASE_CONVENTIONS = (
    EdgeCaseConvention(
        "input has N == 0 or K == 0", "raise ValueError before metric computation",
        "An empty evaluation matrix is not a valid run.",
    ),
    EdgeCaseConvention(
        "empty true and predicted label set for one instance",
        "Instance-F1 = Instance Jaccard = 1.0", "An empty set is predicted exactly.",
    ),
    EdgeCaseConvention(
        "no decided label-instance positions",
        "Selective Hamming Accuracy and Risk at Coverage = NaN",
        "A zero denominator must not reward all-abstain.",
    ),
    EdgeCaseConvention(
        "no decided positions for Selective Micro-F1", "Selective Micro-F1 = 0",
        "The conservative score prevents all-abstain from appearing successful.",
    ),
    EdgeCaseConvention(
        "one label has no decided examples",
        "that label contributes 0 to Selective Macro-F1",
        "The conservative rule prevents selective-score inflation.",
    ),
    EdgeCaseConvention(
        "IL or DL group contains zero labels",
        "all group metrics = NaN except Group Label Count = 0",
        "An empty group is undefined rather than a failed prediction.",
    ),
    EdgeCaseConvention(
        "a label has no rejected positions",
        "its rejected diagnostic = NaN and is excluded from rejected macro aggregation",
        "Absence of reviewed examples is not perfect performance.",
    ),
    EdgeCaseConvention(
        "Rejected Error Rate denominator is zero",
        "Rejected Error Rate = NaN and raw rejected counts are exported",
        "There is no rejected set on which to measure error concentration.",
    ),
    EdgeCaseConvention(
        "Error Capture Rate denominator is zero",
        "Error Capture Rate = NaN and Full Error Count = 0 is exported",
        "A complete predictor with no errors has no errors to capture.",
    ),
    EdgeCaseConvention(
        "positive-class precision/recall/F1 denominator is zero",
        "zero_division = 0, including optimistic metrics",
        "This matches the locked benchmark convention.",
    ),
)


def canonical_metric_names():
    """Return every aggregate schema-v3 metric name/template in stable order."""

    return (
        COMPLETE_METRIC_NAMES
        + SELECTIVE_METRIC_NAMES
        + REJECTED_METRIC_NAMES
        + OPTIMISTIC_METRIC_NAMES
        + ORACLE_GAIN_METRIC_NAMES
        + GROUP_METRIC_NAMES
    )


def group_metric_names(group_name):
    """Materialize canonical group fields, for example ``IL Coverage``."""

    if not isinstance(group_name, str) or not group_name.strip():
        raise ValueError("group_name must be a non-empty string.")
    prefix = group_name.strip()
    return tuple(name.replace("Group", prefix, 1) for name in GROUP_METRIC_NAMES)


def validate_metric_contract():
    """Raise ``ValueError`` if the schema-v3 contract is internally invalid."""

    names = canonical_metric_names()
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate canonical metric names: {duplicates}")

    defined_names = tuple(definition.name for definition in METRIC_DEFINITIONS)
    if set(defined_names) != set(names):
        missing = sorted(set(names) - set(defined_names))
        unexpected = sorted(set(defined_names) - set(names))
        raise ValueError(
            f"Definition/name mismatch; missing={missing}, unexpected={unexpected}"
        )
    if len(defined_names) != len(set(defined_names)):
        raise ValueError("Every canonical metric must have exactly one definition.")
    for definition in METRIC_DEFINITIONS:
        if definition.scope not in SCOPES:
            raise ValueError(f"Unknown scope for {definition.name}: {definition.scope}")
        if not definition.formula or not definition.denominator or not definition.direction:
            raise ValueError(f"Incomplete metric definition: {definition.name}")

    missing_targets = sorted(set(LEGACY_ALIASES.values()) - set(names))
    if missing_targets:
        raise ValueError(f"Legacy aliases target unknown metrics: {missing_targets}")
    if DEFAULT_ABSTAIN_VALUE in (0, 1):
        raise ValueError("The abstention value must differ from binary outputs.")
    return True
