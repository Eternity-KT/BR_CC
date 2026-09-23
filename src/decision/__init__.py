"""Decision-policy layer kept independent from probability estimators."""

from .base import DecisionPolicy
from .configuration import create_configured_policy
from .fbeta import FbetaBOPPolicy
from .hamming import HammingBOPPolicy
from .jaccard import JaccardBOPPolicy
from .macro_f1 import PerLabelMacroF1Policy
from .registry import (
    available_policies,
    canonical_policy_name,
    create_policy,
    register_policy,
)


register_policy(
    "hamming",
    HammingBOPPolicy,
    aliases=("hamming_bop", "sep", "par"),
)
register_policy(
    "fbeta",
    FbetaBOPPolicy,
    aliases=("fbeta_bop", "f1", "f1_bop"),
)
register_policy(
    "jaccard",
    JaccardBOPPolicy,
    aliases=("jaccard_bop", "iou", "intersection_over_union"),
)
register_policy(
    "macro_f1",
    PerLabelMacroF1Policy,
    aliases=("per_label_macro_f1", "macro_f1_bop", "per_label_f1"),
)


__all__ = [
    "DecisionPolicy",
    "FbetaBOPPolicy",
    "HammingBOPPolicy",
    "JaccardBOPPolicy",
    "PerLabelMacroF1Policy",
    "available_policies",
    "canonical_policy_name",
    "create_configured_policy",
    "create_policy",
    "register_policy",
]
