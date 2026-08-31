"""Decision-policy layer kept independent from probability estimators."""

from .base import DecisionPolicy
from .fbeta import FbetaBOPPolicy
from .hamming import HammingBOPPolicy
from .jaccard import JaccardBOPPolicy
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


__all__ = [
    "DecisionPolicy",
    "FbetaBOPPolicy",
    "HammingBOPPolicy",
    "JaccardBOPPolicy",
    "available_policies",
    "canonical_policy_name",
    "create_policy",
    "register_policy",
]
