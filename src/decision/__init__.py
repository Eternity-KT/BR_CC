"""Decision-policy layer kept independent from probability estimators."""

from .base import DecisionPolicy
from .hamming import HammingBOPPolicy
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
__all__ = [
    "DecisionPolicy",
    "HammingBOPPolicy",
    "available_policies",
    "canonical_policy_name",
    "create_policy",
    "register_policy",
]
