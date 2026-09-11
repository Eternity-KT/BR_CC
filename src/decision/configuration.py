"""Constructor adapter for policy families with different parameters."""

from .registry import canonical_policy_name, create_policy


def create_configured_policy(
    name,
    *,
    cost=0.3,
    penalty="linear",
    beta=1.0,
    allow_abstention=True,
    abstain_value=-1,
    hamming_boundary="minimum_loss",
):
    """Create any registered built-in policy from one uniform configuration."""

    canonical = canonical_policy_name(name)
    common = {
        "cost": cost,
        "penalty": penalty,
        "abstain_value": abstain_value,
    }
    if canonical == "hamming":
        return create_policy(
            canonical,
            **common,
            linear_boundary=hamming_boundary,
        )
    if canonical == "fbeta":
        return create_policy(
            canonical,
            **common,
            beta=beta,
            allow_abstention=allow_abstention,
        )
    if canonical == "jaccard":
        return create_policy(
            canonical,
            **common,
            allow_abstention=allow_abstention,
        )
    if canonical in ("macro_f1", "per_label_macro_f1"):
        return create_policy(
            canonical,
            **common,
            allow_abstention=allow_abstention,
        )
    raise ValueError(f"No configuration adapter exists for policy '{canonical}'.")
