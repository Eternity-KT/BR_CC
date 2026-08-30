"""Schema-v3 abstention metrics (Phase Q0 API skeleton)."""


def compute_abstention_metrics(
    y_true,
    y_partial,
    y_full,
    *,
    cost,
    penalty="linear",
    abstain_value=-1,
):
    """Compute selective, rejected and optimistic metrics.

    The concrete implementation is scheduled for Phase Q1.  ``y_full`` is
    explicit because rejected-set diagnostics need the counterfactual complete
    prediction instead of treating abstentions as zero.
    """

    raise NotImplementedError("compute_abstention_metrics is scheduled for Phase Q1")
