"""Schema-v3 complete-prediction metrics (Phase Q0 API skeleton).

Implementation is intentionally scheduled for Phase Q1.  Keeping the skeleton
separate ensures Q0 does not change the production results emitted by
``src.evaluation.metrics.compute_all_metrics``.
"""


def compute_complete_metrics(y_true, y_full):
    """Compute canonical complete metrics for two binary label matrices.

    Returns
    -------
    dict[str, float]
        The names declared in ``metric_contract.COMPLETE_METRIC_NAMES``.
    """

    raise NotImplementedError("compute_complete_metrics is scheduled for Phase Q1")
