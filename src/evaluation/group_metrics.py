"""Schema-v3 per-label and IL/DL group metrics (Phase Q0 API skeleton)."""


def compute_per_label_metrics(y_true, y_full, y_partial=None, *, abstain_value=-1):
    """Return per-label support, quality and optional coverage diagnostics."""

    raise NotImplementedError("compute_per_label_metrics is scheduled for Phase Q1")


def compute_group_metrics(
    y_true,
    y_full,
    y_partial,
    label_groups,
    *,
    abstain_value=-1,
):
    """Aggregate per-label metrics over named groups such as IL and DL."""

    raise NotImplementedError("compute_group_metrics is scheduled for Phase Q1")
