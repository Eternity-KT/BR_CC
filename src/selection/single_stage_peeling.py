"""Locked single-stage IL/DL partition used by the v6.3 ablation."""

from typing import Any, Callable, Optional

from .cv_peeling import (
    CVPeelingConfig,
    CVPeelingResult,
    CVStratifiedPeelingSelector,
)


class SingleStagePeelingSelector:
    """Run exactly one leakage-free CV peeling stage.

    v6.3 deliberately disables the v6.2 singleton-promotion rule.  The
    residual DL order is therefore the correlation-ascending order returned
    by :class:`CVStratifiedPeelingSelector`, even when only one DL remains.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.75,
        n_folds: int = 5,
        cost: float = 0.30,
        metric: str = "selective_f1",
        aug_normalization: str = "matching",
        random_state: int = 42,
        base_estimator_factory: Optional[Callable[[], Any]] = None,
    ):
        self.config = CVPeelingConfig(
            threshold=threshold,
            n_folds=n_folds,
            max_depth=1,
            cost=cost,
            metric=metric,
            decaying_threshold=False,
            dl_order_direction="ascending",
            aug_normalization=aug_normalization,
            min_labels_residual=0,
            promote_singleton_dl=False,
            random_state=random_state,
        )
        self.base_estimator_factory = base_estimator_factory

    def fit_partition(self, X, Y) -> CVPeelingResult:
        selector = CVStratifiedPeelingSelector(
            config=self.config,
            base_estimator_factory=self.base_estimator_factory,
        )
        result = selector.fit_partition(X, Y)
        if result.num_stages_executed != 1:
            raise RuntimeError(
                "The v6.3 selector contract requires exactly one executed stage."
            )
        if len(result.independent_layers) > 1:
            raise RuntimeError("v6.3 produced more than one independent layer.")
        return result


__all__ = ["SingleStagePeelingSelector"]
