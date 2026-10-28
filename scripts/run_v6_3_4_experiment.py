"""Run the focused v6.3.3/no-SAMG/SAMG/SAMG2 comparison."""

import argparse

from run_v6_3_3_experiment import (
    DEFAULT_DATASETS,
    run_v6_3_3_benchmark,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--tag", default="v6_3_4_all10ds")
    parser.add_argument("--base_learner", default="logistic")
    args = parser.parse_args()
    run_v6_3_3_benchmark(
        datasets=args.datasets,
        tag=args.tag,
        base_learner=args.base_learner,
        model_set="samg_audit",
    )

