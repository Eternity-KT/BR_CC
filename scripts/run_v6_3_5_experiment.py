"""Compare all-label rejection with rejection restricted to DL labels."""

import argparse

from run_v6_3_3_experiment import DEFAULT_DATASETS, run_v6_3_3_benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--tag", default="v6_3_5_dl_only_rejection_all10ds")
    parser.add_argument("--base_learner", default="logistic")
    args = parser.parse_args()
    run_v6_3_3_benchmark(
        datasets=args.datasets,
        tag=args.tag,
        base_learner=args.base_learner,
        model_set="dl_only_rejection",
    )
