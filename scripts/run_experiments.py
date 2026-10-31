"""Run the research benchmark with reproducible settings and isolated outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_v6_3_3_final import execute

BENCHMARK_CONFIG = ROOT / "configs/v6_3_3_final.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BENCHMARK_CONFIG, help="Benchmark protocol configuration.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/reproductions", help="Directory for new experiment runs.")
    parser.add_argument("--datasets", nargs="+", help="Dataset names; omit to run the complete collection.")
    parser.add_argument("--base-learners", nargs="+", help="Select logistic, svm_calibrated or mlp.")
    parser.add_argument("--folds", nargs="+", type=int, help="Outer fold numbers; omit to run all folds.")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow uncommitted source changes for exploratory runs.")
    args = parser.parse_args(argv)
    execute(args.config, args.output_root, args.datasets, args.base_learners, args.folds, args.allow_dirty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
