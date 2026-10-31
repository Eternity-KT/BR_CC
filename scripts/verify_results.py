"""Verify saved experiment predictions, metrics and artifact checksums."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_final import RELEASE, verify_artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path, default=RELEASE, help="Run directory; defaults to the saved benchmark reference.")
    parser.add_argument("--timeout", type=int, default=600, help="Verification deadline in seconds.")
    args = parser.parse_args(argv)
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    verify_artifacts(args.run_dir.resolve(), args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
