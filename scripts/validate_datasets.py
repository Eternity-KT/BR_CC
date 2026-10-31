"""Validate benchmark dataset files, checksums, dimensions and labels."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_v6_3_3_datasets import main


if __name__ == "__main__":
    raise SystemExit(main())
