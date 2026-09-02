"""Verify a frozen experiment file and run it through the public pipeline."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from main import run_experiment  # noqa: E402


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_config(config_path, checksum_path=None):
    """Load a frozen config only after its companion SHA-256 matches."""

    path = Path(config_path)
    checksum = (
        Path(checksum_path)
        if checksum_path is not None
        else path.with_suffix(".sha256")
    )
    expected = checksum.read_text(encoding="utf-8").split()[0].lower()
    actual = file_sha256(path)
    if expected != actual:
        raise ValueError(
            f"Frozen config checksum mismatch: expected {expected}, got {actual}."
        )
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema_version") != 1 or payload.get("status") != "frozen":
        raise ValueError("Experiment config must use schema v1 and status=frozen.")
    settings = payload.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("Frozen experiment config must contain a settings object.")
    if settings.get("result_schema") != 3:
        raise ValueError("Frozen primary experiments must use result_schema=3.")
    return payload, actual


def main():
    parser = argparse.ArgumentParser(
        description="Run a checksum-verified schema-v3 experiment config."
    )
    parser.add_argument(
        "--config", default="configs/full_run.json", help="Frozen config JSON."
    )
    parser.add_argument(
        "--checksum", default=None, help="Optional companion SHA-256 file."
    )
    parser.add_argument(
        "--max-new-folds",
        type=int,
        default=None,
        help="Safe quota checkpoint; does not alter the frozen scientific config.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Verify and print settings only."
    )
    args = parser.parse_args()
    payload, digest = verify_frozen_config(args.config, args.checksum)
    settings = dict(payload["settings"])
    print(f"Verified frozen config SHA-256: {digest}", flush=True)
    if args.dry_run:
        print(json.dumps(settings, indent=2, ensure_ascii=False), flush=True)
        return
    settings["max_new_folds"] = args.max_new_folds
    run_experiment(**settings)


if __name__ == "__main__":
    main()
