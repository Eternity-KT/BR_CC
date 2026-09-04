"""Verify and resume the preregistered Q12 ablation job matrix."""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from main import run_experiment  # noqa: E402
from scripts.run_frozen_experiment import file_sha256  # noqa: E402


def verify_ablation_config(config_path, checksum_path=None):
    """Load a frozen ablation manifest after checking its SHA-256."""

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
            f"Frozen ablation checksum mismatch: expected {expected}, got {actual}."
        )
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema_version") != 1 or payload.get("status") != "frozen":
        raise ValueError("Ablation config must use schema v1 and status=frozen.")
    if payload.get("common_settings", {}).get("result_schema") != 3:
        raise ValueError("Frozen ablations must use result_schema=3.")
    seeds = payload.get("random_matched_ablation", {}).get(
        "partition_seeds", []
    )
    if len(seeds) < 30 or len(seeds) != len(set(seeds)):
        raise ValueError("Random-matched ablation requires >=30 unique seeds.")
    return payload, actual


def iter_ablation_jobs(payload):
    """Yield deterministic job IDs and public-pipeline settings."""

    common = payload["common_settings"]
    output_root = Path(payload["output_root"])

    objective = payload["objective_ablation"]
    for name in objective["objectives"]:
        settings = deepcopy(common)
        settings.update({
            "models": list(objective["models"]),
            "gsi_selection_objective": name,
            "output_dir": str(output_root / "objective" / name),
        })
        yield f"objective/{name}", settings

    partition = payload["partition_ablation"]
    for mode in partition["modes"]:
        settings = deepcopy(common)
        settings.update({
            "models": list(partition["models"]),
            "gsi_partition_mode": mode,
            "output_dir": str(output_root / "partition" / mode),
        })
        yield f"partition/{mode}", settings

    random_matched = payload["random_matched_ablation"]
    for seed in random_matched["partition_seeds"]:
        settings = deepcopy(common)
        settings.update({
            "models": list(random_matched["models"]),
            "gsi_partition_mode": "random_matched",
            "gsi_partition_random_state": int(seed),
            "output_dir": str(output_root / "random_matched" / f"seed_{seed}"),
        })
        yield f"random_matched/seed_{seed}", settings


def checkpointed_fold_count(output_dir):
    """Count atomically persisted folds in one isolated ablation job."""

    total = 0
    root = Path(output_dir) / "checkpoints"
    for path in root.glob("*/*.json"):
        with path.open("r", encoding="utf-8") as stream:
            checkpoint = json.load(stream)
        if checkpoint.get("schema_version") == 3:
            total += len(checkpoint.get("folds", {}))
    return total


def run_jobs(payload, max_new_folds=None):
    """Resume jobs in preregistered order under one operational fold quota."""

    remaining = max_new_folds
    for job_id, settings in iter_ablation_jobs(payload):
        before = checkpointed_fold_count(settings["output_dir"])
        print(f"Ablation job: {job_id} (checkpointed={before})", flush=True)
        call_settings = dict(settings)
        call_settings["max_new_folds"] = remaining
        run_experiment(**call_settings)
        after = checkpointed_fold_count(settings["output_dir"])
        added = after - before
        if added < 0:
            raise RuntimeError(f"Checkpoint count regressed for {job_id}.")
        if remaining is not None:
            remaining -= added
            if remaining <= 0:
                return


def main():
    parser = argparse.ArgumentParser(
        description="Run the checksum-verified Q12 ablation matrix."
    )
    parser.add_argument(
        "--config", default="configs/ablation_run.json", help="Frozen config JSON."
    )
    parser.add_argument(
        "--checksum", default=None, help="Optional companion SHA-256 file."
    )
    parser.add_argument(
        "--max-new-folds",
        type=int,
        default=None,
        help="Operational quota across the whole job queue.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Verify and print expanded jobs."
    )
    args = parser.parse_args()
    if args.max_new_folds is not None and args.max_new_folds < 1:
        parser.error("--max-new-folds must be positive.")
    payload, digest = verify_ablation_config(args.config, args.checksum)
    jobs = list(iter_ablation_jobs(payload))
    print(f"Verified frozen ablation SHA-256: {digest}", flush=True)
    print(f"Expanded ablation jobs: {len(jobs)}", flush=True)
    if args.dry_run:
        for job_id, settings in jobs:
            print(
                f"{job_id}: models={len(settings['models'])}, "
                f"datasets={len(settings['datasets'])}, "
                f"folds={settings['n_splits']}",
                flush=True,
            )
        return
    run_jobs(payload, max_new_folds=args.max_new_folds)


if __name__ == "__main__":
    main()
