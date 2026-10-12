"""Validate v6.3 artifacts against the frozen specification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.data.loader import load_dataset
from src.evaluation.result_store_v6 import TABLE_SCHEMAS, V63ResultStore, sha256_file


FIGURE_NAMES = [
    "complete_hamming_accuracy.png",
    "complete_subset_accuracy.png",
    "complete_micro_f1.png",
    "complete_macro_f1.png",
    "complete_instance_f1.png",
    "selective_hamming_accuracy.png",
    "selective_subset_accuracy.png",
    "selective_micro_f1.png",
    "selective_macro_f1.png",
    "selective_instance_f1.png",
    "selective_generalized_loss.png",
    "rejection_cost_generalized_loss.png",
    "rejection_cost_hamming_accuracy.png",
    "rejection_cost_micro_f1.png",
    "rejection_cost_macro_f1.png",
    "rejection_cost_instance_f1.png",
]

NON_ABSTAINING_BASELINES = {"BR", "CC"}


def validate(config_path: Path, *, require_figures: bool = True):
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    settings = config["settings"]
    root = WORKSPACE_ROOT / settings["output_dir"]
    store = V63ResultStore(
        root,
        settings["base_learners"],
        settings["inherited_models"] + settings["new_models"],
    )
    errors = []
    checks = {
        "computed_checkpoints": 0,
        "inherited_checkpoints": 0,
        "computed_folds": 0,
        "inherited_folds": 0,
        "prediction_files": 0,
        "tables": 0,
        "figures": 0,
        "complete_reference_fold_rows": 0,
        "complete_reference_mismatches": 0,
    }
    sample_counts = {}
    for dataset in settings["datasets"]:
        X, _, _, _ = load_dataset(dataset, base_dir=str(WORKSPACE_ROOT))
        sample_counts[dataset] = int(X.shape[0])

    for learner in settings["base_learners"]:
        for dataset in settings["datasets"]:
            for model in settings["inherited_models"]:
                checkpoint = store.load_checkpoint(learner, model, dataset)
                if checkpoint is None:
                    errors.append(f"missing inherited checkpoint: {learner}/{model}/{dataset}")
                    continue
                checks["inherited_checkpoints"] += 1
                folds = checkpoint.get("folds", [])
                checks["inherited_folds"] += len(folds)
                if len(folds) != settings["n_splits"]:
                    errors.append(f"inherited fold count: {learner}/{model}/{dataset}={len(folds)}")
                if checkpoint.get("provenance_status") not in (
                    "inherited_matched",
                    "inherited_unmatched",
                ):
                    errors.append(f"invalid provenance: {learner}/{model}/{dataset}")

            fold_orders = {fold: [] for fold in range(1, settings["n_splits"] + 1)}
            for model in settings["new_models"]:
                checkpoint = store.load_checkpoint(learner, model, dataset)
                if checkpoint is None:
                    errors.append(f"missing computed checkpoint: {learner}/{model}/{dataset}")
                    continue
                checks["computed_checkpoints"] += 1
                folds = checkpoint.get("folds", [])
                checks["computed_folds"] += len(folds)
                if checkpoint.get("status") != "complete" or len(folds) != settings["n_splits"]:
                    errors.append(f"incomplete checkpoint: {learner}/{model}/{dataset}")
                for fold in folds:
                    if fold.get("status") != "complete":
                        errors.append(f"incomplete fold: {learner}/{model}/{dataset}/{fold.get('fold')}")
                    order = tuple(fold.get("dl_labels", []))
                    fold_orders[int(fold["fold"])].append((model, order))
                    if "PCC" in model:
                        sampling = fold.get("pcc_sampling") or {}
                        if sampling.get("num_samples_per_test_x") != 1000:
                            errors.append(f"PCC sample count: {learner}/{model}/{dataset}/{fold['fold']}")
                prediction = store.prediction_path(learner, model, dataset)
                if not prediction.exists():
                    errors.append(f"missing predictions: {learner}/{model}/{dataset}")
                    continue
                checks["prediction_files"] += 1
                expected_hash = checkpoint.get("prediction", {}).get("sha256")
                if expected_hash != sha256_file(prediction):
                    errors.append(f"prediction hash mismatch: {learner}/{model}/{dataset}")
                with np.load(prediction, allow_pickle=False) as archive:
                    required = {"test_indices", "y_true", "probabilities", "y_full", "fold_ids"}
                    missing = required - set(archive.files)
                    if missing:
                        errors.append(f"prediction keys {missing}: {learner}/{model}/{dataset}")
                        continue
                    indices = archive["test_indices"]
                    if len(indices) != sample_counts[dataset]:
                        errors.append(f"prediction row count: {learner}/{model}/{dataset}")
                    if not np.array_equal(np.sort(indices), np.arange(sample_counts[dataset])):
                        errors.append(f"test index coverage: {learner}/{model}/{dataset}")
                    if set(np.unique(archive["fold_ids"])) != set(
                        range(1, settings["n_splits"] + 1)
                    ):
                        errors.append(f"fold id coverage: {learner}/{model}/{dataset}")
                    if np.any((archive["probabilities"] < 0) | (archive["probabilities"] > 1)):
                        errors.append(f"probability range: {learner}/{model}/{dataset}")

            for fold, values in fold_orders.items():
                if values:
                    unique_orders = {order for _, order in values}
                    if len(unique_orders) != 1:
                        errors.append(
                            f"shared DL order mismatch: {learner}/{dataset}/fold={fold}: {values}"
                        )

        table_root = store.learner_root(learner) / "hamming_bop" / "tables"
        for filename, schema in TABLE_SCHEMAS.items():
            path = table_root / filename
            if not path.exists():
                errors.append(f"missing table: {path}")
                continue
            checks["tables"] += 1
            columns = pd.read_csv(path, nrows=0).columns.tolist()
            if columns != schema:
                errors.append(f"table schema mismatch: {path}")

        complete_frame = pd.read_csv(
            table_root / "complete_metric.csv", na_values=["-"]
        )
        selective_frame = pd.read_csv(
            table_root / "selective_metric.csv", na_values=["-"]
        )
        reference = selective_frame[
            (selective_frame["aggregation"] == "fold")
            & selective_frame["model"].isin(NON_ABSTAINING_BASELINES)
        ].copy()
        reference["cost"] = pd.to_numeric(reference["cost"], errors="coerce")
        checks["complete_reference_fold_rows"] += len(reference)
        expected_reference_rows = (
            len(NON_ABSTAINING_BASELINES)
            * len(settings["datasets"])
            * settings["n_splits"]
            * len(settings["abstention_costs"])
        )
        if len(reference) != expected_reference_rows:
            errors.append(
                f"BR/CC complete-reference row count: {learner}="
                f"{len(reference)} expected={expected_reference_rows}"
            )

        expected_costs = np.asarray(settings["abstention_costs"], dtype=float)
        for identity, group in reference.groupby(["model", "dataset", "fold"]):
            observed = np.sort(group["cost"].dropna().unique())
            if observed.shape != expected_costs.shape or not np.allclose(
                observed, expected_costs
            ):
                errors.append(f"BR/CC complete-reference cost grid: {learner}/{identity}")

        complete_reference = complete_frame[
            (complete_frame["aggregation"] == "fold")
            & complete_frame["model"].isin(NON_ABSTAINING_BASELINES)
        ][
            [
                "model",
                "dataset",
                "fold",
                "hamming_accuracy",
                "subset_accuracy",
                "micro_f1",
                "macro_f1",
                "instance_f1",
            ]
        ]
        merged_reference = reference.merge(
            complete_reference,
            on=["model", "dataset", "fold"],
            how="left",
            suffixes=("", "_complete"),
            validate="many_to_one",
        )
        mismatch = np.zeros(len(merged_reference), dtype=bool)
        for selective_name, complete_name in (
            ("selective_hamming_accuracy", "hamming_accuracy"),
            ("selective_subset_accuracy", "subset_accuracy"),
            ("selective_micro_f1", "micro_f1"),
            ("selective_macro_f1", "macro_f1"),
            ("selective_instance_f1", "instance_f1"),
        ):
            left = pd.to_numeric(merged_reference[selective_name], errors="coerce")
            right = pd.to_numeric(merged_reference[complete_name], errors="coerce")
            mismatch |= ~np.isclose(left, right, equal_nan=True)
        hamming_loss = 1.0 - pd.to_numeric(
            merged_reference["hamming_accuracy"], errors="coerce"
        )
        mismatch |= ~np.isclose(
            pd.to_numeric(merged_reference["generalized_loss"], errors="coerce"),
            hamming_loss,
            equal_nan=True,
        )
        mismatch |= ~np.isclose(
            pd.to_numeric(merged_reference["coverage"], errors="coerce"), 1.0
        )
        mismatch |= ~np.isclose(
            pd.to_numeric(merged_reference["abs"], errors="coerce"), 0.0
        )
        mismatch |= ~np.isclose(
            pd.to_numeric(merged_reference["aabs"], errors="coerce"), 0.0
        )
        mismatch |= (
            merged_reference["decision_policy"].astype(str)
            != "none_complete_reference"
        ).to_numpy()
        mismatch_count = int(mismatch.sum())
        checks["complete_reference_mismatches"] += mismatch_count
        if mismatch_count:
            errors.append(
                f"BR/CC complete-reference value mismatch: {learner}={mismatch_count}"
            )

    for scope in [store.LEARNER_DIR[name] for name in settings["base_learners"]] + ["all"]:
        f_measure = root / scope / "f_measure_bop"
        files = [path for path in f_measure.rglob("*") if path.is_file()]
        if files:
            errors.append(f"f_measure_bop must be empty: {scope}: {files}")
        table_root = root / scope / "hamming_bop" / "tables"
        if scope == "all":
            for filename, schema in TABLE_SCHEMAS.items():
                path = table_root / filename
                if not path.exists():
                    errors.append(f"missing all table: {path}")
                    continue
                checks["tables"] += 1
                if pd.read_csv(path, nrows=0).columns.tolist() != schema:
                    errors.append(f"all table schema mismatch: {path}")
        if require_figures:
            for filename in FIGURE_NAMES:
                path = root / scope / "hamming_bop" / "figures" / filename
                if path.exists() and path.stat().st_size > 0:
                    checks["figures"] += 1
                else:
                    errors.append(f"missing figure: {path}")

    report = {
        "version": "v6_3",
        "status": "pass" if not errors else "fail",
        "checks": checks,
        "expected": {
            "computed_checkpoints": len(settings["base_learners"])
            * len(settings["datasets"])
            * len(settings["new_models"]),
            "inherited_checkpoints": len(settings["base_learners"])
            * len(settings["datasets"])
            * len(settings["inherited_models"]),
            "computed_folds": len(settings["base_learners"])
            * len(settings["datasets"])
            * len(settings["new_models"])
            * settings["n_splits"],
            "inherited_folds": len(settings["base_learners"])
            * len(settings["datasets"])
            * len(settings["inherited_models"])
            * settings["n_splits"],
            "prediction_files": len(settings["base_learners"])
            * len(settings["datasets"])
            * len(settings["new_models"]),
            "tables": 20,
            "figures": 64 if require_figures else 0,
            "complete_reference_fold_rows": len(settings["base_learners"])
            * len(NON_ABSTAINING_BASELINES)
            * len(settings["datasets"])
            * settings["n_splits"]
            * len(settings["abstention_costs"]),
            "complete_reference_mismatches": 0,
        },
        "errors": errors,
    }
    report_path = root / "validation_report.json"
    with report_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return not errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=WORKSPACE_ROOT / "configs" / "v6_3_run.json",
    )
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    if not validate(args.config, require_figures=not args.skip_figures):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
