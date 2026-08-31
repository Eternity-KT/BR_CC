"""Opt-in schema-v3 CV runner and fold-summary pipeline."""

import hashlib
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MaxAbsScaler

from ..decision import create_configured_policy
from .cache_v3 import (
    CACHE_SCHEMA_VERSION_V3,
    completed_fold_indices,
    compute_config_hash,
    load_or_create_fold_checkpoint,
    mark_checkpoint_complete,
    save_completed_fold,
)
from .export_v3 import export_v3_artifacts
from .metric_contract import METRIC_CONTRACT_VERSION


class V3OutputIsolationError(ValueError):
    """Raised when schema-v3 output targets a legacy result directory."""


def _evaluation_policy_config(
    model_name,
    report_cost,
    abstention_penalty,
    gsi_decision_policy="hamming",
    gsi_beta=1.0,
    gsi_penalty="linear",
):
    """Describe the exact decision layer that affects one model's outputs."""

    configuration = {
        "full_threshold": 0.5,
        "acceptance_confidence": "2*abs(probability-0.5)",
        "calibration": "none",
        "partial_policy": None,
    }
    if model_name in ("MLC_PA", "GSI_MLC_PA"):
        policy_name = (
            "hamming" if model_name == "MLC_PA" else gsi_decision_policy
        )
        penalty = (
            abstention_penalty if model_name == "MLC_PA" else gsi_penalty
        )
        configuration["partial_policy"] = create_configured_policy(
            policy_name,
            cost=report_cost,
            penalty=penalty,
            beta=gsi_beta,
            allow_abstention=True,
            hamming_boundary=(
                "symmetric_thresholds"
                if model_name == "GSI_MLC_PA"
                else "minimum_loss"
            ),
        ).get_config()
    return configuration


def ensure_v3_output_directory(output_dir):
    """Reject known v2 directories and directories containing legacy tables."""

    output_path = Path(output_dir)
    resolved = output_path.resolve()
    forbidden = {Path("results").resolve(), Path("results_pa").resolve()}
    if resolved in forbidden:
        raise V3OutputIsolationError(
            "Schema v3 must use an isolated directory such as results_pa_v3."
        )
    if (output_path / "tables" / "raw_results.json").exists():
        raise V3OutputIsolationError(
            "Target contains legacy raw_results.json; choose a new v3 directory."
        )
    return output_path


def _numeric_summary(rows):
    raw_rows = [dict(row) for row in rows]
    metric_names = []
    for row in raw_rows:
        for name, value in row.items():
            if name == "Fold" or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and name not in metric_names:
                metric_names.append(name)
    means = {}
    standard_deviations = {}
    for name in metric_names:
        values = [
            float(row[name])
            for row in raw_rows
            if isinstance(row.get(name), (int, float))
            and not isinstance(row.get(name), bool)
            and math.isfinite(float(row[name]))
        ]
        means[name] = float(np.mean(values)) if values else float("nan")
        standard_deviations[name] = (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ) if values else float("nan")
    return {"Mean": means, "Std": standard_deviations, "Raw Folds": raw_rows}


def _per_label_records(ordered_folds, selector):
    records = []
    for fold_index, metrics in ordered_folds:
        for record in selector(metrics):
            records.append({"Fold": fold_index, **record})
    return {"Raw Records": records}


def _group_records(ordered_folds, selector):
    records = []
    for fold_index, metrics in ordered_folds:
        for group_name, values in selector(metrics).items():
            records.append({"Fold": fold_index, "Group": group_name, **values})
    return {"Raw Records": records}


def summarize_v3_checkpoint(checkpoint):
    """Aggregate scalar scopes while retaining fold-level label/group records."""

    ordered_folds = [
        (int(key), entry["metrics"])
        for key, entry in sorted(
            checkpoint.get("folds", {}).items(), key=lambda item: int(item[0])
        )
    ]
    full_rows = [
        {"Fold": fold_index, **metrics["Full"]}
        for fold_index, metrics in ordered_folds
    ]
    summary = {
        "Status": checkpoint.get("status", "in_progress"),
        "Config Hash": checkpoint["config_hash"],
        "Expected Fold Count": int(checkpoint["config"]["n_splits"]),
        "Completed Folds": [fold for fold, _ in ordered_folds],
        "Full": _numeric_summary(full_rows),
        "Per Label": _per_label_records(
            ordered_folds, lambda metrics: metrics.get("Per Label", [])
        ),
        "Groups": _group_records(
            ordered_folds, lambda metrics: metrics.get("Groups", {})
        ),
        "Critical Labels": [
            {"Fold": fold_index, **metrics.get("Critical Labels", {})}
            for fold_index, metrics in ordered_folds
        ],
        "Costs": {},
        "Fold Metadata": [
            {"Fold": int(key), **entry.get("metadata", {})}
            for key, entry in sorted(
                checkpoint.get("folds", {}).items(), key=lambda item: int(item[0])
            )
        ],
    }
    cost_keys = sorted(
        {
            cost
            for _, metrics in ordered_folds
            for cost in metrics.get("Costs", {})
        },
        key=float,
    )
    for cost in cost_keys:
        entries = [
            (fold_index, metrics["Costs"][cost])
            for fold_index, metrics in ordered_folds
            if cost in metrics.get("Costs", {})
        ]
        cost_summary = {}
        for scope in ("Selective", "Rejected", "Optimistic", "Diagnostics"):
            cost_summary[scope] = _numeric_summary(
                [
                    {"Fold": fold_index, **entry.get(scope, {})}
                    for fold_index, entry in entries
                ]
            )
        cost_summary["Per Label"] = _per_label_records(
            entries, lambda entry: entry.get("Per Label", [])
        )
        cost_summary["Groups"] = _group_records(
            entries, lambda entry: entry.get("Groups", {})
        )
        cost_summary["Critical Labels"] = [
            {"Fold": fold_index, **entry.get("Critical Labels", {})}
            for fold_index, entry in entries
        ]
        summary["Costs"][cost] = cost_summary
    return summary


def _pair_config(
    dataset_name,
    model_name,
    x_data,
    y_data,
    label_names,
    n_splits,
    random_state,
    abstention_costs,
    report_cost,
    abstention_penalty,
    mlc_pa_base,
    gsi_validation_size,
    gsi_selection_objective,
    gsi_decision_policy,
    gsi_beta,
    gsi_penalty,
    critical_labels,
    model_signature,
    dataset_fingerprint,
    split_hash,
):
    return {
        "schema_version": CACHE_SCHEMA_VERSION_V3,
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "dataset": dataset_name,
        "model": model_name,
        "dataset_shape": {
            "samples": int(x_data.shape[0]),
            "features": int(x_data.shape[1]),
            "labels": int(y_data.shape[1]),
        },
        "dataset_fingerprint": dataset_fingerprint,
        "split_hash": split_hash,
        "label_names": list(label_names),
        "n_splits": int(n_splits),
        "random_state": int(random_state),
        "scaler": "MaxAbsScaler",
        "abstention_costs": [float(cost) for cost in abstention_costs],
        "report_cost": float(report_cost),
        "abstention_penalty": abstention_penalty,
        "mlc_pa_base": mlc_pa_base,
        "gsi_validation_size": float(gsi_validation_size),
        "gsi_selection_objective": gsi_selection_objective,
        "gsi_decision_policy": gsi_decision_policy,
        "gsi_beta": float(gsi_beta),
        "gsi_penalty": gsi_penalty,
        "critical_labels": critical_labels,
        "model_signature": model_signature,
        "evaluation_policy": _evaluation_policy_config(
            model_name,
            report_cost,
            abstention_penalty,
            gsi_decision_policy,
            gsi_beta,
            gsi_penalty,
        ),
    }


def _model_signature(classifier):
    signature = {
        "class": (
            f"{classifier.__class__.__module__}."
            f"{classifier.__class__.__qualname__}"
        ),
        "backend": getattr(classifier, "backend_", None),
    }
    if hasattr(classifier, "get_params"):
        try:
            signature["parameters"] = classifier.get_params(deep=True)
        except (TypeError, ValueError):
            signature["parameters"] = classifier.get_params(deep=False)
    else:
        signature["parameters"] = {}
    return signature


def _array_fingerprint(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()[:16]


def _split_hash(splits):
    return compute_config_hash(
        {
            "folds": [
                {
                    "train": np.asarray(train_indices, dtype=np.int64),
                    "test": np.asarray(test_indices, dtype=np.int64),
                }
                for train_indices, test_indices in splits
            ]
        }
    )


def run_experiment_v3(
    *,
    datasets,
    models,
    n_splits,
    random_state,
    output_dir,
    abstention_costs,
    report_cost,
    abstention_penalty,
    mlc_pa_base,
    gsi_validation_size,
    dataset_loader,
    cv_factory,
    model_factory,
    evaluator,
    critical_labels=None,
    max_new_folds=None,
    gsi_selection_objective="full_macro_f1",
    gsi_decision_policy="hamming",
    gsi_beta=1.0,
    gsi_penalty="linear",
):
    """Run/resume schema-v3 folds and export strict JSON plus scope CSVs."""

    output_path = ensure_v3_output_directory(output_dir)
    if max_new_folds is not None:
        if (
            isinstance(max_new_folds, (bool, np.bool_))
            or not isinstance(max_new_folds, (int, np.integer))
            or int(max_new_folds) < 1
        ):
            raise ValueError("max_new_folds must be a positive integer or None.")
        max_new_folds = int(max_new_folds)
    checkpoints_dir = output_path / "checkpoints"
    results = {}
    new_fold_count = 0
    stopped_early = False

    model_signatures = {}
    for model_name in models:
        prototype = model_factory(
            model_name,
            random_state=random_state,
            abstention_cost=report_cost,
            abstention_penalty=abstention_penalty,
            mlc_pa_base=mlc_pa_base,
            gsi_validation_size=gsi_validation_size,
            gsi_selection_objective=gsi_selection_objective,
            gsi_decision_policy=gsi_decision_policy,
            gsi_beta=gsi_beta,
            gsi_penalty=gsi_penalty,
        )
        model_signatures[model_name] = _model_signature(prototype)

    for dataset_name in datasets:
        x_data, y_data, _, label_names = dataset_loader(dataset_name)
        label_names = list(label_names)
        splits = list(
            cv_factory(n_splits=n_splits, random_state=random_state).split(
                x_data, y_data
            )
        )
        dataset_fingerprint = _array_fingerprint(x_data, y_data)
        split_hash = _split_hash(splits)
        results.setdefault(dataset_name, {})
        for model_name in models:
            config = _pair_config(
                dataset_name,
                model_name,
                x_data,
                y_data,
                label_names,
                n_splits,
                random_state,
                abstention_costs,
                report_cost,
                abstention_penalty,
                mlc_pa_base,
                gsi_validation_size,
                gsi_selection_objective,
                gsi_decision_policy,
                gsi_beta,
                gsi_penalty,
                critical_labels,
                model_signatures[model_name],
                dataset_fingerprint,
                split_hash,
            )
            checkpoint_path, checkpoint = load_or_create_fold_checkpoint(
                checkpoints_dir, model_name, dataset_name, config
            )
            completed = set(completed_fold_indices(checkpoint))
            for fold_index, (train_indices, test_indices) in enumerate(splits, 1):
                if fold_index in completed:
                    continue
                scaler = MaxAbsScaler()
                x_train = scaler.fit_transform(x_data[train_indices])
                x_test = scaler.transform(x_data[test_indices])
                y_train = y_data[train_indices]
                y_test = y_data[test_indices]
                classifier = model_factory(
                    model_name,
                    random_state=random_state,
                    abstention_cost=report_cost,
                    abstention_penalty=abstention_penalty,
                    mlc_pa_base=mlc_pa_base,
                    gsi_validation_size=gsi_validation_size,
                    gsi_selection_objective=gsi_selection_objective,
                    gsi_decision_policy=gsi_decision_policy,
                    gsi_beta=gsi_beta,
                    gsi_penalty=gsi_penalty,
                )
                started = time.time()
                classifier.fit(x_train, y_train)
                metrics = evaluator(
                    model_name,
                    classifier,
                    x_test,
                    y_test,
                    abstention_costs,
                    metric_schema=3,
                    label_names=label_names,
                    critical_labels=critical_labels,
                )
                metadata = {
                    "Train And Evaluate Seconds": float(time.time() - started),
                    "Train Size": int(len(train_indices)),
                    "Test Size": int(len(test_indices)),
                }
                save_completed_fold(
                    checkpoint_path, checkpoint, fold_index, metrics, metadata
                )
                completed.add(fold_index)
                new_fold_count += 1
                print(
                    f"  -> v3 fold {fold_index}/{len(splits)} "
                    f"{dataset_name}/{model_name} checkpointed",
                    flush=True,
                )
                if max_new_folds is not None and new_fold_count >= max_new_folds:
                    stopped_early = True
                    break
            if (
                completed == set(range(1, len(splits) + 1))
                and checkpoint.get("status") != "complete"
            ):
                mark_checkpoint_complete(checkpoint_path, checkpoint, len(splits))
            results[dataset_name][model_name] = summarize_v3_checkpoint(checkpoint)
            if stopped_early:
                break
        if stopped_early:
            break

    run_config = {
        "schema_version": CACHE_SCHEMA_VERSION_V3,
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "datasets": list(datasets),
        "models": list(models),
        "n_splits": int(n_splits),
        "random_state": int(random_state),
        "abstention_costs": [float(cost) for cost in abstention_costs],
        "report_cost": float(report_cost),
        "abstention_penalty": abstention_penalty,
        "mlc_pa_base": mlc_pa_base,
        "gsi_validation_size": float(gsi_validation_size),
        "gsi_selection_objective": gsi_selection_objective,
        "gsi_decision_policy": gsi_decision_policy,
        "gsi_beta": float(gsi_beta),
        "gsi_penalty": gsi_penalty,
        "critical_labels": critical_labels,
        "scaler": "MaxAbsScaler",
        "model_signatures": model_signatures,
        "evaluation_policy": {
            model_name: _evaluation_policy_config(
                model_name,
                report_cost,
                abstention_penalty,
                gsi_decision_policy,
                gsi_beta,
                gsi_penalty,
            )
            for model_name in models
        },
    }
    every_pair_complete = (
        set(results) == set(datasets)
        and all(set(results[dataset]) == set(models) for dataset in datasets)
        and all(
            summary.get("Status") == "complete"
            and summary.get("Completed Folds")
            == list(range(1, summary.get("Expected Fold Count", 0) + 1))
            for dataset_results in results.values()
            for summary in dataset_results.values()
        )
    )
    run_config_hash = compute_config_hash(run_config)
    document = {
        "Schema Version": CACHE_SCHEMA_VERSION_V3,
        "Metric Contract Version": METRIC_CONTRACT_VERSION,
        "Config Hash": run_config_hash,
        "Status": "complete" if every_pair_complete else "partial",
        "Generated At": datetime.now(timezone.utc).isoformat(),
        "Settings": run_config,
        "Results": results,
    }
    document["Artifacts"] = export_v3_artifacts(
        document, output_path / "tables" / run_config_hash
    )
    return document
