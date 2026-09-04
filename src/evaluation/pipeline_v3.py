"""Opt-in schema-v3 CV runner and fold-summary pipeline."""

import hashlib
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

from ..decision import create_configured_policy
from ..models.registry import model_family
from ..visualization import generate_deployment_plots
from .deployment import (
    OPERATING_POINT_RULES,
    compute_deployment_metrics,
    operating_point_record,
    select_operating_point,
)
from .label_policy import (
    get_dataset_label_policy,
    label_policy_hash,
    load_label_policy_config,
)
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
    family = model_family(model_name)
    if family in ("MLC_PA", "GSI_MLC_PA"):
        policy_name = (
            "hamming" if family == "MLC_PA" else gsi_decision_policy
        )
        penalty = (
            abstention_penalty if family == "MLC_PA" else gsi_penalty
        )
        configuration["partial_policy"] = create_configured_policy(
            policy_name,
            cost=report_cost,
            penalty=penalty,
            beta=gsi_beta,
            allow_abstention=True,
            hamming_boundary=(
                "symmetric_thresholds"
                if family == "GSI_MLC_PA"
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


def _partition_audit(ordered_folds):
    """Summarize GSI partition size, order, timing and fold stability."""

    records = []
    independent_sets = []
    for fold_index, metrics in ordered_folds:
        metadata = metrics.get("Model Metadata", {})
        if "Partition Mode" not in metadata:
            continue
        independent = tuple(
            sorted(int(label) for label in metadata.get("Independent Labels", []))
        )
        independent_sets.append(set(independent))
        records.append({
            "Fold": int(fold_index),
            "Partition Mode": metadata.get("Partition Mode"),
            "Independent Labels": list(independent),
            "Dependent Labels": [
                int(label) for label in metadata.get("Dependent Labels", [])
            ],
            "Independent Label Count": int(
                metadata.get("Independent Label Count", len(independent))
            ),
            "Dependent Label Count": int(
                metadata.get(
                    "Dependent Label Count",
                    len(metadata.get("Dependent Labels", [])),
                )
            ),
            "Final Order Strategy": metadata.get("Final Order Strategy"),
            "Final Order": [
                int(label) for label in metadata.get("Final Order", [])
            ],
            "Selection Seconds": float(metadata.get("Selection Seconds", 0.0)),
            "Probability Inference Seconds": float(
                metadata.get("Probability Inference Seconds", 0.0)
            ),
        })

    pairwise_scores = []
    for left_index, left in enumerate(independent_sets):
        for right in independent_sets[left_index + 1:]:
            union = left | right
            pairwise_scores.append(
                1.0 if not union else float(len(left & right) / len(union))
            )
    if pairwise_scores:
        stability = float(np.mean(pairwise_scores))
    elif independent_sets:
        stability = 1.0
    else:
        stability = float("nan")
    return {
        "Pairwise IL Jaccard Stability": stability,
        "Summary": _numeric_summary(records),
        "Raw Records": records,
    }


def _calibration_summary(ordered_folds):
    metric_rows = []
    per_label_records = []
    reliability_records = []
    unavailable = []
    for fold_index, metrics in ordered_folds:
        calibration = metrics.get("Calibration", {})
        if "Metrics" not in calibration:
            unavailable.append({
                "Fold": int(fold_index),
                "Status": calibration.get("Status", "Unavailable"),
                "Reason": calibration.get("Reason"),
            })
            continue
        metric_rows.append({
            "Fold": int(fold_index),
            **calibration["Metrics"],
        })
        for row in calibration.get("Per Label", []):
            per_label_records.append({"Fold": int(fold_index), **row})
        for row in calibration.get("Reliability", []):
            reliability_records.append({"Fold": int(fold_index), **row})
    return {
        "Metrics": _numeric_summary(metric_rows),
        "Per Label": {"Raw Records": per_label_records},
        "Reliability": {"Raw Records": reliability_records},
        "Unavailable": unavailable,
    }


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
        "Partition Audit": _partition_audit(ordered_folds),
        "Calibration": _calibration_summary(ordered_folds),
        "Critical Labels": [
            {"Fold": fold_index, **metrics.get("Critical Labels", {})}
            for fold_index, metrics in ordered_folds
        ],
        "Model Metadata": [
            {"Fold": fold_index, **metrics.get("Model Metadata", {})}
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
        cost_summary["Deployment"] = _numeric_summary(
            [
                {
                    "Fold": fold_index,
                    **entry.get("Deployment", {}).get("Metrics", {}),
                }
                for fold_index, entry in entries
            ]
        )
        cost_summary["Deployment"]["Reviewer Scenarios"] = [
            {"Fold": fold_index, **row}
            for fold_index, entry in entries
            for row in entry.get("Deployment", {})
            .get("Reviewer Scenarios", {})
            .get("Records", [])
        ]
        cost_summary["Deployment"]["Critical Labels"] = [
            {
                "Fold": fold_index,
                **entry.get("Deployment", {}).get("Critical Labels", {}),
            }
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
    gsi_partition_mode,
    gsi_partition_random_state,
    gsi_fixed_independent_labels,
    gsi_final_order,
    critical_labels,
    model_signature,
    dataset_fingerprint,
    split_hash,
    label_policy_digest,
    operating_point_rule,
    operating_coverage_gamma,
    operating_risk_epsilon,
    operating_validation_size,
):
    config = {
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
        "gsi_partition_mode": gsi_partition_mode,
        "gsi_fixed_independent_labels": (
            None
            if gsi_fixed_independent_labels is None
            else [int(label) for label in gsi_fixed_independent_labels]
        ),
        "gsi_final_order": gsi_final_order,
        "critical_labels": critical_labels,
        "model_signature": model_signature,
        "label_policy_hash": label_policy_digest,
        "operating_point": {
            "rule": operating_point_rule,
            "coverage_gamma": float(operating_coverage_gamma),
            "risk_epsilon": float(operating_risk_epsilon),
            "validation_size": float(operating_validation_size),
            "selection_scope": "inner_validation",
        },
        "evaluation_policy": _evaluation_policy_config(
            model_name,
            report_cost,
            abstention_penalty,
            gsi_decision_policy,
            gsi_beta,
            gsi_penalty,
        ),
    }
    if gsi_partition_random_state is not None:
        config["gsi_partition_random_state"] = int(gsi_partition_random_state)
    return config


def _model_signature(classifier):
    experiment_manifest = getattr(classifier, "experiment_manifest_", None)
    signature = {
        "class": (
            f"{classifier.__class__.__module__}."
            f"{classifier.__class__.__qualname__}"
        ),
        "backend": getattr(classifier, "backend_", None),
    }
    if experiment_manifest is not None:
        signature["experiment_manifest"] = experiment_manifest
        signature["parameters"] = experiment_manifest.get(
            "model_parameters", {}
        )
    elif hasattr(classifier, "get_params"):
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


def _select_inner_operating_point(
    *,
    model_name,
    x_train,
    y_train,
    abstention_costs,
    rule,
    coverage_gamma,
    risk_epsilon,
    validation_size,
    random_state,
    model_factory,
    model_factory_kwargs,
    label_names,
    label_policy,
):
    """Fit a selector only on inner-train and score costs on inner-validation."""

    if rule is None:
        return None
    if not 0.0 < float(validation_size) < 1.0:
        raise ValueError("operating_validation_size must lie strictly in (0, 1).")
    indices = np.arange(len(x_train))
    inner_train, inner_validation = train_test_split(
        indices,
        test_size=float(validation_size),
        random_state=int(random_state),
        shuffle=True,
    )
    inner_scaler = MaxAbsScaler()
    inner_x_train = inner_scaler.fit_transform(x_train[inner_train])
    inner_x_validation = inner_scaler.transform(x_train[inner_validation])
    selector = model_factory(
        model_name,
        random_state=random_state,
        **model_factory_kwargs,
    )
    selector.fit(inner_x_train, y_train[inner_train])
    probabilities = np.asarray(
        selector.predict_proba(inner_x_validation), dtype=np.float64
    )
    full_prediction = selector.predict_full_from_proba(probabilities)
    family = model_family(model_name)
    policy = create_configured_policy(
        getattr(selector, "decision_policy", "hamming"),
        cost=getattr(selector, "cost", 0.3),
        penalty=getattr(selector, "penalty", "linear"),
        beta=getattr(selector, "beta", 1.0),
        allow_abstention=True,
        abstain_value=selector.abstain_value,
        hamming_boundary=(
            "symmetric_thresholds"
            if family == "GSI_MLC_PA"
            else "minimum_loss"
        ),
    )
    records = []
    for cost in abstention_costs:
        partial = policy.predict_from_proba(probabilities, cost=cost)
        deployment = compute_deployment_metrics(
            y_train[inner_validation],
            full_prediction,
            partial,
            cost=cost,
            penalty=getattr(selector, "penalty", "linear"),
            abstain_value=selector.abstain_value,
            label_names=label_names,
            label_policy=label_policy,
            acceptance_confidence=np.abs(probabilities - 0.5) * 2.0,
        )
        records.append(
            operating_point_record(
                cost, deployment, data_scope="inner_validation"
            )
        )
    selection = select_operating_point(
        records,
        rule,
        data_scope="inner_validation",
        coverage_gamma=coverage_gamma,
        risk_epsilon=risk_epsilon,
    )
    selection.update({
        "Inner Train Size": int(len(inner_train)),
        "Inner Validation Size": int(len(inner_validation)),
        "Inner Split Seed": int(random_state),
        "Candidate Records": records,
        "Outer Test Access": False,
    })
    return selection


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
    gsi_partition_mode="learned",
    gsi_partition_random_state=None,
    gsi_fixed_independent_labels=None,
    gsi_final_order="correlation",
    label_policy_path=None,
    operating_point_rule=None,
    operating_coverage_gamma=0.8,
    operating_risk_epsilon=0.1,
    operating_validation_size=0.2,
):
    """Run/resume schema-v3 folds and export strict JSON plus scope CSVs."""

    output_path = ensure_v3_output_directory(output_dir)
    if operating_point_rule == "none":
        operating_point_rule = None
    if (
        operating_point_rule is not None
        and operating_point_rule not in OPERATING_POINT_RULES
    ):
        raise ValueError(
            f"operating_point_rule must be one of {OPERATING_POINT_RULES} or None."
        )
    if not 0.0 <= float(operating_coverage_gamma) <= 1.0:
        raise ValueError("operating_coverage_gamma must lie in [0, 1].")
    if not 0.0 <= float(operating_risk_epsilon) <= 1.0:
        raise ValueError("operating_risk_epsilon must lie in [0, 1].")
    if not 0.0 < float(operating_validation_size) < 1.0:
        raise ValueError("operating_validation_size must lie strictly in (0, 1).")
    label_policy_config = load_label_policy_config(label_policy_path)
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
    # This run-level field must describe the complete requested queue, not only
    # datasets reached before a quota stop.  Validation against actual label
    # names still happens when each dataset is loaded below.
    dataset_policy_hashes = {
        dataset_name: label_policy_hash(
            get_dataset_label_policy(label_policy_config, dataset_name)
        )
        for dataset_name in datasets
    }
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
            gsi_partition_mode=gsi_partition_mode,
            gsi_partition_random_state=gsi_partition_random_state,
            gsi_fixed_independent_labels=gsi_fixed_independent_labels,
            gsi_final_order=gsi_final_order,
        )
        model_signatures[model_name] = _model_signature(prototype)

    for dataset_name in datasets:
        x_data, y_data, _, label_names = dataset_loader(dataset_name)
        label_names = list(label_names)
        dataset_label_policy = get_dataset_label_policy(
            label_policy_config,
            dataset_name,
            label_names=label_names,
        )
        effective_critical_labels = (
            critical_labels
            if critical_labels is not None
            else (
                None
                if dataset_label_policy is None
                else dataset_label_policy.get("critical_labels") or None
            )
        )
        dataset_policy_hash = label_policy_hash(dataset_label_policy)
        if dataset_policy_hash != dataset_policy_hashes[dataset_name]:
            raise RuntimeError(
                f"Normalized label policy changed while loading {dataset_name}."
            )
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
                gsi_partition_mode,
                gsi_partition_random_state,
                gsi_fixed_independent_labels,
                gsi_final_order,
                effective_critical_labels,
                model_signatures[model_name],
                dataset_fingerprint,
                split_hash,
                dataset_policy_hash,
                operating_point_rule,
                operating_coverage_gamma,
                operating_risk_epsilon,
                operating_validation_size,
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
                    gsi_partition_mode=gsi_partition_mode,
                    gsi_partition_random_state=gsi_partition_random_state,
                    gsi_fixed_independent_labels=gsi_fixed_independent_labels,
                    gsi_final_order=gsi_final_order,
                )
                operating_selection = None
                if (
                    operating_point_rule is not None
                    and model_family(model_name) in ("MLC_PA", "GSI_MLC_PA")
                ):
                    operating_selection = _select_inner_operating_point(
                        model_name=model_name,
                        x_train=x_data[train_indices],
                        y_train=y_train,
                        abstention_costs=abstention_costs,
                        rule=operating_point_rule,
                        coverage_gamma=operating_coverage_gamma,
                        risk_epsilon=operating_risk_epsilon,
                        validation_size=operating_validation_size,
                        random_state=random_state,
                        model_factory=model_factory,
                        model_factory_kwargs={
                            "abstention_cost": report_cost,
                            "abstention_penalty": abstention_penalty,
                            "mlc_pa_base": mlc_pa_base,
                            "gsi_validation_size": gsi_validation_size,
                            "gsi_selection_objective": gsi_selection_objective,
                            "gsi_decision_policy": gsi_decision_policy,
                            "gsi_beta": gsi_beta,
                            "gsi_penalty": gsi_penalty,
                            "gsi_partition_mode": gsi_partition_mode,
                            "gsi_partition_random_state": (
                                gsi_partition_random_state
                            ),
                            "gsi_fixed_independent_labels": (
                                gsi_fixed_independent_labels
                            ),
                            "gsi_final_order": gsi_final_order,
                        },
                        label_names=label_names,
                        label_policy=dataset_label_policy,
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
                    critical_labels=effective_critical_labels,
                    label_policy=dataset_label_policy,
                )
                if operating_selection is not None:
                    metrics.setdefault("Model Metadata", {})[
                        "Operating Point Selection"
                    ] = operating_selection
                metadata = {
                    "Train And Evaluate Seconds": float(time.time() - started),
                    "Train Size": int(len(train_indices)),
                    "Test Size": int(len(test_indices)),
                }
                model_metadata = metrics.get("Model Metadata", {})
                if "Partition Mode" in model_metadata:
                    metadata.update({
                        "Partition Mode": model_metadata["Partition Mode"],
                        "Independent Label Count": model_metadata[
                            "Independent Label Count"
                        ],
                        "Dependent Label Count": model_metadata[
                            "Dependent Label Count"
                        ],
                        "Selection Seconds": model_metadata["Selection Seconds"],
                        "Probability Inference Seconds": model_metadata[
                            "Probability Inference Seconds"
                        ],
                    })
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
        "gsi_partition_mode": gsi_partition_mode,
        "gsi_fixed_independent_labels": (
            None
            if gsi_fixed_independent_labels is None
            else [int(label) for label in gsi_fixed_independent_labels]
        ),
        "gsi_final_order": gsi_final_order,
        "critical_labels": critical_labels,
        "label_policy_status": label_policy_config.get("status"),
        "label_policy_hashes": dataset_policy_hashes,
        "operating_point": {
            "rule": operating_point_rule,
            "coverage_gamma": float(operating_coverage_gamma),
            "risk_epsilon": float(operating_risk_epsilon),
            "validation_size": float(operating_validation_size),
            "selection_scope": "inner_validation",
        },
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
    if gsi_partition_random_state is not None:
        run_config["gsi_partition_random_state"] = int(
            gsi_partition_random_state
        )
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
        "Label Policy Source": label_policy_config.get("source_path"),
        "Results": results,
    }
    document["Artifacts"] = generate_deployment_plots(
        document, output_path / "figures" / run_config_hash
    )
    document["Artifacts"] = export_v3_artifacts(
        document, output_path / "tables" / run_config_hash
    )
    return document
