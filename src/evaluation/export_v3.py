"""Strict JSON and scope-specific CSV exports for schema-v3 results."""

import csv
import hashlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .cache_v3 import atomic_json_dump_v3, json_safe
from .metric_contract import (
    BSS_SPCC_FULL_METRIC_NAMES,
    BSS_SPCC_METRIC_PROFILE_VERSION,
    BSS_SPCC_SELECTIVE_METRIC_NAMES,
)


def _atomic_csv_dump(rows, path, leading_fields):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    normalized_rows = [json_safe(row) for row in rows]
    extra_fields = sorted(
        {
            key
            for row in normalized_rows
            for key in row
            if key not in leading_fields
        }
    )
    fieldnames = list(leading_fields) + extra_fields
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}_", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(normalized_rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return target


def _scope_rows(run_document):
    complete_rows = []
    selective_rows = []
    per_label_rows = []
    group_rows = []
    partition_rows = []
    calibration_rows = []
    reliability_rows = []
    deployment_rows = []
    critical_deployment_rows = []
    for dataset_name, models in run_document.get("Results", {}).items():
        for model_name, summary in models.items():
            prefix = {"Dataset": dataset_name, "Model": model_name}
            for row in summary.get("Full", {}).get("Raw Folds", []):
                complete_rows.append({**prefix, **row})
            for row in summary.get("Per Label", {}).get("Raw Records", []):
                per_label_rows.append({**prefix, "Cost": None, **row})
            for row in summary.get("Groups", {}).get("Raw Records", []):
                group_rows.append({**prefix, "Cost": None, **row})
            for row in summary.get("Partition Audit", {}).get("Raw Records", []):
                partition_rows.append({**prefix, **row})
            calibration = summary.get("Calibration", {})
            for row in calibration.get("Metrics", {}).get("Raw Folds", []):
                calibration_rows.append({**prefix, "Scope": "Aggregate", **row})
            for row in calibration.get("Per Label", {}).get("Raw Records", []):
                calibration_rows.append({**prefix, "Scope": "Per Label", **row})
            for row in calibration.get("Reliability", {}).get("Raw Records", []):
                reliability_rows.append({**prefix, **row})

            for cost, cost_summary in summary.get("Costs", {}).items():
                scope_by_fold = {}
                for scope in ("Selective", "Rejected", "Optimistic", "Diagnostics"):
                    for row in cost_summary.get(scope, {}).get("Raw Folds", []):
                        fold = row.get("Fold")
                        scope_by_fold.setdefault(fold, {}).update(row)
                for fold, values in sorted(scope_by_fold.items()):
                    selective_rows.append(
                        {**prefix, "Cost": cost, "Fold": fold, **values}
                    )
                for row in cost_summary.get("Per Label", {}).get("Raw Records", []):
                    per_label_rows.append({**prefix, "Cost": cost, **row})
                for row in cost_summary.get("Groups", {}).get("Raw Records", []):
                    group_rows.append({**prefix, "Cost": cost, **row})
                deployment = cost_summary.get("Deployment", {})
                for row in deployment.get("Raw Folds", []):
                    deployment_rows.append({
                        **prefix,
                        "Cost": cost,
                        "Scope": "Operating Point",
                        **row,
                    })
                for row in deployment.get("Reviewer Scenarios", []):
                    deployment_rows.append({
                        **prefix,
                        "Cost": cost,
                        "Scope": "Reviewer Scenario",
                        **row,
                    })
                for row in deployment.get("Critical Labels", []):
                    metrics = row.get("Metrics", {})
                    critical_deployment_rows.append({
                        **prefix,
                        "Cost": cost,
                        "Fold": row.get("Fold"),
                        "Status": row.get("Status"),
                        "Reason": row.get("Reason"),
                        "Critical Labels": row.get("Critical Labels", []),
                        **metrics,
                    })
    return (
        complete_rows,
        selective_rows,
        per_label_rows,
        group_rows,
        partition_rows,
        calibration_rows,
        reliability_rows,
        deployment_rows,
        critical_deployment_rows,
    )


def export_v3_artifacts(run_document, tables_dir):
    """Export the v3 manifest plus complete/selective/per-label/group CSV files."""

    tables_path = Path(tables_dir)
    (
        complete,
        selective,
        per_label,
        groups,
        partitions,
        calibration,
        reliability,
        deployment,
        critical_deployment,
    ) = _scope_rows(run_document)
    paths = {
        "json": tables_path / "results_v3.json",
        "complete_csv": tables_path / "complete_metrics.csv",
        "selective_csv": tables_path / "selective_metrics.csv",
        "per_label_csv": tables_path / "per_label_metrics.csv",
        "group_csv": tables_path / "group_metrics.csv",
        "partition_csv": tables_path / "partition_audit.csv",
        "calibration_csv": tables_path / "calibration_metrics.csv",
        "reliability_csv": tables_path / "reliability_data.csv",
        "deployment_csv": tables_path / "deployment_metrics.csv",
        "critical_deployment_csv": tables_path / "critical_label_deployment.csv",
    }
    artifacts = {
        **run_document.get("Artifacts", {}),
        **{name: str(path) for name, path in paths.items()},
    }
    run_document["Artifacts"] = artifacts
    atomic_json_dump_v3(run_document, paths["json"])
    _atomic_csv_dump(
        complete,
        paths["complete_csv"],
        ("Dataset", "Model", "Fold"),
    )
    _atomic_csv_dump(
        selective,
        paths["selective_csv"],
        ("Dataset", "Model", "Cost", "Fold"),
    )
    _atomic_csv_dump(
        per_label,
        paths["per_label_csv"],
        ("Dataset", "Model", "Cost", "Fold", "Label Index", "Label Name"),
    )
    _atomic_csv_dump(
        groups,
        paths["group_csv"],
        ("Dataset", "Model", "Cost", "Fold", "Group"),
    )
    _atomic_csv_dump(
        partitions,
        paths["partition_csv"],
        ("Dataset", "Model", "Fold", "Partition Mode"),
    )
    _atomic_csv_dump(
        calibration,
        paths["calibration_csv"],
        ("Dataset", "Model", "Fold", "Scope", "Label Index", "Label Name"),
    )
    _atomic_csv_dump(
        reliability,
        paths["reliability_csv"],
        ("Dataset", "Model", "Fold", "Bin"),
    )
    _atomic_csv_dump(
        deployment,
        paths["deployment_csv"],
        ("Dataset", "Model", "Cost", "Fold", "Scope", "Reviewer Accuracy"),
    )
    _atomic_csv_dump(
        critical_deployment,
        paths["critical_deployment_csv"],
        ("Dataset", "Model", "Cost", "Fold", "Status"),
    )
    return artifacts


def _completed_bss_results(run_document):
    completed = {}
    for dataset_name, models in run_document.get("Results", {}).items():
        for model_name, summary in models.items():
            expected = int(summary.get("Expected Fold Count", 0))
            folds = list(summary.get("Completed Folds", []))
            if (
                summary.get("Status") != "complete"
                or expected <= 0
                or folds != list(range(1, expected + 1))
            ):
                continue
            full = summary.get("Full", {})
            filtered = {
                "Status": "complete",
                "Config Hash": summary.get("Config Hash"),
                "Expected Fold Count": expected,
                "Completed Folds": folds,
                "Full": {
                    "Mean": {
                        name: full.get("Mean", {}).get(name)
                        for name in BSS_SPCC_FULL_METRIC_NAMES
                    },
                    "Std": {
                        name: full.get("Std", {}).get(name)
                        for name in BSS_SPCC_FULL_METRIC_NAMES
                    },
                    "Raw Folds": [
                        {
                            "Fold": row.get("Fold"),
                            **{
                                name: row.get(name)
                                for name in BSS_SPCC_FULL_METRIC_NAMES
                            },
                        }
                        for row in full.get("Raw Folds", [])
                    ],
                },
                "Costs": {},
            }
            for cost, cost_summary in summary.get("Costs", {}).items():
                selective = cost_summary.get("Selective", {})
                filtered["Costs"][cost] = {
                    "Selective": {
                        "Mean": {
                            name: selective.get("Mean", {}).get(name)
                            for name in BSS_SPCC_SELECTIVE_METRIC_NAMES
                        },
                        "Std": {
                            name: selective.get("Std", {}).get(name)
                            for name in BSS_SPCC_SELECTIVE_METRIC_NAMES
                        },
                        "Raw Folds": [
                            {
                                "Fold": row.get("Fold"),
                                **{
                                    name: row.get(name)
                                    for name in BSS_SPCC_SELECTIVE_METRIC_NAMES
                                },
                            }
                            for row in selective.get("Raw Folds", [])
                        ],
                    }
                }
            completed.setdefault(dataset_name, {})[model_name] = filtered
    return completed


def export_bss_spcc_artifacts(run_document, tables_dir):
    """Export only the locked BSS-SPCC metric profile and structure audit."""

    tables_path = Path(tables_dir)
    completed = _completed_bss_results(run_document)
    filtered_document = {
        "Schema Version": run_document.get("Schema Version"),
        "Metric Profile": BSS_SPCC_METRIC_PROFILE_VERSION,
        "Config Hash": run_document.get("Config Hash"),
        "Status": run_document.get("Status"),
        "Generated At": run_document.get("Generated At"),
        "Settings": run_document.get("Settings", {}),
        "Results": completed,
    }
    fold_rows = []
    summary_rows = []
    structure = {
        "Metric Profile": BSS_SPCC_METRIC_PROFILE_VERSION,
        "Config Hash": run_document.get("Config Hash"),
        "Datasets": {},
    }
    for dataset_name, models in completed.items():
        for model_name, summary in models.items():
            prefix = {"Dataset": dataset_name, "Model": model_name}
            for row in summary["Full"]["Raw Folds"]:
                fold_rows.append(
                    {**prefix, "Scope": "Full", "Cost": None, **row}
                )
            summary_rows.append({
                **prefix,
                "Scope": "Full",
                "Cost": None,
                **{
                    f"{name} Mean": summary["Full"]["Mean"].get(name)
                    for name in BSS_SPCC_FULL_METRIC_NAMES
                },
                **{
                    f"{name} Std": summary["Full"]["Std"].get(name)
                    for name in BSS_SPCC_FULL_METRIC_NAMES
                },
            })
            for cost, cost_summary in summary["Costs"].items():
                selective = cost_summary["Selective"]
                for row in selective["Raw Folds"]:
                    fold_rows.append({
                        **prefix,
                        "Scope": "Selective",
                        "Cost": cost,
                        **row,
                    })
                summary_rows.append({
                    **prefix,
                    "Scope": "Selective",
                    "Cost": cost,
                    **{
                        f"{name} Mean": selective["Mean"].get(name)
                        for name in BSS_SPCC_SELECTIVE_METRIC_NAMES
                    },
                    **{
                        f"{name} Std": selective["Std"].get(name)
                        for name in BSS_SPCC_SELECTIVE_METRIC_NAMES
                    },
                })
            original = run_document["Results"][dataset_name][model_name]
            structure["Datasets"].setdefault(dataset_name, {})[model_name] = [
                {
                    "Fold": record.get("Fold"),
                    "Structure Audit": record.get("Structure Audit"),
                }
                for record in original.get("Model Metadata", [])
                if record.get("Structure Audit") is not None
            ]

    paths = {
        "results_json": tables_path / "results.json",
        "fold_results_csv": tables_path / "fold_results.csv",
        "summary_results_csv": tables_path / "summary_results.csv",
        "structure_audit_json": tables_path / "structure_audit.json",
        "artifact_manifest_json": tables_path / "artifact_manifest.json",
    }
    if "Comparison" in run_document:
        paths["comparison_audit_json"] = tables_path / "comparison_audit.json"
    artifacts = {
        **run_document.get("Artifacts", {}),
        **{name: str(path) for name, path in paths.items()},
    }
    filtered_document["Artifacts"] = artifacts
    atomic_json_dump_v3(filtered_document, paths["results_json"])
    atomic_json_dump_v3(structure, paths["structure_audit_json"])
    if "comparison_audit_json" in paths:
        atomic_json_dump_v3(
            run_document["Comparison"], paths["comparison_audit_json"]
        )
    _atomic_csv_dump(
        fold_rows,
        paths["fold_results_csv"],
        ("Dataset", "Model", "Scope", "Cost", "Fold"),
    )
    _atomic_csv_dump(
        summary_rows,
        paths["summary_results_csv"],
        ("Dataset", "Model", "Scope", "Cost"),
    )
    hashes = {}
    for name, value in artifacts.items():
        candidate = Path(value)
        if name == "artifact_manifest_json" or not candidate.is_file():
            continue
        digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        hashes[name] = digest.hexdigest()
    atomic_json_dump_v3(
        {
            "Metric Profile": BSS_SPCC_METRIC_PROFILE_VERSION,
            "Config Hash": run_document.get("Config Hash"),
            "Completed Datasets": list(completed),
            "Report Cost": run_document.get("Settings", {}).get("report_cost"),
            "Cost Grid": run_document.get("Settings", {}).get("abstention_costs", []),
            "Generated At": datetime.now(timezone.utc).isoformat(),
            "SHA-256": hashes,
        },
        paths["artifact_manifest_json"],
    )
    return artifacts
