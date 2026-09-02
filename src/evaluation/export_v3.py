"""Strict JSON and scope-specific CSV exports for schema-v3 results."""

import csv
import os
import tempfile
from pathlib import Path

from .cache_v3 import atomic_json_dump_v3, json_safe


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
    return complete_rows, selective_rows, per_label_rows, group_rows, partition_rows


def export_v3_artifacts(run_document, tables_dir):
    """Export the v3 manifest plus complete/selective/per-label/group CSV files."""

    tables_path = Path(tables_dir)
    complete, selective, per_label, groups, partitions = _scope_rows(run_document)
    paths = {
        "json": tables_path / "results_v3.json",
        "complete_csv": tables_path / "complete_metrics.csv",
        "selective_csv": tables_path / "selective_metrics.csv",
        "per_label_csv": tables_path / "per_label_metrics.csv",
        "group_csv": tables_path / "group_metrics.csv",
        "partition_csv": tables_path / "partition_audit.csv",
    }
    artifacts = {name: str(path) for name, path in paths.items()}
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
    return artifacts
