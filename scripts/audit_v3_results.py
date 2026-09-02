"""Machine-check schema-v3 manifests, checkpoints, CSVs and figures."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.evaluation.cache_v3 import compute_config_hash  # noqa: E402
from src.models.registry import MATCHED_MODEL_IDS, model_family  # noqa: E402


TABLE_FILES = {
    "complete_csv": "complete_metrics.csv",
    "selective_csv": "selective_metrics.csv",
    "per_label_csv": "per_label_metrics.csv",
    "group_csv": "group_metrics.csv",
    "partition_csv": "partition_audit.csv",
    "calibration_csv": "calibration_metrics.csv",
    "reliability_csv": "reliability_data.csv",
    "deployment_csv": "deployment_metrics.csv",
    "critical_deployment_csv": "critical_label_deployment.csv",
}
FIGURE_FILES = {
    "risk_coverage_plot": "risk_coverage.png",
    "optimistic_gain_review_plot": "optimistic_gain_vs_review_load.png",
    "error_capture_review_plot": "error_capture_vs_review_load.png",
    "critical_label_plot": "per_label_critical_metrics.png",
    "il_dl_ablation_plot": "il_dl_ablation.png",
    "objective_comparison_plot": "objective_comparison.png",
    "calibration_reliability_plot": "calibration_reliability.png",
}


def _reject_nonfinite(token):
    raise ValueError(f"Non-standard JSON constant: {token}")


def _load_strict_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream, parse_constant=_reject_nonfinite)


def _cost_key(value):
    return f"{float(value):.2f}"


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return tuple(reader.fieldnames or ()), list(reader)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _artifact_path_matches(recorded, expected, output_root):
    value = Path(recorded)
    if value.is_absolute():
        candidates = [value.resolve()]
    else:
        candidates = [
            (Path.cwd() / value).resolve(),
            (WORKSPACE_ROOT / value).resolve(),
            (output_root / value).resolve(),
        ]
    return expected.resolve() in candidates


def _find_manifest(output_root, config_hash=None):
    candidates = sorted(output_root.glob("tables/*/results_v3.json"))
    _require(candidates, f"No results_v3.json found below {output_root}.")
    if config_hash is not None:
        candidates = [
            path for path in candidates
            if path.parent.name == str(config_hash)
        ]
        _require(candidates, f"Config hash {config_hash} was not found.")
    _require(
        len(candidates) == 1,
        "Multiple manifests found; pass --config-hash explicitly: "
        + ", ".join(path.parent.name for path in candidates),
    )
    return candidates[0]


def _audit_checkpoint(output_root, dataset, model, summary, folds, settings):
    candidates = sorted((output_root / "checkpoints" / model).glob(
        f"{dataset}.*.json"
    ))
    payloads = [_load_strict_json(path) for path in candidates]
    matches = [
        payload for payload in payloads
        if payload.get("config_hash") == summary.get("Config Hash")
    ]
    _require(len(matches) == 1, f"Expected one matching checkpoint for {dataset}/{model}.")
    checkpoint = matches[0]
    _require(checkpoint.get("schema_version") == 3, "Checkpoint schema is not v3.")
    _require(checkpoint.get("status") == "complete", f"Incomplete checkpoint: {dataset}/{model}.")
    _require(
        compute_config_hash(checkpoint["config"]) == checkpoint["config_hash"],
        f"Checkpoint config hash mismatch: {dataset}/{model}.",
    )
    _require(
        sorted(int(key) for key in checkpoint.get("folds", {}))
        == list(range(1, folds + 1)),
        f"Checkpoint folds mismatch: {dataset}/{model}.",
    )
    config = checkpoint["config"]
    for field in (
        "dataset_fingerprint",
        "split_hash",
        "model_signature",
        "label_policy_hash",
        "operating_point",
        "evaluation_policy",
    ):
        _require(field in config, f"Checkpoint lacks {field}: {dataset}/{model}.")
    _require(config.get("random_state") == settings["random_state"], "Seed drift in checkpoint.")
    _require(config.get("n_splits") == folds, "Fold-count drift in checkpoint.")


def audit_results(
    output_dir,
    *,
    config_hash=None,
    expected_datasets=None,
    expected_models=None,
    expected_folds=None,
    expected_costs=None,
):
    """Raise on an invariant violation and return a compact audit summary."""

    output_root = Path(output_dir).resolve()
    manifest_path = _find_manifest(output_root, config_hash=config_hash)
    document = _load_strict_json(manifest_path)
    settings = document.get("Settings", {})
    run_hash = document.get("Config Hash")
    _require(document.get("Schema Version") == 3, "Manifest schema is not v3.")
    _require(document.get("Status") == "complete", "Run manifest is not complete.")
    _require(compute_config_hash(settings) == run_hash, "Run config hash mismatch.")
    _require(manifest_path.parent.name == run_hash, "Manifest directory/hash mismatch.")

    datasets = list(expected_datasets or settings.get("datasets", []))
    models = list(expected_models or settings.get("models", []))
    folds = int(expected_folds or settings.get("n_splits", 0))
    costs = [float(value) for value in (
        expected_costs or settings.get("abstention_costs", [])
    )]
    _require(datasets and models and folds > 0 and costs, "Expected run dimensions are empty.")
    _require(settings.get("datasets") == datasets, "Dataset manifest differs from expectation.")
    _require(settings.get("models") == models, "Model manifest differs from expectation.")
    _require(settings.get("n_splits") == folds, "Fold manifest differs from expectation.")
    _require(settings.get("abstention_costs") == costs, "Cost grid differs from expectation.")
    _require(set(settings.get("model_signatures", {})) == set(models), "Model signatures are incomplete.")

    results = document.get("Results", {})
    _require(set(results) == set(datasets), "Result datasets are incomplete or unexpected.")
    selective_pairs = 0
    for dataset in datasets:
        _require(set(results[dataset]) == set(models), f"Model matrix incomplete for {dataset}.")
        for model in models:
            summary = results[dataset][model]
            _require(summary.get("Status") == "complete", f"Incomplete pair: {dataset}/{model}.")
            _require(
                summary.get("Completed Folds") == list(range(1, folds + 1)),
                f"Fold summary mismatch: {dataset}/{model}.",
            )
            full_rows = summary.get("Full", {}).get("Raw Folds", [])
            _require(len(full_rows) == folds, f"Full row count mismatch: {dataset}/{model}.")
            for row in full_rows:
                accuracy = row.get("Hamming Accuracy")
                loss = row.get("Hamming Loss")
                _require(
                    isinstance(accuracy, (int, float))
                    and isinstance(loss, (int, float))
                    and math.isclose(float(accuracy) + float(loss), 1.0, abs_tol=1e-12),
                    f"Hamming identity failed: {dataset}/{model}/fold {row.get('Fold')}.",
                )
            family = model_family(model)
            cost_summaries = summary.get("Costs", {})
            if family in ("MLC_PA", "GSI_MLC_PA"):
                selective_pairs += 1
                _require(
                    set(cost_summaries) == {_cost_key(cost) for cost in costs},
                    f"Cost grid incomplete: {dataset}/{model}.",
                )
                if any(math.isclose(cost, 0.5) for cost in costs):
                    sanity = cost_summaries["0.50"]["Selective"]["Raw Folds"]
                    _require(len(sanity) == folds, f"Missing c=0.5 rows: {dataset}/{model}.")
                    for row in sanity:
                        _require(
                            math.isclose(float(row["Coverage"]), 1.0, abs_tol=1e-12)
                            and math.isclose(float(row["AABS"]), 0.0, abs_tol=1e-12),
                            f"c=0.5 is not no-abstention: {dataset}/{model}.",
                        )
            else:
                _require(not cost_summaries, f"Complete model has selective rows: {dataset}/{model}.")
            _audit_checkpoint(output_root, dataset, model, summary, folds, settings)

    table_dir = output_root / "tables" / run_hash
    figure_dir = output_root / "figures" / run_hash
    artifacts = document.get("Artifacts", {})
    _require(set(TABLE_FILES) | set(FIGURE_FILES) | {"json"} <= set(artifacts), "Artifact manifest is incomplete.")
    _require(
        _artifact_path_matches(artifacts["json"], manifest_path, output_root),
        "Manifest records the wrong results_v3.json path.",
    )
    csv_rows = {}
    for key, filename in TABLE_FILES.items():
        path = table_dir / filename
        _require(
            _artifact_path_matches(artifacts[key], path, output_root),
            f"Artifact manifest path mismatch for {key}.",
        )
        _require(path.exists() and path.stat().st_size > 0, f"Missing table: {path}.")
        header, rows = _read_csv(path)
        _require(header, f"CSV has no header: {path}.")
        csv_rows[key] = rows
    for key, filename in FIGURE_FILES.items():
        path = figure_dir / filename
        _require(
            _artifact_path_matches(artifacts[key], path, output_root),
            f"Artifact manifest path mismatch for {key}.",
        )
        _require(path.exists() and path.stat().st_size > 0, f"Missing figure: {path}.")
    _require(
        len(csv_rows["complete_csv"]) == len(datasets) * len(models) * folds,
        "complete_metrics.csv row count mismatch.",
    )
    expected_deployment = selective_pairs * len(costs) * folds
    operating_rows = [
        row for row in csv_rows["deployment_csv"]
        if row.get("Scope") == "Operating Point"
    ]
    _require(len(operating_rows) == expected_deployment, "Deployment row count mismatch.")
    _require(
        len(csv_rows["selective_csv"]) == expected_deployment,
        "selective_metrics.csv row count mismatch.",
    )
    _require(
        len(csv_rows["critical_deployment_csv"]) == expected_deployment,
        "critical_label_deployment.csv row count mismatch.",
    )
    _require(
        all(row.get("Scope") != "Full" for row in csv_rows["deployment_csv"]),
        "Deployment CSV mixes Full and Selective denominators.",
    )
    for row in operating_rows:
        _require(row.get("Coverage") != "", "Deployment row lacks coverage.")
        coverage = float(row["Coverage"])
        _require(0.0 <= coverage <= 1.0, "Deployment coverage is outside [0, 1].")
        if coverage > 0.0:
            _require(
                row.get("Selective Risk") != "",
                "Positive-coverage deployment row lacks selective risk.",
            )
    return {
        "status": "PASS",
        "config_hash": run_hash,
        "datasets": len(datasets),
        "models": len(models),
        "folds_per_pair": folds,
        "model_dataset_pairs": len(datasets) * len(models),
        "selective_pairs": selective_pairs,
        "costs": costs,
        "csv_artifacts": len(TABLE_FILES),
        "figure_artifacts": len(FIGURE_FILES),
        "manifest": str(manifest_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="Schema-v3 output root to audit.")
    parser.add_argument("--config-hash", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--costs", nargs="+", type=float, default=None)
    parser.add_argument(
        "--expect-primary-models",
        action="store_true",
        help="Require the frozen 12-ID matched model order.",
    )
    args = parser.parse_args()
    expected_models = list(MATCHED_MODEL_IDS) if args.expect_primary_models else args.models
    summary = audit_results(
        args.output_dir,
        config_hash=args.config_hash,
        expected_datasets=args.datasets,
        expected_models=expected_models,
        expected_folds=args.folds,
        expected_costs=args.costs,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
