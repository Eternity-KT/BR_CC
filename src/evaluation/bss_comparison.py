"""Strict adapters for Logistic BR/CC/MLC-PA comparison figures.

The adapters deliberately keep comparison data outside the BSS metric result
document.  Legacy schema-v2 results are reused only when their model identity
and scientific settings prove that the base estimator is Logistic.  Missing
metrics remain missing; they are never replaced by zero or approximated.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .metric_contract import (
    BSS_SPCC_FULL_METRIC_NAMES,
    BSS_SPCC_SELECTIVE_METRIC_NAMES,
)


COMPARISON_MODEL_ORDER = (
    "BR_Logistic",
    "CC_Logistic",
    "MLC_PA_Logistic",
)

_LEGACY_FULL_MAP = {
    "Subset Accuracy": "Subset Accuracy",
    "Macro-F1": "Macro-F1",
    "Micro-F1": "Micro-F1",
    "Instance-F1": "Example-F1",
}

_LEGACY_SELECTIVE_MAP = {
    "Coverage": "Coverage",
    "AABS": "AABS",
    "ABS": "ABS",
    "Generalized Loss": "Generalized Loss",
    "Selective Macro-F1": "Selective Macro-F1",
    "Selective Micro-F1": "Selective Micro-F1",
}


def _load_json(path):
    target = Path(path)
    with target.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path):
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _summary_from_rows(rows, metric_names):
    normalized_rows = []
    for fold_index, row in enumerate(rows, 1):
        normalized = {"Fold": int(row.get("Fold", fold_index))}
        for metric in metric_names:
            numeric = _finite_float(row.get(metric))
            if numeric is not None:
                normalized[metric] = numeric
        normalized_rows.append(normalized)
    mean, std = {}, {}
    for metric in metric_names:
        values = [row[metric] for row in normalized_rows if metric in row]
        if values:
            mean[metric] = float(np.mean(values))
            std[metric] = float(np.std(values))
    return {"Mean": mean, "Std": std, "Raw Folds": normalized_rows}


def _legacy_full_summary(result):
    rows = []
    for fold_index, source in enumerate(result.get("raw_folds", []), 1):
        hamming_loss = _finite_float(source.get("Hamming Loss"))
        row = {"Fold": fold_index}
        if hamming_loss is not None:
            row["Hamming Accuracy"] = 1.0 - hamming_loss
        for target, legacy in _LEGACY_FULL_MAP.items():
            value = _finite_float(source.get(legacy))
            if value is not None:
                row[target] = value
        rows.append(row)
    return _summary_from_rows(rows, BSS_SPCC_FULL_METRIC_NAMES)


def _legacy_selective_summary(result):
    rows = []
    for fold_index, source in enumerate(result.get("raw_folds", []), 1):
        row = {"Fold": fold_index}
        for target, legacy in _LEGACY_SELECTIVE_MAP.items():
            value = _finite_float(source.get(legacy))
            if value is not None:
                row[target] = value
        selective_loss = _finite_float(source.get("Selective Hamming Loss"))
        if selective_loss is not None:
            row["Selective Hamming Accuracy"] = 1.0 - selective_loss
        # The schema-v2 MLC-PA cache did not compute Selective Instance-F1.
        # Its absence is intentional and must remain visible to the plotter.
        rows.append(row)
    return _summary_from_rows(rows, BSS_SPCC_SELECTIVE_METRIC_NAMES)


def _canonical_v3_full(summary):
    source_rows = summary.get("Full", {}).get("Raw Folds", [])
    rows = []
    for fold_index, source in enumerate(source_rows, 1):
        row = {"Fold": int(source.get("Fold", fold_index))}
        for metric in BSS_SPCC_FULL_METRIC_NAMES:
            value = _finite_float(source.get(metric))
            if value is not None:
                row[metric] = value
        rows.append(row)
    return _summary_from_rows(rows, BSS_SPCC_FULL_METRIC_NAMES)


def _v3_model_documents(output_dir, model_name):
    root = Path(output_dir)
    if not root.is_dir():
        return []
    documents = []
    for result_path in sorted(root.glob("tables/*/results_v3.json")):
        try:
            document = _load_json(result_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        settings = document.get("Settings", {})
        if settings.get("models") != [model_name]:
            continue
        documents.append((result_path, document))
    return documents


def comparison_status(
    config,
    datasets,
    n_splits=None,
    random_state=None,
    abstention_costs=None,
    report_cost=None,
    abstention_penalty=None,
):
    """Return source availability without loading any datasets or training."""

    datasets = list(datasets)
    legacy_path = Path(config["legacy_results_path"])
    mlc_path = Path(config["mlc_pa_cache_path"])
    status = {
        "BR_Logistic": {"source": str(legacy_path), "available_datasets": []},
        "CC_Logistic": {
            "source": str(Path(config["cc_logistic_output_dir"])),
            "available_datasets": [],
        },
        "MLC_PA_Logistic": {
            "source": [
                str(mlc_path),
                str(Path(config["mlc_pa_logistic_output_dir"])),
            ],
            "available_datasets": [],
        },
    }
    if legacy_path.is_file():
        legacy = _load_json(legacy_path)
        status["BR_Logistic"]["available_datasets"] = [
            name
            for name in datasets
            if isinstance(legacy.get(name, {}).get("BR_Logistic"), dict)
            and (
                n_splits is None
                or len(
                    legacy[name]["BR_Logistic"].get("raw_folds", [])
                )
                == int(n_splits)
            )
        ]
    if mlc_path.is_file():
        cache = _load_json(mlc_path)
        settings = cache.get("settings", {})
        compatible = settings.get("base_estimator") == "logistic"
        if n_splits is not None:
            compatible = compatible and settings.get("n_splits") == int(n_splits)
        if random_state is not None:
            compatible = compatible and settings.get("random_state") == int(
                random_state
            )
        if abstention_costs is not None:
            compatible = compatible and [
                float(value) for value in settings.get("abstention_costs", [])
            ] == [float(value) for value in abstention_costs]
        if report_cost is not None:
            compatible = compatible and float(
                settings.get("report_cost", -1.0)
            ) == float(report_cost)
        if abstention_penalty is not None:
            compatible = compatible and settings.get(
                "abstention_penalty"
            ) == abstention_penalty
        if compatible:
            status["MLC_PA_Logistic"]["available_datasets"] = [
                name
                for name in datasets
                if name in cache.get("datasets", {})
                and (
                    n_splits is None
                    or len(
                        cache["datasets"][name]
                        .get("full", {})
                        .get("raw_folds", [])
                    )
                    == int(n_splits)
                )
            ]
    cc_available = set()
    for _, document in _v3_model_documents(
        config["cc_logistic_output_dir"], "CC_Logistic"
    ):
        settings = document.get("Settings", {})
        if n_splits is not None and settings.get("n_splits") != int(n_splits):
            continue
        if random_state is not None and settings.get("random_state") != int(
            random_state
        ):
            continue
        for dataset, models in document.get("Results", {}).items():
            summary = models.get("CC_Logistic", {})
            expected = int(summary.get("Expected Fold Count", 0))
            if (
                summary.get("Status") == "complete"
                and summary.get("Completed Folds")
                == list(range(1, expected + 1))
            ):
                cc_available.add(dataset)
    status["CC_Logistic"]["available_datasets"] = [
        name for name in datasets if name in cc_available
    ]
    mlc_available = set(status["MLC_PA_Logistic"]["available_datasets"])
    mlc_instance_available = set()
    for _, document in _v3_model_documents(
        config["mlc_pa_logistic_output_dir"], "MLC_PA_Logistic"
    ):
        settings = document.get("Settings", {})
        if settings.get("mlc_pa_base") != "logistic":
            continue
        if n_splits is not None and settings.get("n_splits") != int(n_splits):
            continue
        if random_state is not None and settings.get("random_state") != int(
            random_state
        ):
            continue
        if abstention_costs is not None and [
            float(value) for value in settings.get("abstention_costs", [])
        ] != [float(value) for value in abstention_costs]:
            continue
        if report_cost is not None and float(
            settings.get("report_cost", -1.0)
        ) != float(report_cost):
            continue
        if abstention_penalty is not None and settings.get(
            "abstention_penalty"
        ) != abstention_penalty:
            continue
        for dataset, models in document.get("Results", {}).items():
            summary = models.get("MLC_PA_Logistic", {})
            expected = int(summary.get("Expected Fold Count", 0))
            if (
                summary.get("Status") == "complete"
                and summary.get("Completed Folds")
                == list(range(1, expected + 1))
            ):
                mlc_available.add(dataset)
                if settings.get("include_selective_instance_f1") is True:
                    cost_keys = (
                        [f"{float(value):.2f}" for value in abstention_costs]
                        if abstention_costs is not None
                        else list(summary.get("Costs", {}))
                    )
                    if cost_keys and all(
                        "Selective Instance-F1"
                        in summary.get("Costs", {})
                        .get(cost_key, {})
                        .get("Selective", {})
                        .get("Mean", {})
                        for cost_key in cost_keys
                    ):
                        mlc_instance_available.add(dataset)
    status["MLC_PA_Logistic"]["available_datasets"] = [
        name for name in datasets if name in mlc_available
    ]
    status["MLC_PA_Logistic"][
        "selective_instance_f1_available_datasets"
    ] = [name for name in datasets if name in mlc_instance_available]
    status["MLC_PA_Logistic"][
        "missing_selective_instance_f1_datasets"
    ] = [name for name in datasets if name not in mlc_instance_available]
    for model in COMPARISON_MODEL_ORDER:
        available = set(status[model]["available_datasets"])
        status[model]["missing_datasets"] = [
            name for name in datasets if name not in available
        ]
    return status


def _validate_mlc_settings(settings, expected):
    required = {
        "base_estimator": "logistic",
        "n_splits": int(expected["n_splits"]),
        "random_state": int(expected["random_state"]),
        "abstention_penalty": expected["abstention_penalty"],
    }
    mismatches = {
        key: {"expected": value, "actual": settings.get(key)}
        for key, value in required.items()
        if settings.get(key) != value
    }
    actual_costs = [float(value) for value in settings.get("abstention_costs", [])]
    expected_costs = [float(value) for value in expected["abstention_costs"]]
    if actual_costs != expected_costs:
        mismatches["abstention_costs"] = {
            "expected": expected_costs,
            "actual": actual_costs,
        }
    if float(settings.get("report_cost", -1.0)) != float(expected["report_cost"]):
        mismatches["report_cost"] = {
            "expected": float(expected["report_cost"]),
            "actual": settings.get("report_cost"),
        }
    if mismatches:
        raise ValueError(
            "MLC_PA legacy cache is not scientifically compatible: "
            f"{mismatches}"
        )


def load_bss_comparison_bundle(config, datasets, expected_settings):
    """Load compatible results into a canonical, auditable comparison bundle."""

    datasets = list(datasets)
    bundle = {
        "Schema Version": 1,
        "Base Learner": "logistic",
        "Model Order": list(COMPARISON_MODEL_ORDER),
        "Models": {},
        "Missing": {},
        "Warnings": [],
    }

    legacy_path = Path(config["legacy_results_path"])
    if not legacy_path.is_file():
        raise FileNotFoundError(f"Missing BR Logistic cache: {legacy_path}")
    legacy = _load_json(legacy_path)
    br_datasets = {}
    for dataset in datasets:
        source = legacy.get(dataset, {}).get("BR_Logistic")
        if (
            isinstance(source, dict)
            and len(source.get("raw_folds", []))
            == int(expected_settings["n_splits"])
        ):
            br_datasets[dataset] = {
                "Full": _legacy_full_summary(source),
                "Costs": {},
            }
    bundle["Models"]["BR_Logistic"] = {
        "Kind": "complete_only",
        "Source": {
            "Path": str(legacy_path.resolve()),
            "SHA-256": _sha256(legacy_path),
            "Schema": "legacy raw_results.json",
            "Model Key": "BR_Logistic",
        },
        "Datasets": br_datasets,
    }
    bundle["Warnings"].append(
        "Legacy BR_Logistic has no embedded seed/settings manifest; model key "
        "and expected outer-fold count were verified."
    )

    mlc_path = Path(config["mlc_pa_cache_path"])
    if not mlc_path.is_file():
        raise FileNotFoundError(f"Missing MLC-PA Logistic cache: {mlc_path}")
    mlc = _load_json(mlc_path)
    _validate_mlc_settings(mlc.get("settings", {}), expected_settings)
    mlc_datasets = {}
    for dataset in datasets:
        source = mlc.get("datasets", {}).get(dataset)
        if (
            not isinstance(source, dict)
            or len(source.get("full", {}).get("raw_folds", []))
            != int(expected_settings["n_splits"])
        ):
            continue
        if any(
            len(cost_result.get("raw_folds", []))
            != int(expected_settings["n_splits"])
            for cost_result in source.get("costs", {}).values()
        ):
            continue
        costs = {
            f"{float(cost):.2f}": {
                "Selective": _legacy_selective_summary(cost_result)
            }
            for cost, cost_result in source.get("costs", {}).items()
        }
        mlc_datasets[dataset] = {
            "Full": _legacy_full_summary(source.get("full", {})),
            "Costs": costs,
        }
    mlc_sources = [{
            "Path": str(mlc_path.resolve()),
            "SHA-256": _sha256(mlc_path),
            "Schema": mlc.get("schema_version"),
            "Model Key": mlc.get("model"),
            "Verified Settings": mlc.get("settings", {}),
    }]
    mlc_documents = _v3_model_documents(
        config["mlc_pa_logistic_output_dir"], "MLC_PA_Logistic"
    )
    mlc_documents.sort(
        key=lambda item: bool(
            item[1]
            .get("Settings", {})
            .get("include_selective_instance_f1")
        )
    )
    for result_path, document in mlc_documents:
        settings = document.get("Settings", {})
        if (
            settings.get("mlc_pa_base") != "logistic"
            or int(settings.get("n_splits", -1))
            != int(expected_settings["n_splits"])
            or int(settings.get("random_state", -1))
            != int(expected_settings["random_state"])
            or [float(value) for value in settings.get("abstention_costs", [])]
            != [float(value) for value in expected_settings["abstention_costs"]]
            or float(settings.get("report_cost", -1.0))
            != float(expected_settings["report_cost"])
            or settings.get("abstention_penalty")
            != expected_settings["abstention_penalty"]
        ):
            continue
        used = False
        for dataset in datasets:
            summary = document.get("Results", {}).get(dataset, {}).get(
                "MLC_PA_Logistic"
            )
            if not isinstance(summary, dict):
                continue
            expected = int(summary.get("Expected Fold Count", 0))
            if (
                summary.get("Status") != "complete"
                or summary.get("Completed Folds")
                != list(range(1, expected + 1))
            ):
                continue
            costs = {}
            for cost, cost_result in summary.get("Costs", {}).items():
                source_rows = cost_result.get("Selective", {}).get("Raw Folds", [])
                rows = []
                for fold_index, source in enumerate(source_rows, 1):
                    row = {"Fold": int(source.get("Fold", fold_index))}
                    for metric in BSS_SPCC_SELECTIVE_METRIC_NAMES:
                        value = _finite_float(source.get(metric))
                        if value is not None:
                            row[metric] = value
                    rows.append(row)
                costs[f"{float(cost):.2f}"] = {
                    "Selective": _summary_from_rows(
                        rows, BSS_SPCC_SELECTIVE_METRIC_NAMES
                    )
                }
            mlc_datasets[dataset] = {
                "Full": _canonical_v3_full(summary),
                "Costs": costs,
            }
            used = True
        if used:
            mlc_sources.append({
                "Path": str(result_path.resolve()),
                "SHA-256": _sha256(result_path),
                "Config Hash": document.get("Config Hash"),
                "Schema": document.get("Schema Version"),
                "Model Key": "MLC_PA_Logistic",
                "Verified Settings": settings,
            })
    bundle["Models"]["MLC_PA_Logistic"] = {
        "Kind": "selective",
        "Source": mlc_sources,
        "Datasets": mlc_datasets,
    }
    missing_instance_f1 = []
    expected_cost_keys = [
        f"{float(value):.2f}" for value in expected_settings["abstention_costs"]
    ]
    for dataset in datasets:
        data = mlc_datasets.get(dataset, {})
        if not data or not all(
            "Selective Instance-F1"
            in data.get("Costs", {})
            .get(cost_key, {})
            .get("Selective", {})
            .get("Mean", {})
            for cost_key in expected_cost_keys
        ):
            missing_instance_f1.append(dataset)
    bundle["Missing Selective Instance-F1"] = {
        "MLC_PA_Logistic": missing_instance_f1
    }
    if missing_instance_f1:
        bundle["Warnings"].append(
            "MLC_PA Selective Instance-F1 is unavailable for: "
            + ", ".join(missing_instance_f1)
        )

    cc_datasets = {}
    cc_sources = []
    for result_path, document in _v3_model_documents(
        config["cc_logistic_output_dir"], "CC_Logistic"
    ):
        settings = document.get("Settings", {})
        if (
            int(settings.get("n_splits", -1)) != int(expected_settings["n_splits"])
            or int(settings.get("random_state", -1))
            != int(expected_settings["random_state"])
        ):
            continue
        used = False
        for dataset in datasets:
            summary = document.get("Results", {}).get(dataset, {}).get(
                "CC_Logistic"
            )
            if not isinstance(summary, dict):
                continue
            expected = int(summary.get("Expected Fold Count", 0))
            if (
                summary.get("Status") != "complete"
                or summary.get("Completed Folds")
                != list(range(1, expected + 1))
            ):
                continue
            cc_datasets[dataset] = {
                "Full": _canonical_v3_full(summary),
                "Costs": {},
            }
            used = True
        if used:
            cc_sources.append({
                "Path": str(result_path.resolve()),
                "SHA-256": _sha256(result_path),
                "Config Hash": document.get("Config Hash"),
                "Schema": document.get("Schema Version"),
                "Model Key": "CC_Logistic",
            })
    bundle["Models"]["CC_Logistic"] = {
        "Kind": "complete_only",
        "Source": cc_sources,
        "Datasets": cc_datasets,
    }

    for model in COMPARISON_MODEL_ORDER:
        available = set(bundle["Models"][model]["Datasets"])
        bundle["Missing"][model] = [
            dataset for dataset in datasets if dataset not in available
        ]
    return bundle
