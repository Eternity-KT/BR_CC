"""Validate and run the official BSS-UG-SPCC-PA Logistic experiment."""

import argparse
import json
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from main import run_experiment  # noqa: E402
from src.data.loader import DATASET_CONFIG  # noqa: E402
from src.evaluation.cache_v3 import compute_config_hash  # noqa: E402
from src.evaluation.bss_comparison import (  # noqa: E402
    COMPARISON_MODEL_ORDER,
    comparison_status,
)
from src.evaluation.metric_contract import (  # noqa: E402
    BSS_SPCC_METRIC_PROFILE_VERSION,
)


MODEL_PARAMETER_NAMES = (
    "inner_oof_splits",
    "alpha_grid",
    "ug_threshold",
    "parent_gain_threshold",
    "q_max",
    "epsilon",
    "alpha_tie_tolerance",
    "calibration",
    "calibration_splits",
    "calibration_tolerance",
)


def load_and_validate_config(path):
    """Load the primary config and fail closed on scientific drift."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("schema_version") != 1 or config.get("status") != "primary":
        raise ValueError("BSS config must use schema_version=1 and status=primary.")
    if config.get("models") != ["BSS_UG_SPCC_PA_Logistic"]:
        raise ValueError("Primary BSS config must contain its single registered model.")
    if config.get("base_learner") != "logistic":
        raise ValueError(
            "Primary BSS run is Logistic-only. Optional learners require a "
            "separate future config/model ID and output directory."
        )
    datasets = list(config.get("datasets", []))
    if not datasets or len(datasets) != len(set(datasets)):
        raise ValueError("datasets must be a non-empty unique list.")
    unknown = sorted(set(datasets) - set(DATASET_CONFIG))
    if unknown:
        raise ValueError(f"Unknown datasets in BSS config: {unknown}.")
    costs = [float(cost) for cost in config.get("abstention_costs", [])]
    if (
        not costs
        or costs != sorted(set(costs))
        or any(cost <= 0.0 or cost >= 0.5 for cost in costs)
    ):
        raise ValueError("abstention_costs must be unique, sorted and in (0, 0.5).")
    report_cost = float(config.get("report_cost"))
    if report_cost not in costs:
        raise ValueError("report_cost must occur in abstention_costs.")
    if config.get("abstention_penalty") != "linear":
        raise ValueError("The primary BSS run requires the linear SEP penalty.")
    if config.get("metric_profile") != BSS_SPCC_METRIC_PROFILE_VERSION:
        raise ValueError("Unsupported BSS metric_profile.")
    if int(config.get("n_splits", 0)) < 2:
        raise ValueError("n_splits must be at least 2.")
    comparison = config.get("comparison", {})
    if comparison.get("enabled") is not True:
        raise ValueError("The official run requires Logistic comparison figures.")
    if comparison.get("base_learner") != "logistic":
        raise ValueError("Comparison models must use the Logistic base learner.")
    if tuple(comparison.get("models", ())) != COMPARISON_MODEL_ORDER:
        raise ValueError(
            "Comparison order must be BR_Logistic, CC_Logistic, MLC_PA_Logistic."
        )
    for key in (
        "legacy_results_path",
        "mlc_pa_cache_path",
        "cc_logistic_output_dir",
        "mlc_pa_logistic_output_dir",
    ):
        if not comparison.get(key):
            raise ValueError(f"comparison.{key} is required.")
    if not isinstance(
        comparison.get("require_mlc_pa_selective_instance_f1", False), bool
    ):
        raise ValueError(
            "comparison.require_mlc_pa_selective_instance_f1 must be boolean."
        )
    return config


def model_parameters(config):
    return {
        name: config[name]
        for name in MODEL_PARAMETER_NAMES
        if name in config
    }


def resolved_comparison_config(config):
    comparison = dict(config["comparison"])
    for key in (
        "legacy_results_path",
        "mlc_pa_cache_path",
        "cc_logistic_output_dir",
        "mlc_pa_logistic_output_dir",
    ):
        path = Path(comparison[key])
        if not path.is_absolute():
            path = WORKSPACE_ROOT / path
        comparison[key] = str(path.resolve())
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Run/resume the official BSS-UG-SPCC-PA Logistic benchmark."
    )
    parser.add_argument(
        "--config",
        default="configs/bss_ug_spcc_pa.json",
        help="Scientific config JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output-directory override; must remain isolated.",
    )
    parser.add_argument(
        "--max-new-folds",
        type=int,
        default=None,
        help="Stop safely after this many newly checkpointed outer folds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the queue/hash without loading datasets.",
    )
    parser.add_argument(
        "--skip-comparison-preparation",
        action="store_true",
        help=(
            "Do not advance missing CC_Logistic/MLC_PA_Logistic checkpoints. "
            "Existing BR/CC/MLC-PA results are still loaded for figures."
        ),
    )
    args = parser.parse_args()

    config = load_and_validate_config(args.config)
    comparison = resolved_comparison_config(config)
    output_dir = args.output_dir or config["output_dir"]
    source_status = comparison_status(
        comparison,
        config["datasets"],
        n_splits=int(config["n_splits"]),
        random_state=int(config["random_state"]),
        abstention_costs=config["abstention_costs"],
        report_cost=float(config["report_cost"]),
        abstention_penalty=config["abstention_penalty"],
    )
    preview = {
        "config_path": str(Path(args.config).resolve()),
        "scientific_config_hash": compute_config_hash(config),
        "datasets": config["datasets"],
        "models": config["models"],
        "outer_folds": int(config["n_splits"]),
        "costs": config["abstention_costs"],
        "report_cost": config["report_cost"],
        "metric_profile": config["metric_profile"],
        "output_dir": str(Path(output_dir)),
        "max_new_folds": args.max_new_folds,
        "comparison": {
            "base_learner": "logistic",
            "sources": source_status,
            "prepare_missing_cc_logistic": bool(
                comparison.get("prepare_missing_cc_logistic", True)
                and not args.skip_comparison_preparation
            ),
            "prepare_missing_mlc_pa_logistic": bool(
                comparison.get("prepare_missing_mlc_pa_logistic", True)
                and not args.skip_comparison_preparation
            ),
            "require_mlc_pa_selective_instance_f1": bool(
                comparison.get("require_mlc_pa_selective_instance_f1", False)
            ),
        },
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    if args.dry_run:
        return preview

    missing_cc = source_status["CC_Logistic"]["missing_datasets"]
    if (
        missing_cc
        and comparison.get("prepare_missing_cc_logistic", True)
        and not args.skip_comparison_preparation
    ):
        print(
            "Advancing CC_Logistic comparison checkpoints for: "
            + ", ".join(missing_cc),
            flush=True,
        )
        run_experiment(
            datasets=config["datasets"],
            models=["CC_Logistic"],
            n_splits=int(config["n_splits"]),
            random_state=int(config["random_state"]),
            output_dir=comparison["cc_logistic_output_dir"],
            abstention_costs=config["abstention_costs"],
            report_cost=float(config["report_cost"]),
            abstention_penalty=config["abstention_penalty"],
            mlc_pa_base="logistic",
            result_schema=3,
            max_new_folds=args.max_new_folds,
        )
    missing_mlc = source_status["MLC_PA_Logistic"]["missing_datasets"]
    if (
        missing_mlc
        and comparison.get("prepare_missing_mlc_pa_logistic", True)
        and not comparison.get("require_mlc_pa_selective_instance_f1", False)
        and not args.skip_comparison_preparation
    ):
        print(
            "Advancing MLC_PA_Logistic comparison checkpoints for: "
            + ", ".join(missing_mlc),
            flush=True,
        )
        run_experiment(
            datasets=missing_mlc,
            models=["MLC_PA_Logistic"],
            n_splits=int(config["n_splits"]),
            random_state=int(config["random_state"]),
            output_dir=comparison["mlc_pa_logistic_output_dir"],
            abstention_costs=config["abstention_costs"],
            report_cost=float(config["report_cost"]),
            abstention_penalty=config["abstention_penalty"],
            mlc_pa_base="logistic",
            result_schema=3,
            max_new_folds=args.max_new_folds,
        )
    missing_mlc_instance_f1 = source_status["MLC_PA_Logistic"].get(
        "missing_selective_instance_f1_datasets", []
    )
    if (
        missing_mlc_instance_f1
        and comparison.get("require_mlc_pa_selective_instance_f1", False)
        and not args.skip_comparison_preparation
    ):
        print(
            "Advancing MLC_PA_Logistic Selective Instance-F1 checkpoints for: "
            + ", ".join(missing_mlc_instance_f1),
            flush=True,
        )
        run_experiment(
            datasets=missing_mlc_instance_f1,
            models=["MLC_PA_Logistic"],
            n_splits=int(config["n_splits"]),
            random_state=int(config["random_state"]),
            output_dir=comparison["mlc_pa_logistic_output_dir"],
            abstention_costs=config["abstention_costs"],
            report_cost=float(config["report_cost"]),
            abstention_penalty=config["abstention_penalty"],
            mlc_pa_base="logistic",
            result_schema=3,
            max_new_folds=args.max_new_folds,
            include_selective_instance_f1=True,
        )

    document = run_experiment(
        datasets=config["datasets"],
        models=config["models"],
        n_splits=int(config["n_splits"]),
        random_state=int(config["random_state"]),
        output_dir=output_dir,
        abstention_costs=config["abstention_costs"],
        report_cost=float(config["report_cost"]),
        abstention_penalty=config["abstention_penalty"],
        result_schema=3,
        max_new_folds=args.max_new_folds,
        bss_parameters=model_parameters(config),
        metric_profile=config["metric_profile"],
        refresh_artifacts_per_dataset=bool(
            config.get("refresh_artifacts_per_dataset", True)
        ),
        comparison_config=comparison,
    )
    print(
        json.dumps(
            {
                "status": document["Status"],
                "run_config_hash": document["Config Hash"],
                "artifacts": document.get("Artifacts", {}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return document


if __name__ == "__main__":
    main()
