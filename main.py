"""Resumable benchmark pipeline for BR, CC, MLC-PA, and GSI-MLC-PA."""

import argparse
import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import MaxAbsScaler

from src.data.loader import DATASET_CONFIG, load_dataset
from src.evaluation.cache import (
    backup_model_cache,
    import_legacy_model_results,
    load_legacy_results,
    load_model_cache,
    save_legacy_results,
    save_model_cache,
)
from src.evaluation.cv import get_multilabel_cv
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_partial_abstention_metrics,
)
from src.models.binary_relevance import (
    BinaryRelevanceClassifier,
    BinaryRelevanceLogisticRegression,
    BinaryRelevanceMLP,
)
from src.models.classifier_chain import ClassifierChainClassifier
from src.models.gsi_mlc_pa import GSIMLCPartialAbstentionClassifier
from src.models.mlc_pa import MLCPartialAbstentionClassifier
from src.models.base_learners import base_learner_manifest
from src.models.registry import (
    MATCHED_MODEL_IDS,
    canonical_registered_model_id,
    create_registered_model,
    get_model_spec,
    is_registered_model_id,
    model_base_learner,
    model_family,
)
from src.visualization.plots import generate_pa_plots


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


DEFAULT_ABSTENTION_COSTS = (0.20, 0.25, 0.30, 0.35, 0.40)
DEFAULT_V3_ABSTENTION_COSTS = (*DEFAULT_ABSTENTION_COSTS, 0.50)


def _is_selective_model(model_name):
    try:
        return model_family(model_name) in ("MLC_PA", "GSI_MLC_PA")
    except ValueError:
        return False


def _is_gsi_model(model_name):
    try:
        return model_family(model_name) == "GSI_MLC_PA"
    except ValueError:
        return False


def _cost_key(cost):
    return f"{float(cost):.2f}"


def _create_model(
    model_name,
    random_state=42,
    abstention_cost=0.3,
    abstention_penalty="linear",
    mlc_pa_base="mlp",
    gsi_validation_size=0.2,
    gsi_selection_objective="full_macro_f1",
    gsi_decision_policy="hamming",
    gsi_beta=1.0,
    gsi_penalty="linear",
    gsi_partition_mode="learned",
    gsi_partition_random_state=None,
    gsi_fixed_independent_labels=None,
    gsi_final_order="correlation",
):
    """Create a supported model from its standardized name."""
    model_key = model_name.upper()
    if is_registered_model_id(model_name):
        return create_registered_model(
            model_name,
            random_state=random_state,
            abstention_cost=abstention_cost,
            abstention_penalty=abstention_penalty,
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
    if model_key in ("BR", "BR_SVC", "BR_LINEARSVC"):
        return BinaryRelevanceClassifier(
            base_estimator="svm", random_state=random_state
        )
    if model_key in ("BR_LOGISTIC", "BR_LR", "BR_LOGREG"):
        return BinaryRelevanceLogisticRegression(random_state=random_state)
    if model_key in ("BR_MLP", "BR_NEURAL_NETWORK", "BR_NN"):
        return BinaryRelevanceMLP(random_state=random_state)
    if model_key in ("CC", "CC_SVC", "CC_LINEARSVC"):
        return ClassifierChainClassifier(
            base_estimator="svm", random_state=random_state
        )
    if model_key in ("CC_LOGISTIC", "CC_LR", "CC_LOGREG"):
        return ClassifierChainClassifier(
            base_estimator="logistic", random_state=random_state
        )
    if model_key in ("CC_MLP", "CC_NEURAL_NETWORK", "CC_NN"):
        return ClassifierChainClassifier(
            base_estimator="mlp", random_state=random_state
        )
    if model_key in ("MLC_PA", "MLCPA", "MLC_PARTIAL_ABSTENTION"):
        return MLCPartialAbstentionClassifier(
            base_estimator=mlc_pa_base,
            cost=abstention_cost,
            penalty=abstention_penalty,
            random_state=random_state,
        )
    if model_key in (
        "GSI_MLC_PA",
        "GSIMLCPA",
        "GSI_MLC_PARTIAL_ABSTENTION",
    ):
        return GSIMLCPartialAbstentionClassifier(
            cost=abstention_cost,
            validation_size=gsi_validation_size,
            random_state=random_state,
            selection_objective=gsi_selection_objective,
            decision_policy=gsi_decision_policy,
            beta=gsi_beta,
            penalty=gsi_penalty,
            partition_mode=gsi_partition_mode,
            partition_random_state=gsi_partition_random_state,
            fixed_independent_labels=gsi_fixed_independent_labels,
            final_order=gsi_final_order,
        )
    raise ValueError(
        f"Unknown model name: {model_name}. Registered: {MATCHED_MODEL_IDS}; "
        "legacy aliases BR, CC, MLC_PA and GSI_MLC_PA remain supported."
    )


def _standardize_model_name(name):
    model_key = name.upper()
    if model_key in ("BR", "BR_SVC", "BR_LINEARSVC"):
        return "BR"
    if model_key in ("CC", "CC_SVC", "CC_LINEARSVC"):
        return "CC"
    if model_key in ("MLC_PA", "MLCPA", "MLC_PARTIAL_ABSTENTION"):
        return "MLC_PA"
    if model_key in (
        "GSI_MLC_PA",
        "GSIMLCPA",
        "GSI_MLC_PARTIAL_ABSTENTION",
    ):
        return "GSI_MLC_PA"
    if is_registered_model_id(name):
        return canonical_registered_model_id(name)
    return name


def _canonicalize_dataset_name(name):
    """Normalize CLI spelling/case and a known Genbase transposition typo."""
    normalized = str(name).strip().lower().replace("_", "-")
    aliases = {
        "gengase": "genbase",
        "reuters_k500": "reuters-k500",
    }
    return aliases.get(normalized, normalized)


def _cache_settings(
    model_name,
    n_splits,
    random_state,
    abstention_costs,
    report_cost,
    abstention_penalty,
    mlc_pa_base,
    gsi_validation_size,
    gsi_selection_objective="full_macro_f1",
    gsi_decision_policy="hamming",
    gsi_beta=1.0,
    gsi_penalty="linear",
    gsi_partition_mode="learned",
    gsi_partition_random_state=None,
    gsi_fixed_independent_labels=None,
    gsi_final_order="correlation",
):
    settings = {
        "n_splits": int(n_splits),
        "random_state": int(random_state),
    }
    family = model_family(model_name)
    configured_base = model_base_learner(
        model_name, legacy_mlc_pa_base=mlc_pa_base
    )
    if family == "MLC_PA":
        settings.update({
            "abstention_costs": [float(cost) for cost in abstention_costs],
            "report_cost": float(report_cost),
            "abstention_penalty": abstention_penalty,
            "base_estimator": configured_base,
            "target_loss": "generalized_hamming",
            "f1_abstention_policy": "full_separate_selective_ignore",
        })
    elif family == "GSI_MLC_PA":
        from src.selection import canonical_selection_objective

        canonical_objective = canonical_selection_objective(
            gsi_selection_objective
        )
        settings.update({
            "abstention_costs": [float(cost) for cost in abstention_costs],
            "report_cost": float(report_cost),
            "base_estimators": (
                f"BR_{str(configured_base).upper()}+"
                f"CC_{str(configured_base).upper()}"
            ),
            "validation_size": float(gsi_validation_size),
            "selection_objective": (
                "complete_macro_f1"
                if canonical_objective == "full_macro_f1"
                else canonical_objective
            ),
            "selection_strategy": "sequential_single_pass",
            "correlation_timing": "after_il_dl_selection",
            "correlation_measure": "phi_pearson_binary",
            "f1_abstention_policy": "full_separate_selective_ignore",
            "marginalization": "two_state_single_parent+mean_field_multi_parent",
            "chain_order": "post_selection_correlation",
            "refit_after_selection": True,
        })
        if (
            canonical_objective != "full_macro_f1"
            or gsi_decision_policy != "hamming"
            or float(gsi_beta) != 1.0
            or gsi_penalty != "linear"
            or gsi_partition_mode != "learned"
            or gsi_partition_random_state is not None
            or gsi_fixed_independent_labels is not None
            or gsi_final_order != "correlation"
        ):
            settings.update({
                "selection_objective_canonical": canonical_objective,
                "decision_policy": gsi_decision_policy,
                "decision_beta": float(gsi_beta),
                "decision_penalty": gsi_penalty,
                "partition_mode": gsi_partition_mode,
                "partition_random_state": (
                    None
                    if gsi_partition_random_state is None
                    else int(gsi_partition_random_state)
                ),
                "fixed_independent_labels": (
                    None
                    if gsi_fixed_independent_labels is None
                    else [int(label) for label in gsi_fixed_independent_labels]
                ),
                "final_order_strategy": gsi_final_order,
            })
    if is_registered_model_id(model_name):
        spec = get_model_spec(model_name)
        settings["model_manifest"] = {
            "model_id": spec.model_id,
            "family": spec.family,
            "base_learner": base_learner_manifest(spec.base_learner),
        }
    return settings


def _selective_cache_is_compatible(cached_settings, expected_settings):
    """Validate caches whose predictions depend on selection/loss settings."""
    if not cached_settings:
        return False
    return all(
        cached_settings.get(key) == value
        for key, value in expected_settings.items()
    )


def _evaluate_model_v3(
    model_name,
    classifier,
    x_test,
    y_test,
    abstention_costs,
    label_names=None,
    critical_labels=None,
    label_policy=None,
):
    """Evaluate one fitted model with isolated schema-v3 metric scopes."""

    from src.decision import create_configured_policy
    from src.evaluation.metric_facade import compute_metric_bundle
    from src.evaluation.deployment import compute_deployment_metrics

    if critical_labels is None and label_policy is not None:
        critical_labels = label_policy.get("critical_labels") or None

    model_metadata = {}
    if hasattr(classifier, "experiment_manifest_"):
        model_metadata["Experiment Manifest"] = classifier.experiment_manifest_
    if hasattr(classifier, "calibration_audit_"):
        model_metadata["Calibration Audit"] = classifier.calibration_audit_
    label_groups = {}
    if _is_gsi_model(model_name):
        independent = [int(label) for label in classifier.independent_labels_]
        dependent = [int(label) for label in classifier.dependent_labels_]
        label_groups = {"IL": independent, "DL": dependent}
        model_metadata.update({
            "Independent Labels": independent,
            "Dependent Labels": dependent,
            "Independent Label Count": int(len(independent)),
            "Dependent Label Count": int(len(dependent)),
            "Partition Mode": getattr(classifier, "partition_mode_", "learned"),
            "Partition Audit": getattr(classifier, "partition_audit_", {}),
            "Reference Independent Labels": getattr(
                classifier, "reference_independent_labels_", None
            ),
            "Final Order Strategy": getattr(
                classifier, "final_order_strategy_", "correlation"
            ),
            "Selection Order": [
                int(label)
                for label in getattr(classifier, "selection_order_", [])
            ],
            "Correlation Order": [
                int(label)
                for label in getattr(classifier, "correlation_order_", [])
            ],
            "Final Order": [
                int(label) for label in getattr(classifier, "order_", [])
            ],
            "Selection Seconds": float(
                getattr(classifier, "selection_time_seconds_", 0.0)
            ),
            "Validation Full Macro-F1": float(
                getattr(
                    classifier,
                    "validation_full_macro_f1_",
                    classifier.validation_objective_,
                )
            ),
            "Validation Objective Score": float(classifier.validation_objective_),
            "Selection Objective": getattr(
                classifier, "selection_objective_name_", "full_macro_f1"
            ),
            "Selection Config": getattr(classifier, "selection_config_", {}),
            "Selection History": getattr(classifier, "selection_history_", []),
        })

    if _is_selective_model(model_name):
        inference_started = time.perf_counter()
        probabilities = np.asarray(classifier.predict_proba(x_test), dtype=np.float64)
        inference_seconds = float(time.perf_counter() - inference_started)
        full_prediction = classifier.predict_full_from_proba(probabilities)
        acceptance_confidence = np.abs(probabilities - 0.5) * 2.0
        decision_policy = create_configured_policy(
            getattr(classifier, "decision_policy", "hamming"),
            cost=getattr(classifier, "cost", 0.3),
            penalty=getattr(classifier, "penalty", "linear"),
            beta=getattr(classifier, "beta", 1.0),
            allow_abstention=True,
            abstain_value=classifier.abstain_value,
            hamming_boundary=(
                "symmetric_thresholds"
                if _is_gsi_model(model_name)
                else "minimum_loss"
            ),
        )
        model_metadata["Decision Policy"] = decision_policy.get_config()
        if _is_gsi_model(model_name):
            model_metadata["Probability Inference Seconds"] = inference_seconds
    else:
        probabilities = (
            np.asarray(classifier.predict_proba(x_test), dtype=np.float64)
            if hasattr(classifier, "predict_proba")
            else None
        )
        acceptance_confidence = None
        full_prediction = classifier.predict(x_test)
        decision_policy = None

    full_bundle = compute_metric_bundle(
        y_test,
        full_prediction,
        acceptance_confidence=acceptance_confidence,
        label_names=label_names,
        label_groups=label_groups,
        critical_labels=critical_labels,
    )
    result = {
        "Schema Version": 3,
        "Metric Contract Version": full_bundle["Metric Contract Version"],
        "Full": dict(full_bundle["Full"]),
        "Per Label": full_bundle["Per Label"],
        "Groups": full_bundle["Groups"],
        "Critical Labels": full_bundle["Critical Labels"],
        "Model Metadata": model_metadata,
        "Costs": {},
    }
    if probabilities is not None:
        from src.evaluation.calibration_metrics import compute_calibration_metrics

        result["Calibration"] = compute_calibration_metrics(
            y_test,
            probabilities,
            label_names=label_names,
        )
    else:
        result["Calibration"] = {
            "Status": "Unavailable",
            "Reason": "classifier_has_no_predict_proba",
        }
    if not _is_selective_model(model_name):
        return result

    for cost in abstention_costs:
        partial_prediction = decision_policy.predict_from_proba(
            probabilities, cost=cost
        )
        bundle = compute_metric_bundle(
            y_test,
            full_prediction,
            y_partial=partial_prediction,
            cost=cost,
            penalty=getattr(classifier, "penalty", "linear"),
            abstain_value=classifier.abstain_value,
            acceptance_confidence=acceptance_confidence,
            label_names=label_names,
            label_groups=label_groups,
            critical_labels=critical_labels,
        )
        result["Costs"][_cost_key(cost)] = {
            "Selective": dict(bundle["Selective"]),
            "Rejected": bundle["Rejected"],
            "Optimistic": bundle["Optimistic"],
            "Diagnostics": bundle["Diagnostics"],
            "Per Label": bundle["Per Label"],
            "Groups": bundle["Groups"],
            "Critical Labels": bundle["Critical Labels"],
            "Deployment": compute_deployment_metrics(
                y_test,
                full_prediction,
                partial_prediction,
                cost=cost,
                penalty=getattr(classifier, "penalty", "linear"),
                abstain_value=classifier.abstain_value,
                label_names=label_names,
                label_policy=label_policy,
                acceptance_confidence=acceptance_confidence,
            ),
        }
    return result


def _evaluate_model(
    model_name,
    classifier,
    x_test,
    y_test,
    abstention_costs,
    metric_schema=2,
    label_names=None,
    critical_labels=None,
    label_policy=None,
):
    """Evaluate one fitted model, applying rejection only after training."""
    if metric_schema == 3:
        return _evaluate_model_v3(
            model_name,
            classifier,
            x_test,
            y_test,
            abstention_costs,
            label_names=label_names,
            critical_labels=critical_labels,
            label_policy=label_policy,
        )
    if metric_schema != 2:
        raise ValueError("metric_schema must be either 2 or 3.")
    if not _is_selective_model(model_name):
        return {
            "full": compute_all_metrics(y_test, classifier.predict(x_test)),
            "costs": {},
        }

    # One probability pass feeds both the complete output and every operating
    # cost.  No cost can affect fit(), IL/DL selection, or correlation.
    inference_started = time.perf_counter()
    test_probabilities = classifier.predict_proba(x_test)
    inference_seconds = float(time.perf_counter() - inference_started)
    full_prediction = classifier.predict_full_from_proba(test_probabilities)
    full_metrics = compute_all_metrics(y_test, full_prediction)
    if _is_gsi_model(model_name):
        full_metrics.update({
            "Independent Label Count": float(len(classifier.independent_labels_)),
            "Dependent Label Count": float(len(classifier.dependent_labels_)),
            "Validation Full Macro-F1": float(classifier.validation_objective_),
            "Selection Time Seconds": float(
                getattr(classifier, "selection_time_seconds_", 0.0)
            ),
            "Inference Time Seconds": inference_seconds,
        })

    cost_metrics = {}
    for cost in abstention_costs:
        partial_prediction = classifier.predict_from_proba(
            test_probabilities, cost=cost
        )
        cost_metrics[_cost_key(cost)] = compute_partial_abstention_metrics(
            y_test,
            partial_prediction,
            cost=cost,
            abstain_value=classifier.abstain_value,
            penalty=getattr(classifier, "penalty", "linear"),
        )
    return {"full": full_metrics, "costs": cost_metrics}


def _validated_costs(abstention_costs, report_cost, *, result_schema=2):
    costs = (
        list(
            DEFAULT_V3_ABSTENTION_COSTS
            if result_schema == 3
            else DEFAULT_ABSTENTION_COSTS
        )
        if abstention_costs is None
        else [float(cost) for cost in abstention_costs]
    )
    if not costs:
        raise ValueError("abstention_costs must contain at least one value.")
    if any(not 0.0 <= cost <= 0.5 for cost in costs):
        raise ValueError("Every abstention cost must lie in [0, 0.5].")
    costs = sorted(set(costs))
    report_cost = float(report_cost)
    if not any(np.isclose(report_cost, cost) for cost in costs):
        raise ValueError("report_cost must be included in abstention_costs.")
    return costs, report_cost


def _normalize_cached_dataset_result(result):
    """Wrap a legacy mean/std/raw_folds result in the schema-v2 shape."""
    if not isinstance(result, dict):
        return result
    if "full" in result and "costs" in result:
        return result
    if "mean" in result and "std" in result:
        return {"full": result, "costs": {}}
    return result


def _summarize_fold_results(fold_results, abstention_costs):
    full_frame = pd.DataFrame([fold_result["full"] for fold_result in fold_results])
    summary = {
        "full": {
            "mean": full_frame.mean(numeric_only=True).to_dict(),
            "std": full_frame.std(numeric_only=True).fillna(0.0).to_dict(),
            "raw_folds": [fold_result["full"] for fold_result in fold_results],
        },
        "costs": {},
    }
    for cost in abstention_costs:
        key = _cost_key(cost)
        rows = [
            fold_result["costs"][key]
            for fold_result in fold_results
            if key in fold_result["costs"]
        ]
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        summary["costs"][key] = {
            "mean": frame.mean(numeric_only=True).to_dict(),
            "std": frame.std(numeric_only=True).fillna(0.0).to_dict(),
            "raw_folds": rows,
        }
    return summary


def run_experiment(
    datasets=None,
    models=None,
    n_splits=5,
    random_state=42,
    output_dir=None,
    abstention_costs=None,
    report_cost=0.3,
    abstention_penalty="linear",
    mlc_pa_base="mlp",
    gsi_validation_size=0.2,
    gsi_selection_objective="full_macro_f1",
    gsi_decision_policy="hamming",
    gsi_beta=1.0,
    gsi_penalty="linear",
    result_schema=2,
    critical_labels=None,
    max_new_folds=None,
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
    """Run only missing model/dataset pairs and then rebuild all plots.

    Schema v2 remains the default and preserves its existing caches/plots.
    Schema v3 is opt-in, writes to an isolated directory, and checkpoints each
    completed fold before continuing.
    """
    if result_schema not in (2, 3):
        raise ValueError("result_schema must be either 2 or 3.")
    if result_schema == 2 and max_new_folds is not None:
        raise ValueError("max_new_folds is available only with result_schema=3.")
    if result_schema == 2 and critical_labels is not None:
        raise ValueError("critical_labels is available only with result_schema=3.")
    if result_schema == 2 and label_policy_path is not None:
        raise ValueError("label_policy_path is available only with result_schema=3.")
    if result_schema == 2 and operating_point_rule not in (None, "none"):
        raise ValueError(
            "operating_point_rule is available only with result_schema=3."
        )
    if output_dir is None:
        output_dir = "results_pa_v3" if result_schema == 3 else "results_pa"
    if datasets is None:
        datasets = list(DATASET_CONFIG.keys())
    else:
        datasets = [_canonicalize_dataset_name(name) for name in datasets]
    if models is None:
        models = list(MATCHED_MODEL_IDS)
    abstention_costs, report_cost = _validated_costs(
        abstention_costs, report_cost, result_schema=result_schema
    )

    standardized_models = [_standardize_model_name(model) for model in models]
    if len(standardized_models) != len(set(standardized_models)):
        raise ValueError("The model list contains duplicate aliases.")
    unknown_datasets = sorted(set(datasets) - set(DATASET_CONFIG))
    if unknown_datasets:
        raise ValueError(f"Unknown datasets: {unknown_datasets}")

    if result_schema == 3:
        from src.evaluation.pipeline_v3 import run_experiment_v3

        return run_experiment_v3(
            datasets=datasets,
            models=standardized_models,
            n_splits=n_splits,
            random_state=random_state,
            output_dir=output_dir,
            abstention_costs=abstention_costs,
            report_cost=report_cost,
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
            dataset_loader=load_dataset,
            cv_factory=get_multilabel_cv,
            model_factory=_create_model,
            evaluator=_evaluate_model,
            critical_labels=critical_labels,
            max_new_folds=max_new_folds,
            label_policy_path=label_policy_path,
            operating_point_rule=operating_point_rule,
            operating_coverage_gamma=operating_coverage_gamma,
            operating_risk_epsilon=operating_risk_epsilon,
            operating_validation_size=operating_validation_size,
        )

    figures_dir = os.path.join(output_dir, "plots_pa")
    tables_dir = os.path.join(output_dir, "tables")
    legacy_json_path = os.path.join(tables_dir, "raw_results.json")
    legacy_results = load_legacy_results(legacy_json_path)
    if not legacy_results:
        shared_legacy_path = os.path.join("results", "tables", "raw_results.json")
        if (
            os.path.abspath(shared_legacy_path) != os.path.abspath(legacy_json_path)
            and os.path.exists(shared_legacy_path)
        ):
            legacy_results = load_legacy_results(shared_legacy_path)
            print(
                f"Using shared legacy cache: {shared_legacy_path}",
                flush=True,
            )

    model_caches = {}
    for model_name in standardized_models:
        expected_settings = _cache_settings(
            model_name,
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
        )
        cache = load_model_cache(tables_dir, model_name)
        legacy_dataset_shape = any(
            isinstance(result, dict)
            and "mean" in result
            and "full" not in result
            for result in cache.get("datasets", {}).values()
        )
        cache["datasets"] = {
            dataset_name: _normalize_cached_dataset_result(result)
            for dataset_name, result in cache.get("datasets", {}).items()
        }
        if (
            _is_selective_model(model_name)
            and cache["datasets"]
            and not _selective_cache_is_compatible(
                cache.get("settings", {}), expected_settings
            )
        ):
            backup_path = backup_model_cache(
                tables_dir, model_name, reason="pre_cost_sweep_schema"
            )
            print(
                f"{model_name} cache settings differ; recomputing it. "
                f"Backup: {backup_path}",
                flush=True,
            )
            cache["datasets"] = {}

        # The legacy file has no selective-model cost/selection metadata, so
        # importing such rows could silently reuse an incompatible decision
        # configuration. BR/CC complete predictions do not have this ambiguity.
        imported = (
            import_legacy_model_results(
                legacy_results, model_name, source_model=model_name
            )
            if not _is_selective_model(model_name)
            else {}
        )
        imported_any = False
        for dataset_name, result in imported.items():
            if dataset_name not in cache["datasets"]:
                cache["datasets"][dataset_name] = _normalize_cached_dataset_result(
                    result
                )
                imported_any = True

        if imported_any or (
            legacy_dataset_shape and not _is_selective_model(model_name)
        ):
            if legacy_dataset_shape and not _is_selective_model(model_name):
                backup_model_cache(
                    tables_dir, model_name, reason="pre_schema_v2"
                )
            cache["settings"] = {
                **expected_settings,
                "source": "legacy_cache_or_raw_results",
            }
            cache_path = save_model_cache(
                tables_dir,
                model_name,
                cache["datasets"],
                settings=cache["settings"],
            )
            print(
                f"Migrated/imported cached {model_name} results -> {cache_path}",
                flush=True,
            )
        model_caches[model_name] = cache

    print("=" * 90, flush=True)
    print(" RESUMABLE MULTI-LABEL CLASSIFICATION BENCHMARK", flush=True)
    print(
        f" Models ({len(standardized_models)}): "
        f"{', '.join(standardized_models)}",
        flush=True,
    )
    print(f" Datasets ({len(datasets)}): {', '.join(datasets)}", flush=True)
    print(
        f" Evaluation: {n_splits}-Fold Multilabel Stratified Cross-Validation",
        flush=True,
    )
    print(f" Output Dir: {output_dir}", flush=True)
    print(f" Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 90, flush=True)

    total_start_time = time.time()
    for dataset_index, dataset_name in enumerate(datasets, 1):
        dataset_start_time = time.time()
        missing_models = [
            model_name
            for model_name in standardized_models
            if dataset_name not in model_caches[model_name]["datasets"]
        ]
        cached_models = [
            model_name
            for model_name in standardized_models
            if model_name not in missing_models
        ]
        print(
            f"\n[{dataset_index}/{len(datasets)}] Dataset: {dataset_name.upper()}",
            flush=True,
        )
        if cached_models:
            print(f"  -> Cached: {', '.join(cached_models)}", flush=True)
        if not missing_models:
            print("  -> All requested results are cached; skipping CV.", flush=True)
            continue
        print(f"  -> Running: {', '.join(missing_models)}", flush=True)

        try:
            x_data, y_data, _, _ = load_dataset(dataset_name)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Cannot compute missing results for {dataset_name}: {exc}. "
                "Existing caches were preserved; add the dataset files and rerun."
            ) from exc

        n_samples, n_features = x_data.shape
        n_labels = y_data.shape[1]
        cardinality = float(y_data.sum(axis=1).mean())
        density = float(y_data.mean())
        print(
            f"  -> Samples: {n_samples:,} | Features: {n_features:,} | "
            f"Labels: {n_labels} | Cardinality: {cardinality:.2f} | "
            f"Density: {density:.4f}",
            flush=True,
        )

        cv = get_multilabel_cv(
            n_splits=n_splits, random_state=random_state
        )
        splits = list(cv.split(x_data, y_data))
        fold_metrics = {model_name: [] for model_name in missing_models}
        fold_metadata = {model_name: [] for model_name in missing_models}

        for fold_index, (train_indices, test_indices) in enumerate(splits, 1):
            scaler = MaxAbsScaler()
            x_train = scaler.fit_transform(x_data[train_indices])
            x_test = scaler.transform(x_data[test_indices])
            y_train = y_data[train_indices]
            y_test = y_data[test_indices]

            for model_name in missing_models:
                started = time.time()
                classifier = _create_model(
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
                classifier.fit(x_train, y_train)
                metrics = _evaluate_model(
                    model_name,
                    classifier,
                    x_test,
                    y_test,
                    abstention_costs,
                )
                metrics["full"]["train_time"] = time.time() - started
                fold_metrics[model_name].append(metrics)
                if _is_gsi_model(model_name):
                    fold_metadata[model_name].append({
                        "fold": int(fold_index),
                        "independent_labels": [
                            int(label)
                            for label in classifier.independent_labels_
                        ],
                        "dependent_labels": [
                            int(label)
                            for label in classifier.dependent_labels_
                        ],
                        "partition_mode": classifier.partition_mode_,
                        "partition_audit": classifier.partition_audit_,
                        "final_order_strategy": classifier.final_order_strategy_,
                        "selection_time_seconds": float(
                            classifier.selection_time_seconds_
                        ),
                        "selection_history": classifier.selection_history_,
                        "selection_config": getattr(
                            classifier, "selection_config_", {}
                        ),
                        "selection_order": [
                            int(label) for label in classifier.selection_order_
                        ],
                        "correlation_order": [
                            int(label) for label in classifier.correlation_order_
                        ],
                        "final_order": [
                            int(label) for label in classifier.order_
                        ],
                        "dependent_parent_map": {
                            str(label): (
                                None if parent is None else int(parent)
                            )
                            for label, parent in classifier.dependent_parent_map_.items()
                        },
                        "label_correlation": (
                            classifier.label_correlation_.tolist()
                        ),
                        "selection_train_size": int(
                            classifier.selection_train_size_
                        ),
                        "validation_size": int(classifier.validation_size_),
                        "refit_train_size": int(classifier.refit_train_size_),
                        "evaluated_configurations": int(
                            classifier.evaluated_configurations_
                        ),
                    })
                print(
                    f"  -> Fold {fold_index}/{n_splits} {model_name} done",
                    flush=True,
                )

        for model_name in missing_models:
            dataset_result = _summarize_fold_results(
                fold_metrics[model_name], abstention_costs
            )
            if fold_metadata[model_name]:
                dataset_result["fold_metadata"] = fold_metadata[model_name]
            model_caches[model_name]["datasets"][dataset_name] = dataset_result
            settings = _cache_settings(
                model_name,
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
            )
            model_caches[model_name]["settings"] = settings
            cache_path = save_model_cache(
                tables_dir,
                model_name,
                model_caches[model_name]["datasets"],
                settings=settings,
            )
            print(f"  -> Saved cache: {cache_path}", flush=True)

        elapsed = time.time() - dataset_start_time
        summary = " | ".join(
            f"{model_name} Macro-F1: "
            f"{model_caches[model_name]['datasets'][dataset_name]['full']['mean']['Macro-F1']:.4f}"
            for model_name in missing_models
        )
        print(f"  -> Done in {elapsed:.2f}s | {summary}", flush=True)

    all_results = {
        dataset_name: {
            model_name: model_caches[model_name]["datasets"][dataset_name]
            for model_name in standardized_models
        }
        for dataset_name in datasets
    }

    # Preserve every legacy result, including models/datasets not selected now.
    combined_results = dict(legacy_results)
    for dataset_name, dataset_results in all_results.items():
        combined_results.setdefault(dataset_name, {}).update(dataset_results)
    save_legacy_results(legacy_json_path, combined_results)

    elapsed_total = time.time() - total_start_time
    print("\n" + "=" * 90, flush=True)
    print(
        f" EXPERIMENTS COMPLETED in {elapsed_total:.2f}s "
        f"({elapsed_total / 60:.1f} min)",
        flush=True,
    )
    print("=" * 90, flush=True)

    print(f"\nGenerating plots and summary tables in '{output_dir}'...", flush=True)
    generated_files, csv_path = generate_pa_plots(
        all_results,
        abstention_costs=abstention_costs,
        report_cost=report_cost,
        output_dir=figures_dir,
        tables_dir=tables_dir,
    )
    print(f"\nSaved CSV Results: {csv_path}", flush=True)
    print(f"Saved Raw JSON:    {legacy_json_path}", flush=True)
    print(f"Generated Figures ({len(generated_files)} files):", flush=True)
    for generated_file in generated_files:
        print(f"  - {generated_file}", flush=True)

    print("\n" + "=" * 110, flush=True)
    print(
        f"{'DATASET':<15} {'MODEL':<14} {'MACRO-F1':<18} "
        f"{'MICRO-F1':<18} {'HAMMING LOSS':<18} {'SUBSET ACC':<18}",
        flush=True,
    )
    print("-" * 110, flush=True)
    for dataset_name in datasets:
        for model_name in standardized_models:
            mean = all_results[dataset_name][model_name]["full"]["mean"]
            std = all_results[dataset_name][model_name]["full"]["std"]
            macro = f"{mean['Macro-F1']:.3f} +/- {std['Macro-F1']:.3f}"
            micro = f"{mean['Micro-F1']:.3f} +/- {std['Micro-F1']:.3f}"
            hamming = f"{mean['Hamming Loss']:.3f} +/- {std['Hamming Loss']:.3f}"
            subset = f"{mean['Subset Accuracy']:.3f} +/- {std['Subset Accuracy']:.3f}"
            print(
                f"{dataset_name.upper():<15} {model_name:<14} {macro:<18} "
                f"{micro:<18} {hamming:<18} {subset:<18}",
                flush=True,
            )
        print("-" * 110, flush=True)
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Resumable BR/CC/MLC-PA/GSI-MLC-PA multi-label benchmark pipeline"
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to evaluate; defaults to all configured datasets.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MATCHED_MODEL_IDS),
        help=(
            "Models to evaluate. Existing per-model or raw_results.json "
            "entries are reused. Defaults to the 12 matched Logistic/MLP/"
            "calibrated-SVM IDs."
        ),
    )
    parser.add_argument(
        "--n_splits", type=int, default=5, help="CV folds (default: 5)."
    )
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random seed (default: 42)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory; defaults to results_pa or isolated results_pa_v3.",
    )
    parser.add_argument(
        "--result_schema",
        type=int,
        choices=[2, 3],
        default=2,
        help="Result/cache schema. v3 enables fold checkpoints (default: 2).",
    )
    parser.add_argument(
        "--max_new_folds",
        type=int,
        default=None,
        help="Stop safely after this many newly completed v3 folds.",
    )
    parser.add_argument(
        "--critical_labels",
        nargs="+",
        default=None,
        help="Optional global critical label names for schema-v3 audit.",
    )
    parser.add_argument(
        "--label_policy_path",
        type=str,
        default=None,
        help="Dataset-specific critical-label/cost policy JSON for schema v3.",
    )
    parser.add_argument(
        "--operating_point_rule",
        choices=[
            "none",
            "min_generalized_loss",
            "max_utility_at_coverage",
            "max_coverage_at_risk",
        ],
        default="none",
        help="Select a cost on inner validation only (schema v3).",
    )
    parser.add_argument(
        "--operating_coverage_gamma",
        type=float,
        default=0.8,
        help="Minimum coverage for max-utility operating-point selection.",
    )
    parser.add_argument(
        "--operating_risk_epsilon",
        type=float,
        default=0.1,
        help="Maximum selective risk for max-coverage selection.",
    )
    parser.add_argument(
        "--operating_validation_size",
        type=float,
        default=0.2,
        help="Outer-train fraction reserved for operating-point selection.",
    )
    parser.add_argument(
        "--abstention_costs",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Decision-time costs (schema v2 default: 0.20..0.40; "
            "schema v3 also includes the no-abstention sanity point 0.50)."
        ),
    )
    parser.add_argument(
        "--report_cost",
        type=float,
        default=0.3,
        help="Cost used in the three per-dataset comparison figures.",
    )
    parser.add_argument(
        "--abstention_cost",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--abstention_penalty",
        choices=["linear", "concave"],
        default="linear",
        help="MLC-PA penalty: linear=SEP or concave=PAR.",
    )
    parser.add_argument(
        "--mlc_pa_base",
        choices=["mlp", "logistic", "svm"],
        default="logistic",
        help="Marginal probability estimator below MLC-PA (default: logistic).",
    )
    parser.add_argument(
        "--gsi_validation_size",
        type=float,
        default=0.2,
        help="Internal outer-train fraction for GSI IL/DL selection (default: 0.2).",
    )
    parser.add_argument(
        "--gsi_selection_objective",
        choices=[
            "full_macro_f1",
            "immediate_instance_f1",
            "bop_instance_f1",
            "bop_jaccard",
            "macro_precision",
            "macro_recall",
            "f_beta_0_5",
            "f_beta_2",
        ],
        default="full_macro_f1",
        help="Inner-validation objective for learned GSI partitions.",
    )
    parser.add_argument(
        "--gsi_decision_policy",
        choices=["hamming", "fbeta", "jaccard"],
        default="hamming",
        help="Final GSI decision policy used after probability inference.",
    )
    parser.add_argument(
        "--gsi_beta",
        type=float,
        default=1.0,
        help="Beta for the final GSI F-beta decision policy (default: 1).",
    )
    parser.add_argument(
        "--gsi_penalty",
        choices=["linear", "concave"],
        default="linear",
        help="Abstention penalty for GSI selection/final policy.",
    )
    parser.add_argument(
        "--gsi_partition_mode",
        choices=[
            "learned",
            "learned_no_correlation_order",
            "all_il",
            "all_dl",
            "fixed",
            "random_matched",
        ],
        default="learned",
        help="GSI IL/DL provider used for partition ablations.",
    )
    parser.add_argument(
        "--gsi_fixed_independent_labels",
        type=int,
        nargs="*",
        default=None,
        help="Zero-based IL indices required by --gsi_partition_mode fixed.",
    )
    parser.add_argument(
        "--gsi_partition_random_state",
        type=int,
        default=None,
        help=(
            "Seed used only for random-matched IL/DL sampling; outer folds, "
            "inner splits and estimator seeds remain controlled by --random_state."
        ),
    )
    parser.add_argument(
        "--gsi_final_order",
        choices=["correlation", "selection", "natural"],
        default="correlation",
        help="Final GSI chain order after the IL/DL partition is frozen.",
    )
    args = parser.parse_args()

    resolved_costs = args.abstention_costs
    resolved_report_cost = args.report_cost
    if args.abstention_cost is not None:
        resolved_costs = [args.abstention_cost]
        resolved_report_cost = args.abstention_cost

    run_experiment(
        datasets=args.datasets,
        models=args.models,
        n_splits=args.n_splits,
        random_state=args.random_state,
        output_dir=args.output_dir,
        abstention_costs=resolved_costs,
        report_cost=resolved_report_cost,
        abstention_penalty=args.abstention_penalty,
        mlc_pa_base=args.mlc_pa_base,
        gsi_validation_size=args.gsi_validation_size,
        gsi_selection_objective=args.gsi_selection_objective,
        gsi_decision_policy=args.gsi_decision_policy,
        gsi_beta=args.gsi_beta,
        gsi_penalty=args.gsi_penalty,
        gsi_partition_mode=args.gsi_partition_mode,
        gsi_partition_random_state=args.gsi_partition_random_state,
        gsi_fixed_independent_labels=args.gsi_fixed_independent_labels,
        gsi_final_order=args.gsi_final_order,
        result_schema=args.result_schema,
        critical_labels=args.critical_labels,
        max_new_folds=args.max_new_folds,
        label_policy_path=args.label_policy_path,
        operating_point_rule=args.operating_point_rule,
        operating_coverage_gamma=args.operating_coverage_gamma,
        operating_risk_epsilon=args.operating_risk_epsilon,
        operating_validation_size=args.operating_validation_size,
    )


if __name__ == "__main__":
    main()
