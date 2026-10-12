"""Run the frozen GSI-MLC-PA v6.3 single-stage ablation benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.data.loader import load_dataset
from src.decision.hamming import HammingBOPPolicy
from src.evaluation.calibration_metrics import compute_calibration_metrics
from src.evaluation.complete_metrics import compute_complete_metrics
from src.evaluation.cv import get_multilabel_cv
from src.evaluation.layer_metrics import (
    build_layer_rows,
    compute_v6_3_selective_metrics,
)
from src.evaluation.result_store_v6 import (
    V63ResultStore,
    sha256_file,
    stable_hash,
)
from src.models.base_learners import create_binary_estimator
from src.models.gsi_v6_3 import GSIMLCPAv6_3Classifier
from src.selection.single_stage_peeling import SingleStagePeelingSelector


MODEL_SETTINGS = {
    "GSI_v6_3_1L_ResidualBR": ("residual_br", "x_il"),
    "GSI_v6_3_CC_X": ("cc", "x"),
    "GSI_v6_3_PCC_X": ("pcc", "x"),
    "GSI_v6_3_CC_X_IL": ("cc", "x_il"),
    "GSI_v6_3_PCC_X_IL": ("pcc", "x_il"),
}

NON_ABSTAINING_BASELINES = {"BR", "CC"}

LEARNER_DISPLAY = {
    "logistic": "Logistic",
    "svm_calibrated": "SVM",
    "mlp": "MLP",
}


def _dataset_fingerprint(X, Y) -> str:
    digest = hashlib.sha256()
    for matrix in (np.asarray(X), np.asarray(Y)):
        digest.update(str(matrix.shape).encode("ascii"))
        digest.update(str(matrix.dtype).encode("ascii"))
        digest.update(np.ascontiguousarray(matrix).view(np.uint8))
    return digest.hexdigest()


def _mean_ir(Y) -> float:
    positives = np.asarray(Y, dtype=np.int64).sum(axis=0)
    valid = positives > 0
    if not np.any(valid):
        return float("nan")
    maximum = float(np.max(positives[valid]))
    return float(np.mean(maximum / positives[valid]))


def _split_hash(train_indices, test_indices) -> str:
    return stable_hash(
        {
            "train": [int(value) for value in train_indices],
            "test": [int(value) for value in test_indices],
        }
    )


def _complete_metrics(y_true, y_full):
    metrics = compute_complete_metrics(y_true, y_full)
    return {
        "hamming_accuracy": metrics["Hamming Accuracy"],
        "subset_accuracy": metrics["Subset Accuracy"],
        "micro_f1": metrics["Micro-F1"],
        "macro_f1": metrics["Macro-F1"],
        "instance_f1": metrics["Instance-F1"],
    }


def _calibration_metrics(y_true, probabilities, bins):
    result = compute_calibration_metrics(y_true, probabilities, n_bins=bins)
    return {
        "brier_score": result["Metrics"]["Brier Score"],
        "ece": result["Metrics"]["ECE"],
        "log_loss": result["Metrics"]["Log Loss"],
        "ece_bin_count": int(result["Bin Count"]),
    }


def _parent_map(model):
    if model.dl_inference == "residual_br":
        return {
            str(label): [int(parent) for parent in model.dependency_graph_.get(label, [])]
            for label in model.dependent_labels_
        }
    order = list(model.dependent_labels_)
    return {
        str(label): [int(parent) for parent in order[:index]]
        for index, label in enumerate(order)
    }


def _layer_payload(rows):
    # Common identity fields are supplied by the checkpoint flattener.
    excluded = {
        "version",
        "base_learner",
        "decision_policy",
        "model",
        "dataset",
        "mean_ir",
        "fold",
        "aggregation",
        "source_version",
        "provenance_status",
        "config_hash",
        "split_hash",
    }
    return [{key: value for key, value in row.items() if key not in excluded} for row in rows]


def _inherit_v6_2(
    *,
    store: V63ResultStore,
    config: dict,
    learner: str,
    dataset: str,
    mean_ir: float,
    dataset_fingerprint: str,
):
    source_path = WORKSPACE_ROOT / config["settings"]["inheritance_source"]
    source = pd.read_csv(source_path)
    source = source[
        (source["dataset"] == dataset)
        & (source["learner"] == LEARNER_DISPLAY[learner])
    ]
    source_hash = sha256_file(source_path)
    non_abstaining_models = NON_ABSTAINING_BASELINES
    for model_name in config["settings"]["inherited_models"]:
        model_rows = source[source["model"] == model_name].sort_values("fold")
        folds = []
        for _, row in model_rows.iterrows():
            complete = {
                "hamming_accuracy": 1.0 - float(row["Hamming_Loss"]),
                "subset_accuracy": float(row["Subset_Accuracy"]),
                "micro_f1": float(row["Full_Micro_F1"]),
                "macro_f1": float(row["Full_Macro_F1"]),
                "instance_f1": None,
            }
            selective = {
                "cost": float(config["settings"]["report_cost"]),
                "selective_hamming_accuracy": 1.0
                - float(row["Selective_Hamming_Loss"]),
                "selective_subset_accuracy": None,
                "selective_micro_f1": float(row["Selective_Micro_F1"]),
                "selective_macro_f1": float(row["Selective_Macro_F1"]),
                "selective_instance_f1": None,
                "generalized_loss": None,
                "coverage": float(row["Coverage"]),
                "abs": None,
                "aabs": None,
            }
            complete_reference = {
                "selective_hamming_accuracy": complete["hamming_accuracy"],
                "selective_subset_accuracy": complete["subset_accuracy"],
                "selective_micro_f1": complete["micro_f1"],
                "selective_macro_f1": complete["macro_f1"],
                "selective_instance_f1": complete["instance_f1"],
                # With zero abstentions, generalized Hamming loss is exactly
                # the ordinary complete-prediction Hamming loss.
                "generalized_loss": float(row["Hamming_Loss"]),
                "coverage": 1.0,
                "abs": 0.0,
                "aabs": 0.0,
            }
            folds.append(
                {
                    "fold": int(row["fold"]),
                    "status": "inherited",
                    "provenance_status": "inherited_unmatched",
                    "split_hash": "-",
                    "complete_metrics": complete,
                    # BR/CC remain visible as no-abstention references. Their
                    # values come strictly from complete metrics and are copied
                    # across the cost grid, so rejection-cost plots show a
                    # constant baseline rather than a simulated abstention run.
                    "selective_metrics": (
                        [
                            {"cost": float(cost), **complete_reference}
                            for cost in config["settings"]["abstention_costs"]
                        ]
                        if model_name in non_abstaining_models
                        else [selective]
                    ),
                    "calibration_metrics": {},
                    "layer_metrics": [
                        {
                            "layer_type": layer,
                            "layer_index": 1 if layer == "IL" else 0,
                            "label_indices": "-",
                            "label_count": None,
                            "cumulative_label_count": None,
                            "full_macro_f1": None,
                            "selective_macro_f1": None,
                            "coverage": None,
                            "cumulative_full_macro_f1": None,
                            "cumulative_selective_macro_f1": None,
                            "cumulative_coverage": None,
                            "status": "missing_inherited_metric",
                        }
                        for layer in ("IL", "DL")
                    ],
                    "timing": {
                        "fit_time_sec": float(row["train_time_sec"]),
                        "inference_time_sec": float(row["infer_time_sec"]),
                        "il_time_sec": None,
                        "dl_time_sec": None,
                    },
                    "il_count": None,
                    "dl_count": None,
                    "dl_parent_count": None,
                    "dl_parent_map": None,
                    "chain_order": None,
                }
            )
        payload = {
            "version": "v6_3",
            "source_version": "v6_2",
            "provenance_status": "inherited_unmatched",
            "status": "inherited" if folds else "missing",
            "base_learner": learner,
            "decision_policy": (
                "none_complete_reference"
                if model_name in non_abstaining_models
                else "hamming_bop"
            ),
            "model": model_name,
            "dataset": dataset,
            "mean_ir": mean_ir,
            "dataset_fingerprint": dataset_fingerprint,
            "expected_fold_count": int(config["settings"]["n_splits"]),
            "config": config,
            "config_hash": stable_hash(config),
            "source_artifact": str(source_path.relative_to(WORKSPACE_ROOT)).replace(
                "\\", "/"
            ),
            "source_sha256": source_hash,
            "source_split_hash": "-",
            "supports_abstention": model_name not in non_abstaining_models,
            "folds": folds,
        }
        store.save_inherited_checkpoint(
            learner=learner, model=model_name, dataset=dataset, payload=payload
        )


def _model_from_config(model_name: str, learner: str, settings: dict, seed: int):
    dl_inference, feature_space = MODEL_SETTINGS[model_name]
    return GSIMLCPAv6_3Classifier(
        base_learner=learner,
        dl_inference=dl_inference,
        feature_space=feature_space,
        stratified_threshold=settings["stratified_threshold"],
        residual_corr_threshold=settings["residual_corr_threshold"],
        cv_folds=settings["inner_folds"],
        peeling_metric=settings["peeling_metric"],
        aug_normalization=settings["augmentation_normalization"],
        pcc_num_samples=settings["pcc_num_samples"],
        pcc_sampling_batch_size=settings["pcc_sampling_batch_size"],
        cost=settings["report_cost"],
        abstention_penalty=settings["abstention_penalty"],
        linear_boundary=settings["hamming_boundary"],
        random_state=seed,
    )


def run(config_path: Path, datasets=None, learners=None, models=None, max_new_model_folds=None):
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    settings = config["settings"]
    datasets = list(datasets or settings["datasets"])
    learners = list(learners or settings["base_learners"])
    models = list(models or settings["new_models"])
    unknown = sorted(set(models) - set(MODEL_SETTINGS))
    if unknown:
        raise ValueError(f"Unknown v6.3 models: {unknown}")

    all_models = list(settings["inherited_models"]) + list(settings["new_models"])
    output_root = WORKSPACE_ROOT / settings["output_dir"]
    store = V63ResultStore(output_root, learners, all_models)
    new_model_fold_count = 0

    print("=" * 88, flush=True)
    print("GSI-MLC-PA v6.3: single-stage IL, shared-order CC/PCC ablation", flush=True)
    print(
        f"Datasets={datasets} | Learners={learners} | Models={models} | "
        f"PCC ancestral samples/test x={settings['pcc_num_samples']}",
        flush=True,
    )
    print("=" * 88, flush=True)

    for dataset in datasets:
        X, Y, _, label_names = load_dataset(dataset, base_dir=str(WORKSPACE_ROOT))
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int32)
        fingerprint = _dataset_fingerprint(X, Y)
        mean_ir = _mean_ir(Y)
        print(
            f"\n[{dataset}] n={X.shape[0]} d={X.shape[1]} L={Y.shape[1]} MeanIR={mean_ir:.4f}",
            flush=True,
        )

        split_provider = get_multilabel_cv(
            n_splits=settings["n_splits"],
            shuffle=True,
            random_state=settings["random_state"],
        )
        splits = list(split_provider.split(X, Y))

        for learner in learners:
            _inherit_v6_2(
                store=store,
                config=config,
                learner=learner,
                dataset=dataset,
                mean_ir=mean_ir,
                dataset_fingerprint=fingerprint,
            )
            print(f"  learner={learner}", flush=True)
            for fold_index, (train_indices, test_indices) in enumerate(splits, start=1):
                split_hash = _split_hash(train_indices, test_indices)
                pending = [
                    model
                    for model in models
                    if not store.is_fold_complete(
                        learner,
                        model,
                        dataset,
                        fold_index,
                        expected_config_hash=stable_hash(
                            {
                                "run": config,
                                "model": {
                                    "model": model,
                                    "dl_inference": MODEL_SETTINGS[model][0],
                                    "feature_space": MODEL_SETTINGS[model][1],
                                    # The order is fold-specific and audited separately.
                                    "shared_dl_order": None,
                                    "pcc_num_samples": (
                                        settings["pcc_num_samples"]
                                        if MODEL_SETTINGS[model][0] == "pcc"
                                        else None
                                    ),
                                },
                            }
                        ),
                        expected_split_hash=split_hash,
                    )
                ]
                if not pending:
                    print(f"    fold={fold_index}: resume hit", flush=True)
                    continue
                if max_new_model_folds is not None:
                    remaining = max_new_model_folds - new_model_fold_count
                    if remaining <= 0:
                        store.rebuild_tables(learner)
                        store.rebuild_all_tables()
                        return
                    pending = pending[:remaining]

                scaler = MaxAbsScaler()
                X_train = scaler.fit_transform(X[train_indices]).astype(np.float32)
                X_test = scaler.transform(X[test_indices]).astype(np.float32)
                Y_train, Y_test = Y[train_indices], Y[test_indices]
                fold_seed = int(settings["random_state"] + fold_index - 1)

                partition_start = perf_counter()
                selector = SingleStagePeelingSelector(
                    threshold=settings["stratified_threshold"],
                    n_folds=settings["inner_folds"],
                    cost=settings["report_cost"],
                    metric=settings["peeling_metric"],
                    aug_normalization=settings["augmentation_normalization"],
                    random_state=fold_seed,
                    base_estimator_factory=lambda learner=learner, fold_seed=fold_seed: create_binary_estimator(
                        learner, random_state=fold_seed
                    ),
                )
                partition = selector.fit_partition(X_train, Y_train)
                partition_time = float(perf_counter() - partition_start)
                shared_il_models = None
                shared_il_time = 0.0

                print(
                    f"    fold={fold_index}: IL={list(partition.all_independent_labels)} "
                    f"DL(shared order)={list(partition.dependent_residual_labels)}",
                    flush=True,
                )
                for model_name in pending:
                    model = _model_from_config(model_name, learner, settings, fold_seed)
                    fit_start = perf_counter()
                    model.fit(
                        X_train,
                        Y_train,
                        partition_result=partition,
                        shared_il_models=shared_il_models,
                    )
                    variant_fit_wall = float(perf_counter() - fit_start)
                    if shared_il_models is None:
                        shared_il_models = model.il_models_
                        shared_il_time = model.il_wall_time_seconds_

                    inference_start = perf_counter()
                    probabilities = model.predict_proba(X_test)
                    y_full = (probabilities >= 0.5).astype(np.int32)
                    partial_by_cost = {}
                    selective_metrics = []
                    for cost in settings["abstention_costs"]:
                        policy = HammingBOPPolicy(
                            cost=cost,
                            penalty=settings["abstention_penalty"],
                            linear_boundary=settings["hamming_boundary"],
                        )
                        partial = policy.predict(probabilities)
                        partial_by_cost[float(cost)] = partial
                        selective_metrics.append(
                            {
                                "cost": float(cost),
                                **compute_v6_3_selective_metrics(
                                    Y_test,
                                    partial,
                                    y_full,
                                    cost=cost,
                                    penalty=settings["abstention_penalty"],
                                ),
                            }
                        )
                    inference_time = float(perf_counter() - inference_start)

                    report_partial = partial_by_cost[float(settings["report_cost"])]
                    common = {
                        "version": "v6_3",
                        "base_learner": learner,
                        "decision_policy": "hamming_bop",
                        "model": model_name,
                        "dataset": dataset,
                        "mean_ir": mean_ir,
                        "fold": fold_index,
                        "aggregation": "fold",
                        "source_version": "v6_3",
                        "provenance_status": "computed",
                        "config_hash": stable_hash(config),
                        "split_hash": split_hash,
                    }
                    layers = build_layer_rows(
                        common=common,
                        y_true=Y_test,
                        y_full=y_full,
                        y_partial=report_partial,
                        independent_labels=model.independent_labels_,
                        dependent_labels=model.dependent_labels_,
                    )
                    parents = _parent_map(model)
                    parent_counts = [len(value) for value in parents.values()]
                    logical_il_time = partition_time + shared_il_time
                    logical_fit_time = partition_time + shared_il_time + model.dl_wall_time_seconds_
                    model_config = {
                        "model": model_name,
                        "dl_inference": model.dl_inference,
                        "feature_space": model.feature_space,
                        "shared_dl_order": list(model.dependent_labels_),
                        "pcc_num_samples": (
                            settings["pcc_num_samples"]
                            if model.dl_inference == "pcc"
                            else None
                        ),
                    }
                    hash_model_config = dict(model_config)
                    hash_model_config["shared_dl_order"] = None
                    checkpoint_base = {
                        "version": "v6_3",
                        "source_version": "v6_3",
                        "provenance_status": "computed",
                        "base_learner": learner,
                        "model": model_name,
                        "dataset": dataset,
                        "mean_ir": mean_ir,
                        "dataset_fingerprint": fingerprint,
                        "expected_fold_count": int(settings["n_splits"]),
                        "config": config,
                        "model_config": model_config,
                        "config_hash": stable_hash(
                            {"run": config, "model": hash_model_config}
                        ),
                        "pcc_num_samples": model_config["pcc_num_samples"],
                        "sampling_seed_rule": "random_state + outer_fold - 1",
                    }
                    fold_record = {
                        "fold": fold_index,
                        "provenance_status": "computed",
                        "split_hash": split_hash,
                        "train_size": int(len(train_indices)),
                        "test_size": int(len(test_indices)),
                        "partition": model.peeling_result_.as_dict(),
                        "il_labels": list(model.independent_labels_),
                        "dl_labels": list(model.dependent_labels_),
                        "partition_scores": (
                            model.peeling_result_.stage_diagnostics[0].get(
                                "stage_scores", {}
                            )
                        ),
                        "chain_order": (
                            list(model.dependent_labels_)
                            if model.dl_inference in ("cc", "pcc")
                            else None
                        ),
                        "pcc_sampling": (
                            {
                                "num_samples_per_test_x": settings["pcc_num_samples"],
                                "batch_size": settings["pcc_sampling_batch_size"],
                                "seed": fold_seed,
                            }
                            if model.dl_inference == "pcc"
                            else None
                        ),
                        "complete_metrics": _complete_metrics(Y_test, y_full),
                        "selective_metrics": selective_metrics,
                        "calibration_metrics": _calibration_metrics(
                            Y_test, probabilities, settings["calibration_bins"]
                        ),
                        "layer_metrics": _layer_payload(layers),
                        "timing": {
                            "fit_time_sec": logical_fit_time,
                            "actual_variant_fit_wall_sec": variant_fit_wall,
                            "inference_time_sec": inference_time,
                            "il_time_sec": logical_il_time,
                            "dl_time_sec": model.dl_wall_time_seconds_,
                            "shared_partition_time_sec": partition_time,
                            "shared_il_fit_time_sec": shared_il_time,
                        },
                        "il_count": len(model.independent_labels_),
                        "dl_count": len(model.dependent_labels_),
                        "dl_parent_count": (
                            float(np.mean(parent_counts)) if parent_counts else 0.0
                        ),
                        "dl_parent_map": parents,
                    }
                    fold_arrays = {
                        "test_indices": np.asarray(test_indices, dtype=np.int64),
                        "y_true": Y_test.astype(np.int8),
                        "probabilities": probabilities.astype(np.float32),
                        "y_full": y_full.astype(np.int8),
                        "fold_ids": np.full(len(test_indices), fold_index, dtype=np.int8),
                    }
                    for cost, partial in partial_by_cost.items():
                        fold_arrays[f"y_partial_c_{cost:.2f}"] = partial.astype(np.int8)
                    store.save_computed_fold(
                        learner=learner,
                        model=model_name,
                        dataset=dataset,
                        checkpoint_base=checkpoint_base,
                        fold_record=fold_record,
                        fold_arrays=fold_arrays,
                    )
                    new_model_fold_count += 1
                    print(
                        f"      {model_name}: saved (fit={logical_fit_time:.2f}s, "
                        f"infer={inference_time:.2f}s)",
                        flush=True,
                    )
                    if (
                        max_new_model_folds is not None
                        and new_model_fold_count >= max_new_model_folds
                    ):
                        store.rebuild_tables(learner)
                        store.rebuild_all_tables()
                        return
            store.rebuild_tables(learner)
        store.rebuild_all_tables()

    for learner in learners:
        store.rebuild_tables(learner)
    store.rebuild_all_tables()
    print(f"\n[DONE] v6.3 artifacts: {output_root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=WORKSPACE_ROOT / "configs" / "v6_3_run.json",
    )
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument(
        "--base-learners", nargs="+", choices=["logistic", "svm_calibrated", "mlp"]
    )
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_SETTINGS))
    parser.add_argument(
        "--max-new-model-folds",
        type=int,
        help="Stop after N newly computed model-folds (smoke/resume support).",
    )
    args = parser.parse_args()
    run(
        args.config,
        datasets=args.datasets,
        learners=args.base_learners,
        models=args.models,
        max_new_model_folds=args.max_new_model_folds,
    )


if __name__ == "__main__":
    main()
