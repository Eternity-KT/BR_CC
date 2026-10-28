"""Unified paper benchmark for BR, CC, MLC-PA, v6.3.3 and v6.3.5.

All models use Logistic Regression, identical iterative-stratified folds,
fold-local MaxAbs scaling, and random_state=42.  The output is the sole
quantitative source for the v6.3.5 LaTeX report.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MaxAbsScaler

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.data.loader import load_dataset
from src.evaluation.cv import get_multilabel_cv
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_partial_abstention_metrics,
    compute_selective_macro_f1,
    compute_selective_micro_f1,
)
from src.models.base_learners import create_binary_estimator, create_multilabel_estimator
from src.models.classifier_chain import ClassifierChainClassifier
from src.models.gsi_v6_3_3 import GSIMLCPAv6_3_3Classifier
from src.models.gsi_v6_3_5 import GSIMLCPAv6_3_5Classifier
from src.models.mlc_pa import MLCPartialAbstentionClassifier

DATASETS = [
    "emotions", "scene", "yeast", "plantpseaac", "humanpseaac",
    "chd49", "music", "gpositivepseaac", "genbase", "viruspseaac",
]


def _selective_class_metrics(y_true, y_partial):
    positive_f1 = []
    negative_f1 = []
    positive_precision = []
    positive_recall = []
    for label in range(y_true.shape[1]):
        decided = y_partial[:, label] != -1
        if not np.any(decided):
            positive_f1.append(0.0)
            negative_f1.append(0.0)
            positive_precision.append(0.0)
            positive_recall.append(0.0)
            continue
        truth = y_true[decided, label]
        pred = y_partial[decided, label]
        positive_f1.append(f1_score(truth, pred, pos_label=1, zero_division=0))
        negative_f1.append(f1_score(truth, pred, pos_label=0, zero_division=0))
        positive_precision.append(
            precision_score(truth, pred, pos_label=1, zero_division=0)
        )
        positive_recall.append(
            recall_score(truth, pred, pos_label=1, zero_division=0)
        )
    return {
        "Selective_Positive_Macro_F1": float(np.mean(positive_f1)),
        "Selective_Negative_Macro_F1": float(np.mean(negative_f1)),
        "Selective_Positive_Macro_Precision": float(np.mean(positive_precision)),
        "Selective_Positive_Macro_Recall": float(np.mean(positive_recall)),
    }


def evaluate(y_true, model, x_test, partial_model):
    if hasattr(model, "predict_full"):
        full = np.asarray(model.predict_full(x_test), dtype=np.int32)
    else:
        full = np.asarray(model.predict(x_test), dtype=np.int32)

    if partial_model:
        partial = np.asarray(model.predict(x_test), dtype=np.int32)
    else:
        partial = full.copy()

    decided = partial != -1
    rejected = ~decided
    full_errors = full != y_true
    full_error_count = int(np.sum(full_errors))
    rejected_count = int(np.sum(rejected))
    positive_positions = y_true == 1
    negative_positions = ~positive_positions
    full_metrics = compute_all_metrics(y_true, full)
    partial_metrics = compute_partial_abstention_metrics(
        y_true, partial, cost=0.30, penalty="linear"
    )
    selective_hamming = (
        float(np.mean(partial[decided] != y_true[decided]))
        if np.any(decided)
        else np.nan
    )

    result = {
        "Full_Macro_F1": float(full_metrics["Macro-F1"]),
        "Full_Micro_F1": float(full_metrics["Micro-F1"]),
        "Full_Example_F1": float(full_metrics["Example-F1"]),
        "Subset_Accuracy": float(full_metrics["Subset Accuracy"]),
        "Hamming_Loss": float(full_metrics["Hamming Loss"]),
        "Selective_Macro_F1": float(compute_selective_macro_f1(y_true, partial)),
        "Selective_Micro_F1": float(compute_selective_micro_f1(y_true, partial)),
        "Selective_Instance_F1": float(partial_metrics["Selective Instance-F1"]),
        "Selective_Hamming_Loss": selective_hamming,
        "Generalized_Loss": float(partial_metrics["Generalized Loss"]),
        "Coverage": float(np.mean(decided)),
        "Abstention_Rate": float(np.mean(rejected)),
        "Sample_Abstention_Rate_ABS": float(partial_metrics["ABS"]),
        "Positive_Coverage": float(np.mean(decided[positive_positions])),
        "Negative_Coverage": float(np.mean(decided[negative_positions])),
        "Error_Capture_Rate": (
            float(np.sum(full_errors & rejected) / full_error_count)
            if full_error_count
            else 0.0
        ),
        "Rejected_Error_Rate": (
            float(np.sum(full_errors & rejected) / rejected_count)
            if rejected_count
            else np.nan
        ),
    }
    result.update(_selective_class_metrics(y_true, partial))

    for group, indices in (
        ("IL", getattr(model, "independent_labels_", [])),
        ("DL", getattr(model, "dependent_labels_", [])),
    ):
        if indices:
            group_partial = partial[:, indices]
            result[f"{group}_Label_Count"] = len(indices)
            result[f"{group}_Coverage"] = float(np.mean(group_partial != -1))
            result[f"{group}_Selective_Macro_F1"] = float(
                compute_selective_macro_f1(y_true[:, indices], group_partial)
            )
        else:
            result[f"{group}_Label_Count"] = 0
            result[f"{group}_Coverage"] = np.nan
            result[f"{group}_Selective_Macro_F1"] = np.nan
    return result, full, partial


def run_benchmark(datasets, output_dir, n_splits=5, random_state=42):
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_rows = []
    per_label_rows = []
    dataset_rows = []

    for dataset in datasets:
        x_data, y_data, _, _ = load_dataset(dataset, base_dir=str(WORKSPACE_ROOT))
        if hasattr(x_data, "toarray"):
            x_data = x_data.toarray()
        x_data = np.asarray(x_data, dtype=np.float32)
        y_data = np.asarray(y_data, dtype=np.int32)
        priors = np.mean(y_data, axis=0)
        dataset_rows.append({
            "Dataset": dataset,
            "Samples": x_data.shape[0],
            "Features": x_data.shape[1],
            "Labels": y_data.shape[1],
            "Label_Cardinality": float(np.mean(np.sum(y_data, axis=1))),
            "Label_Density": float(np.mean(y_data)),
            "Mean_Positive_Prior": float(np.mean(priors)),
            "Min_Positive_Prior": float(np.min(priors)),
            "Max_Positive_Prior": float(np.max(priors)),
        })

        folds = list(
            get_multilabel_cv(n_splits=n_splits, random_state=random_state).split(
                x_data, y_data
            )
        )
        print(f"\n[{dataset}] N={x_data.shape[0]}, d={x_data.shape[1]}, K={y_data.shape[1]}", flush=True)
        for fold_index, (train_index, test_index) in enumerate(folds, start=1):
            scaler = MaxAbsScaler()
            x_train = scaler.fit_transform(x_data[train_index])
            x_test = scaler.transform(x_data[test_index])
            y_train = y_data[train_index]
            y_test = y_data[test_index]
            seed = random_state + fold_index - 1
            common_gsi = dict(
                base_learner="logistic", mf_alpha=0.25, mf_z_max=0.5,
                tau_balance=0.85, blend_kappa=20.0, cost=0.30,
                gamma_min=0.70, tau_1_cap=0.55, tau_0_floor=0.45,
                separation_aware=True, random_state=seed,
            )
            models = {
                "BR": (
                    create_multilabel_estimator("logistic", random_state=seed), False
                ),
                "CC": (
                    ClassifierChainClassifier(
                        base_estimator=create_binary_estimator(
                            "logistic", random_state=seed
                        ),
                        random_state=seed,
                    ),
                    False,
                ),
                "MLC_PA": (
                    MLCPartialAbstentionClassifier(
                        base_estimator="logistic", cost=0.30, random_state=seed
                    ),
                    True,
                ),
                "GSI_v6_3_3": (GSIMLCPAv6_3_3Classifier(**common_gsi), True),
                "GSI_v6_3_5": (GSIMLCPAv6_3_5Classifier(**common_gsi), True),
            }

            fold_scores = []
            for model_name, (model, partial_model) in models.items():
                started = time.perf_counter()
                model.fit(x_train, y_train)
                fit_time = time.perf_counter() - started
                started = time.perf_counter()
                metrics, full, partial = evaluate(
                    y_test, model, x_test, partial_model=partial_model
                )
                predict_time = time.perf_counter() - started
                detailed_rows.append({
                    "Dataset": dataset,
                    "Fold": fold_index,
                    "Model": model_name,
                    "Base_Learner": "Logistic",
                    "Fit_Time_s": fit_time,
                    "Predict_Time_s": predict_time,
                    **metrics,
                })
                fold_scores.append(f"{model_name}={metrics['Selective_Macro_F1']:.4f}")

                il_set = set(getattr(model, "independent_labels_", []))
                dl_set = set(getattr(model, "dependent_labels_", []))
                for label in range(y_data.shape[1]):
                    decided = partial[:, label] != -1
                    truth_decided = y_test[decided, label]
                    pred_decided = partial[decided, label]
                    group = "IL" if label in il_set else ("DL" if label in dl_set else "NA")
                    per_label_rows.append({
                        "Dataset": dataset,
                        "Fold": fold_index,
                        "Model": model_name,
                        "Label": label,
                        "Group": group,
                        "Train_Prior": float(np.mean(y_train[:, label])),
                        "Positive_Support": int(np.sum(y_test[:, label] == 1)),
                        "Negative_Support": int(np.sum(y_test[:, label] == 0)),
                        "Coverage": float(np.mean(decided)),
                        "F1_Positive": (
                            float(f1_score(truth_decided, pred_decided, pos_label=1, zero_division=0))
                            if np.any(decided) else 0.0
                        ),
                        "F1_Negative": (
                            float(f1_score(truth_decided, pred_decided, pos_label=0, zero_division=0))
                            if np.any(decided) else 0.0
                        ),
                    })
            print(f"  Fold {fold_index}/{n_splits}: " + " | ".join(fold_scores), flush=True)

    detailed = pd.DataFrame(detailed_rows)
    metric_columns = [
        column for column in detailed.columns
        if column not in {"Dataset", "Fold", "Model", "Base_Learner"}
    ]
    summary = (
        detailed.groupby(["Dataset", "Model"], sort=False)[metric_columns]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    detailed.to_csv(output_dir / "unified_detailed_folds.csv", index=False)
    summary.to_csv(output_dir / "unified_dataset_summary.csv", index=False)
    pd.DataFrame(per_label_rows).to_csv(
        output_dir / "unified_per_label_folds.csv", index=False
    )
    pd.DataFrame(dataset_rows).to_csv(
        output_dir / "dataset_characteristics.csv", index=False
    )
    print(f"\nSaved unified benchmark to {output_dir}", flush=True)
    return detailed, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT / "results_v6_3" / "paper_v6_3_5",
    )
    args = parser.parse_args()
    run_benchmark(args.datasets, args.output_dir)
