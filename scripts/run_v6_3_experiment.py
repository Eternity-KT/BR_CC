"""
Benchmark Experiment for GSI-MLC-PA v6.3 across 5 Benchmark Datasets and 3 Base Learners.

Evaluates and compares:
1. GSI_v6_3: v6.3 (Prior-Calibrated Asymmetric Negative Verification & Adaptive Confidence Abstention)
2. GSI_v6_2: v6.2 (Symmetric Chow Partial Abstention)
3. BR: Binary Relevance Baseline

Datasets:
- emotions (balanced, audio)
- scene (moderate imbalance, vision)
- yeast (balanced/moderate, biology)
- plantpseaac (high imbalance ~8%, protein)
- humanpseaac (extreme imbalance ~3%, protein)

Base Learners:
- logistic: Logistic Regression
- svm_calibrated: Calibrated LinearSVC
- mlp: PyTorch Multi-Layer Perceptron (GPU accelerated)

Outputs strictly isolated to results_v6_3/:
- results_v6_3/v6_3_5ds_summary.csv
- results_v6_3/v6_3_5ds_detailed_folds.csv
- results_v6_3/v6_3_thresholds_audit.csv
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.data.loader import load_dataset
from src.evaluation.cv import get_multilabel_cv
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_selective_macro_f1,
    compute_selective_micro_f1,
)
from src.models.binary_relevance import BinaryRelevanceClassifier
from src.models.gsi_v6_2 import GSIMLCPAv6_2Classifier
from src.models.gsi_v6_3 import GSIMLCPAv6_3Classifier

DATASETS_5 = [
    "emotions",
    "scene",
    "yeast",
    "plantpseaac",
    "humanpseaac",
]

REMAINING_5 = [
    "chd49",
    "music",
    "gpositivepseaac",
    "genbase",
    "viruspseaac",
]

ALL_10_DATASETS = [
    "emotions",
    "scene",
    "chd49",
    "music",
    "gpositivepseaac",
    "genbase",
    "humanpseaac",
    "plantpseaac",
    "viruspseaac",
    "yeast",
]

BASE_LEARNERS_3 = ["logistic", "svm_calibrated", "mlp"]


def evaluate_model_fold(
    y_test: np.ndarray,
    model: Any,
    X_test: np.ndarray,
    cost: float = 0.30,
) -> Dict[str, float]:
    """Compute complete set of full and selective metrics for one model on test fold."""
    if hasattr(model, "predict_full"):
        full_preds = model.predict_full(X_test)
        try:
            sel_preds = model.predict(X_test, cost=cost)
        except TypeError:
            sel_preds = model.predict(X_test)
    elif hasattr(model, "predict"):
        full_preds = model.predict(X_test)
        sel_preds = full_preds
    else:
        raise ValueError("Model does not expose predict()")

    full_metrics = compute_all_metrics(y_test, full_preds)

    if isinstance(model, (GSIMLCPAv6_2Classifier, GSIMLCPAv6_3Classifier)):
        sel_macro_f1 = compute_selective_macro_f1(y_test, sel_preds)
        sel_micro_f1 = compute_selective_micro_f1(y_test, sel_preds)
        coverage = float(np.mean(sel_preds != -1))
        decided = sel_preds != -1
        sel_hamming = (
            float(np.mean(y_test[decided] != sel_preds[decided]))
            if np.any(decided)
            else 0.0
        )
    else:
        sel_macro_f1 = full_metrics["Macro-F1"]
        sel_micro_f1 = full_metrics["Micro-F1"]
        coverage = 1.0
        sel_hamming = full_metrics["Hamming Loss"]

    return {
        "Full_Macro_F1": float(full_metrics["Macro-F1"]),
        "Selective_Macro_F1": float(sel_macro_f1),
        "Coverage": float(coverage),
        "Full_Micro_F1": float(full_metrics["Micro-F1"]),
        "Selective_Micro_F1": float(sel_micro_f1),
        "Hamming_Loss": float(full_metrics["Hamming Loss"]),
        "Selective_Hamming_Loss": float(sel_hamming),
        "Subset_Accuracy": float(full_metrics["Subset Accuracy"]),
    }


def run_v6_3_benchmark(
    datasets: Optional[List[str]] = None,
    base_learners: Optional[List[str]] = None,
    stratified_threshold: float = 0.75,
    residual_corr_threshold: float = 0.25,
    n_splits: int = 5,
    cost: float = 0.30,
    gamma_min: float = 0.70,
    output_dir: Path = WORKSPACE_ROOT / "results_v6_3",
    random_state: int = 42,
) -> pd.DataFrame:
    """Run CV benchmark comparing GSI v6.3 vs GSI v6.2 vs BR across 5 datasets and 3 learners."""
    if datasets is None:
        datasets = DATASETS_5
    if base_learners is None:
        base_learners = BASE_LEARNERS_3

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = output_dir / "v6_3_all10ds_summary.csv"
    detailed_csv_path = output_dir / "v6_3_all10ds_detailed_folds.csv"
    thresholds_csv_path = output_dir / "v6_3_thresholds_audit.csv"

    # Also keep compatibility with 5ds summary file
    init_5ds_summary = output_dir / "v6_3_5ds_summary.csv"
    init_5ds_detailed = output_dir / "v6_3_5ds_detailed_folds.csv"

    detailed_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    threshold_records: List[Dict[str, Any]] = []

    # Pre-load existing summary rows
    source_summary = summary_csv_path if summary_csv_path.exists() else (init_5ds_summary if init_5ds_summary.exists() else None)
    if source_summary is not None:
        try:
            df_old_sum = pd.read_csv(source_summary)
            # Remove any overlapping entries we are about to re-run
            summary_rows = df_old_sum.to_dict("records")
        except Exception:
            summary_rows = []

    source_detailed = detailed_csv_path if detailed_csv_path.exists() else (init_5ds_detailed if init_5ds_detailed.exists() else None)
    if source_detailed is not None:
        try:
            df_old_det = pd.read_csv(source_detailed)
            detailed_rows = df_old_det.to_dict("records")
        except Exception:
            detailed_rows = []

    if thresholds_csv_path.exists():
        try:
            df_old_th = pd.read_csv(thresholds_csv_path)
            threshold_records = df_old_th.to_dict("records")
        except Exception:
            threshold_records = []

    print("\n" + "=" * 90, flush=True)
    print("BENCHMARK GSI-MLC-PA v6.3: Prior-Calibrated Asymmetric Negative Verification", flush=True)
    print(f"Datasets ({len(datasets)}): {datasets}", flush=True)
    print(f"Base Learners ({len(base_learners)}): {base_learners}", flush=True)
    print(f"Outer CV: {n_splits}-Fold | tau_f1: {stratified_threshold} | tau_corr: {residual_corr_threshold} | c: {cost} | gamma_min: {gamma_min}", flush=True)
    print("=" * 90, flush=True)

    total_start_time = time.time()

    for learner in base_learners:
        display_learner = {
            "logistic": "Logistic",
            "svm_calibrated": "SVM",
            "mlp": "MLP",
        }.get(learner, learner.upper())

        print(f"\n=======================================================", flush=True)
        print(f" >>> BASE LEARNER: {display_learner} ({learner})", flush=True)
        print(f"=======================================================", flush=True)

        for dataset_name in datasets:
            print(f"\n---> Loading dataset: {dataset_name} ...", flush=True)
            X, Y, _, _ = load_dataset(dataset_name, base_dir=str(WORKSPACE_ROOT))
            if hasattr(X, "toarray"):
                X = X.toarray()
            X = np.asarray(X, dtype=np.float32)
            Y = np.asarray(Y, dtype=np.int32)

            N, d = X.shape
            K = Y.shape[1]
            priors = [float(np.mean(Y[:, j])) for j in range(K)]
            avg_prior = float(np.mean(priors))

            print(f"     N={N}, d={d}, K={K}, Avg Positive Prior={avg_prior:.4f} (Min={min(priors):.4f}, Max={max(priors):.4f})", flush=True)

            # Deduplicate existing rows for this dataset and learner
            summary_rows = [
                r for r in summary_rows
                if not (r.get("dataset") == dataset_name and r.get("learner") == display_learner)
            ]
            detailed_rows = [
                r for r in detailed_rows
                if not (r.get("dataset") == dataset_name and r.get("learner") == display_learner)
            ]
            threshold_records = [
                r for r in threshold_records
                if not (r.get("dataset") == dataset_name and r.get("learner") == display_learner)
            ]

            cv = get_multilabel_cv(n_splits=n_splits, random_state=random_state)
            fold_indices = list(cv.split(X, Y))

            model_names = ["BR", "GSI_v6_2", "GSI_v6_3"]
            fold_metrics_by_model: Dict[str, List[Dict[str, float]]] = {m: [] for m in model_names}
            fold_times_by_model: Dict[str, List[float]] = {m: [] for m in model_names}

            for fold_idx, (train_idx, test_idx) in enumerate(fold_indices):
                X_tr_raw, X_te_raw = X[train_idx], X[test_idx]
                Y_tr, Y_te = Y[train_idx], Y[test_idx]

                # Safe scaling strictly fitted on training fold
                scaler = MaxAbsScaler()
                X_tr = scaler.fit_transform(X_tr_raw)
                X_te = scaler.transform(X_te_raw)

                # Initialize models
                models = {
                    "BR": BinaryRelevanceClassifier(
                        base_estimator=learner,
                        random_state=random_state,
                    ),
                    "GSI_v6_2": GSIMLCPAv6_2Classifier(
                        base_learner=learner,
                        stratified_threshold=stratified_threshold,
                        residual_corr_threshold=residual_corr_threshold,
                        cost=cost,
                        cv_folds=5,
                        decaying_threshold=True,
                        random_state=random_state,
                    ),
                    "GSI_v6_3": GSIMLCPAv6_3Classifier(
                        base_learner=learner,
                        stratified_threshold=stratified_threshold,
                        residual_corr_threshold=residual_corr_threshold,
                        cost=cost,
                        gamma_min=gamma_min,
                        cv_folds=5,
                        decaying_threshold=True,
                        use_prior_adaptive=True,
                        calibrate_tail=True,
                        random_state=random_state,
                    ),
                }

                for m_name, model in models.items():
                    t0 = time.time()
                    model.fit(X_tr, Y_tr)
                    fit_time = time.time() - t0

                    metrics = evaluate_model_fold(
                        y_test=Y_te,
                        model=model,
                        X_test=X_te,
                        cost=cost,
                    )
                    metrics["Train_Time_s"] = fit_time
                    fold_metrics_by_model[m_name].append(metrics)
                    fold_times_by_model[m_name].append(fit_time)

                    # Record detailed row
                    detailed_rows.append({
                        "dataset": dataset_name,
                        "learner": display_learner,
                        "model": m_name,
                        "fold": fold_idx + 1,
                        **metrics,
                    })

                    # Record threshold audit from v6.3 on fold 0
                    if m_name == "GSI_v6_3" and fold_idx == 0:
                        for j in range(K):
                            th = model.adaptive_thresholds_[j]
                            threshold_records.append({
                                "dataset": dataset_name,
                                "learner": display_learner,
                                "label_idx": j,
                                "prior": th["prior"],
                                "tau_0": th["tau_0"],
                                "tau_1": th["tau_1"],
                                "coverage_guard": th["coverage_guard"],
                                "empirical_coverage": th["empirical_coverage"],
                            })

                print(
                    f"     [Fold {fold_idx + 1}/{n_splits}] "
                    f"BR Sel-F1={fold_metrics_by_model['BR'][-1]['Selective_Macro_F1']:.4f} | "
                    f"v6.2 Sel-F1={fold_metrics_by_model['GSI_v6_2'][-1]['Selective_Macro_F1']:.4f} (Cov={fold_metrics_by_model['GSI_v6_2'][-1]['Coverage']:.3f}) | "
                    f"v6.3 Sel-F1={fold_metrics_by_model['GSI_v6_3'][-1]['Selective_Macro_F1']:.4f} (Cov={fold_metrics_by_model['GSI_v6_3'][-1]['Coverage']:.3f})",
                    flush=True,
                )

            # Aggregate fold metrics for dataset and learner
            for m_name in model_names:
                df_f = pd.DataFrame(fold_metrics_by_model[m_name])
                summary_rows.append({
                    "dataset": dataset_name,
                    "learner": display_learner,
                    "model": m_name,
                    "Selective_Macro_F1_mean": df_f["Selective_Macro_F1"].mean(),
                    "Selective_Macro_F1_std": df_f["Selective_Macro_F1"].std(),
                    "Coverage_mean": df_f["Coverage"].mean(),
                    "Coverage_std": df_f["Coverage"].std(),
                    "Full_Macro_F1_mean": df_f["Full_Macro_F1"].mean(),
                    "Full_Macro_F1_std": df_f["Full_Macro_F1"].std(),
                    "Selective_Micro_F1_mean": df_f["Selective_Micro_F1"].mean(),
                    "Full_Micro_F1_mean": df_f["Full_Micro_F1"].mean(),
                    "Hamming_Loss_mean": df_f["Hamming_Loss"].mean(),
                    "Selective_Hamming_Loss_mean": df_f["Selective_Hamming_Loss"].mean(),
                    "Train_Time_s_mean": df_f["Train_Time_s"].mean(),
                })

            # Save checkpoints incrementally
            pd.DataFrame(detailed_rows).to_csv(detailed_csv_path, index=False)
            pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
            pd.DataFrame(threshold_records).to_csv(thresholds_csv_path, index=False)

    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 90, flush=True)
    print(f"BENCHMARK COMPLETED in {total_elapsed / 60.0:.2f} minutes!", flush=True)
    print(f"Results saved to:\n - {summary_csv_path}\n - {detailed_csv_path}\n - {thresholds_csv_path}", flush=True)
    print("=" * 90, flush=True)

    return pd.DataFrame(summary_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GSI v6.3 Benchmark")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--remaining", action="store_true", help="Run the 5 remaining datasets")
    parser.add_argument("--all10", action="store_true", help="Run all 10 datasets")
    parser.add_argument("--learners", nargs="+", default=BASE_LEARNERS_3)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--cost", type=float, default=0.30)
    parser.add_argument("--gamma_min", type=float, default=0.70)
    args = parser.parse_args()

    if args.remaining:
        chosen_datasets = REMAINING_5
    elif args.all10:
        chosen_datasets = ALL_10_DATASETS
    elif args.datasets:
        chosen_datasets = args.datasets
    else:
        chosen_datasets = DATASETS_5

    run_v6_3_benchmark(
        datasets=chosen_datasets,
        base_learners=args.learners,
        n_splits=args.splits,
        cost=args.cost,
        gamma_min=args.gamma_min,
    )
