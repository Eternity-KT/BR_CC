"""
Benchmark Experiment for GSI-MLC-PA v6.3.1 (Precision Guard & Balanced-Root Calibration).
Evaluates and compares:
1. GSI_v6_3_1: v6.3.1 (Balanced-Root Sqrt-Platt Calibration + Precision Guard)
2. GSI_v6_3: v6.3.0 (Original Asymmetric Likelihood Ratio + Full-Weighted Platt)
3. GSI_v6_2: v6.2.1 (Decaying Peeling + Symmetric Chow Rejection)
4. BR: Binary Relevance Baseline

Datasets:
- emotions (balanced, audio)
- scene (vision)
- yeast (biology)
- plantpseaac (high imbalance, protein)
- humanpseaac (extreme imbalance, protein)
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
from src.models.gsi_v6_3_1 import GSIMLCPAv6_3_1Classifier

ALL_10_DATASETS = [
    "emotions",
    "scene",
    "yeast",
    "plantpseaac",
    "humanpseaac",
    "chd49",
    "music",
    "gpositivepseaac",
    "genbase",
    "viruspseaac",
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

    if isinstance(model, (GSIMLCPAv6_2Classifier, GSIMLCPAv6_3Classifier, GSIMLCPAv6_3_1Classifier)):
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


def run_v6_3_1_benchmark(
    datasets: Optional[List[str]] = None,
    base_learners: Optional[List[str]] = None,
    model_names: Optional[List[str]] = None,
    stratified_threshold: float = 0.75,
    residual_corr_threshold: float = 0.25,
    n_splits: int = 5,
    cost: float = 0.30,
    gamma_min: float = 0.70,
    calibration_method: str = "sqrt_platt",
    output_dir: Path = WORKSPACE_ROOT / "results_v6_3",
    random_state: int = 42,
) -> pd.DataFrame:
    if datasets is None:
        datasets = ALL_10_DATASETS
    if base_learners is None:
        base_learners = BASE_LEARNERS_3
    if model_names is None:
        model_names = ["GSI_v6_3_1"]

    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_csv_path = output_dir / "v6_3_1_all10ds_detailed_folds.csv"
    summary_csv_path = output_dir / "v6_3_1_all10ds_summary.csv"

    detailed_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    # If detailed CSV already exists, load existing rows
    if detailed_csv_path.exists():
        try:
            df_old = pd.read_csv(detailed_csv_path)
            detailed_rows = df_old.to_dict(orient="records")
            print(f"Loaded {len(detailed_rows)} existing fold records from {detailed_csv_path.name}")
        except Exception:
            pass

    print("\n" + "=" * 90, flush=True)
    print("BENCHMARK GSI-MLC-PA v6.3.1 (ALL 10 DATASETS): Balanced-Root & Precision Guard", flush=True)
    print(f"Datasets ({len(datasets)}): {datasets}", flush=True)
    print(f"Base Learners ({len(base_learners)}): {base_learners}", flush=True)
    print(f"Models to evaluate: {model_names}", flush=True)
    print(f"Calibration Method: {calibration_method} | Precision Guard: True (min_tau_1=0.50)", flush=True)
    print("=" * 90, flush=True)

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

            cv = get_multilabel_cv(n_splits=n_splits, random_state=random_state)
            fold_indices = list(cv.split(X, Y))

            fold_metrics_by_model: Dict[str, List[Dict[str, float]]] = {m: [] for m in model_names}

            for fold_idx, (train_idx, test_idx) in enumerate(fold_indices):
                # Check if this fold is already done
                already_done = any(
                    r.get("dataset") == dataset_name and r.get("learner") == display_learner and r.get("model") in model_names and r.get("fold") == fold_idx
                    for r in detailed_rows
                )
                if already_done and len(model_names) == 1:
                    # Retrieve existing
                    for r in detailed_rows:
                        if r.get("dataset") == dataset_name and r.get("learner") == display_learner and r.get("model") == model_names[0] and r.get("fold") == fold_idx:
                            fold_metrics_by_model[model_names[0]].append(r)
                    continue

                X_tr_raw, X_te_raw = X[train_idx], X[test_idx]
                Y_tr, Y_te = Y[train_idx], Y[test_idx]

                scaler = MaxAbsScaler()
                X_tr = scaler.fit_transform(X_tr_raw)
                X_te = scaler.transform(X_te_raw)

                models_dict = {}
                if "GSI_v6_2" in model_names:
                    models_dict["GSI_v6_2"] = GSIMLCPAv6_2Classifier(
                        base_learner=learner,
                        stratified_threshold=stratified_threshold,
                        residual_corr_threshold=residual_corr_threshold,
                        cost=cost,
                        cv_folds=5,
                        decaying_threshold=True,
                        random_state=random_state,
                    )
                if "GSI_v6_3" in model_names:
                    models_dict["GSI_v6_3"] = GSIMLCPAv6_3Classifier(
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
                    )
                if "GSI_v6_3_1" in model_names:
                    models_dict["GSI_v6_3_1"] = GSIMLCPAv6_3_1Classifier(
                        base_learner=learner,
                        stratified_threshold=stratified_threshold,
                        residual_corr_threshold=residual_corr_threshold,
                        cost=cost,
                        gamma_min=gamma_min,
                        cv_folds=5,
                        decaying_threshold=True,
                        use_prior_adaptive=True,
                        calibrate_tail=True,
                        calibration_method=calibration_method,
                        precision_guard=True,
                        min_tau_1=0.50,
                        random_state=random_state,
                    )

                for m_name, model in models_dict.items():
                    t0 = time.time()
                    model.fit(X_tr, Y_tr)
                    t_fit = time.time() - t0

                    metrics = evaluate_model_fold(
                        y_test=Y_te,
                        model=model,
                        X_test=X_te,
                        cost=cost,
                    )
                    metrics["Train_Time_s"] = float(t_fit)
                    fold_metrics_by_model[m_name].append(metrics)

                    det_row = {
                        "dataset": dataset_name,
                        "learner": display_learner,
                        "model": m_name,
                        "fold": fold_idx,
                        **metrics,
                    }
                    detailed_rows.append(det_row)

                # Incremental flush of detailed CSV
                pd.DataFrame(detailed_rows).to_csv(detailed_csv_path, index=False)

            # Summarize across folds
            for m_name in model_names:
                m_list = fold_metrics_by_model[m_name]
                if not m_list:
                    continue
                summary_row = {
                    "dataset": dataset_name,
                    "learner": display_learner,
                    "model": m_name,
                }
                for k in ["Full_Macro_F1", "Selective_Macro_F1", "Coverage", "Full_Micro_F1", "Selective_Micro_F1", "Hamming_Loss", "Selective_Hamming_Loss", "Subset_Accuracy", "Train_Time_s"]:
                    if k in m_list[0]:
                        vals = [r[k] for r in m_list]
                        summary_row[f"{k}_mean"] = float(np.mean(vals))
                        summary_row[f"{k}_std"] = float(np.std(vals))
                summary_rows.append(summary_row)

                print(f"   [{dataset_name} | {display_learner} | {m_name}] Sel-F1={summary_row.get('Selective_Macro_F1_mean', 0.0):.4f} | Cov={summary_row.get('Coverage_mean', 0.0)*100:.1f}% | SA={summary_row.get('Subset_Accuracy_mean', 0.0):.4f} | HL={summary_row.get('Hamming_Loss_mean', 0.0):.4f}", flush=True)

    df_sum = pd.DataFrame(summary_rows)
    df_det = pd.DataFrame(detailed_rows)
    df_sum.to_csv(summary_csv_path, index=False)
    df_det.to_csv(detailed_csv_path, index=False)
    print(f"\nSaved summary to: {summary_csv_path}", flush=True)
    return df_sum


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learners", nargs="+", default=BASE_LEARNERS_3)
    parser.add_argument("--datasets", nargs="+", default=ALL_10_DATASETS)
    parser.add_argument("--models", nargs="+", default=["GSI_v6_3_1"])
    parser.add_argument("--calibration", type=str, default="sqrt_platt")
    args = parser.parse_args()

    run_v6_3_1_benchmark(
        datasets=args.datasets,
        base_learners=args.learners,
        model_names=args.models,
        calibration_method=args.calibration,
    )

