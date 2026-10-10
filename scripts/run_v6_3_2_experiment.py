"""
Benchmark Experiment for GSI-MLC-PA v6.3.2 across 5 Benchmark Datasets.
Compares:
1. GSI_v6_3_1: Baseline (Conditioned BR on DL + Precision Guard + Balanced-Root Platt)
2. GSI_v6_3_2: Version 6.3.2 (One-Step Normalized Mean-Field on DL + Precision Guard + Balanced-Root Platt)

Outputs:
- results_v6_3/v6_3_2_5ds_summary.csv
- results_v6_3/v6_3_2_5ds_detailed_folds.csv
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

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
from src.models.gsi_v6_3_1 import GSIMLCPAv6_3_1Classifier
from src.models.gsi_v6_3_2 import GSIMLCPAv6_3_2Classifier

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

ALL_10_DATASETS = DATASETS_5 + REMAINING_5

def evaluate_fold(y_test: np.ndarray, model: Any, X_test: np.ndarray, cost: float = 0.30) -> Dict[str, float]:
    full_preds = model.predict_full(X_test)
    sel_preds = model.predict(X_test, cost=cost)

    full_metrics = compute_all_metrics(y_test, full_preds)
    sel_macro_f1 = compute_selective_macro_f1(y_test, sel_preds)
    sel_micro_f1 = compute_selective_micro_f1(y_test, sel_preds)
    coverage = float(np.mean(sel_preds != -1))
    decided = (sel_preds != -1)
    sel_hamming = (
        float(np.mean(y_test[decided] != sel_preds[decided]))
        if np.any(decided)
        else 0.0
    )

    return {
        "Selective_Macro_F1": float(sel_macro_f1),
        "Coverage": float(coverage),
        "Full_Macro_F1": float(full_metrics["Macro-F1"]),
        "Selective_Micro_F1": float(sel_micro_f1),
        "Full_Micro_F1": float(full_metrics["Micro-F1"]),
        "Hamming_Loss": float(full_metrics["Hamming Loss"]),
        "Selective_Hamming_Loss": float(sel_hamming),
        "Subset_Accuracy": float(full_metrics["Subset Accuracy"]),
    }

def run_v6_3_2_benchmark(
    datasets: List[str] = REMAINING_5,
    tag: str = "remaining5ds",
    base_learner: str = "logistic",
    n_splits: int = 5,
    cost: float = 0.30,
    random_state: int = 42,
    output_dir: Path = WORKSPACE_ROOT / "results_v6_3",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_csv = output_dir / f"v6_3_2_{tag}_detailed_folds.csv"
    summary_csv = output_dir / f"v6_3_2_{tag}_summary.csv"

    detailed_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 95)
    print(f"BENCHMARK: GSI-MLC-PA v6.3.1 vs. v6.3.2 (One-Step Mean-Field) [{tag}]")
    print(f"Datasets: {datasets}")
    print(f"Base Learner: {base_learner} | Folds: {n_splits} | Rejection Cost: {cost}")
    print("=" * 95)

    for dataset_name in datasets:
        print(f"\n---> Loading: {dataset_name} ...", flush=True)
        X, Y, _, _ = load_dataset(dataset_name, base_dir=str(WORKSPACE_ROOT))
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int32)

        cv = get_multilabel_cv(n_splits=n_splits, random_state=random_state)
        fold_indices = list(cv.split(X, Y))

        models_to_test = {
            "GSI_v6_3_1": lambda seed: GSIMLCPAv6_3_1Classifier(
                base_learner=base_learner,
                random_state=seed,
            ),
            "GSI_v6_3_2": lambda seed: GSIMLCPAv6_3_2Classifier(
                base_learner=base_learner,
                mf_alpha=0.25,
                mf_z_max=0.5,
                random_state=seed,
            ),
        }

        dataset_fold_results: Dict[str, List[Dict[str, float]]] = {m: [] for m in models_to_test}

        for fold_idx, (train_idx, test_idx) in enumerate(fold_indices):
            X_tr_raw, X_te_raw = X[train_idx], X[test_idx]
            Y_tr, Y_te = Y[train_idx], Y[test_idx]

            scaler = MaxAbsScaler()
            X_tr = scaler.fit_transform(X_tr_raw)
            X_te = scaler.transform(X_te_raw)

            seed = random_state + fold_idx

            for m_name, model_fn in models_to_test.items():
                m = model_fn(seed)
                t0 = time.time()
                m.fit(X_tr, Y_tr)
                t_fit = time.time() - t0

                metrics = evaluate_fold(Y_te, m, X_te, cost=cost)
                metrics["Train_Time_s"] = float(t_fit)
                metrics["DL_Count"] = len(m.dependent_labels_)
                dataset_fold_results[m_name].append(metrics)

                detailed_rows.append({
                    "dataset": dataset_name,
                    "model": m_name,
                    "fold": fold_idx,
                    **metrics,
                })

            base_res = dataset_fold_results["GSI_v6_3_1"][-1]
            mf_res = dataset_fold_results["GSI_v6_3_2"][-1]
            print(
                f"  Fold {fold_idx}: "
                f"v6.3.1 Sel-F1={base_res['Selective_Macro_F1']:.4f} (Cov={base_res['Coverage']:.4f}, t={base_res['Train_Time_s']:.2f}s) | "
                f"v6.3.2 Sel-F1={mf_res['Selective_Macro_F1']:.4f} (Cov={mf_res['Coverage']:.4f}, t={mf_res['Train_Time_s']:.2f}s)",
                flush=True,
            )

        # Dataset summary
        for m_name in models_to_test:
            res_list = dataset_fold_results[m_name]
            s_row = {
                "dataset": dataset_name,
                "model": m_name,
                "Selective_Macro_F1": float(np.mean([r["Selective_Macro_F1"] for r in res_list])),
                "Coverage": float(np.mean([r["Coverage"] for r in res_list])),
                "Full_Macro_F1": float(np.mean([r["Full_Macro_F1"] for r in res_list])),
                "Selective_Micro_F1": float(np.mean([r["Selective_Micro_F1"] for r in res_list])),
                "Full_Micro_F1": float(np.mean([r["Full_Micro_F1"] for r in res_list])),
                "Hamming_Loss": float(np.mean([r["Hamming_Loss"] for r in res_list])),
                "Selective_Hamming_Loss": float(np.mean([r["Selective_Hamming_Loss"] for r in res_list])),
                "Subset_Accuracy": float(np.mean([r["Subset_Accuracy"] for r in res_list])),
                "Train_Time_s": float(np.mean([r["Train_Time_s"] for r in res_list])),
                "Avg_DL_Count": float(np.mean([r["DL_Count"] for r in res_list])),
            }
            summary_rows.append(s_row)

        b_s = next(r for r in summary_rows if r["dataset"] == dataset_name and r["model"] == "GSI_v6_3_1")
        mf_s = next(r for r in summary_rows if r["dataset"] == dataset_name and r["model"] == "GSI_v6_3_2")
        delta_f1 = mf_s["Selective_Macro_F1"] - b_s["Selective_Macro_F1"]
        delta_cov = mf_s["Coverage"] - b_s["Coverage"]
        speedup = (b_s["Train_Time_s"] - mf_s["Train_Time_s"]) / b_s["Train_Time_s"] * 100 if b_s["Train_Time_s"] > 0 else 0
        print(f"--> [Summary {dataset_name}] Delta Sel-F1: {delta_f1:+.4f} | Delta Cov: {delta_cov:+.4f} | Speedup: {speedup:+.1f}%")

    df_det = pd.DataFrame(detailed_rows)
    df_det.to_csv(detailed_csv, index=False)
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(summary_csv, index=False)
    print("\n" + "=" * 95)
    print(f"Saved detailed fold results to: {detailed_csv.name}")
    print(f"Saved summary results to: {summary_csv.name}")

    # If remaining5ds finished and 5ds exists, auto-merge into all10ds
    f5_sum_path = output_dir / "v6_3_2_5ds_summary.csv"
    f5_det_path = output_dir / "v6_3_2_5ds_detailed_folds.csv"
    if tag == "remaining5ds" and f5_sum_path.exists() and f5_det_path.exists():
        f5_sum = pd.read_csv(f5_sum_path)
        f5_det = pd.read_csv(f5_det_path)
        all_sum = pd.concat([f5_sum, df_sum], ignore_index=True)
        all_det = pd.concat([f5_det, df_det], ignore_index=True)
        all_sum_path = output_dir / "v6_3_2_all10ds_summary.csv"
        all_det_path = output_dir / "v6_3_2_all10ds_detailed_folds.csv"
        all_sum.to_csv(all_sum_path, index=False)
        all_det.to_csv(all_det_path, index=False)
        print(f"Auto-merged all 10 datasets to: {all_sum_path.name}")
        print(f"Auto-merged detailed folds to: {all_det_path.name}")

    print("=" * 95)
    return df_sum

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run GSI-MLC-PA v6.3.2 benchmarks.")
    parser.add_argument(
        "--mode",
        choices=["remaining", "5ds", "all"],
        default="remaining",
        help="Benchmark dataset split: remaining (5 remaining), 5ds (first 5), all (all 10)",
    )
    args = parser.parse_args()

    if args.mode == "remaining":
        run_v6_3_2_benchmark(datasets=REMAINING_5, tag="remaining5ds")
    elif args.mode == "5ds":
        run_v6_3_2_benchmark(datasets=DATASETS_5, tag="5ds")
    elif args.mode == "all":
        run_v6_3_2_benchmark(datasets=ALL_10_DATASETS, tag="all10ds")
