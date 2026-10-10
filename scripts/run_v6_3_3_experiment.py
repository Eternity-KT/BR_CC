"""
Benchmark Experiment for GSI-MLC-PA v6.3.3 across Benchmark Datasets.

Compares:
1. GSI_v6_3_2: Baseline (OS-NMF Mean-Field + Precision Guard + Balanced-Root Platt)
2. GSI_v6_3_3: Proposed (Adaptive Tri-Regime Inference + Smooth Boundary Blending + Scale-Aligned OS-NMF)

Outputs:
- results_v6_3/v6_3_3_benchmark_summary.csv
- results_v6_3/v6_3_3_benchmark_detailed_folds.csv
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
from src.models.gsi_v6_3_2 import GSIMLCPAv6_3_2Classifier
from src.models.gsi_v6_3_3 import GSIMLCPAv6_3_3Classifier

DEFAULT_DATASETS = [
    "chd49",
    "viruspseaac",
    "plantpseaac",
    "humanpseaac",
    "emotions",
    "scene",
    "yeast",
    "music",
    "gpositivepseaac",
    "genbase",
]


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


def run_v6_3_3_benchmark(
    datasets: List[str] = DEFAULT_DATASETS,
    tag: str = "v6_3_3_eval",
    base_learner: str = "logistic",
    n_splits: int = 5,
    cost: float = 0.30,
    random_state: int = 42,
    output_dir: Path = WORKSPACE_ROOT / "results_v6_3",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_csv = output_dir / f"{tag}_detailed_folds.csv"
    summary_csv = output_dir / f"{tag}_summary.csv"

    detailed_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 95)
    print(f"BENCHMARK: GSI-MLC-PA v6.3.2 vs. v6.3.3 (Adaptive Tri-Regime) [{tag}]")
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
            "GSI_v6_3_2": lambda seed: GSIMLCPAv6_3_2Classifier(
                base_learner=base_learner,
                mf_alpha=0.25,
                mf_z_max=0.5,
                random_state=seed,
            ),
            "GSI_v6_3_3": lambda seed: GSIMLCPAv6_3_3Classifier(
                base_learner=base_learner,
                mf_alpha=0.25,
                mf_z_max=0.5,
                tau_balance=0.75,
                blend_kappa=20.0,
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
                fit_t = time.time() - t0

                res = evaluate_fold(Y_te, m, X_te, cost=cost)
                res["Fit_Time_s"] = fit_t
                dataset_fold_results[m_name].append(res)

                detailed_rows.append({
                    "Dataset": dataset_name,
                    "Fold": fold_idx + 1,
                    "Model": m_name,
                    "Base_Learner": base_learner,
                    **res,
                })

            v2_sel_f1 = dataset_fold_results["GSI_v6_3_2"][-1]["Selective_Macro_F1"]
            v3_sel_f1 = dataset_fold_results["GSI_v6_3_3"][-1]["Selective_Macro_F1"]
            diff = v3_sel_f1 - v2_sel_f1
            sign = "+" if diff >= 0 else ""
            print(
                f"  Fold {fold_idx + 1}/{n_splits} | "
                f"v6.3.2 Sel-F1: {v2_sel_f1:.4f} | "
                f"v6.3.3 Sel-F1: {v3_sel_f1:.4f} ({sign}{diff:.4f})",
                flush=True,
            )

        # Dataset summary
        for m_name in models_to_test:
            folds = dataset_fold_results[m_name]
            summary_rows.append({
                "Dataset": dataset_name,
                "Model": m_name,
                "Base_Learner": base_learner,
                "Selective_Macro_F1_Mean": float(np.mean([f["Selective_Macro_F1"] for f in folds])),
                "Selective_Macro_F1_Std": float(np.std([f["Selective_Macro_F1"] for f in folds])),
                "Coverage_Mean": float(np.mean([f["Coverage"] for f in folds])),
                "Coverage_Std": float(np.std([f["Coverage"] for f in folds])),
                "Subset_Accuracy_Mean": float(np.mean([f["Subset_Accuracy"] for f in folds])),
                "Hamming_Loss_Mean": float(np.mean([f["Hamming_Loss"] for f in folds])),
                "Selective_Micro_F1_Mean": float(np.mean([f["Selective_Micro_F1"] for f in folds])),
                "Fit_Time_Mean": float(np.mean([f["Fit_Time_s"] for f in folds])),
            })

        # Print comparison for this dataset
        df_curr = pd.DataFrame(summary_rows)
        ds_curr = df_curr[df_curr["Dataset"] == dataset_name]
        print(f"\nSummary for {dataset_name}:")
        print(ds_curr[["Model", "Selective_Macro_F1_Mean", "Coverage_Mean", "Subset_Accuracy_Mean", "Hamming_Loss_Mean"]].to_string(index=False))

    df_detailed = pd.DataFrame(detailed_rows)
    df_summary = pd.DataFrame(summary_rows)

    df_detailed.to_csv(detailed_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)

    print("\n" + "=" * 95)
    print(f"BENCHMARK COMPLETED: {tag}")
    print(f"Detailed CSV: {detailed_csv}")
    print(f"Summary CSV:  {summary_csv}")
    print("=" * 95)
    return df_summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["chd49", "viruspseaac", "plantpseaac", "humanpseaac", "emotions"])
    parser.add_argument("--tag", type=str, default="v6_3_3_eval")
    parser.add_argument("--base_learner", type=str, default="logistic")
    args = parser.parse_args()

    run_v6_3_3_benchmark(
        datasets=args.datasets,
        tag=args.tag,
        base_learner=args.base_learner,
    )
