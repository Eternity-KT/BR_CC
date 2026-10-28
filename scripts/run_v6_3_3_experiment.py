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
from src.models.gsi_v6_3_4 import GSIMLCPAv6_3_4Classifier
from src.models.gsi_v6_3_5 import GSIMLCPAv6_3_5Classifier

DEFAULT_DATASETS = [
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


def _nanmean_or_nan(values) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else np.nan


def evaluate_fold(y_test: np.ndarray, model: Any, X_test: np.ndarray, cost: Any = None) -> Dict[str, float]:
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

    result = {
        "Selective_Macro_F1": float(sel_macro_f1),
        "Coverage": float(coverage),
        "Full_Macro_F1": float(full_metrics["Macro-F1"]),
        "Selective_Micro_F1": float(sel_micro_f1),
        "Full_Micro_F1": float(full_metrics["Micro-F1"]),
        "Hamming_Loss": float(full_metrics["Hamming Loss"]),
        "Selective_Hamming_Loss": float(sel_hamming),
        "Subset_Accuracy": float(full_metrics["Subset Accuracy"]),
    }
    for group_name, label_indices in (
        ("IL", model.independent_labels_),
        ("DL", model.dependent_labels_),
    ):
        if label_indices:
            group_truth = y_test[:, label_indices]
            group_preds = sel_preds[:, label_indices]
            result[f"{group_name}_Label_Count"] = int(len(label_indices))
            result[f"{group_name}_Coverage"] = float(np.mean(group_preds != -1))
            result[f"{group_name}_Selective_Macro_F1"] = float(
                compute_selective_macro_f1(group_truth, group_preds)
            )
        else:
            result[f"{group_name}_Label_Count"] = 0
            result[f"{group_name}_Coverage"] = np.nan
            result[f"{group_name}_Selective_Macro_F1"] = np.nan
    return result


def run_v6_3_3_benchmark(
    datasets: List[str] = DEFAULT_DATASETS,
    tag: str = "v6_3_3_all10ds",
    base_learner: str = "logistic",
    n_splits: int = 5,
    cost: Any = None,
    random_state: int = 42,
    model_set: str = "legacy",
    output_dir: Path = WORKSPACE_ROOT / "results_v6_3",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_csv = output_dir / f"{tag}_detailed_folds.csv"
    summary_csv = output_dir / f"{tag}_summary.csv"

    detailed_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    benchmark_title = (
        "GSI-MLC-PA v6.3.2 vs. v6.3.3 (Adaptive Tri-Regime)"
        if model_set == "legacy"
        else "GSI-MLC-PA selective rejection audit"
    )
    print("\n" + "=" * 95)
    print(f"BENCHMARK: {benchmark_title} [{tag}]")
    print(f"Datasets: {datasets}")
    print(f"Base Learner: {base_learner} | Folds: {n_splits}")
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

        common_v633 = dict(
            base_learner=base_learner,
            mf_alpha=0.25,
            mf_z_max=0.5,
            tau_balance=0.85,
            blend_kappa=20.0,
            cost=0.30,
            gamma_min=0.70,
            tau_1_cap=0.55,
            tau_0_floor=0.45,
        )
        if model_set == "legacy":
            models_to_test = {
                "GSI_v6_3_2": lambda seed: GSIMLCPAv6_3_2Classifier(
                    base_learner=base_learner,
                    mf_alpha=0.25,
                    mf_z_max=0.5,
                    random_state=seed,
                ),
                "GSI_v6_3_3": lambda seed: GSIMLCPAv6_3_3Classifier(
                    **common_v633,
                    random_state=seed,
                ),
            }
        elif model_set == "samg_audit":
            models_to_test = {
                "GSI_v6_3_3_No_SAMG": lambda seed: GSIMLCPAv6_3_3Classifier(
                    **common_v633,
                    separation_aware=False,
                    random_state=seed,
                ),
                "GSI_v6_3_3_SAMG": lambda seed: GSIMLCPAv6_3_3Classifier(
                    **common_v633,
                    separation_aware=True,
                    random_state=seed,
                ),
                "GSI_v6_3_4_SAMG2": lambda seed: GSIMLCPAv6_3_4Classifier(
                    **common_v633,
                    separation_aware=True,
                    random_state=seed,
                ),
            }
        elif model_set == "dl_only_rejection":
            models_to_test = {
                "GSI_v6_3_3_All_Label_Rejection": lambda seed: GSIMLCPAv6_3_3Classifier(
                    **common_v633,
                    separation_aware=True,
                    random_state=seed,
                ),
                "GSI_v6_3_4_All_Label_Rejection": lambda seed: GSIMLCPAv6_3_4Classifier(
                    **common_v633,
                    separation_aware=True,
                    random_state=seed,
                ),
                "GSI_v6_3_5_DL_Only_Rejection": lambda seed: GSIMLCPAv6_3_5Classifier(
                    **common_v633,
                    separation_aware=True,
                    random_state=seed,
                ),
            }
        else:
            raise ValueError(f"Unknown model_set={model_set!r}")

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

            fold_scores = " | ".join(
                f"{name}: {dataset_fold_results[name][-1]['Selective_Macro_F1']:.4f}"
                for name in models_to_test
            )
            print(f"  Fold {fold_idx + 1}/{n_splits} | {fold_scores}", flush=True)

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
                "IL_Label_Count_Mean": float(np.mean([f["IL_Label_Count"] for f in folds])),
                "IL_Coverage_Mean": _nanmean_or_nan([f["IL_Coverage"] for f in folds]),
                "IL_Selective_Macro_F1_Mean": _nanmean_or_nan([f["IL_Selective_Macro_F1"] for f in folds]),
                "DL_Label_Count_Mean": float(np.mean([f["DL_Label_Count"] for f in folds])),
                "DL_Coverage_Mean": _nanmean_or_nan([f["DL_Coverage"] for f in folds]),
                "DL_Selective_Macro_F1_Mean": _nanmean_or_nan([f["DL_Selective_Macro_F1"] for f in folds]),
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
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--tag", type=str, default="v6_3_3_all10ds")
    parser.add_argument("--base_learner", type=str, default="logistic")
    parser.add_argument(
        "--model_set",
        choices=("legacy", "samg_audit", "dl_only_rejection"),
        default="legacy",
    )
    args = parser.parse_args()

    run_v6_3_3_benchmark(
        datasets=args.datasets,
        tag=args.tag,
        base_learner=args.base_learner,
        model_set=args.model_set,
    )
