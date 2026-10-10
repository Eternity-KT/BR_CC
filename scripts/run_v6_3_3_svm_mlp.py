"""
Benchmark script to run GSI-MLC-PA v6.3.3 on SVM and MLP across all 10 datasets.
Saves incremental results to results_v6_3/v6_3_3_svm_mlp_detailed_folds.csv.
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
from src.models.gsi_v6_3_3 import GSIMLCPAv6_3_3Classifier

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

TARGET_LEARNERS = [
    ("svm_calibrated", "SVM"),
    ("mlp", "MLP"),
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
        "Full_Macro_F1": float(full_metrics["Macro-F1"]),
        "Selective_Macro_F1": float(sel_macro_f1),
        "Coverage": float(coverage),
        "Full_Micro_F1": float(full_metrics["Micro-F1"]),
        "Selective_Micro_F1": float(sel_micro_f1),
        "Hamming_Loss": float(full_metrics["Hamming Loss"]),
        "Selective_Hamming_Loss": float(sel_hamming),
        "Subset_Accuracy": float(full_metrics["Subset Accuracy"]),
    }

def main():
    output_dir = WORKSPACE_ROOT / "results_v6_3"
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_csv = output_dir / "v6_3_3_svm_mlp_detailed_folds.csv"

    rows: List[Dict[str, Any]] = []
    completed_keys = set()
    if detailed_csv.exists():
        try:
            df_old = pd.read_csv(detailed_csv)
            rows = df_old.to_dict(orient="records")
            for r in rows:
                completed_keys.add((r["dataset"], r["learner"], r["fold"]))
            print(f"Loaded {len(rows)} existing rows from {detailed_csv.name}", flush=True)
        except Exception as e:
            print(f"Could not load existing CSV: {e}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("STARTING GSI-MLC-PA v6.3.3 BENCHMARK ON SVM AND MLP", flush=True)
    print("=" * 90, flush=True)

    for learner_key, display_learner in TARGET_LEARNERS:
        print(f"\n#######################################################", flush=True)
        print(f" >>> BASE LEARNER: {display_learner} ({learner_key})", flush=True)
        print(f"#######################################################", flush=True)

        for dataset_name in ALL_10_DATASETS:
            # Check if all 5 folds already completed
            if all((dataset_name, display_learner, f) in completed_keys for f in range(5)):
                print(f"---> Dataset: {dataset_name} | Learner: {display_learner} ALREADY DONE. Skipping.", flush=True)
                continue

            print(f"\n---> Dataset: {dataset_name} | Learner: {display_learner} ...", flush=True)
            X, Y, _, _ = load_dataset(dataset_name, base_dir=str(WORKSPACE_ROOT))
            if hasattr(X, "toarray"):
                X = X.toarray()
            X = np.asarray(X, dtype=np.float32)
            Y = np.asarray(Y, dtype=np.int32)

            cv = get_multilabel_cv(n_splits=5, random_state=42)
            for fold, (train_idx, test_idx) in enumerate(cv.split(X, Y)):
                if (dataset_name, display_learner, fold) in completed_keys:
                    continue

                X_train, y_train = X[train_idx], Y[train_idx]
                X_test, y_test = X[test_idx], Y[test_idx]

                scaler = MaxAbsScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                t0 = time.time()
                model = GSIMLCPAv6_3_3Classifier(
                    base_learner=learner_key,
                    random_state=42 + fold,
                )
                model.fit(X_train, y_train)
                fit_time = time.time() - t0

                metrics = evaluate_fold(y_test, model, X_test, cost=0.30)
                record = {
                    "dataset": dataset_name,
                    "learner": display_learner,
                    "model": "GSI_v6_3_3",
                    "fold": fold,
                    **metrics,
                    "Train_Time_s": fit_time,
                }
                rows.append(record)
                completed_keys.add((dataset_name, display_learner, fold))
                print(
                    f"  Fold {fold+1}/5: Sel-F1={metrics['Selective_Macro_F1']:.4f}, "
                    f"Cov={metrics['Coverage']*100:.1f}%, SubAcc={metrics['Subset_Accuracy']:.4f}, "
                    f"Hamming={metrics['Hamming_Loss']:.4f} ({fit_time:.2f}s)",
                    flush=True,
                )

                # Save incrementally
                pd.DataFrame(rows).to_csv(detailed_csv, index=False)

    print("\n" + "=" * 90, flush=True)
    print("FINISHED ALL RUNS. SAVED TO:", detailed_csv, flush=True)
    print("=" * 90, flush=True)

if __name__ == "__main__":
    main()
