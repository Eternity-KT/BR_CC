"""
Benchmark script to run GSI-MLC-PA v6.3.2 on SVM and MLP across all 10 datasets.
Saves incremental results to results_v6_3/v6_3_2_svm_mlp_detailed_folds.csv.
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
    detailed_csv = output_dir / "v6_3_2_svm_mlp_detailed_folds.csv"

    rows: List[Dict[str, Any]] = []
    if detailed_csv.exists():
        try:
            df_old = pd.read_csv(detailed_csv)
            rows = df_old.to_dict(orient="records")
            print(f"Loaded {len(rows)} existing rows from {detailed_csv.name}", flush=True)
        except Exception as e:
            print(f"Could not load existing CSV: {e}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("STARTING GSI-MLC-PA v6.3.2 BENCHMARK ON SVM AND MLP", flush=True)
    print("=" * 90, flush=True)

    for learner_key, display_learner in TARGET_LEARNERS:
        print(f"\n#######################################################", flush=True)
        print(f" >>> BASE LEARNER: {display_learner} ({learner_key})", flush=True)
        print(f"#######################################################", flush=True)

        for dataset_name in ALL_10_DATASETS:
            print(f"\n---> Dataset: {dataset_name} | Learner: {display_learner} ...", flush=True)
            X, Y, _, _ = load_dataset(dataset_name, base_dir=str(WORKSPACE_ROOT))
            if hasattr(X, "toarray"):
                X = X.toarray()
            X = np.asarray(X, dtype=np.float32)
            Y = np.asarray(Y, dtype=np.int32)

            cv = get_multilabel_cv(n_splits=5, random_state=42)
            fold_indices = list(cv.split(X, Y))

            for fold_idx, (train_idx, test_idx) in enumerate(fold_indices):
                # Check if already computed
                done = any(
                    r.get("dataset") == dataset_name
                    and r.get("learner") == display_learner
                    and r.get("model") == "GSI_v6_3_2"
                    and int(r.get("fold", -1)) == fold_idx
                    for r in rows
                )
                if done:
                    print(f"  Fold {fold_idx}: ALREADY DONE (Skipping)", flush=True)
                    continue

                X_tr_raw, X_te_raw = X[train_idx], X[test_idx]
                Y_tr, Y_te = Y[train_idx], Y[test_idx]

                scaler = MaxAbsScaler()
                X_tr = scaler.fit_transform(X_tr_raw)
                X_te = scaler.transform(X_te_raw)

                seed = 42 + fold_idx
                clf = GSIMLCPAv6_3_2Classifier(
                    base_learner=learner_key,
                    mf_alpha=0.25,
                    mf_z_max=0.5,
                    random_state=seed,
                )

                t0 = time.time()
                clf.fit(X_tr, Y_tr)
                t_fit = time.time() - t0

                m = evaluate_fold(Y_te, clf, X_te, cost=0.30)
                m["Train_Time_s"] = float(t_fit)

                rec = {
                    "dataset": dataset_name,
                    "learner": display_learner,
                    "model": "GSI_v6_3_2",
                    "fold": fold_idx,
                    **m,
                }
                rows.append(rec)

                print(
                    f"  Fold {fold_idx}: Sel-F1={m['Selective_Macro_F1']:.4f} "
                    f"| Cov={m['Coverage']:.4f} | SubAcc={m['Subset_Accuracy']:.4f} "
                    f"| Time={t_fit:.2f}s",
                    flush=True,
                )

                # Save immediately after each fold
                pd.DataFrame(rows).to_csv(detailed_csv, index=False)

    print("\n" + "=" * 90, flush=True)
    print(f"COMPLETED ALL 10 DATASETS FOR SVM AND MLP! Output: {detailed_csv.name}", flush=True)
    print("=" * 90, flush=True)

if __name__ == "__main__":
    main()
