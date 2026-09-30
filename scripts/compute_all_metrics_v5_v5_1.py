"""Compute complete metric suite comparing GSI v5 vs GSI v5.1 across 10 datasets.

Configured with decision_policy="macro_f1" to exactly match v5 and v5.1 benchmarks.

Metrics evaluated:
- Hamming Loss (Full & Selective)
- Micro-F1 (Full & Selective)
- Subset Accuracy (0/1 Exact Match)
- Precision (Macro Precision & Micro Precision)
- Accuracy (Hamming Accuracy & Example Jaccard Accuracy)
- Macro-F1 (Full & Selective)
- Coverage
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import (
    hamming_loss,
    accuracy_score,
    precision_score,
    f1_score,
    jaccard_score,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.data.loader import load_dataset
from src.evaluation.cv import get_multilabel_cv
from src.evaluation.metrics import (
    compute_selective_macro_f1,
    compute_selective_micro_f1,
)
from src.models.gsi_mlc_pa import GSIMLCPartialAbstentionClassifier

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

def compute_metrics_for_predictions(y_true, y_full, y_sel):
    decided = y_sel != -1
    cov = float(np.mean(decided))
    
    # Hamming loss
    hl_full = float(hamming_loss(y_true, y_full))
    if np.any(decided):
        hl_sel = float(np.mean(y_true[decided] != y_sel[decided]))
    else:
        hl_sel = 0.0
    
    # Accuracy
    hamming_acc = float(1.0 - hl_full)
    hamming_acc_sel = float(1.0 - hl_sel)
    example_acc = float(jaccard_score(y_true, y_full, average="samples", zero_division=1))
    
    # Subset 0/1
    subset_acc = float(accuracy_score(y_true, y_full))
    
    # Precision
    macro_prec = float(precision_score(y_true, y_full, average="macro", zero_division=0))
    micro_prec = float(precision_score(y_true, y_full, average="micro", zero_division=0))
    
    # Selective precision
    sel_prec_per_label = []
    for j in range(y_true.shape[1]):
        dec_j = y_sel[:, j] != -1
        if np.any(dec_j):
            sel_prec_per_label.append(precision_score(y_true[dec_j, j], y_sel[dec_j, j], zero_division=0))
        else:
            sel_prec_per_label.append(0.0)
    sel_macro_prec = float(np.mean(sel_prec_per_label)) if sel_prec_per_label else 0.0
    
    # Micro F1
    micro_f1_full = float(f1_score(y_true, y_full, average="micro", zero_division=0))
    sel_micro_f1 = float(compute_selective_micro_f1(y_true, y_sel))
    
    # Macro F1
    macro_f1_full = float(f1_score(y_true, y_full, average="macro", zero_division=0))
    sel_macro_f1 = float(compute_selective_macro_f1(y_true, y_sel))
    
    return {
        "Coverage": cov,
        "Hamming_Loss_Full": hl_full,
        "Hamming_Loss_Sel": hl_sel,
        "Hamming_Accuracy": hamming_acc,
        "Hamming_Accuracy_Sel": hamming_acc_sel,
        "Example_Accuracy": example_acc,
        "Subset_01_Accuracy": subset_acc,
        "Macro_Precision_Full": macro_prec,
        "Macro_Precision_Sel": sel_macro_prec,
        "Micro_Precision_Full": micro_prec,
        "Micro_F1_Full": micro_f1_full,
        "Micro_F1_Sel": sel_micro_f1,
        "Macro_F1_Full": macro_f1_full,
        "Macro_F1_Sel": sel_macro_f1,
    }

def run_evaluation_for_learner(base_learner="logistic", threshold=0.70):
    results = []
    print(f"\n================ Running for Base Learner: {base_learner} (threshold={threshold}) ================", flush=True)
    for ds_name in ALL_10_DATASETS:
        t0 = time.time()
        X, Y, _, _ = load_dataset(ds_name)
        cv = get_multilabel_cv(n_splits=5, random_state=42)
        
        fold_metrics = {"v5": [], "v5_1": []}
        
        for fold_idx, (tr, te) in enumerate(cv.split(X, Y)):
            scaler = MaxAbsScaler()
            X_tr = scaler.fit_transform(X[tr])
            X_te = scaler.transform(X[te])
            Y_tr, Y_te = Y[tr], Y[te]
            
            # v5 Greedy with decision_policy="macro_f1"
            m_v5 = GSIMLCPartialAbstentionClassifier(
                base_learner=base_learner,
                partition_mode="learned",
                final_order="correlation",
                selection_objective="full_macro_f1",
                decision_policy="macro_f1",
                cost=0.30,
                random_state=42,
            )
            m_v5.fit(X_tr, Y_tr)
            p_v5_full = m_v5.predict_full(X_te)
            p_v5_sel = m_v5.predict(X_te, cost=0.30)
            fold_metrics["v5"].append(compute_metrics_for_predictions(Y_te, p_v5_full, p_v5_sel))
            
            # v5.1 Stratified Peeling + Ascending CC with decision_policy="macro_f1"
            m_v5_1 = GSIMLCPartialAbstentionClassifier(
                base_learner=base_learner,
                partition_mode="stratified_peeling",
                stratified_threshold=threshold,
                dl_order_direction="ascending",
                final_order="ascending_correlation",
                decision_policy="macro_f1",
                cost=0.30,
                random_state=42,
                use_complexity_penalty=False,
            )
            m_v5_1.fit(X_tr, Y_tr)
            p_v5_1_full = m_v5_1.predict_full(X_te)
            p_v5_1_sel = m_v5_1.predict(X_te, cost=0.30)
            fold_metrics["v5_1"].append(compute_metrics_for_predictions(Y_te, p_v5_1_full, p_v5_1_sel))
            
        # Average across 5 folds
        row_v5 = {"Dataset": ds_name, "Base_Learner": base_learner, "Model": "GSI_v5_Greedy"}
        row_v5_1 = {"Dataset": ds_name, "Base_Learner": base_learner, "Model": "GSI_v5_1_Stratified"}
        
        for k in fold_metrics["v5"][0].keys():
            row_v5[k] = round(float(np.mean([m[k] for m in fold_metrics["v5"]])), 4)
            row_v5_1[k] = round(float(np.mean([m[k] for m in fold_metrics["v5_1"]])), 4)
            
        results.append(row_v5)
        results.append(row_v5_1)
        print(f"[{base_learner}] {ds_name:<16s} completed in {time.time()-t0:.2f}s | Sel F1: {row_v5['Macro_F1_Sel']:.4f} -> {row_v5_1['Macro_F1_Sel']:.4f} | HL: {row_v5['Hamming_Loss_Full']:.4f} -> {row_v5_1['Hamming_Loss_Full']:.4f} | Subset: {row_v5['Subset_01_Accuracy']:.4f} -> {row_v5_1['Subset_01_Accuracy']:.4f}", flush=True)
        
    df = pd.DataFrame(results)
    out_path = WORKSPACE_ROOT / f"results_v5_1_test/complete_metrics_{base_learner}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {base_learner} results to {out_path}", flush=True)
    return df

if __name__ == "__main__":
    bl = sys.argv[1] if len(sys.argv) > 1 else "logistic"
    run_evaluation_for_learner(bl)
