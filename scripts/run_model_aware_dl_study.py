"""Empirical Benchmark: Model-Aware Label Correlation and Confidence Ordering in DL (GSI-MLC-PA)

This script implements an isolated study comparing:
- Arm 0 (Baseline v5.1.1): Ground-Truth Phi correlation, ascending correlation order, Sparse CC (theta=0.75).
- Arm 1 (Predicted Prob Corr): Phi_pred on soft predicted probabilities, ascending Phi_pred order, Sparse CC (theta=0.75).
- Arm 2 (Residual Error Corr): Phi_resid on prediction residuals (Y - P_hat), ascending Phi_resid order, Sparse CC (theta=0.25).
- Arm 3 (Confidence Ordering + GT Corr): Validation F1 descending order, Sparse CC on GT Phi (theta=0.75).
- Arm 4 (Full Model-Aware): Validation F1 descending order, Sparse CC on Phi_resid (theta=0.25).
- Arm 5 (Dense CC Baseline): Dense CC (theta=0.0) with context [X, P_IL].

Features used: [X, P_hat(IL)] across all arms.
Gating protocol: 5-Fold Multilabel Stratified CV, c=0.30, tau=0.70.
10 benchmark datasets x 3 base learners (Logistic, LinearSVC, PyTorch MLP GPU).

Results are saved to: results_v5_1_test/Model_Aware_DL_results/
"""

import sys
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.data.loader import load_dataset
from src.evaluation.cv import get_multilabel_cv
from src.evaluation.metrics import (
    compute_selective_macro_f1,
    compute_selective_micro_f1,
)
from src.selection.stratified_peeling import (
    StratifiedPeelingConfig,
    StratifiedPeelingSelector,
    compute_label_correlation_matrix,
    order_dl_by_correlation,
)
from src.selection.model_aware_correlation import (
    compute_predicted_correlation_matrix,
    compute_residual_correlation_matrix,
    order_dl_by_confidence,
    order_dl_by_model_correlation,
)
from src.models.base_learners import create_binary_estimator
from src.models.sparse_chain import SparseClassifierChainClassifier, _ConstantClassifier

BENCHMARK_DATASETS = [
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

BASE_LEARNERS = ["logistic", "svm", "mlp"]

OUTPUT_DIR = WORKSPACE_ROOT / "results_v5_1_test" / "Model_Aware_DL_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _compute_selective_macro_precision(
    y_true: np.ndarray,
    y_partial: np.ndarray,
    abstain_value: int = -1,
) -> float:
    """Compute Macro-Precision restricted to decided positions."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_partial = np.asarray(y_partial, dtype=np.int32)
    n_labels = y_true.shape[1]
    precisions = []

    for l in range(n_labels):
        mask = (y_partial[:, l] != abstain_value)
        if np.sum(mask) == 0:
            continue
        yt = y_true[mask, l]
        yp = y_partial[mask, l]
        if np.sum(yp == 1) == 0:
            p = 1.0 if np.sum(yt == 1) == 0 else 0.0
        else:
            p = float(precision_score(yt, yp, zero_division=0))
        precisions.append(p)

    return float(np.mean(precisions)) if precisions else 0.0


def apply_bop(probabilities: np.ndarray, cost: float = 0.30) -> Tuple[np.ndarray, np.ndarray]:
    """Bayes-Optimal Prediction for MLC-PA with linear penalty."""
    probs = np.asarray(probabilities, dtype=np.float32)
    tau_low = cost
    tau_high = 1.0 - cost

    y_full = (probs >= 0.50).astype(np.int32)
    y_partial = np.full(probs.shape, -1, dtype=np.int32)
    y_partial[probs <= tau_low] = 0
    y_partial[probs >= tau_high] = 1
    return y_full, y_partial


def evaluate_label_metrics(
    y_true: np.ndarray,
    y_full: np.ndarray,
    y_partial: np.ndarray,
    cost: float = 0.30,
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics on decided subspace."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_full = np.asarray(y_full, dtype=np.int32)
    y_partial = np.asarray(y_partial, dtype=np.int32)
    n_samples, n_labels = y_true.shape

    if n_labels == 0:
        return {
            "num_labels": 0,
            "coverage": np.nan,
            "selective_macro_f1": np.nan,
            "full_macro_f1": np.nan,
            "selective_macro_precision": np.nan,
            "full_macro_precision": np.nan,
            "selective_micro_f1": np.nan,
            "full_micro_f1": np.nan,
            "subset_01_accuracy": np.nan,
            "example_accuracy": np.nan,
            "hamming_loss_full": np.nan,
            "hamming_loss_sel": np.nan,
            "hamming_accuracy_sel": np.nan,
            "generalized_loss": np.nan,
        }

    decided_mask = (y_partial != -1)
    coverage = float(np.mean(decided_mask))

    sel_macro_f1 = compute_selective_macro_f1(y_true, y_partial)
    full_macro_f1 = float(f1_score(y_true, y_full, average="macro", zero_division=0))

    sel_macro_prec = _compute_selective_macro_precision(y_true, y_partial)
    full_macro_prec = float(precision_score(y_true, y_full, average="macro", zero_division=0))

    sel_micro_f1 = compute_selective_micro_f1(y_true, y_partial)
    full_micro_f1 = float(f1_score(y_true, y_full, average="micro", zero_division=0))

    if np.sum(decided_mask) > 0:
        hl_sel = float(np.mean(y_true[decided_mask] != y_partial[decided_mask]))
        hl_acc_sel = 1.0 - hl_sel
    else:
        hl_sel = 0.0
        hl_acc_sel = 1.0

    hl_full = float(hamming_loss(y_true, y_full))
    subset_01 = float(accuracy_score(y_true, y_full))

    # Example accuracy (Jaccard)
    intersection = np.logical_and(y_true == 1, y_full == 1).sum(axis=1)
    union = np.logical_or(y_true == 1, y_full == 1).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union == 0, 1.0, intersection / union)
    example_acc = float(np.mean(jaccard))

    # Generalized loss
    error_count = np.sum((y_partial != -1) & (y_partial != y_true))
    abstain_count = np.sum(y_partial == -1)
    gen_loss = float((error_count + cost * abstain_count) / (n_samples * n_labels))

    return {
        "num_labels": n_labels,
        "coverage": coverage,
        "selective_macro_f1": sel_macro_f1,
        "full_macro_f1": full_macro_f1,
        "selective_macro_precision": sel_macro_prec,
        "full_macro_precision": full_macro_prec,
        "selective_micro_f1": sel_micro_f1,
        "full_micro_f1": full_micro_f1,
        "subset_01_accuracy": subset_01,
        "example_accuracy": example_acc,
        "hamming_loss_full": hl_full,
        "hamming_loss_sel": hl_sel,
        "hamming_accuracy_sel": hl_acc_sel,
        "generalized_loss": gen_loss,
    }


def evaluate_model_aware_for_dataset(
    dataset_name: str,
    base_learner: str,
    n_splits: int = 5,
    stratified_threshold: float = 0.70,
    cost: float = 0.30,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """Run model-aware correlation study for a single dataset and base learner."""
    print(f"\n=======================================================", flush=True)
    print(f"Dataset: {dataset_name} | Base Learner: {base_learner}", flush=True)
    print(f"=======================================================", flush=True)

    X_full, Y_full, _, _ = load_dataset(dataset_name)
    n_samples, n_labels = Y_full.shape

    cv = get_multilabel_cv(
        n_splits=n_splits,
        random_state=random_state,
        shuffle=True,
    )

    records: List[Dict[str, Any]] = []
    t_start = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_full, Y_full)):
        fold_t0 = time.time()

        # Fold splitting and scaling
        X_tr_raw, X_te_raw = X_full[train_idx], X_full[test_idx]
        Y_tr, Y_te = Y_full[train_idx], Y_full[test_idx]

        scaler = MaxAbsScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)

        n_tr = X_tr.shape[0]
        n_val = max(1, int(0.20 * n_tr))
        rng = np.random.default_rng(random_state + fold_idx)
        val_indices = rng.choice(n_tr, size=n_val, replace=False)
        sub_tr_mask = np.ones(n_tr, dtype=bool)
        sub_tr_mask[val_indices] = False
        sub_tr_indices = np.where(sub_tr_mask)[0]

        peeling_cfg = StratifiedPeelingConfig(
            threshold=stratified_threshold,
            max_depth=3,
            dl_order_direction="ascending",
        )
        peeling_selector = StratifiedPeelingSelector(
            config=peeling_cfg,
            base_estimator_factory=lambda: create_binary_estimator(base_learner, random_state=random_state),
        )
        peeling_res = peeling_selector.fit_partition(
            X_tr[sub_tr_indices],
            Y_tr[sub_tr_indices],
            X_tr[val_indices],
            Y_tr[val_indices],
        )

        all_il = sorted(list(peeling_res.all_independent_labels))
        dl_labels = sorted(list(peeling_res.dependent_residual_labels))

        # -------------------------------------------------------------
        # 2. Fit Independent Labels (IL) via BR on X_tr
        # -------------------------------------------------------------
        il_probs_tr = np.zeros((X_tr.shape[0], len(all_il)), dtype=np.float32)
        il_probs_te = np.zeros((X_te.shape[0], len(all_il)), dtype=np.float32)

        for idx, lbl in enumerate(all_il):
            unique_classes = np.unique(Y_tr[:, lbl])
            if len(unique_classes) <= 1:
                const_val = int(unique_classes[0]) if len(unique_classes) == 1 else 0
                clf = _ConstantClassifier(const_val)
            else:
                clf = create_binary_estimator(base_learner, random_state=random_state)
                clf.fit(X_tr, Y_tr[:, lbl])
            if hasattr(clf, "predict_proba"):
                probs_tr = clf.predict_proba(X_tr)
                probs_te = clf.predict_proba(X_te)
                il_probs_tr[:, idx] = probs_tr[:, 1] if probs_tr.ndim == 2 and probs_tr.shape[1] > 1 else probs_tr.ravel()
                il_probs_te[:, idx] = probs_te[:, 1] if probs_te.ndim == 2 and probs_te.shape[1] > 1 else probs_te.ravel()
            elif hasattr(clf, "decision_function"):
                s_tr = clf.decision_function(X_tr)
                s_te = clf.decision_function(X_te)
                il_probs_tr[:, idx] = 1.0 / (1.0 + np.exp(-np.clip(s_tr, -30.0, 30.0)))
                il_probs_te[:, idx] = 1.0 / (1.0 + np.exp(-np.clip(s_te, -30.0, 30.0)))
            else:
                il_probs_tr[:, idx] = clf.predict(X_tr)
                il_probs_te[:, idx] = clf.predict(X_te)

        # Build Context Feature matrix: [X, P_hat(Y_IL)]
        if len(all_il) > 0:
            X_context_tr = np.hstack([X_tr, il_probs_tr])
            X_context_te = np.hstack([X_te, il_probs_te])
        else:
            X_context_tr = X_tr
            X_context_te = X_te

        # -------------------------------------------------------------
        # 3. Model-Aware Matrix Estimation on Internal Validation Split
        # -------------------------------------------------------------
        if len(dl_labels) > 0:
            # Internal 80/20 train/val split of training fold to estimate P_hat without leakage
            rng = np.random.RandomState(random_state + fold_idx * 17)
            n_tr = X_context_tr.shape[0]
            val_size = max(1, int(0.20 * n_tr))
            shuffled_idx = rng.permutation(n_tr)
            sub_tr_idx = shuffled_idx[val_size:]
            sub_val_idx = shuffled_idx[:val_size]

            X_sub_tr, X_sub_val = X_context_tr[sub_tr_idx], X_context_tr[sub_val_idx]
            Y_sub_tr, Y_sub_val = Y_tr[sub_tr_idx], Y_tr[sub_val_idx]

            # Fit base estimators on sub_tr to generate probability predictions on sub_val
            dl_probs_val = np.zeros((len(sub_val_idx), n_labels), dtype=np.float32)
            dl_confidence_f1: Dict[int, float] = {}

            for lbl in dl_labels:
                unique_classes = np.unique(Y_sub_tr[:, lbl])
                if len(unique_classes) <= 1:
                    const_val = int(unique_classes[0]) if len(unique_classes) == 1 else 0
                    c_clf = _ConstantClassifier(const_val)
                    preds_val = np.full(len(sub_val_idx), const_val, dtype=np.float32)
                else:
                    c_clf = create_binary_estimator(base_learner, random_state=random_state)
                    c_clf.fit(X_sub_tr, Y_sub_tr[:, lbl])
                    if hasattr(c_clf, "predict_proba"):
                        pv = c_clf.predict_proba(X_sub_val)
                        preds_val = pv[:, 1] if pv.ndim == 2 and pv.shape[1] > 1 else pv.ravel()
                    elif hasattr(c_clf, "decision_function"):
                        sv = c_clf.decision_function(X_sub_val)
                        preds_val = 1.0 / (1.0 + np.exp(-np.clip(sv, -30.0, 30.0)))
                    else:
                        preds_val = c_clf.predict(X_sub_val).astype(np.float32)

                dl_probs_val[:, lbl] = preds_val
                # Calculate internal validation F1
                bin_preds = (preds_val >= 0.50).astype(np.int32)
                dl_confidence_f1[lbl] = float(f1_score(Y_sub_val[:, lbl], bin_preds, zero_division=0))

            # Matrices:
            # 1. Ground Truth Phi correlation
            gt_corr_mat = compute_label_correlation_matrix(Y_tr)
            gt_order = order_dl_by_correlation(gt_corr_mat, dl_labels, direction="ascending")

            # 2. Predicted Prob Correlation (Phi_pred)
            pred_corr_mat = compute_predicted_correlation_matrix(dl_probs_val)
            pred_order = order_dl_by_model_correlation(pred_corr_mat, dl_labels, direction="ascending")

            # 3. Residual Error Correlation (Phi_resid)
            resid_corr_mat = compute_residual_correlation_matrix(Y_sub_val, dl_probs_val)
            resid_order = order_dl_by_model_correlation(resid_corr_mat, dl_labels, direction="ascending")

            # 4. Confidence Ordering (Validation F1 descending)
            conf_order = order_dl_by_confidence(dl_confidence_f1, dl_labels, direction="descending")

            # -------------------------------------------------------------
            # 4. Evaluate the 6 Target Experimental Arms
            # -------------------------------------------------------------
            arms = [
                # Arm 0: Baseline v5.1.1
                ("Arm_0_GT_Corr_Ascending_Sparse_075", gt_corr_mat, gt_order, 0.75, "Baseline_GT"),
                # Arm 1: Predicted Probability Correlation
                ("Arm_1_Pred_Prob_Corr_Sparse_075", pred_corr_mat, pred_order, 0.75, "Model_Aware_Pred"),
                # Arm 2: Residual Error Correlation
                ("Arm_2_Residual_Corr_Sparse_025", resid_corr_mat, resid_order, 0.25, "Model_Aware_Resid"),
                # Arm 3: Confidence Ordering + GT Correlation
                ("Arm_3_Confidence_Order_GT_Sparse_075", gt_corr_mat, conf_order, 0.75, "Confidence_Order"),
                # Arm 4: Full Model-Aware (Confidence Order + Residual Sparsification)
                ("Arm_4_Full_Model_Aware_Sparse_025", resid_corr_mat, conf_order, 0.25, "Full_Model_Aware"),
                # Arm 5: Dense CC Baseline
                ("Arm_5_Dense_CC", gt_corr_mat, gt_order, 0.00, "Dense_CC"),
            ]

            for arm_name, c_matrix, chain_order, c_thresh, arm_category in arms:
                clf_dl = SparseClassifierChainClassifier(
                    base_estimator=base_learner,
                    order=chain_order,
                    correlation_matrix=c_matrix,
                    correlation_threshold=c_thresh,
                    random_state=random_state,
                )
                clf_dl.fit(X_context_tr, Y_tr)
                p_dl_all = clf_dl.predict_proba(X_context_te)
                p_dl_te = p_dl_all[:, dl_labels]

                y_dl_full, y_dl_partial = apply_bop(p_dl_te, cost=cost)
                m_dl = evaluate_label_metrics(Y_te[:, dl_labels], y_dl_full, y_dl_partial, cost=cost)

                # Count active edges in chain
                active_edges = sum(len(parents) for parents in clf_dl.active_parents_map_.values())
                avg_parents = active_edges / len(dl_labels) if len(dl_labels) > 0 else 0.0

                records.append({
                    "Dataset": dataset_name,
                    "Base_Learner": base_learner,
                    "Fold": fold_idx,
                    "Arm": arm_name,
                    "Arm_Category": arm_category,
                    "Corr_Threshold": c_thresh,
                    "Avg_Active_Parents": avg_parents,
                    **m_dl,
                })
        else:
            # DL is empty (all labels IL)
            for arm_name, cat in [
                ("Arm_0_GT_Corr_Ascending_Sparse_075", "Baseline_GT"),
                ("Arm_1_Pred_Prob_Corr_Sparse_075", "Model_Aware_Pred"),
                ("Arm_2_Residual_Corr_Sparse_025", "Model_Aware_Resid"),
                ("Arm_3_Confidence_Order_GT_Sparse_075", "Confidence_Order"),
                ("Arm_4_Full_Model_Aware_Sparse_025", "Full_Model_Aware"),
                ("Arm_5_Dense_CC", "Dense_CC"),
            ]:
                records.append({
                    "Dataset": dataset_name,
                    "Base_Learner": base_learner,
                    "Fold": fold_idx,
                    "Arm": arm_name,
                    "Arm_Category": cat,
                    "Corr_Threshold": 0.75,
                    "Avg_Active_Parents": 0.0,
                    "num_labels": 0,
                    "coverage": np.nan,
                    "selective_macro_f1": np.nan,
                    "full_macro_f1": np.nan,
                    "selective_macro_precision": np.nan,
                    "full_macro_precision": np.nan,
                    "selective_micro_f1": np.nan,
                    "full_micro_f1": np.nan,
                    "subset_01_accuracy": np.nan,
                    "example_accuracy": np.nan,
                    "hamming_loss_full": np.nan,
                    "hamming_loss_sel": np.nan,
                    "hamming_accuracy_sel": np.nan,
                    "generalized_loss": np.nan,
                })

        fold_elapsed = time.time() - fold_t0
        print(f"  Fold {fold_idx+1}/5 finished ({fold_elapsed:.2f}s) | DL labels: {len(dl_labels)}", flush=True)

    print(f"[Done] {dataset_name} ({base_learner}) finished in {time.time() - t_start:.2f}s", flush=True)
    return records


def main():
    print("=" * 70)
    print("STARTING MODEL-AWARE LABEL CORRELATION BENCHMARK (GSI-MLC-PA)")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Datasets: {BENCHMARK_DATASETS}")
    print(f"Base Learners: {BASE_LEARNERS}")
    print("=" * 70, flush=True)

    all_records: List[Dict[str, Any]] = []
    global_t0 = time.time()

    for ds in BENCHMARK_DATASETS:
        for bl in BASE_LEARNERS:
            recs = evaluate_model_aware_for_dataset(
                dataset_name=ds,
                base_learner=bl,
                n_splits=5,
                stratified_threshold=0.70,
                cost=0.30,
                random_state=42,
            )
            all_records.extend(recs)

            # Checkpoint detailed folds
            df_folds = pd.DataFrame(all_records)
            df_folds.to_csv(OUTPUT_DIR / "model_aware_detailed_folds.csv", index=False)

    df_folds = pd.DataFrame(all_records)
    df_folds.to_csv(OUTPUT_DIR / "model_aware_detailed_folds.csv", index=False)

    # Aggregate across 5 folds
    agg_cols = [
        "coverage",
        "selective_macro_f1",
        "full_macro_f1",
        "selective_macro_precision",
        "full_macro_precision",
        "selective_micro_f1",
        "full_micro_f1",
        "subset_01_accuracy",
        "example_accuracy",
        "hamming_loss_full",
        "hamming_loss_sel",
        "hamming_accuracy_sel",
        "generalized_loss",
        "Avg_Active_Parents",
    ]
    summary_df = (
        df_folds.groupby(["Dataset", "Base_Learner", "Arm", "Arm_Category", "Corr_Threshold"])[agg_cols]
        .mean()
        .reset_index()
    )
    summary_df.to_csv(OUTPUT_DIR / "model_aware_summary.csv", index=False)

    # Create Comparison Pivot vs Baseline Arm 0
    piv_f1 = summary_df.pivot_table(
        index=["Dataset", "Base_Learner"],
        columns="Arm",
        values="selective_macro_f1",
    )
    piv_f1.to_csv(OUTPUT_DIR / "table_arm_f1_comparison.csv")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"Total execution time: {time.time() - global_t0:.2f}s")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
