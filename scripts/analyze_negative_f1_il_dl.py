"""
Comprehensive script to evaluate Positive F1, Negative F1, and Binary Macro F1
across IL and DL partitions, with a special focus on Rare Labels (Rare Positive vs Rare Negative).
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MaxAbsScaler

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.data.loader import load_dataset
from src.evaluation.cv import get_multilabel_cv
from src.models.gsi_v6_3_3 import GSIMLCPAv6_3_3Classifier

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


def run_negative_f1_analysis(
    datasets: List[str] = DEFAULT_DATASETS,
    base_learner: str = "logistic",
    n_splits: int = 5,
    random_state: int = 42,
    output_dir: Path = WORKSPACE_ROOT / "results_v6_3" / "negative_f1_analysis",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    label_records: List[Dict[str, Any]] = []

    print("\n" + "=" * 90)
    print("ANALYSIS OF POSITIVE VS NEGATIVE F1 ON RARE LABELS ACROSS IL & DL")
    print("=" * 90)

    for d_name in datasets:
        print(f"\n---> Dataset: {d_name} ...", flush=True)
        X, Y, _, _ = load_dataset(d_name, base_dir=str(WORKSPACE_ROOT))
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int32)
        n_samples, K = Y.shape

        cv = get_multilabel_cv(n_splits=n_splits, random_state=random_state)
        fold_indices = list(cv.split(X, Y))

        for fold_idx, (train_idx, test_idx) in enumerate(fold_indices):
            scaler = MaxAbsScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
            Y_tr, Y_te = Y[train_idx], Y[test_idx]

            seed = random_state + fold_idx
            model = GSIMLCPAv6_3_3Classifier(
                base_learner=base_learner,
                random_state=seed,
            )
            model.fit(X_tr, Y_tr)
            preds = model.predict(X_te)

            il_set = set(model.independent_labels_)
            dl_set = set(model.dependent_labels_)

            for j in range(K):
                group = "IL" if j in il_set else "DL"
                y_true_col = Y_te[:, j]
                p_col = preds[:, j]
                decided = p_col != -1
                cov_j = float(np.mean(decided))

                pi_train = float(np.mean(Y_tr[:, j]))
                pi_test = float(np.mean(y_true_col))

                # Categorize label rarity based on training prior pi
                if pi_train < 0.20:
                    rarity_category = "Rare_Positive (pi < 0.20)"
                elif pi_train > 0.575:
                    rarity_category = "Rare_Negative (pi > 0.575)"
                else:
                    rarity_category = "Balanced (0.20 <= pi <= 0.575)"

                if np.any(decided):
                    y_dec = y_true_col[decided]
                    p_dec = p_col[decided]

                    f1_pos = float(f1_score(y_dec, p_dec, pos_label=1, zero_division=0))
                    f1_neg = float(f1_score(y_dec, p_dec, pos_label=0, zero_division=0))
                    prec_pos = float(precision_score(y_dec, p_dec, pos_label=1, zero_division=0))
                    prec_neg = float(precision_score(y_dec, p_dec, pos_label=0, zero_division=0))
                    rec_pos = float(recall_score(y_dec, p_dec, pos_label=1, zero_division=0))
                    rec_neg = float(recall_score(y_dec, p_dec, pos_label=0, zero_division=0))
                else:
                    f1_pos = 0.0
                    f1_neg = 0.0
                    prec_pos = 0.0
                    prec_neg = 0.0
                    rec_pos = 0.0
                    rec_neg = 0.0

                f1_binary_macro = 0.5 * (f1_pos + f1_neg)

                threshold_info = model.adaptive_thresholds_.get(j, {})
                t0 = threshold_info.get("tau_0", 0.30)
                t1 = threshold_info.get("tau_1", 0.70)
                regime = threshold_info.get("regime", "Unknown")

                label_records.append({
                    "Dataset": d_name,
                    "Fold": fold_idx + 1,
                    "Label_Idx": j,
                    "Group": group,
                    "Prior_Train": pi_train,
                    "Prior_Test": pi_test,
                    "Rarity_Category": rarity_category,
                    "Regime": regime,
                    "Tau_0": t0,
                    "Tau_1": t1,
                    "Coverage": cov_j,
                    "F1_Positive": f1_pos,
                    "F1_Negative": f1_neg,
                    "Binary_Macro_F1": f1_binary_macro,
                    "Prec_Positive": prec_pos,
                    "Prec_Negative": prec_neg,
                    "Recall_Positive": rec_pos,
                    "Recall_Negative": rec_neg,
                })

    df = pd.DataFrame(label_records)
    raw_csv = output_dir / "label_level_detailed_metrics.csv"
    df.to_csv(raw_csv, index=False)
    print(f"\n[Saved] Label detailed records: {raw_csv}")

    # Aggregations
    # 1. By Dataset & Group (IL vs DL)
    ds_group = df.groupby(["Dataset", "Group"]).agg({
        "F1_Positive": "mean",
        "F1_Negative": "mean",
        "Binary_Macro_F1": "mean",
        "Prec_Positive": "mean",
        "Prec_Negative": "mean",
        "Recall_Positive": "mean",
        "Recall_Negative": "mean",
        "Coverage": "mean",
        "Prior_Train": "mean",
    }).reset_index()
    ds_group.to_csv(output_dir / "summary_by_dataset_and_group.csv", index=False)

    # 2. By Rarity Category & Group (IL vs DL) across all benchmarks
    rarity_group = df.groupby(["Rarity_Category", "Group"]).agg({
        "F1_Positive": ["mean", "std", "count"],
        "F1_Negative": ["mean", "std"],
        "Binary_Macro_F1": ["mean", "std"],
        "Prec_Positive": "mean",
        "Prec_Negative": "mean",
        "Recall_Positive": "mean",
        "Recall_Negative": "mean",
        "Coverage": "mean",
        "Prior_Train": "mean",
    }).reset_index()
    rarity_group.to_csv(output_dir / "summary_by_rarity_and_group.csv", index=False)

    print("\n" + "=" * 90)
    print("SUMMARY BY RARITY CATEGORY AND GROUP (IL vs DL):")
    print("=" * 90)
    print(rarity_group.to_string())

    # 3. Rare Labels Focus: Protein & Imbalanced datasets breakdown
    rare_df = df[df["Rarity_Category"] == "Rare_Positive (pi < 0.20)"]
    print("\n" + "=" * 90)
    print("FOCUS ON RARE POSITIVE LABELS (pi < 0.20):")
    print(f"Total rare-positive evaluations: {len(rare_df)}")
    rare_summary = rare_df.groupby(["Dataset", "Group"]).agg({
        "F1_Positive": "mean",
        "F1_Negative": "mean",
        "Prec_Negative": "mean",
        "Recall_Negative": "mean",
        "Prec_Positive": "mean",
        "Recall_Positive": "mean",
        "Coverage": "mean",
        "Prior_Train": "mean",
    }).reset_index()
    print(rare_summary.to_string(index=False))
    rare_summary.to_csv(output_dir / "rare_positive_labels_summary.csv", index=False)

    # 4. Rare Negative Focus (pi > 0.575):
    rare_neg_df = df[df["Rarity_Category"] == "Rare_Negative (pi > 0.575)"]
    print("\n" + "=" * 90)
    print("FOCUS ON RARE NEGATIVE LABELS (pi > 0.575):")
    print(f"Total rare-negative evaluations: {len(rare_neg_df)}")
    rare_neg_summary = rare_neg_df.groupby(["Dataset", "Group"]).agg({
        "F1_Positive": "mean",
        "F1_Negative": "mean",
        "Prec_Negative": "mean",
        "Recall_Negative": "mean",
        "Prec_Positive": "mean",
        "Recall_Positive": "mean",
        "Coverage": "mean",
        "Prior_Train": "mean",
    }).reset_index()
    print(rare_neg_summary.to_string(index=False))
    rare_neg_summary.to_csv(output_dir / "rare_negative_labels_summary.csv", index=False)

    return df


if __name__ == "__main__":
    run_negative_f1_analysis()
