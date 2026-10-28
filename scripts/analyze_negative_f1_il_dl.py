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


def run_negative_f1_analysis(
    datasets: List[str] = DEFAULT_DATASETS,
    base_learner: str = "logistic",
    n_splits: int = 5,
    random_state: int = 42,
    model_version: str = "v6_3_3",
    output_dir: Path = WORKSPACE_ROOT / "results_v6_3" / "negative_f1_analysis",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    label_records: List[Dict[str, Any]] = []
    pooled_predictions: Dict[Any, Dict[str, Any]] = {}
    model_classes = {
        "v6_3_3": GSIMLCPAv6_3_3Classifier,
        "v6_3_4": GSIMLCPAv6_3_4Classifier,
        "v6_3_5": GSIMLCPAv6_3_5Classifier,
    }
    if model_version not in model_classes:
        raise ValueError(f"Unknown model_version={model_version!r}")
    model_class = model_classes[model_version]

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
            model = model_class(
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
                positive_mask = y_true_col == 1
                negative_mask = ~positive_mask
                positive_coverage = (
                    float(np.mean(decided[positive_mask])) if np.any(positive_mask) else np.nan
                )
                negative_coverage = (
                    float(np.mean(decided[negative_mask])) if np.any(negative_mask) else np.nan
                )

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
                    tp = int(np.sum((y_dec == 1) & (p_dec == 1)))
                    fp = int(np.sum((y_dec == 0) & (p_dec == 1)))
                    tn = int(np.sum((y_dec == 0) & (p_dec == 0)))
                    fn_decided = int(np.sum((y_dec == 1) & (p_dec == 0)))
                else:
                    f1_pos = 0.0
                    f1_neg = 0.0
                    prec_pos = 0.0
                    prec_neg = 0.0
                    rec_pos = 0.0
                    rec_neg = 0.0
                    tp = fp = tn = fn_decided = 0

                support_positive = int(np.sum(positive_mask))
                support_negative = int(np.sum(negative_mask))
                positive_abstained = int(np.sum(positive_mask & ~decided))
                negative_abstained = int(np.sum(negative_mask & ~decided))
                recall_positive_end_to_end = (
                    float(tp / support_positive) if support_positive else np.nan
                )

                f1_binary_macro = 0.5 * (f1_pos + f1_neg)

                threshold_info = model.adaptive_thresholds_.get(j, {})
                t0 = threshold_info.get("tau_0", 0.30)
                t1 = threshold_info.get("tau_1", 0.70)
                regime = threshold_info.get("regime", "Unknown")

                pooled_key = (d_name, j)
                pooled_entry = pooled_predictions.setdefault(
                    pooled_key,
                    {
                        "Dataset": d_name,
                        "Label_Idx": j,
                        "Group_Values": [],
                        "Y_True": [],
                        "Y_Pred": [],
                        "Prior_Train": [],
                    },
                )
                pooled_entry["Group_Values"].append(group)
                pooled_entry["Y_True"].append(y_true_col.copy())
                pooled_entry["Y_Pred"].append(p_col.copy())
                pooled_entry["Prior_Train"].append(pi_train)

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
                    "Positive_Coverage": positive_coverage,
                    "Negative_Coverage": negative_coverage,
                    "Support_Positive": support_positive,
                    "Support_Negative": support_negative,
                    "TP": tp,
                    "FP": fp,
                    "TN": tn,
                    "FN_Decided": fn_decided,
                    "Positive_Abstained": positive_abstained,
                    "Negative_Abstained": negative_abstained,
                    "F1_Positive": f1_pos,
                    "F1_Negative": f1_neg,
                    "Binary_Macro_F1": f1_binary_macro,
                    "Prec_Positive": prec_pos,
                    "Prec_Negative": prec_neg,
                    "Recall_Positive": rec_pos,
                    "Recall_Positive_EndToEnd": recall_positive_end_to_end,
                    "Recall_Negative": rec_neg,
                    "SAMG_Strategy": threshold_info.get("samg_strategy", "unknown"),
                    "SAMG_Applied": threshold_info.get("samg_applied", False),
                    "SAMG_Reason": threshold_info.get("samg_reason", "unknown"),
                    "SAMG_Probability_Source": threshold_info.get(
                        "samg_probability_source", "unknown"
                    ),
                    "SAMG_Pos_Floor": threshold_info.get("samg_pos_floor", np.nan),
                    "SAMG_Neg_Ceiling": threshold_info.get("samg_neg_ceiling", np.nan),
                })

    df = pd.DataFrame(label_records)
    raw_csv = output_dir / "label_level_detailed_metrics.csv"
    df.to_csv(raw_csv, index=False)
    print(f"\n[Saved] Label detailed records: {raw_csv}")

    pooled_rows: List[Dict[str, Any]] = []
    for entry in pooled_predictions.values():
        y_true = np.concatenate(entry["Y_True"])
        y_pred = np.concatenate(entry["Y_Pred"])
        decided = y_pred != -1
        y_dec = y_true[decided]
        p_dec = y_pred[decided]
        positive_mask = y_true == 1
        negative_mask = ~positive_mask
        tp = int(np.sum((y_dec == 1) & (p_dec == 1)))
        support_positive = int(np.sum(positive_mask))
        pooled_rows.append({
            "Dataset": entry["Dataset"],
            "Label_Idx": entry["Label_Idx"],
            "Group": "/".join(sorted(set(entry["Group_Values"]))),
            "Prior_Train": float(np.mean(entry["Prior_Train"])),
            "Support_Positive": support_positive,
            "Support_Negative": int(np.sum(negative_mask)),
            "Coverage": float(np.mean(decided)),
            "Positive_Coverage": float(np.mean(decided[positive_mask])) if np.any(positive_mask) else np.nan,
            "Negative_Coverage": float(np.mean(decided[negative_mask])) if np.any(negative_mask) else np.nan,
            "F1_Positive": float(f1_score(y_dec, p_dec, pos_label=1, zero_division=0)),
            "F1_Negative": float(f1_score(y_dec, p_dec, pos_label=0, zero_division=0)),
            "Recall_Positive_Selective": float(recall_score(y_dec, p_dec, pos_label=1, zero_division=0)),
            "Recall_Positive_EndToEnd": float(tp / support_positive) if support_positive else np.nan,
            "Insufficient_Support": support_positive < n_splits,
        })
    pd.DataFrame(pooled_rows).to_csv(
        output_dir / "pooled_label_metrics.csv", index=False
    )

    threshold_columns = [
        "Dataset", "Fold", "Label_Idx", "Group", "Prior_Train", "Tau_0", "Tau_1",
        "SAMG_Strategy", "SAMG_Applied", "SAMG_Reason", "SAMG_Probability_Source",
        "SAMG_Pos_Floor", "SAMG_Neg_Ceiling",
    ]
    df[threshold_columns].to_csv(output_dir / "samg_threshold_audit.csv", index=False)

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
        "Positive_Coverage": "mean",
        "Negative_Coverage": "mean",
        "Recall_Positive_EndToEnd": "mean",
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_version", choices=("v6_3_3", "v6_3_4", "v6_3_5"), default="v6_3_3"
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()
    selected_output = args.output_dir or (
        WORKSPACE_ROOT / "results_v6_3" / f"negative_f1_analysis_{args.model_version}"
    )
    run_negative_f1_analysis(
        datasets=args.datasets,
        model_version=args.model_version,
        output_dir=selected_output,
    )
