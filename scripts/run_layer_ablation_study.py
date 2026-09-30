"""
Script to execute the Layer-Wise Ablation Study (6 Regimes) across datasets for GSI-MLC-PA v5.1.1.

Evaluates:
- ABL_1_ONLY_IL1
- ABL_2_ONLY_IL2
- ABL_2_ACCUM_IL12
- ABL_3_ONLY_IL3
- ABL_3_ALL_IL
- ABL_3_ONLY_DL
- FULL_SYSTEM

Metrics:
- Num Labels
- Coverage
- Selective Macro-F1 & Full Macro-F1
- Selective Macro-Precision & Full Macro-Precision
- Selective Micro-F1 & Full Micro-F1
- Subset 0/1 Accuracy
- Hamming Loss (Full & Selective)
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.data.loader import load_dataset
from src.evaluation.cv import get_multilabel_cv
from src.evaluation.layer_ablation import LayerAblationEvaluator
from src.models.gsi_mlc_pa import GSIMLCPartialAbstentionClassifier

BENCHMARK_DATASETS = [
    "emotions",
    "scene",
    "music",
    "yeast",
    "chd49",
    "gpositivepseaac",
    "genbase",
    "humanpseaac",
    "plantpseaac",
    "viruspseaac",
]


def run_ablation_for_dataset(
    dataset_name: str,
    base_learner: str = "logistic",
    stratified_threshold: float = 0.70,
    sparse_cc_threshold: float = 0.75,
    aug_normalization: str = "matching",
    cost: float = 0.30,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run 5-fold layer ablation evaluation for one dataset."""
    print(f"\n[Ablation] Running {dataset_name} ({base_learner}) | tau={stratified_threshold}, theta={sparse_cc_threshold}...", flush=True)
    t0 = time.time()
    X, Y, _, _ = load_dataset(dataset_name)
    n_samples, n_labels = Y.shape
    cv = get_multilabel_cv(n_splits=n_splits, random_state=random_state)

    fold_records = []

    for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X, Y)):
        scaler = MaxAbsScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        Y_tr, Y_te = Y[tr_idx], Y[te_idx]

        clf = GSIMLCPartialAbstentionClassifier(
            base_learner=base_learner,
            partition_mode="stratified_peeling",
            stratified_threshold=stratified_threshold,
            sparse_cc_threshold=sparse_cc_threshold,
            aug_normalization=aug_normalization,
            dl_order_direction="ascending",
            final_order="ascending_correlation",
            decision_policy="macro_f1",
            cost=cost,
            random_state=random_state,
        )
        clf.fit(X_tr, Y_tr)

        p_full = clf.predict_full(X_te)
        p_sel = clf.predict(X_te, cost=cost)

        audit = clf.stratified_peeling_audit_ or {}
        ind_layers = audit.get("independent_layers", [])
        dl_labels = audit.get("dependent_residual_labels", [])

        evaluator = LayerAblationEvaluator(
            independent_layers=ind_layers,
            dependent_residual_labels=dl_labels,
            total_labels=n_labels,
            abstain_value=clf.abstain_value,
        )

        regime_results = evaluator.evaluate_all_regimes(Y_te, p_full, p_sel)

        for regime_name, res in regime_results.items():
            record = {
                "Dataset": dataset_name,
                "Base_Learner": base_learner,
                "Fold": fold_idx,
                "Regime": regime_name,
                **res,
            }
            fold_records.append(record)

    df_folds = pd.DataFrame(fold_records)

    # Average metrics across folds
    metric_cols = [
        "num_labels",
        "coverage",
        "selective_macro_f1",
        "full_macro_f1",
        "selective_macro_precision",
        "full_macro_precision",
        "selective_micro_f1",
        "full_micro_f1",
        "subset_01_accuracy",
        "hamming_loss_full",
        "hamming_loss_sel",
        "hamming_accuracy_sel",
    ]

    mean_df = (
        df_folds.groupby(["Dataset", "Base_Learner", "Regime"])[metric_cols]
        .mean()
        .reset_index()
    )

    elapsed = time.time() - t0
    print(f"[Ablation] Finished {dataset_name} in {elapsed:.2f}s.", flush=True)
    return mean_df


def main():
    parser = argparse.ArgumentParser(description="Run Layer-Wise Ablation Study for GSI-MLC-PA v5.1.1.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["emotions", "scene"],
        help="Datasets to evaluate (default: emotions scene)",
    )
    parser.add_argument("--base_learner", default="logistic", choices=["logistic", "svm", "mlp"])
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--corr_threshold", type=float, default=0.75)
    parser.add_argument("--aug_norm", default="matching", choices=["matching", "centered", "standard", "logit", "none"])
    parser.add_argument("--cost", type=float, default=0.30)
    args = parser.parse_args()

    all_dfs = []
    for ds in args.datasets:
        df_ds = run_ablation_for_dataset(
            dataset_name=ds,
            base_learner=args.base_learner,
            stratified_threshold=args.threshold,
            sparse_cc_threshold=args.corr_threshold,
            aug_normalization=args.aug_norm,
            cost=args.cost,
        )
        all_dfs.append(df_ds)

    total_df = pd.concat(all_dfs, ignore_index=True)
    out_dir = WORKSPACE_ROOT / "results_v5_1_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"layer_ablation_{args.base_learner}.csv"
    total_df.to_csv(out_path, index=False)
    print(f"\n[Saved] Ablation results saved to: {out_path}")

    # Print summary table
    print("\n" + "=" * 110)
    print(f"BẢNG KẾT QUẢ THỰC NGHIỆM BÓC TÁCH TẦNG NHÃN (ABLATION REGIMES) - Base Learner: {args.base_learner}")
    print("=" * 110)
    display_cols = ["Dataset", "Regime", "num_labels", "coverage", "selective_macro_f1", "full_macro_f1", "selective_macro_precision", "subset_01_accuracy", "hamming_loss_full"]
    print(total_df[display_cols].round(4).to_string(index=False))
    print("=" * 110)


if __name__ == "__main__":
    main()
