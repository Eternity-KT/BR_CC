"""
Benchmark Script for GSI-MLC-PA v6.1.1 across all 10 datasets.

v6.1.1 Architecture:
- StratifiedPeelingSelector (v5.1 validation-based label peeling, tau=0.70, val_size=0.20, max_depth=3).
- Independent Labels (IL): Binary Relevance (BR) with abstention cost c = 0.30.
- Dependent Labels (DL): Feature-Augmented Ensemble of Classifier Chains (ECC, M=10 chains).
  Augmentation: X_aug = [X, Normalize(P_IL)].
  Prediction: Average probability across chains, abstention cost c = 0.30.
- Preprocessing: MaxAbsScaler fit on train fold, applied to test fold.
- Evaluation: 5-fold CV (seed=42) on all 10 benchmark datasets.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.data.loader import load_dataset
from src.evaluation.cv import get_multilabel_cv
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_selective_macro_f1,
    compute_selective_micro_f1,
)
from src.models.gsi_v6_1_1 import GSIMLCPAv6_1_1Classifier

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

BASE_LEARNER = "logistic"
STRATIFIED_THRESHOLD = 0.70
VALIDATION_SIZE = 0.20
MAX_PEELING_DEPTH = 3
N_CHAINS = 10
COST = 0.30
N_FOLDS = 5
RANDOM_STATE = 42

OUTPUT_DIR = WORKSPACE_ROOT / "results_v6_1_1"


def evaluate_fold(y_test: np.ndarray, model: GSIMLCPAv6_1_1Classifier, X_test: np.ndarray, cost: float = 0.30) -> Dict[str, float]:
    full_preds = model.predict_full(X_test)
    sel_preds = model.predict(X_test, cost=cost)

    full_metrics = compute_all_metrics(y_test, full_preds)
    sel_macro_f1 = compute_selective_macro_f1(y_test, sel_preds, abstain_value=-1)
    sel_micro_f1 = compute_selective_micro_f1(y_test, sel_preds, abstain_value=-1)
    coverage = float(np.mean(sel_preds != -1))
    decided = sel_preds != -1
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detailed_csv_path = OUTPUT_DIR / "v6_1_1_detailed_folds.csv"
    summary_csv_path = OUTPUT_DIR / "v6_1_1_summary.csv"
    comparison_csv_path = OUTPUT_DIR / "v6_1_1_vs_baselines_10datasets.csv"
    report_md_path = OUTPUT_DIR / "v6_1_1_benchmark_report.md"

    # Baseline references
    # 1. v5.1.1 baseline from results_v5_1_test/layer_ablation_logistic.csv
    v511_df = pd.read_csv(WORKSPACE_ROOT / "results_v5_1_test/layer_ablation_logistic.csv")
    v511_full = v511_df[v511_df["Regime"] == "FULL_SYSTEM"].set_index("Dataset")

    # 2. v5.1 baseline from results_v5_1_test/benchmark_3_base_learners_summary.csv
    v5_df = pd.read_csv(WORKSPACE_ROOT / "results_v5_1_test/benchmark_3_base_learners_summary.csv")
    v5_log = v5_df[v5_df["Model"] == "GSI_v5_1_Stratified_Logistic"].set_index("Dataset")

    # 3. v6.1 baseline from results_v6_1/v6_1_summary.csv
    v61_df = pd.read_csv(WORKSPACE_ROOT / "results_v6_1/v6_1_summary.csv")
    v61_log = v61_df[(v61_df["model"] == "GSI_v6_1") & (v61_df["base_learner"] == "Logistic")].set_index("dataset")

    detailed_records: List[Dict[str, Any]] = []
    summary_records: List[Dict[str, Any]] = []

    print("\n" + "=" * 90, flush=True)
    print("STARTING FULL BENCHMARK: GSI-MLC-PA v6.1.1 ON ALL 10 DATASETS", flush=True)
    print(f"Base Learner: {BASE_LEARNER} | Cost: {COST} | Chains: {N_CHAINS} | Tau: {STRATIFIED_THRESHOLD} | Max Depth: {MAX_PEELING_DEPTH}", flush=True)
    print("=" * 90, flush=True)

    for d_idx, dataset_name in enumerate(ALL_10_DATASETS, 1):
        print(f"\n[{d_idx}/{len(ALL_10_DATASETS)}] Loading dataset: {dataset_name} ...", flush=True)
        X, Y, _, _ = load_dataset(dataset_name, base_dir=str(WORKSPACE_ROOT))
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int32)
        n_samples, n_labels = Y.shape
        n_features = X.shape[1]
        print(f"     N = {n_samples}, d = {n_features}, K = {n_labels}", flush=True)

        cv = get_multilabel_cv(n_splits=N_FOLDS, random_state=RANDOM_STATE)

        fold_res_list = []
        fold_times = []
        fold_il_counts = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, Y)):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            # Standard MaxAbsScaler strictly fit on training fold
            scaler = MaxAbsScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = GSIMLCPAv6_1_1Classifier(
                base_learner=BASE_LEARNER,
                stratified_threshold=STRATIFIED_THRESHOLD,
                validation_size=VALIDATION_SIZE,
                max_peeling_depth=MAX_PEELING_DEPTH,
                aug_normalization="matching",
                n_chains=N_CHAINS,
                cost=COST,
                random_state=RANDOM_STATE + fold_idx * 17,
            )

            t0 = time.perf_counter()
            model.fit(X_train, Y_train)
            fit_time = time.perf_counter() - t0

            eval_res = evaluate_fold(Y_test, model, X_test, cost=COST)
            il_count = len(getattr(model, "independent_labels_", []))

            fold_res_list.append(eval_res)
            fold_times.append(fit_time)
            fold_il_counts.append(il_count)

            detailed_record = {
                "dataset": dataset_name,
                "base_learner": "Logistic",
                "fold": fold_idx + 1,
                "model": "GSI_v6_1_1",
                **eval_res,
                "il_count": il_count,
                "fit_time_sec": fit_time,
            }
            detailed_records.append(detailed_record)

        mean_sel_f1 = float(np.mean([r["Selective_Macro_F1"] for r in fold_res_list]))
        std_sel_f1 = float(np.std([r["Selective_Macro_F1"] for r in fold_res_list]))
        mean_full_f1 = float(np.mean([r["Full_Macro_F1"] for r in fold_res_list]))
        mean_cov = float(np.mean([r["Coverage"] for r in fold_res_list]))
        mean_subset = float(np.mean([r["Subset_Accuracy"] for r in fold_res_list]))
        mean_hamming = float(np.mean([r["Hamming_Loss"] for r in fold_res_list]))
        mean_time = float(np.mean(fold_times))
        mean_il = float(np.mean(fold_il_counts))

        sum_rec = {
            "dataset": dataset_name,
            "base_learner": "Logistic",
            "model": "GSI_v6_1_1",
            "n_samples": n_samples,
            "n_labels": n_labels,
            "selective_macro_f1": mean_sel_f1,
            "selective_macro_f1_std": std_sel_f1,
            "coverage": mean_cov,
            "full_macro_f1": mean_full_f1,
            "subset_accuracy": mean_subset,
            "hamming_loss": mean_hamming,
            "mean_il_labels": mean_il,
            "fit_time_sec": mean_time,
        }
        summary_records.append(sum_rec)

        v511_sel = v511_full.loc[dataset_name, "selective_macro_f1"] if dataset_name in v511_full.index else np.nan
        v511_full_val = v511_full.loc[dataset_name, "full_macro_f1"] if dataset_name in v511_full.index else np.nan
        v511_cov_val = v511_full.loc[dataset_name, "coverage"] if dataset_name in v511_full.index else np.nan

        print(
            f"   -> v6.1.1: Sel-F1 = {mean_sel_f1:.4f}±{std_sel_f1:.4f} | "
            f"Cov = {mean_cov*100:5.1f}% | Full-F1 = {mean_full_f1:.4f} | IL = {mean_il:.1f}/{n_labels}",
            flush=True,
        )
        print(
            f"   -> v5.1.1: Sel-F1 = {v511_sel:.4f} | Cov = {v511_cov_val*100:5.1f}% | Full-F1 = {v511_full_val:.4f}",
            flush=True,
        )

    # Save detailed and summary CSVs
    df_detailed = pd.DataFrame(detailed_records)
    df_summary = pd.DataFrame(summary_records)
    df_detailed.to_csv(detailed_csv_path, index=False)
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"\n[OK] Saved detailed fold records to: {detailed_csv_path}", flush=True)
    print(f"[OK] Saved summary records to: {summary_csv_path}", flush=True)

    # Build Comparison Table vs v5.1, v5.1.1, v6.1
    comp_rows = []
    for d in ALL_10_DATASETS:
        v611_row = df_summary[df_summary["dataset"] == d].iloc[0]
        v511_sel = float(v511_full.loc[d, "selective_macro_f1"]) if d in v511_full.index else np.nan
        v511_full_val = float(v511_full.loc[d, "full_macro_f1"]) if d in v511_full.index else np.nan
        v511_cov_val = float(v511_full.loc[d, "coverage"]) if d in v511_full.index else np.nan

        v5_sel = float(v5_log.loc[d, "Selective_Macro_F1"]) if d in v5_log.index else np.nan
        v5_full_val = float(v5_log.loc[d, "Full_Macro_F1"]) if d in v5_log.index else np.nan
        v5_cov_val = float(v5_log.loc[d, "Coverage"]) if d in v5_log.index else np.nan

        v61_sel = float(v61_log.loc[d, "selective_macro_f1"]) if d in v61_log.index else np.nan
        v61_full_val = float(v61_log.loc[d, "full_macro_f1"]) if d in v61_log.index else np.nan
        v61_cov_val = float(v61_log.loc[d, "coverage"]) if d in v61_log.index else np.nan

        # Comparison v6.1.1 vs v5.1.1
        diff_sel = v611_row["selective_macro_f1"] - v511_sel
        diff_full = v611_row["full_macro_f1"] - v511_full_val
        diff_cov = v611_row["coverage"] - v511_cov_val

        # Status (>3% win, <-3% loss, else competitive)
        rel_diff_full = (diff_full / v511_full_val) * 100 if v511_full_val > 0 else 0.0
        rel_diff_sel = (diff_sel / v511_sel) * 100 if v511_sel > 0 else 0.0

        if rel_diff_full >= 3.0:
            full_status = "WIN"
        elif rel_diff_full <= -3.0:
            full_status = "LOSS"
        else:
            full_status = "COMPETITIVE"

        if rel_diff_sel >= 3.0:
            sel_status = "WIN"
        elif rel_diff_sel <= -3.0:
            sel_status = "LOSS"
        else:
            sel_status = "COMPETITIVE"

        comp_rows.append({
            "dataset": d,
            "v5_1_sel_f1": v5_sel,
            "v5_1_1_sel_f1": v511_sel,
            "v6_1_sel_f1": v61_sel,
            "v6_1_1_sel_f1": v611_row["selective_macro_f1"],
            "diff_sel_f1": diff_sel,
            "rel_diff_sel_pct": rel_diff_sel,
            "sel_status": sel_status,
            "v5_1_full_f1": v5_full_val,
            "v5_1_1_full_f1": v511_full_val,
            "v6_1_full_f1": v61_full_val,
            "v6_1_1_full_f1": v611_row["full_macro_f1"],
            "diff_full_f1": diff_full,
            "rel_diff_full_pct": rel_diff_full,
            "full_status": full_status,
            "v5_1_cov": v5_cov_val,
            "v5_1_1_cov": v511_cov_val,
            "v6_1_cov": v61_cov_val,
            "v6_1_1_cov": v611_row["coverage"],
            "diff_cov": diff_cov,
        })

    df_comp = pd.DataFrame(comp_rows)
    df_comp.to_csv(comparison_csv_path, index=False)
    print(f"[OK] Saved comparison results to: {comparison_csv_path}", flush=True)

    print("\n" + "=" * 90)
    print("COMPARISON: v6.1.1 vs v5.1.1 vs v6.1 (FULL MACRO-F1)")
    print("=" * 90)
    print(df_comp[["dataset", "v5_1_1_full_f1", "v6_1_full_f1", "v6_1_1_full_f1", "rel_diff_full_pct", "full_status"]].to_string(index=False))

    print("\n" + "=" * 90)
    print("COMPARISON: v6.1.1 vs v5.1.1 vs v6.1 (SELECTIVE MACRO-F1)")
    print("=" * 90)
    print(df_comp[["dataset", "v5_1_1_sel_f1", "v6_1_sel_f1", "v6_1_1_sel_f1", "rel_diff_sel_pct", "sel_status"]].to_string(index=False))

    print("\n" + "=" * 90)
    print("COMPARISON: COVERAGE")
    print("=" * 90)
    print(df_comp[["dataset", "v5_1_1_cov", "v6_1_cov", "v6_1_1_cov", "diff_cov"]].to_string(index=False))

    print("\n" + "=" * 90)
    print("AVERAGES ACROSS ALL 10 DATASETS:")
    print("=" * 90)
    print(f"v5.1:   Selective F1 = {df_comp['v5_1_sel_f1'].mean():.4f} | Full F1 = {df_comp['v5_1_full_f1'].mean():.4f} | Coverage = {df_comp['v5_1_cov'].mean():.2%}")
    print(f"v5.1.1: Selective F1 = {df_comp['v5_1_1_sel_f1'].mean():.4f} | Full F1 = {df_comp['v5_1_1_full_f1'].mean():.4f} | Coverage = {df_comp['v5_1_1_cov'].mean():.2%}")
    print(f"v6.1:   Selective F1 = {df_comp['v6_1_sel_f1'].mean():.4f} | Full F1 = {df_comp['v6_1_full_f1'].mean():.4f} | Coverage = {df_comp['v6_1_cov'].mean():.2%}")
    print(f"v6.1.1: Selective F1 = {df_comp['v6_1_1_sel_f1'].mean():.4f} | Full F1 = {df_comp['v6_1_1_full_f1'].mean():.4f} | Coverage = {df_comp['v6_1_1_cov'].mean():.2%}")


if __name__ == "__main__":
    main()
