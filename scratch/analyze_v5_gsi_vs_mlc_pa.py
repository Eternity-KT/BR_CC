"""Rigorous statistical analysis script comparing GSI-MLC-PA vs MLC-PA on v5 benchmark."""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


def find_latest_v5_table_dir(results_dir="results_pa_v5"):
    tables_dir = Path(results_dir) / "tables"
    if not tables_dir.exists():
        return None
    subdirs = [d for d in tables_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    # Sort by modification time
    subdirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return subdirs[0]


def load_data(table_dir):
    json_path = table_dir / "results_v3.json"
    selective_csv = table_dir / "selective_metrics.csv"
    complete_csv = table_dir / "complete_metrics.csv"

    run_data = None
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            run_data = json.load(f)

    df_selective = pd.read_csv(selective_csv) if selective_csv.exists() else None
    df_complete = pd.read_csv(complete_csv) if complete_csv.exists() else None

    return run_data, df_selective, df_complete


def analyze_comparison(table_dir):
    print(f"=== ANALYZING RESULTS FROM: {table_dir} ===\n")
    run_data, df_sel, df_comp = load_data(table_dir)

    if df_sel is None or df_comp is None:
        print("CSV files not found!")
        return

    # Average over folds first if 'Fold' column exists
    if "Fold" in df_sel.columns:
        df_sel_agg = df_sel.groupby(["Dataset", "Model", "Cost"]).mean(numeric_only=True).reset_index()
    else:
        df_sel_agg = df_sel

    if "Fold" in df_comp.columns:
        df_comp_agg = df_comp.groupby(["Dataset", "Model"]).mean(numeric_only=True).reset_index()
    else:
        df_comp_agg = df_comp

    # Metric column names normalization
    sel_f1_col = "Selective Macro-F1"
    cov_col = "Coverage"
    full_f1_col = "Macro-F1"

    base_learners = ["Logistic", "SVM", "MLP"]
    costs = sorted(df_sel_agg["Cost"].unique())

    print("=" * 105)
    print("1. MACRO-F1 & COVERAGE ACROSS ABSTENTION COSTS (c) - HEAD TO HEAD GSI vs MLC-PA")
    print("=" * 105)

    all_improvements = []

    for bl in base_learners:
        mlc_name = f"MLC_PA_{bl}"
        gsi_name = f"GSI_MLC_PA_{bl}"

        print(f"\n### Base Learner: {bl.upper()} (MLC_PA vs GSI_MLC_PA)")
        print(f"{'Cost':<6} | {'MLC Sel F1':<12} | {'GSI Sel F1':<12} | {'Δ F1 (abs)':<11} | {'Δ F1 (%)':<10} | {'MLC Cov':<9} | {'GSI Cov':<9} | {'Δ Cov':<8} | {'Win/Tie/Loss'}")
        print("-" * 105)

        for c in costs:
            sub = df_sel_agg[df_sel_agg["Cost"] == c]
            mlc_sub = sub[sub["Model"] == mlc_name].set_index("Dataset")
            gsi_sub = sub[sub["Model"] == gsi_name].set_index("Dataset")

            common_datasets = sorted(list(set(mlc_sub.index) & set(gsi_sub.index)))
            if not common_datasets:
                continue

            mlc_f1 = mlc_sub.loc[common_datasets, sel_f1_col].values
            gsi_f1 = gsi_sub.loc[common_datasets, sel_f1_col].values
            mlc_cov = mlc_sub.loc[common_datasets, cov_col].values
            gsi_cov = gsi_sub.loc[common_datasets, cov_col].values

            mean_mlc_f1 = np.mean(mlc_f1)
            mean_gsi_f1 = np.mean(gsi_f1)
            diff_f1 = mean_gsi_f1 - mean_mlc_f1
            diff_f1_pct = (diff_f1 / mean_mlc_f1 * 100) if mean_mlc_f1 > 0 else 0

            mean_mlc_cov = np.mean(mlc_cov)
            mean_gsi_cov = np.mean(gsi_cov)
            diff_cov = mean_gsi_cov - mean_mlc_cov

            wins = np.sum(gsi_f1 > mlc_f1 + 1e-4)
            ties = np.sum(np.abs(gsi_f1 - mlc_f1) <= 1e-4)
            losses = np.sum(gsi_f1 < mlc_f1 - 1e-4)

            print(f"{c:<6.2f} | {mean_mlc_f1:<12.4f} | {mean_gsi_f1:<12.4f} | {diff_f1:>+10.4f} | {diff_f1_pct:>+8.2f}%  | {mean_mlc_cov:<9.4f} | {mean_gsi_cov:<9.4f} | {diff_cov:>+7.4f} | {wins}/{ties}/{losses}")

    print("\n" + "=" * 105)
    print("2. DATASET-BY-DATASET BREAKDOWN AT STANDARD COST c = 0.30")
    print("=" * 105)

    for bl in base_learners:
        mlc_name = f"MLC_PA_{bl}"
        gsi_name = f"GSI_MLC_PA_{bl}"

        sub = df_sel_agg[df_sel_agg["Cost"] == 0.30]
        mlc_sub = sub[sub["Model"] == mlc_name].set_index("Dataset")
        gsi_sub = sub[sub["Model"] == gsi_name].set_index("Dataset")
        common_datasets = sorted(list(set(mlc_sub.index) & set(gsi_sub.index)))

        print(f"\n### Detailed Breakdown for {bl.upper()} at c = 0.30")
        print(f"{'Dataset':<16} | {'MLC F1':<8} | {'GSI F1':<8} | {'Δ F1':<8} | {'MLC Cov':<8} | {'GSI Cov':<8} | {'Δ Cov':<8} | {'Outcome'}")
        print("-" * 88)

        for ds in common_datasets:
            m_f1 = mlc_sub.loc[ds, sel_f1_col]
            g_f1 = gsi_sub.loc[ds, sel_f1_col]
            m_cov = mlc_sub.loc[ds, cov_col]
            g_cov = gsi_sub.loc[ds, cov_col]
            d_f1 = g_f1 - m_f1
            d_cov = g_cov - m_cov
            outcome = "WIN (GSI)" if d_f1 > 0.001 else ("LOSS (GSI)" if d_f1 < -0.001 else "TIE")
            print(f"{ds:<16} | {m_f1:<8.4f} | {g_f1:<8.4f} | {d_f1:>+7.4f} | {m_cov:<8.4f} | {g_cov:<8.4f} | {d_cov:>+7.4f} | {outcome}")

    print("\n" + "=" * 105)
    print("3. COMPLETE METRICS COMPARISON (FULL EVALUATION / UNCONSTRAINED)")
    print("=" * 105)

    for bl in base_learners:
        mlc_name = f"MLC_PA_{bl}"
        gsi_name = f"GSI_MLC_PA_{bl}"

        mlc_c = df_comp_agg[df_comp_agg["Model"] == mlc_name].set_index("Dataset")
        gsi_c = df_comp_agg[df_comp_agg["Model"] == gsi_name].set_index("Dataset")
        common_ds = sorted(list(set(mlc_c.index) & set(gsi_c.index)))

        print(f"\n### Full Metrics for {bl.upper()}:")
        print(f"{'Metric':<22} | {'MLC_PA':<12} | {'GSI_MLC_PA':<12} | {'Absolute Δ':<12} | {'Relative Δ (%)'}")
        print("-" * 75)

        for metric in ["Macro F1", "Micro F1", "Subset Accuracy", "Hamming Loss", "Macro-F1", "Micro-F1"]:
            if metric in mlc_c.columns and metric in gsi_c.columns:
                m_val = mlc_c.loc[common_ds, metric].mean()
                g_val = gsi_c.loc[common_ds, metric].mean()
                d_val = g_val - m_val
                pct = (d_val / m_val * 100) if m_val != 0 else 0
                print(f"{metric:<22} | {m_val:<12.4f} | {g_val:<12.4f} | {d_val:>+11.4f} | {pct:>+13.2f}%")

    print("\n" + "=" * 105)
    print("4. OVERALL STATISTICAL HYPOTHESIS TESTING (GSI vs MLC-PA across all datasets)")
    print("=" * 105)

    for bl in base_learners:
        mlc_name = f"MLC_PA_{bl}"
        gsi_name = f"GSI_MLC_PA_{bl}"
        sub = df_sel_agg[df_sel_agg["Cost"] == 0.30]
        mlc_sub = sub[sub["Model"] == mlc_name].set_index("Dataset")
        gsi_sub = sub[sub["Model"] == gsi_name].set_index("Dataset")
        common_ds = sorted(list(set(mlc_sub.index) & set(gsi_sub.index)))

        if len(common_ds) >= 5:
            m_f1 = mlc_sub.loc[common_ds, sel_f1_col].values
            g_f1 = gsi_sub.loc[common_ds, sel_f1_col].values
            diff = g_f1 - m_f1

            t_stat, p_t = stats.ttest_rel(g_f1, m_f1)
            try:
                w_stat, p_w = stats.wilcoxon(g_f1, m_f1)
            except Exception:
                w_stat, p_w = float("nan"), float("nan")

            print(f"{bl.upper():<10} | Mean Diff: {np.mean(diff):>+7.4f} | Paired t-test p: {p_t:.4f} | Wilcoxon p: {p_w:.4f}")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    t_dir = find_latest_v5_table_dir()
    if t_dir:
        analyze_comparison(t_dir)
    else:
        print("No v5 table directory found yet.")
