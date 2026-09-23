"""Analysis script for v3.1 benchmark comparing decoupled DL against v3 and baselines."""

import os
import glob
import pandas as pd
import numpy as np

table_dirs = glob.glob("results_pa_v3_1/tables/*")
scope_dir = [d for d in table_dirs if os.path.isdir(d)][0]
print(f"Loading results from: {scope_dir}")

complete_csv = os.path.join(scope_dir, "complete_metrics.csv")
selective_csv = os.path.join(scope_dir, "selective_metrics.csv")
partition_csv = os.path.join(scope_dir, "partition_audit.csv")

df_comp = pd.read_csv(complete_csv)
df_sel = pd.read_csv(selective_csv)
df_part = pd.read_csv(partition_csv)

print("\n" + "="*80)
print("1. PARTITION AUDIT SUMMARY (|IL| vs |DL| across datasets for GSI models)")
print("="*80)
part_summary = df_part.groupby(["Dataset", "Model"])[["Independent Label Count", "Dependent Label Count"]].mean().reset_index()
print(part_summary.to_string(index=False))

print("\n" + "="*80)
print("2. COMPLETE METRICS COMPARISON (Macro-F1, Micro-F1, Hamming Loss)")
print("="*80)

df_comp_mean = df_comp.groupby(["Dataset", "Model"])[["Macro-F1", "Micro-F1", "Hamming Loss", "Subset Accuracy"]].mean().reset_index()

macro_pivot = df_comp_mean.pivot(index="Dataset", columns="Model", values="Macro-F1")

logistic_models = ["BR_Logistic", "CC_Logistic", "MLC_PA_Logistic", "GSI_MLC_PA_Logistic", "GSI_MLC_PA_v3_1_Logistic"]
mlp_models = ["BR_MLP", "CC_MLP", "MLC_PA_MLP", "GSI_MLC_PA_MLP", "GSI_MLC_PA_v3_1_MLP"]
svm_models = ["BR_SVM", "CC_SVM", "MLC_PA_SVM", "GSI_MLC_PA_SVM", "GSI_MLC_PA_v3_1_SVM"]

print("\n--- LOGISTIC MODELS: Macro-F1 ---")
print(macro_pivot[[m for m in logistic_models if m in macro_pivot.columns]].round(4).to_string())

print("\n--- MLP MODELS: Macro-F1 ---")
print(macro_pivot[[m for m in mlp_models if m in macro_pivot.columns]].round(4).to_string())

print("\n--- SVM MODELS: Macro-F1 ---")
print(macro_pivot[[m for m in svm_models if m in svm_pivot.columns] if 'svm_pivot' in locals() else [m for m in svm_models if m in macro_pivot.columns]].round(4).to_string())

# Pivot for Micro-F1
micro_pivot = df_comp_mean.pivot(index="Dataset", columns="Model", values="Micro-F1")
print("\n--- LOGISTIC MODELS: Micro-F1 ---")
print(micro_pivot[[m for m in logistic_models if m in micro_pivot.columns]].round(4).to_string())

print("\n--- MLP MODELS: Micro-F1 ---")
print(micro_pivot[[m for m in mlp_models if m in micro_pivot.columns]].round(4).to_string())

# Pivot for Hamming Loss
hl_pivot = df_comp_mean.pivot(index="Dataset", columns="Model", values="Hamming Loss")
print("\n--- LOGISTIC MODELS: Hamming Loss (Lower is better) ---")
print(hl_pivot[[m for m in logistic_models if m in hl_pivot.columns]].round(4).to_string())

print("\n--- MLP MODELS: Hamming Loss (Lower is better) ---")
print(hl_pivot[[m for m in mlp_models if m in mlp_pivot.columns] if 'mlp_pivot' in locals() else [m for m in mlp_models if m in hl_pivot.columns]].round(4).to_string())

print("\n" + "="*80)
print("3. SELECTIVE METRICS COMPARISON (Cost = 0.30)")
print("="*80)
df_sel_03 = df_sel[np.isclose(df_sel["Cost"], 0.3)].copy()

sel_mean = df_sel_03.groupby(["Dataset", "Model"])[["Selective Macro-F1", "Coverage", "Risk at Coverage"]].mean().reset_index()

cov_pivot = sel_mean.pivot(index="Dataset", columns="Model", values="Coverage")
print("\n--- Coverage at cost 0.30 ---")
gsi_cols = [c for c in ["MLC_PA_Logistic", "GSI_MLC_PA_Logistic", "GSI_MLC_PA_v3_1_Logistic", "MLC_PA_MLP", "GSI_MLC_PA_MLP", "GSI_MLC_PA_v3_1_MLP"] if c in cov_pivot.columns]
print(cov_pivot[gsi_cols].round(4).to_string())

sel_f1_pivot = sel_mean.pivot(index="Dataset", columns="Model", values="Selective Macro-F1")
print("\n--- Selective Macro-F1 at cost 0.30 ---")
print(sel_f1_pivot[gsi_cols].round(4).to_string())

print("\n" + "="*80)
print("4. HEAD-TO-HEAD WIN/TIE/LOSS: v3.1 vs v3")
print("="*80)

def compare_pair(model_v3, model_v3_1, metric_df, metric_name, higher_is_better=True):
    v3_vals = metric_df[model_v3]
    v3_1_vals = metric_df[model_v3_1]
    diff = v3_1_vals - v3_vals
    if not higher_is_better:
        diff = -diff
    wins = (diff > 1e-4).sum()
    ties = (diff.abs() <= 1e-4).sum()
    losses = (diff < -1e-4).sum()
    mean_diff = diff.mean()
    pct_win = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    print(f"{model_v3_1} vs {model_v3} ({metric_name}): {wins} Wins, {ties} Ties, {losses} Losses | Win Rate (excl ties): {pct_win:.1f}% | Mean Diff: {mean_diff:+.4f}")

print("\n--- Macro-F1 Head-to-Head ---")
compare_pair("GSI_MLC_PA_Logistic", "GSI_MLC_PA_v3_1_Logistic", macro_pivot, "Macro-F1")
compare_pair("GSI_MLC_PA_MLP", "GSI_MLC_PA_v3_1_MLP", macro_pivot, "Macro-F1")
compare_pair("GSI_MLC_PA_SVM", "GSI_MLC_PA_v3_1_SVM", macro_pivot, "Macro-F1")

print("\n--- Micro-F1 Head-to-Head ---")
compare_pair("GSI_MLC_PA_Logistic", "GSI_MLC_PA_v3_1_Logistic", micro_pivot, "Micro-F1")
compare_pair("GSI_MLC_PA_MLP", "GSI_MLC_PA_v3_1_MLP", micro_pivot, "Micro-F1")
compare_pair("GSI_MLC_PA_SVM", "GSI_MLC_PA_v3_1_SVM", micro_pivot, "Micro-F1")

print("\n--- Hamming Loss Head-to-Head (lower is better) ---")
compare_pair("GSI_MLC_PA_Logistic", "GSI_MLC_PA_v3_1_Logistic", hl_pivot, "Hamming Loss", higher_is_better=False)
compare_pair("GSI_MLC_PA_MLP", "GSI_MLC_PA_v3_1_MLP", hl_pivot, "Hamming Loss", higher_is_better=False)
compare_pair("GSI_MLC_PA_SVM", "GSI_MLC_PA_v3_1_SVM", hl_pivot, "Hamming Loss", higher_is_better=False)
