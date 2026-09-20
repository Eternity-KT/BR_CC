"""Generate comprehensive, publication-grade comparison charts for v4 benchmark.

Plots generated:
1. v4_selective_macro_f1_by_dataset.png: Selective Macro-F1 across 10 datasets for BR, CC, MLC-PA, GSI-MLC-PA (3 base learners)
2. v4_coverage_by_dataset.png: Coverage across 10 datasets for BR, CC, MLC-PA, GSI-MLC-PA (3 base learners)
3. v4_cost_sweep_curves.png: Coverage and Selective Macro-F1 vs Cost c (0.20 to 0.50)
4. v4_comprehensive_metrics_summary.png: 5-metric overview across all 4 models and 3 base learners
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 1.2,
    "grid.color": "#ebebeb",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

OUTPUT_DIR = "results_pa_v4/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COMP_PATH = "results_pa_v4/tables/434d53dc787206d6/complete_metrics.csv"
SEL_PATH = "results_pa_v4/tables/434d53dc787206d6/selective_metrics.csv"

comp_df = pd.read_csv(COMP_PATH)
sel_df = pd.read_csv(SEL_PATH)

# Colors for the 4 models
PALETTE = {
    "BR": "#64748b",         # Slate gray
    "CC": "#3b82f6",         # Blue
    "MLC-PA": "#f97316",     # Coral orange
    "GSI-MLC-PA": "#10b981", # Emerald green
}

DATASETS = sorted(comp_df["Dataset"].unique())
DATASET_NAMES = [d.upper() for d in DATASETS]
BASE_LEARNERS = ["Logistic", "SVM", "MLP"]

# --------------------------------------------------------------------------------------------------
# Figure 1: Selective Macro-F1 by Dataset (Grouped Bar Chart, 3 subplots)
# --------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(15, 14), sharex=True)
sel_30 = sel_df[np.isclose(sel_df["Cost"], 0.30)]
comp_mean = comp_df.groupby(["Dataset", "Model"])["Macro-F1"].mean().reset_index()
sel_mean = sel_30.groupby(["Dataset", "Model"])["Selective Macro-F1"].mean().reset_index()

bar_width = 0.20
x = np.arange(len(DATASETS))

for idx, bl in enumerate(BASE_LEARNERS):
    ax = axes[idx]
    models = [f"BR_{bl}", f"CC_{bl}", f"MLC_PA_{bl}", f"GSI_MLC_PA_{bl}"]
    labels = ["BR", "CC", "MLC-PA", "GSI-MLC-PA"]
    
    for m_idx, (model_id, label) in enumerate(zip(models, labels)):
        f1_vals = []
        for d in DATASETS:
            if "MLC_PA" in model_id or "GSI_MLC_PA" in model_id:
                sub = sel_mean[(sel_mean["Dataset"] == d) & (sel_mean["Model"] == model_id)]
                f1_vals.append(sub["Selective Macro-F1"].values[0] if not sub.empty else 0.0)
            else:
                sub = comp_mean[(comp_mean["Dataset"] == d) & (comp_mean["Model"] == model_id)]
                f1_vals.append(sub["Macro-F1"].values[0] if not sub.empty else 0.0)
                
        bars = ax.bar(x + (m_idx - 1.5) * bar_width, f1_vals, bar_width, 
                      label=label, color=PALETTE[label], edgecolor="white", linewidth=1.0, alpha=0.9)
        
    ax.set_ylabel("Selective Macro-F1")
    ax.set_title(f"Base Learner: {bl} (Evaluated at c = 0.30)", fontweight="bold", pad=10)
    ax.set_ylim(0, 0.85)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

axes[-1].set_xticks(x)
axes[-1].set_xticklabels(DATASET_NAMES, rotation=25, ha="right", fontweight="semibold")
axes[-1].set_xlabel("Dataset", fontweight="bold", labelpad=10)

plt.suptitle("Selective Macro-F1 Across 10 Datasets (v4: BR vs CC vs MLC-PA vs GSI-MLC-PA)", 
             fontweight="bold", fontsize=16, y=0.995)
plt.tight_layout()
fig1_path = os.path.join(OUTPUT_DIR, "v4_selective_macro_f1_by_dataset.png")
plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {fig1_path}")

# --------------------------------------------------------------------------------------------------
# Figure 2: Coverage by Dataset (Grouped Bar Chart, 3 subplots)
# --------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(15, 14), sharex=True)
cov_mean = sel_30.groupby(["Dataset", "Model"])["Coverage"].mean().reset_index()

for idx, bl in enumerate(BASE_LEARNERS):
    ax = axes[idx]
    models = [f"BR_{bl}", f"CC_{bl}", f"MLC_PA_{bl}", f"GSI_MLC_PA_{bl}"]
    labels = ["BR", "CC", "MLC-PA", "GSI-MLC-PA"]
    
    for m_idx, (model_id, label) in enumerate(zip(models, labels)):
        cov_vals = []
        for d in DATASETS:
            if "MLC_PA" in model_id or "GSI_MLC_PA" in model_id:
                sub = cov_mean[(cov_mean["Dataset"] == d) & (cov_mean["Model"] == model_id)]
                cov_vals.append(sub["Coverage"].values[0] if not sub.empty else 0.0)
            else:
                cov_vals.append(1.0) # BR and CC always have 100% coverage
                
        bars = ax.bar(x + (m_idx - 1.5) * bar_width, [c * 100 for c in cov_vals], bar_width, 
                      label=label, color=PALETTE[label], edgecolor="white", linewidth=1.0, alpha=0.9)
        
    ax.axhline(80, color="#ef4444", linestyle=":", linewidth=1.5, alpha=0.7, label="min_coverage constraint (80%)")
    ax.set_ylabel("Coverage (%)")
    ax.set_title(f"Base Learner: {bl} (Evaluated at c = 0.30)", fontweight="bold", pad=10)
    ax.set_ylim(0, 110)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

axes[-1].set_xticks(x)
axes[-1].set_xticklabels(DATASET_NAMES, rotation=25, ha="right", fontweight="semibold")
axes[-1].set_xlabel("Dataset", fontweight="bold", labelpad=10)

plt.suptitle("Coverage (%) Across 10 Datasets (v4: BR vs CC vs MLC-PA vs GSI-MLC-PA)", 
             fontweight="bold", fontsize=16, y=0.995)
plt.tight_layout()
fig2_path = os.path.join(OUTPUT_DIR, "v4_coverage_by_dataset.png")
plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {fig2_path}")

# --------------------------------------------------------------------------------------------------
# Figure 3: Cost Sweep Curves (Coverage & Selective Macro-F1 vs Cost c)
# --------------------------------------------------------------------------------------------------
costs = sorted(sel_df["Cost"].unique())
cost_df = sel_df.groupby(["Cost", "Model"])[["Coverage", "Selective Macro-F1"]].mean().reset_index()
comp_model_mean = comp_df.groupby("Model")["Macro-F1"].mean().to_dict()

fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True)

markers = {"BR": "o", "CC": "s", "MLC-PA": "^", "GSI-MLC-PA": "D"}

for idx, bl in enumerate(BASE_LEARNERS):
    ax_cov = axes[0, idx]
    ax_f1 = axes[1, idx]
    
    models = [f"BR_{bl}", f"CC_{bl}", f"MLC_PA_{bl}", f"GSI_MLC_PA_{bl}"]
    labels = ["BR", "CC", "MLC-PA", "GSI-MLC-PA"]
    
    for model_id, label in zip(models, labels):
        if "MLC_PA" in model_id or "GSI_MLC_PA" in model_id:
            m_data = cost_df[cost_df["Model"] == model_id].sort_values("Cost")
            cov_pts = m_data["Coverage"].values * 100
            f1_pts = m_data["Selective Macro-F1"].values
        else:
            cov_pts = np.full(len(costs), 100.0)
            f1_pts = np.full(len(costs), comp_model_mean.get(model_id, 0.0))
            
        ax_cov.plot(costs, cov_pts, marker=markers[label], color=PALETTE[label], 
                    linewidth=2.2, markersize=7, label=label)
        ax_f1.plot(costs, f1_pts, marker=markers[label], color=PALETTE[label], 
                   linewidth=2.2, markersize=7, label=label)
        
    ax_cov.set_title(f"Coverage vs Cost ({bl})", fontweight="bold")
    ax_cov.set_ylabel("Mean Coverage (%)")
    ax_cov.set_ylim(40, 105)
    ax_cov.axhline(80, color="#ef4444", linestyle=":", linewidth=1.5, alpha=0.7)
    ax_cov.legend(loc="lower right", frameon=True)
    
    ax_f1.set_title(f"Selective Macro-F1 vs Cost ({bl})", fontweight="bold")
    ax_f1.set_ylabel("Mean Selective Macro-F1")
    ax_f1.set_xlabel("Abstention Cost (c)")
    ax_f1.set_ylim(0.15, 0.52)
    ax_f1.legend(loc="lower right", frameon=True)

plt.suptitle("Trade-off Dynamics Across Operating Costs c ∈ [0.20, 0.50] (Mean of 10 Datasets)", 
             fontweight="bold", fontsize=16, y=0.995)
plt.tight_layout()
fig3_path = os.path.join(OUTPUT_DIR, "v4_cost_sweep_curves.png")
plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {fig3_path}")

# --------------------------------------------------------------------------------------------------
# Figure 4: Comprehensive Metrics Overview (Full Macro-F1, Subset Acc, Micro-F1, Coverage, Selective F1)
# --------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

metrics_keys = ["Full Macro-F1", "Subset Acc", "Micro-F1", "Coverage", "Selective F1"]
x_indices = np.arange(len(metrics_keys))
b_width = 0.20

comp_all_mean = comp_df.groupby("Model")[["Macro-F1", "Subset Accuracy", "Micro-F1"]].mean()
sel_30_mean = sel_30.groupby("Model")[["Coverage", "Selective Macro-F1"]].mean()

for idx, bl in enumerate(BASE_LEARNERS):
    ax = axes[idx]
    models = [f"BR_{bl}", f"CC_{bl}", f"MLC_PA_{bl}", f"GSI_MLC_PA_{bl}"]
    labels = ["BR", "CC", "MLC-PA", "GSI-MLC-PA"]
    
    for m_idx, (model_id, label) in enumerate(zip(models, labels)):
        full_f1 = comp_all_mean.loc[model_id, "Macro-F1"]
        subset_acc = comp_all_mean.loc[model_id, "Subset Accuracy"]
        micro_f1 = comp_all_mean.loc[model_id, "Micro-F1"]
        
        if "MLC_PA" in model_id or "GSI_MLC_PA" in model_id:
            cov = sel_30_mean.loc[model_id, "Coverage"]
            sel_f1 = sel_30_mean.loc[model_id, "Selective Macro-F1"]
        else:
            cov = 1.0
            sel_f1 = full_f1
            
        metric_vals = [full_f1, subset_acc, micro_f1, cov, sel_f1]
        
        ax.bar(x_indices + (m_idx - 1.5) * b_width, metric_vals, b_width, 
               label=label, color=PALETTE[label], edgecolor="white", linewidth=1.0, alpha=0.9)
        
    ax.set_title(f"Base Learner: {bl}", fontweight="bold", pad=10)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(metrics_keys, fontweight="semibold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", frameon=True)
    if idx == 0:
        ax.set_ylabel("Metric Score [0, 1]", fontweight="bold")

plt.suptitle("Comprehensive Benchmark Comparison Across 5 Key Metrics (v4: 10 Datasets)", 
             fontweight="bold", fontsize=16, y=1.02)
plt.tight_layout()
fig4_path = os.path.join(OUTPUT_DIR, "v4_comprehensive_metrics_summary.png")
plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {fig4_path}")
