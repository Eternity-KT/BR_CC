"""
Scientific visualization generator for Bipartite GSI benchmark results on branch Khanh_2.
Datasets: emotions, music, scene, yeast, enron.
Produces 6 publication-ready figures for ESWA/IEEE/JAIR:
  1. Figure 1: coverage_vs_performance_tradeoff.png (Coverage vs Macro-F1 & Micro-F1 Pareto Trade-off)
  2. Figure 2: bipartite_il_dl_partition.png (IL vs DL Partition Topology & Ratio)
  3. Figure 3: model_comparison_dashboard.png (BR_MLP vs CC_MLP vs GSI_MLC_PA Full vs Selective)
  4. Figure 4: cost_sensitivity_dynamics.png (Coverage, Generalized Loss & AABS vs Cost c)
  5. Figure 5: runtime_and_efficiency_profile.png (Training time, Selection time, Inference latency)
  6. Figure 6: eswa_rejection_cost_panel.png (Mau_Bieu_Do.jpg ESWA Multi-row Panel for Khanh_2)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Set publication style parameters
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.edgecolor": "#334155",
    "axes.linewidth": 0.85,
})

DATASETS = ["emotions", "music", "scene", "yeast", "enron"]
DATASET_LABELS = {
    "emotions": "Emotions (Audio)",
    "music": "Music (Audio)",
    "scene": "Scene (Image)",
    "yeast": "Yeast (Biology)",
    "enron": "Enron (Text)",
}

DATASET_COLORS = {
    "emotions": "#E11D48",  # Rose Red
    "music": "#D97706",     # Amber Orange
    "scene": "#2563EB",     # Royal Blue
    "yeast": "#16A34A",     # Emerald Green
    "enron": "#7C3AED",     # Violet / Purple
}

DATASET_MARKERS = {
    "emotions": "o",
    "music": "s",
    "scene": "^",
    "yeast": "D",
    "enron": "v",
}

COSTS = ["0.20", "0.25", "0.30", "0.35", "0.40"]
COST_COLORS = {
    "0.20": "#DC2626",  # Red
    "0.25": "#2563EB",  # Blue
    "0.30": "#16A34A",  # Green
    "0.35": "#D97706",  # Orange
    "0.40": "#475569",  # Slate
}

def load_data(raw_path="results_khanh_2/tables/raw_results.json"):
    with open(raw_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================================
# FIGURE 1: Coverage vs Performance Trade-off (Accuracy-Rejection Curve)
# =========================================================================
def plot_coverage_tradeoff(data, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))

    for ds in DATASETS:
        gsi_data = data[ds]["GSI_MLC_PA"]
        covs = []
        macro_f1s = []
        micro_f1s = []

        for c in COSTS:
            c_mean = gsi_data["costs"][c]["mean"]
            covs.append(c_mean["Coverage"])
            macro_f1s.append(c_mean["Selective Macro-F1"])
            micro_f1s.append(c_mean["Selective Micro-F1"])

        # Also add Full prediction point (Coverage = 1.0)
        full_mean = gsi_data["full"]["mean"]
        covs.append(1.0)
        macro_f1s.append(full_mean["Macro-F1"])
        micro_f1s.append(full_mean["Micro-F1"])

        # Sort points by coverage
        order = np.argsort(covs)
        covs = np.array(covs)[order]
        macro_f1s = np.array(macro_f1s)[order]
        micro_f1s = np.array(micro_f1s)[order]

        color = DATASET_COLORS[ds]
        marker = DATASET_MARKERS[ds]
        label = DATASET_LABELS[ds]

        # Panel 1: Selective Macro-F1 vs Coverage
        ax1.plot(covs, macro_f1s, marker=marker, markersize=7, linewidth=2.0,
                 color=color, label=label, alpha=0.9, zorder=4)

        # Panel 2: Selective Micro-F1 vs Coverage
        ax2.plot(covs, micro_f1s, marker=marker, markersize=7, linewidth=2.0,
                 color=color, label=label, alpha=0.9, zorder=4)

    # Styling Panel 1
    ax1.set_title("(a) Selective Macro-F1 vs. Coverage", fontweight="bold", fontsize=12, pad=10)
    ax1.set_xlabel("Coverage (Proportion of Accepted Predictions)", fontweight="bold")
    ax1.set_ylabel("Selective Macro-F1 Score", fontweight="bold")
    ax1.set_xlim(-0.02, 1.05)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, linestyle="--", alpha=0.35, zorder=1)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", loc="lower right", fontsize=9.5)

    # Styling Panel 2
    ax2.set_title("(b) Selective Micro-F1 vs. Coverage", fontweight="bold", fontsize=12, pad=10)
    ax2.set_xlabel("Coverage (Proportion of Accepted Predictions)", fontweight="bold")
    ax2.set_ylabel("Selective Micro-F1 Score", fontweight="bold")
    ax2.set_xlim(-0.02, 1.05)
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, linestyle="--", alpha=0.35, zorder=1)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", loc="lower right", fontsize=9.5)

    fig.suptitle("Accuracy-Rejection Trade-off Curves across 5 Benchmark Datasets (Bipartite GSI)",
                 fontsize=13.5, fontweight="bold", y=1.00)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "coverage_vs_performance_tradeoff.png")
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# =========================================================================
# FIGURE 2: Bipartite IL vs DL Partition Topology & Proportion
# =========================================================================
def plot_bipartite_partition(data, out_dir):
    il_counts = []
    dl_counts = []
    total_labels = []

    for ds in DATASETS:
        gsi_data = data[ds]["GSI_MLC_PA"]["full"]["mean"]
        il = gsi_data.get("Independent Label Count", 0.0)
        dl = gsi_data.get("Dependent Label Count", 0.0)
        il_counts.append(il)
        dl_counts.append(dl)
        total_labels.append(il + dl)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    y = np.arange(len(DATASETS))

    # Panel 1: Stacked horizontal bars of absolute counts
    bar_h = 0.52
    ax1.barh(y, il_counts, height=bar_h, color="#3B82F6", edgecolor="#1E3A8A",
             linewidth=0.8, label="Independent Labels (IL) - Anchor Predictors", zorder=3)
    ax1.barh(y, dl_counts, height=bar_h, left=il_counts, color="#F97316", edgecolor="#9A3412",
             linewidth=0.8, label="Dependent Labels (DL) - Conditioned on IL", zorder=3)

    for i, (il, dl, tot) in enumerate(zip(il_counts, dl_counts, total_labels)):
        ax1.text(tot + 0.8, i, f"Total: {int(tot)} (|IL|={il:.1f}, |DL|={dl:.1f})",
                 va="center", fontsize=9, fontweight="semibold", color="#0F172A")

    ax1.set_yticks(y)
    ax1.set_yticklabels([d.upper() for d in DATASETS], fontweight="bold")
    ax1.set_xlabel("Number of Labels", fontweight="bold")
    ax1.set_xlim(0, max(total_labels) * 1.45)
    ax1.set_title("(a) Absolute Label Partition Counts (|IL| vs |DL|)", fontweight="bold", pad=10)
    ax1.grid(axis="x", linestyle="--", alpha=0.35, zorder=1)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9)
    ax1.invert_yaxis()

    # Panel 2: Percentage breakdown (100% stacked horizontal bar)
    il_pct = [il / tot * 100 for il, tot in zip(il_counts, total_labels)]
    dl_pct = [dl / tot * 100 for dl, tot in zip(dl_counts, total_labels)]

    ax2.barh(y, il_pct, height=bar_h, color="#3B82F6", edgecolor="#1E3A8A",
             linewidth=0.8, label="Independent Labels (%)", zorder=3)
    ax2.barh(y, dl_pct, height=bar_h, left=il_pct, color="#F97316", edgecolor="#9A3412",
             linewidth=0.8, label="Dependent Labels (%)", zorder=3)

    for i, (ip, dp) in enumerate(zip(il_pct, dl_pct)):
        if ip > 5:
            ax2.text(ip / 2, i, f"{ip:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        ax2.text(ip + dp / 2, i, f"{dp:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    ax2.set_yticks(y)
    ax2.set_yticklabels([d.upper() for d in DATASETS], fontweight="bold")
    ax2.set_xlabel("Proportion of Total Labels (%)", fontweight="bold")
    ax2.set_xlim(0, 100)
    ax2.set_title("(b) Relative Partition Topology Ratio (IL% vs DL%)", fontweight="bold", pad=10)
    ax2.grid(axis="x", linestyle="--", alpha=0.35, zorder=1)
    ax2.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9)
    ax2.invert_yaxis()

    fig.suptitle("Bipartite Graph Partition Distribution across 5 Benchmark Datasets",
                 fontsize=13.5, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "bipartite_il_dl_partition.png")
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# =========================================================================
# FIGURE 3: Multi-Model Benchmark Comparison (BR vs CC vs MLC-PA vs GSI)
# =========================================================================
def plot_model_comparison_dashboard(data, out_dir):
    models = ["BR_MLP", "CC_MLP", "MLC_PA (c=0.35)", "GSI_Full", "GSI_Selective (c=0.35)"]
    model_colors = ["#2563EB", "#D97706", "#DC2626", "#64748B", "#059669"]

    metrics = [
        ("Macro-F1 Score", "Macro-F1", True),
        ("Micro-F1 Score", "Micro-F1", True),
        ("Subset Accuracy", "Subset Accuracy", True),
        ("Hamming Loss", "Hamming Loss", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    axes = axes.flatten()

    x = np.arange(len(DATASETS))
    n_models = len(models)
    width = 0.16

    for m_idx, (title, metric_name, higher_is_better) in enumerate(metrics):
        ax = axes[m_idx]

        for idx, (m, color) in enumerate(zip(models, model_colors)):
            vals = []
            for ds in DATASETS:
                if m == "BR_MLP":
                    val = data[ds]["BR_MLP"]["full"]["mean"].get(metric_name, 0.0)
                elif m == "CC_MLP":
                    val = data[ds]["CC_MLP"]["full"]["mean"].get(metric_name, 0.0)
                elif m == "MLC_PA (c=0.35)":
                    if "MLC_PA" in data[ds] and "costs" in data[ds]["MLC_PA"]:
                        c_mean = data[ds]["MLC_PA"]["costs"]["0.35"]["mean"]
                        if metric_name == "Macro-F1":
                            val = c_mean["Selective Macro-F1"]
                        elif metric_name == "Micro-F1":
                            val = c_mean["Selective Micro-F1"]
                        elif metric_name == "Subset Accuracy":
                            val = data[ds]["MLC_PA"]["full"]["mean"].get("Subset Accuracy", 0.0)
                        elif metric_name == "Hamming Loss":
                            val = c_mean["Selective Hamming Loss"]
                    else:
                        val = 0.0
                elif m == "GSI_Full":
                    val = data[ds]["GSI_MLC_PA"]["full"]["mean"].get(metric_name, 0.0)
                elif m == "GSI_Selective (c=0.35)":
                    c_mean = data[ds]["GSI_MLC_PA"]["costs"]["0.35"]["mean"]
                    if metric_name == "Macro-F1":
                        val = c_mean["Selective Macro-F1"]
                    elif metric_name == "Micro-F1":
                        val = c_mean["Selective Micro-F1"]
                    elif metric_name == "Subset Accuracy":
                        val = data[ds]["GSI_MLC_PA"]["full"]["mean"].get("Subset Accuracy", 0.0)
                    elif metric_name == "Hamming Loss":
                        val = c_mean["Selective Hamming Loss"]
                vals.append(val)

            offset = (idx - (n_models - 1) / 2.0) * width
            bars = ax.bar(x + offset, vals, width * 0.90, color=color,
                          edgecolor="#1E293B", linewidth=0.75, label=m, zorder=3)

            # Annotations on top of bars
            for bar, val in zip(bars, vals):
                h = bar.get_height()
                ax.annotate(f"{val:.3f}", (bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3.5), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.2, rotation=90, color="#0F172A")

        ax.set_title(f"({chr(97 + m_idx)}) {title}" + (" (Higher is better)" if higher_is_better else " (Lower is better)"),
                     fontweight="bold", fontsize=11, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([d.upper() for d in DATASETS], fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=1)

        y_max = ax.get_ylim()[1]
        ax.set_ylim(0, y_max * 1.22)

        if m_idx == 0:
            ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.5, ncol=2)

    fig.suptitle("Comparative Evaluation: Baselines vs. Bipartite GSI-MLC-PA Across 5 Datasets",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "model_comparison_dashboard.png")
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# =========================================================================
# FIGURE 4: Cost Sensitivity Dynamics (Coverage, GenLoss, AABS vs Cost c)
# =========================================================================
def plot_cost_sensitivity_dynamics(data, out_dir):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8))

    c_vals = [float(c) for c in COSTS]

    for ds in DATASETS:
        gsi_data = data[ds]["GSI_MLC_PA"]["costs"]
        covs = [gsi_data[c]["mean"]["Coverage"] * 100 for c in COSTS]
        gen_losses = [gsi_data[c]["mean"]["Generalized Loss"] for c in COSTS]
        aabs_vals = [gsi_data[c]["mean"]["AABS"] * 100 for c in COSTS]

        color = DATASET_COLORS[ds]
        marker = DATASET_MARKERS[ds]
        label = DATASET_LABELS[ds]

        ax1.plot(c_vals, covs, marker=marker, markersize=6.5, linewidth=1.8,
                 color=color, label=label, zorder=4)
        ax2.plot(c_vals, gen_losses, marker=marker, markersize=6.5, linewidth=1.8,
                 color=color, label=label, zorder=4)
        ax3.plot(c_vals, aabs_vals, marker=marker, markersize=6.5, linewidth=1.8,
                 color=color, label=label, zorder=4)

    # Subplot 1: Coverage
    ax1.set_title("(a) Coverage vs. Rejection Cost $c$", fontweight="bold", fontsize=11, pad=10)
    ax1.set_xlabel("Rejection Cost $c$", fontweight="bold")
    ax1.set_ylabel("Coverage (%)", fontweight="bold")
    ax1.set_xticks(c_vals)
    ax1.set_ylim(-2, 105)
    ax1.grid(True, linestyle="--", alpha=0.35, zorder=1)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.5)

    # Subplot 2: Generalized Loss
    ax2.set_title("(b) Generalized Loss vs. Rejection Cost $c$", fontweight="bold", fontsize=11, pad=10)
    ax2.set_xlabel("Rejection Cost $c$", fontweight="bold")
    ax2.set_ylabel(r"Generalized Loss $\mathcal{L}_{\mathrm{Gen}}$", fontweight="bold")
    ax2.set_xticks(c_vals)
    ax2.grid(True, linestyle="--", alpha=0.35, zorder=1)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.5)

    # Subplot 3: Average Abstention Size (AABS)
    ax3.set_title("(c) Average Abstention Size (AABS) vs. Cost $c$", fontweight="bold", fontsize=11, pad=10)
    ax3.set_xlabel("Rejection Cost $c$", fontweight="bold")
    ax3.set_ylabel("AABS (%)", fontweight="bold")
    ax3.set_xticks(c_vals)
    ax3.set_ylim(-2, 105)
    ax3.grid(True, linestyle="--", alpha=0.35, zorder=1)
    ax3.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.5)

    fig.suptitle(r"Cost Sensitivity Dynamics of Bipartite GSI-MLC-PA Across Rejection Levels $c \in [0.20, 0.40]$",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "cost_sensitivity_dynamics.png")
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# =========================================================================
# FIGURE 5: Runtime & Computational Efficiency Profile
# =========================================================================
def plot_runtime_efficiency(data, out_dir):
    train_times = []
    select_times = []
    infer_times_ms = []

    for ds in DATASETS:
        full_mean = data[ds]["GSI_MLC_PA"]["full"]["mean"]
        train_times.append(full_mean.get("train_time", 0.0))
        select_times.append(full_mean.get("Selection Time Seconds", 0.0))
        infer_times_ms.append(full_mean.get("Inference Time Seconds", 0.0) * 1000.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    x = np.arange(len(DATASETS))
    width = 0.35

    # Panel 1: Training & Selection Time (Log scale)
    bars1 = ax1.bar(x - width/2, train_times, width, color="#2563EB", edgecolor="#1E3A8A",
                    linewidth=0.8, label="Model Refit / Training Time (s)", zorder=3)
    bars2 = ax1.bar(x + width/2, select_times, width, color="#D97706", edgecolor="#9A3412",
                    linewidth=0.8, label="Order & Partition Selection Time (s)", zorder=3)

    ax1.set_yscale("log")
    ax1.set_title("(a) Training & Selection Time (Log Scale)", fontweight="bold", fontsize=11, pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.upper() for d in DATASETS], fontweight="bold")
    ax1.set_ylabel("Execution Time in Seconds (log scale)", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.35, zorder=1)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9)

    for bar, val in zip(bars1, train_times):
        h = bar.get_height()
        ax1.annotate(f"{val:.2f}s", (bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                     fontsize=8, fontweight="semibold", color="#1E3A8A")

    for bar, val in zip(bars2, select_times):
        h = bar.get_height()
        ax1.annotate(f"{val:.2f}s", (bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                     fontsize=8, fontweight="semibold", color="#9A3412")

    # Panel 2: Inference Latency per fold (in milliseconds)
    bars3 = ax2.bar(x, infer_times_ms, 0.45, color="#10B981", edgecolor="#065F46",
                    linewidth=0.8, label="Inference Latency per Fold (ms)", zorder=3)

    ax2.set_title("(b) Inference Latency (Milliseconds per Test Fold)", fontweight="bold", fontsize=11, pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.upper() for d in DATASETS], fontweight="bold")
    ax2.set_ylabel("Inference Latency (ms)", fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.35, zorder=1)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9)

    for bar, val in zip(bars3, infer_times_ms):
        h = bar.get_height()
        ax2.annotate(f"{val:.1f} ms", (bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                     fontsize=8.5, fontweight="bold", color="#065F46")

    ax2.set_ylim(0, max(infer_times_ms) * 1.25)

    fig.suptitle("Computational Efficiency & Scalability Profile of Bipartite GSI-MLC-PA",
                 fontsize=13.5, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "runtime_and_efficiency_profile.png")
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# =========================================================================
# FIGURE 6: Multi-Row Rejection Cost Panel (ESWA Mau_Bieu_Do.jpg format)
# =========================================================================
def plot_eswa_panel(data, out_dir):
    n_rows = len(COSTS)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13.5, 2.5 * n_rows),
                             gridspec_kw={"width_ratios": [3.2, 1.2]})

    x = np.arange(len(DATASETS))
    width = 0.35

    for row, c in enumerate(COSTS):
        ax_left = axes[row, 0]
        ax_right = axes[row, 1]
        c_float = float(c)

        # --- Left Column: Full vs Selective Macro-F1 ---
        full_f1 = [data[ds]["GSI_MLC_PA"]["full"]["mean"]["Macro-F1"] * 100 for ds in DATASETS]
        sel_f1 = [data[ds]["GSI_MLC_PA"]["costs"][c]["mean"]["Selective Macro-F1"] * 100 for ds in DATASETS]

        bars_full = ax_left.bar(x - width/2, full_f1, width, color="#94A3B8",
                                edgecolor="#334155", linewidth=0.75, label="Full Prediction" if row == 0 else "")
        bars_sel = ax_left.bar(x + width/2, sel_f1, width, color=COST_COLORS[c],
                               edgecolor="#1E293B", linewidth=0.75, label=f"Selective ($c={c}$)" if row == 0 else "")

        # Annotations on bars
        for bf, bs in zip(bars_full, bars_sel):
            hf = bf.get_height()
            hs = bs.get_height()
            ax_left.text(bf.get_x() + bf.get_width() / 2, hf + 1.5, f"{hf:.1f}",
                         ha="center", va="bottom", fontsize=7.2, color="#475569")
            ax_left.text(bs.get_x() + bs.get_width() / 2, hs + 1.5, f"{hs:.1f}",
                         ha="center", va="bottom", fontsize=7.2, fontweight="bold", color=COST_COLORS[c])

        ax_left.set_ylim(0, 115)
        ax_left.set_ylabel("Score (%)", fontsize=9.5, fontweight="bold")
        ax_left.set_xticks(x)
        ax_left.set_xticklabels([d.upper() for d in DATASETS], fontweight="bold")
        ax_left.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

        # Cost badge inside left axis
        ax_left.text(0.015, 0.86, f"$c = {c_float:.2f}$", transform=ax_left.transAxes,
                     fontsize=10.5, fontweight="bold", color=COST_COLORS[c],
                     bbox=dict(boxstyle="round,pad=0.25", facecolor="#F8FAFC", edgecolor=COST_COLORS[c], linewidth=1.0))

        if row == 0:
            ax_left.set_title("Selective Macro-F1: Full Prediction vs. With Rejection", fontsize=11.5, pad=8, fontweight="bold")
            ax_left.legend(loc="upper right", frameon=False, fontsize=9, ncol=2)

        # --- Right Column: Rejection Extent (ABS & AABS) for MLC-PA vs GSI-MLC-PA ---
        mlc_abs = np.mean([data[ds]["MLC_PA"]["costs"][c]["mean"]["ABS"] * 100 for ds in DATASETS])
        mlc_aabs = np.mean([data[ds]["MLC_PA"]["costs"][c]["mean"]["AABS"] * 100 for ds in DATASETS])
        gsi_abs = np.mean([data[ds]["GSI_MLC_PA"]["costs"][c]["mean"]["ABS"] * 100 for ds in DATASETS])
        gsi_aabs = np.mean([data[ds]["GSI_MLC_PA"]["costs"][c]["mean"]["AABS"] * 100 for ds in DATASETS])

        ax_right.bar([0, 1], [mlc_abs, gsi_abs], width=0.45, color=COST_COLORS[c], alpha=0.35,
                     hatch="//", edgecolor=COST_COLORS[c], linewidth=1.0, label="ABS" if row == 0 else "")
        ax_right.scatter([0, 1], [mlc_aabs, gsi_aabs], color="white", edgecolor=COST_COLORS[c],
                         s=55, linewidth=1.5, zorder=5, label="AABS" if row == 0 else "")

        ax_right.text(0, mlc_abs + 2.5, f"ABS: {mlc_abs:.1f}%\nAABS: {mlc_aabs:.1f}%",
                      ha="center", va="bottom", fontsize=7.0, fontweight="bold", color=COST_COLORS[c])
        ax_right.text(1, gsi_abs + 2.5, f"ABS: {gsi_abs:.1f}%\nAABS: {gsi_aabs:.1f}%",
                      ha="center", va="bottom", fontsize=7.0, fontweight="bold", color=COST_COLORS[c])

        ax_right.set_xlim(-0.6, 1.6)
        ax_right.set_ylim(0, 130)
        ax_right.set_ylabel("Rate (%)", fontsize=9.5, fontweight="bold")
        ax_right.set_xticks([0, 1])
        ax_right.set_xticklabels(["MLC-PA", "GSI-MLC-PA"], fontweight="bold", fontsize=8.0)
        ax_right.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

        if row == 0:
            ax_right.set_title("Rejection Extent (ABS/AABS)", fontsize=11.5, pad=8, fontweight="bold")
            ax_right.legend(loc="upper right", frameon=False, fontsize=8.5)

    plt.tight_layout(rect=(0, 0.05, 1, 0.98), h_pad=1.8, w_pad=2.0)

    # Caption at bottom
    caption = (
        "Figure: Average Macro-F1 scores (in %, left) and rejection extent (ABS/AABS comparing MLC-PA vs Bipartite GSI-MLC-PA, right) across 5 datasets.\n"
        "Color-coded with respect to rejection cost c: c=0.20 (red), c=0.25 (blue), c=0.30 (green), c=0.35 (amber), and c=0.40 (slate)."
    )
    fig.text(0.04, 0.01, caption, ha="left", va="bottom", fontsize=9, style="italic", color="#1E293B")

    out_path = os.path.join(out_dir, "eswa_rejection_cost_panel.png")
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    out_dir = "results_khanh_2/plots_analysis"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory initialized: {out_dir}")

    data = load_data("results_khanh_2/tables/raw_results.json")

    print("\n--- 1. Generating Figure 1: Coverage vs Performance Trade-off ---")
    plot_coverage_tradeoff(data, out_dir)

    print("\n--- 2. Generating Figure 2: Bipartite IL vs DL Partition Topology ---")
    plot_bipartite_partition(data, out_dir)

    print("\n--- 3. Generating Figure 3: Model Comparison Dashboard ---")
    plot_model_comparison_dashboard(data, out_dir)

    print("\n--- 4. Generating Figure 4: Cost Sensitivity Dynamics ---")
    plot_cost_sensitivity_dynamics(data, out_dir)

    print("\n--- 5. Generating Figure 5: Runtime & Efficiency Profile ---")
    plot_runtime_efficiency(data, out_dir)

    print("\n--- 6. Generating Figure 6: ESWA Multi-Row Rejection Cost Panel ---")
    plot_eswa_panel(data, out_dir)

    print("\nAll 6 publication figures generated successfully in results_khanh_2/plots_analysis/!")

if __name__ == "__main__":
    main()
