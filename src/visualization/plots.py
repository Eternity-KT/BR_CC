"""
Visualization module for BR and CC Multi-Label Classification Results.
Generates:
1. 7 Grouped Bar Charts for each metric (Mean ± Std)
2. 1 Radar Chart for overall comparison
3. Heatmaps for each model across datasets × 7 metrics
4. 1 Summary Table figure
5. CSV export of detailed results
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set high-quality styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

COLORS = {
    "BR": "#2563EB",           # Royal Blue
    "BR_Logistic": "#0D9488",  # Teal / Emerald Green
    "CC": "#EA580C",           # Vibrant Orange
    "CC_Logistic": "#D97706"   # Amber
}

FALLBACK_COLORS = ["#6366F1", "#8B5CF6", "#EC4899", "#14B8A6", "#F59E0B"]

MODEL_LABELS = {
    "BR": "BR (LinearSVC)",
    "BR_Logistic": "BR (Logistic Reg)",
    "CC": "CC (LinearSVC)",
    "CC_Logistic": "CC (Logistic Reg)"
}

HEATMAP_CMAPS = {
    "BR": "Blues",
    "BR_Logistic": "YlGn",
    "CC": "Oranges",
    "CC_Logistic": "YlOrBr"
}

METRIC_FILENAMES = {
    "Macro-F1": "macro_f1_comparison.png",
    "Micro-F1": "micro_f1_comparison.png",
    "Hamming Loss": "hamming_loss_comparison.png",
    "Subset Accuracy": "subset_accuracy_comparison.png",
    "Example-F1": "example_f1_comparison.png",
    "Macro Precision": "macro_precision_comparison.png",
    "Macro Recall": "macro_recall_comparison.png"
}


def _get_model_color(model_name, idx=0):
    return COLORS.get(model_name, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])


def _get_model_label(model_name):
    return MODEL_LABELS.get(model_name, model_name)


def plot_grouped_bar(all_results, metric_name, output_dir):
    """
    Generate a grouped bar chart with error bars comparing models across all datasets.
    """
    datasets = list(all_results.keys())
    first_dataset = datasets[0]
    models = list(all_results[first_dataset].keys())
    n_models = len(models)

    x = np.arange(len(datasets))
    total_width = 0.8
    width = total_width / max(n_models, 1)

    fig_width = max(13, len(datasets) * 1.3)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    all_means = []

    for m_idx, m_name in enumerate(models):
        means = [all_results[d][m_name]["mean"][metric_name] for d in datasets]
        stds = [all_results[d][m_name]["std"][metric_name] for d in datasets]
        all_means.extend(means)

        offset = (m_idx - (n_models - 1) / 2.0) * width
        color = _get_model_color(m_name, m_idx)
        label = _get_model_label(m_name)

        rects = ax.bar(
            x + offset, means, width * 0.92, yerr=stds,
            label=label, color=color,
            alpha=0.9, capsize=3.5, edgecolor="black", linewidth=0.7,
            error_kw={"elinewidth": 1.1}
        )

        for rect, mean_val in zip(rects, means):
            height = rect.get_height()
            ax.annotate(
                f"{mean_val:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5, rotation=90
            )

    ax.set_xlabel("Datasets", fontweight="bold", labelpad=10)
    ax.set_ylabel(f"{metric_name} (Mean ± Std)", fontweight="bold", labelpad=10)
    direction_hint = "(Lower is better)" if metric_name == "Hamming Loss" else "(Higher is better)"
    model_str = " vs ".join([_get_model_label(m) for m in models])
    ax.set_title(f"{metric_name} Comparison: {model_str} across Datasets {direction_hint}",
                 fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in datasets], rotation=30, ha="right", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    max_val = max(all_means) if all_means else 1.0
    ax.set_ylim(0, max(max_val * 1.25, 0.1))

    plt.tight_layout()
    filename = METRIC_FILENAMES.get(metric_name, f"{metric_name.lower().replace(' ', '_')}.png")
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_radar_chart(all_results, output_dir):
    """
    Generate a Radar/Spider Chart comparing overall average performance across all 7 metrics.
    """
    datasets = list(all_results.keys())
    models = list(all_results[datasets[0]].keys())
    metrics = list(METRIC_FILENAMES.keys())

    # Number of variables
    N = len(metrics)
    labels = [m if m != "Hamming Loss" else "1 - Hamming Loss" for m in metrics]

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], labels, color="black", size=11, fontweight="bold")
    ax.set_rlabel_position(30)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=9)
    plt.ylim(0, 1.0)

    for m_idx, m_name in enumerate(models):
        overall = []
        for m in metrics:
            avg_val = np.mean([all_results[d][m_name]["mean"][m] for d in datasets])
            if m == "Hamming Loss":
                val = 1.0 - avg_val
            else:
                val = avg_val
            overall.append(val)

        vals = overall + overall[:1]
        color = _get_model_color(m_name, m_idx)
        label = _get_model_label(m_name)

        ax.plot(angles, vals, linewidth=2.5, linestyle="solid", label=label, color=color)
        ax.fill(angles, vals, color=color, alpha=0.20)

    plt.title("Overall Performance Comparison Across All Metrics\n(Average over Datasets, Higher = Better)",
              size=14, fontweight="bold", y=1.08)
    plt.legend(loc="upper right", bbox_to_anchor=(1.30, 1.1), frameon=True, shadow=True)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "radar_overall.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_heatmaps(all_results, output_dir):
    """
    Generate Heatmaps for each model across all datasets and metrics.
    """
    datasets = list(all_results.keys())
    models = list(all_results[datasets[0]].keys())
    metrics = list(METRIC_FILENAMES.keys())

    generated_paths = []

    for m_name in models:
        matrix = np.zeros((len(datasets), len(metrics)))
        for i, d in enumerate(datasets):
            for j, m in enumerate(metrics):
                matrix[i, j] = all_results[d][m_name]["mean"][m]

        df_m = pd.DataFrame(matrix, index=[d.upper() for d in datasets], columns=metrics)
        cmap = HEATMAP_CMAPS.get(m_name, "Blues")

        fig, ax = plt.subplots(figsize=(11, max(6, len(datasets) * 0.65)))
        sns.heatmap(
            df_m, annot=True, fmt=".3f", cmap=cmap, cbar=True,
            linewidths=1, linecolor="white", ax=ax, annot_kws={"size": 10, "weight": "bold"}
        )
        model_title = _get_model_label(m_name)
        ax.set_title(f"{model_title} Performance Heatmap (Mean across 5 Folds)",
                     fontweight="bold", pad=15)
        plt.xticks(rotation=30, ha="right", fontweight="bold")
        plt.yticks(rotation=0, fontweight="bold")
        plt.tight_layout()

        filename = f"heatmap_{m_name.lower()}.png"
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path)
        plt.close()
        generated_paths.append(out_path)

    return generated_paths


def plot_summary_table(all_results, output_dir):
    """
    Render a clean graphical summary table of all experimental results.
    """
    datasets = list(all_results.keys())
    models = list(all_results[datasets[0]].keys())
    metrics = ["Macro-F1", "Micro-F1", "Hamming Loss", "Subset Accuracy", "Example-F1"]

    rows = []
    for d in datasets:
        for m in models:
            row = [d.upper(), _get_model_label(m)]
            for metric in metrics:
                mean_v = all_results[d][m]["mean"][metric]
                std_v = all_results[d][m]["std"][metric]
                row.append(f"{mean_v:.3f} ± {std_v:.3f}")
            rows.append(row)

    col_labels = ["Dataset", "Model"] + metrics

    n_rows = len(rows)
    fig_height = max(8, n_rows * 0.45 + 2)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9.0)
    table.scale(1.1, 1.35)

    # Alternate background shading per dataset block
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#1E293B")
            cell.set_text_props(color="white", weight="bold")
        else:
            d_idx = (row_idx - 1) // len(models)
            if d_idx % 2 == 0:
                cell.set_facecolor("#F8FAFC")
            else:
                cell.set_facecolor("#EFF6FF")

    plt.title("5-Fold Cross-Validation Performance Summary (Mean ± Std)",
              fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "summary_table.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


def export_results_table(all_results, output_csv_path):
    """
    Export all results (Mean and Std) to a structured CSV file.
    """
    records = []
    datasets = list(all_results.keys())
    models = list(all_results[datasets[0]].keys())
    metrics = list(METRIC_FILENAMES.keys())

    for d in datasets:
        for model in models:
            rec = {
                "Dataset": d,
                "Model": model,
                "Model_Name": _get_model_label(model)
            }
            for m in metrics:
                rec[f"{m}_Mean"] = all_results[d][model]["mean"][m]
                rec[f"{m}_Std"] = all_results[d][model]["std"][m]
            records.append(rec)

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    return df


def generate_all_plots(all_results, output_dir="results/figures", tables_dir="results/tables"):
    """
    Generate all evaluation plots and export CSV tables.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    generated_files = []

    # 1-7: Grouped Bar Charts for each metric
    for metric in METRIC_FILENAMES.keys():
        p = plot_grouped_bar(all_results, metric, output_dir)
        generated_files.append(p)

    # 8: Radar Chart
    p_radar = plot_radar_chart(all_results, output_dir)
    generated_files.append(p_radar)

    # 9+: Heatmaps for each model
    p_heatmaps = plot_heatmaps(all_results, output_dir)
    generated_files.extend(p_heatmaps)

    # Summary Table
    p_tab = plot_summary_table(all_results, output_dir)
    generated_files.append(p_tab)

    # CSV Export
    csv_path = os.path.join(tables_dir, "summary_results.csv")
    export_results_table(all_results, csv_path)

    return generated_files, csv_path

