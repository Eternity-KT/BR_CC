"""
Visualization module for BR and CC Multi-Label Classification Results.
Generates:
1. 5 Grouped Bar Charts for each complete-prediction metric (Mean ± Std)
2. 1 Radar Chart for overall comparison
3. Heatmaps for each model across datasets × 5 metrics
4. 1 Summary Table figure
5. CSV export of detailed results
"""

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
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
    "BR_MLP": "#7C3AED",       # Deep Purple
    "CC": "#EA580C",           # Vibrant Orange
    "CC_Logistic": "#D97706",  # Amber
    "CC_MLP": "#E11D48",       # Rose / Crimson
    "MLC_PA": "#0891B2",       # Cyan
    "MLC_PA_Logistic": "#C2410C",  # Burnt orange
    "BSS_UG_SPCC_PA_Logistic": "#4D7C0F",  # Olive green
    "GSI_MLC_PA": "#16A34A"    # Green
}

FALLBACK_COLORS = ["#6366F1", "#8B5CF6", "#EC4899", "#14B8A6", "#F59E0B", "#10B981"]

MODEL_LABELS = {
    "BR": "BR (LinearSVC)",
    "BR_Logistic": "BR (Logistic Reg)",
    "BR_MLP": "BR (MLP)",
    "CC": "CC (LinearSVC)",
    "CC_Logistic": "CC (Logistic Reg)",
    "CC_MLP": "CC (MLP)",
    "MLC_PA": "MLC-PA",
    "MLC_PA_Logistic": "MLC-PA (Logistic)",
    "BSS_UG_SPCC_PA_Logistic": "BSS-UG-SPCC-PA (Logistic)",
    "GSI_MLC_PA": "GSI-MLC-PA"
}

HEATMAP_CMAPS = {
    "BR": "Blues",
    "BR_Logistic": "YlGn",
    "BR_MLP": "Purples",
    "CC": "Oranges",
    "CC_Logistic": "YlOrBr",
    "CC_MLP": "Reds",
    "MLC_PA": "GnBu",
    "GSI_MLC_PA": "YlGn"
}

METRIC_FILENAMES = {
    "Macro-F1": "macro_f1_comparison.png",
    "Micro-F1": "micro_f1_comparison.png",
    "Hamming Loss": "hamming_loss_comparison.png",
    "Subset Accuracy": "subset_accuracy_comparison.png",
    "Example-F1": "example_f1_comparison.png",
}

PARTIAL_METRIC_FILENAMES = {
    "Generalized Loss": "generalized_loss_comparison.png",
    "Generalized Hamming Loss": "mlc_pa_generalized_hamming_loss.png",
    "Selective Hamming Loss": "mlc_pa_selective_hamming_loss.png",
    "Selective Macro-F1": "selective_macro_f1.png",
    "Selective Micro-F1": "selective_micro_f1.png",
    "GSI Selection Objective": "gsi_selection_objective.png",
    "Independent Label Count": "gsi_independent_label_count.png",
    "Dependent Label Count": "gsi_dependent_label_count.png",
    "Validation Selection Objective": "gsi_validation_objective.png",
    "Coverage": "mlc_pa_coverage.png",
    "Abstention Rate": "mlc_pa_abstention_rate.png",
    "ABS": "abs_comparison.png",
    "AABS": "aabs_comparison.png",
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
    models = [
        model_name
        for model_name in all_results[first_dataset].keys()
        if all(
            metric_name in all_results[dataset_name][model_name]["mean"]
            for dataset_name in datasets
        )
    ]
    if not models:
        return None
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
    lower_is_better = metric_name in {
        "Hamming Loss",
        "Generalized Hamming Loss",
        "Selective Hamming Loss",
    }
    if metric_name in {
        "Coverage",
        "Abstention Rate",
        "Independent Label Count",
        "Dependent Label Count",
        "Validation Selection Objective",
    }:
        direction_hint = "(Descriptive; interpret together with loss)"
    else:
        direction_hint = (
            "(Lower is better)" if lower_is_better else "(Higher is better)"
        )
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
    filename = METRIC_FILENAMES.get(
        metric_name,
        PARTIAL_METRIC_FILENAMES.get(
            metric_name, f"{metric_name.lower().replace(' ', '_')}.png"
        ),
    )
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_radar_chart(all_results, output_dir):
    """
    Generate a radar chart comparing the five complete-prediction metrics.
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
    for metric in PARTIAL_METRIC_FILENAMES:
        if any(
            metric in all_results[d][model]["mean"]
            for d in datasets
            for model in models
        ):
            metrics.append(metric)

    for d in datasets:
        for model in models:
            rec = {
                "Dataset": d,
                "Model": model,
                "Model_Name": _get_model_label(model)
            }
            for m in metrics:
                rec[f"{m}_Mean"] = all_results[d][model]["mean"].get(m, np.nan)
                rec[f"{m}_Std"] = all_results[d][model]["std"].get(m, np.nan)
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
        if p is not None:
            generated_files.append(p)

    # MLC-PA metrics are plotted separately because conventional classifiers
    # do not produce partial predictions and therefore have no comparable
    # generalized abstention risk or coverage.
    for metric in PARTIAL_METRIC_FILENAMES:
        p = plot_grouped_bar(all_results, metric, output_dir)
        if p is not None:
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


# ---------------------------------------------------------------------------
# Partial-abstention benchmark figures (schema v2)
# ---------------------------------------------------------------------------

PA_MODEL_LABELS = {
    "BR_MLP": "BR",
    "BR": "BR",
    "CC_MLP": "CC",
    "CC": "CC",
    "MLC_PA": "MLC-PA",
    "GSI_MLC_PA": "GSI-MLC-PA",
}

PA_MODEL_COLORS = {
    "BR_MLP": "#2563EB",
    "BR": "#2563EB",
    "CC_MLP": "#D97706",
    "CC": "#D97706",
    "MLC_PA": "#C2410C",
    "GSI_MLC_PA": "#4D7C0F",
}

PA_COST_COLORS = ["#2563EB", "#D97706", "#C2410C", "#4D7C0F", "#BE185D"]


def _pa_cost_key(cost):
    return f"{float(cost):.2f}"


def _pa_model_label(model_name):
    return PA_MODEL_LABELS.get(model_name, _get_model_label(model_name))


def _pa_models(all_results):
    first_dataset = next(iter(all_results))
    return list(all_results[first_dataset].keys())


def _pa_common_datasets(all_results, models, required_costs=()):
    required_keys = [_pa_cost_key(cost) for cost in required_costs]
    common = []
    for dataset_name, dataset_results in all_results.items():
        if not all(model in dataset_results for model in models):
            continue
        valid = True
        for model in models:
            result = dataset_results[model]
            if "full" not in result or "mean" not in result["full"]:
                valid = False
                break
            if result.get("costs") and not all(
                key in result["costs"] for key in required_keys
            ):
                valid = False
                break
        if valid:
            common.append(dataset_name)
    return common


def _pa_metric_location(result, metric_name, report_cost):
    """Return the comparable summary and source metric for one model."""
    if result.get("costs"):
        cost_summary = result["costs"][_pa_cost_key(report_cost)]
        return cost_summary, metric_name
    full_metric = {
        "Selective Macro-F1": "Macro-F1",
        "Selective Micro-F1": "Micro-F1",
        "Generalized Loss": "Hamming Loss",
    }[metric_name]
    return result["full"], full_metric


def plot_rejection_cost_comparison(
    all_results, abstention_costs, output_dir
):
    """Create the requested cost-by-cost full/selective and ABS/AABS figure."""
    models = _pa_models(all_results)
    datasets = _pa_common_datasets(
        all_results, models, required_costs=abstention_costs
    )
    if not datasets:
        raise ValueError(
            "No common completed datasets contain every model and abstention cost."
        )
    selective_models = [
        model
        for model in models
        if all(all_results[d][model].get("costs") for d in datasets)
    ]
    if not selective_models:
        raise ValueError("The rejection figure needs at least one selective model.")

    n_rows = len(abstention_costs)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(14, max(4.0 * n_rows, 6.5)),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    model_x = np.arange(len(models), dtype=np.float64)
    width = 0.34

    for row, cost in enumerate(abstention_costs):
        key = _pa_cost_key(cost)
        color = PA_COST_COLORS[row % len(PA_COST_COLORS)]
        left, right = axes[row]

        full_values = [
            100.0
            * float(np.mean([
                all_results[d][model]["full"]["mean"]["Macro-F1"]
                for d in datasets
            ]))
            for model in models
        ]
        left.bar(
            model_x - width / 2,
            full_values,
            width,
            color="#C7DDEA",
            edgecolor="#334155",
            linewidth=0.7,
            label="Full prediction" if row == 0 else None,
        )

        for model_index, model in enumerate(models):
            if model not in selective_models:
                continue
            selective_value = 100.0 * float(np.mean([
                all_results[d][model]["costs"][key]["mean"][
                    "Selective Macro-F1"
                ]
                for d in datasets
            ]))
            bar = left.bar(
                model_x[model_index] + width / 2,
                selective_value,
                width,
                color=color,
                edgecolor="#1F2937",
                linewidth=0.8,
                hatch="//",
                label=(
                    "With rejection"
                    if row == 0 and model == selective_models[0]
                    else None
                ),
            )[0]
            delta = selective_value - full_values[model_index]
            left.annotate(
                f"{delta:+.1f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#111827",
            )

        left.set_ylim(0, 100)
        left.set_ylabel("Macro-F1 (%)")
        left.set_xticks(model_x)
        left.set_xticklabels([_pa_model_label(model) for model in models])
        left.set_title(f"Prediction quality — c = {float(cost):.2f}")
        left.grid(axis="y", linestyle="--", alpha=0.35)
        if row == 0:
            left.legend(loc="upper left", frameon=False, ncol=2)

        selective_x = np.arange(len(selective_models), dtype=np.float64)
        abs_values = [
            100.0 * float(np.mean([
                all_results[d][model]["costs"][key]["mean"]["ABS"]
                for d in datasets
            ]))
            for model in selective_models
        ]
        aabs_values = [
            100.0 * float(np.mean([
                all_results[d][model]["costs"][key]["mean"]["AABS"]
                for d in datasets
            ]))
            for model in selective_models
        ]
        right.bar(
            selective_x,
            abs_values,
            width=0.58,
            color=color,
            edgecolor="#1F2937",
            linewidth=0.8,
            hatch="//",
            label="ABS" if row == 0 else None,
        )
        right.scatter(
            selective_x,
            aabs_values,
            s=52,
            facecolors="white",
            edgecolors="#111827",
            linewidths=1.2,
            zorder=4,
            label="AABS" if row == 0 else None,
        )
        right.set_ylim(0, 100)
        right.set_ylabel("Rejection rate (%)")
        right.set_xticks(selective_x)
        right.set_xticklabels([
            _pa_model_label(model) for model in selective_models
        ])
        right.set_title(f"Rejection extent — c = {float(cost):.2f}")
        right.grid(axis="y", linestyle="--", alpha=0.35)
        if row == 0:
            right.legend(loc="upper right", frameon=False)

    dataset_text = ", ".join(dataset.upper() for dataset in datasets)
    fig.suptitle(
        "Full prediction versus partial abstention across rejection costs",
        fontsize=17,
        fontweight="bold",
        y=0.997,
    )
    fig.text(
        0.5,
        0.979,
        f"Unweighted mean across {len(datasets)} common datasets: {dataset_text}",
        ha="center",
        va="top",
        fontsize=10,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.963), h_pad=2.0, w_pad=2.2)
    output_path = os.path.join(output_dir, "rejection_cost_comparison.png")
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)
    return output_path


def plot_pa_metric_comparison(
    all_results, metric_name, report_cost, output_dir
):
    """Grouped per-dataset mean +/- fold std for one comparable metric."""
    models = _pa_models(all_results)
    datasets = _pa_common_datasets(
        all_results, models, required_costs=[report_cost]
    )
    if not datasets:
        raise ValueError(
            f"No common completed datasets contain {metric_name} at "
            f"c={float(report_cost):.2f}."
        )

    x = np.arange(len(datasets), dtype=np.float64)
    total_width = 0.82
    width = total_width / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(max(13, 1.35 * len(datasets)), 6.8))
    plotted_values = []

    for model_index, model in enumerate(models):
        means = []
        stds = []
        for dataset in datasets:
            summary, source_metric = _pa_metric_location(
                all_results[dataset][model], metric_name, report_cost
            )
            means.append(float(summary["mean"][source_metric]))
            stds.append(float(summary["std"].get(source_metric, 0.0)))
        plotted_values.extend(
            mean + max(std, 0.0) for mean, std in zip(means, stds)
        )
        offset = (model_index - (len(models) - 1) / 2.0) * width
        bars = ax.bar(
            x + offset,
            means,
            width * 0.92,
            yerr=stds,
            capsize=3.2,
            color=PA_MODEL_COLORS.get(model, _get_model_color(model, model_index)),
            edgecolor="#1F2937",
            linewidth=0.75,
            label=_pa_model_label(model),
            error_kw={"elinewidth": 1.0},
        )
        for bar, value, std in zip(bars, means, stds):
            ax.annotate(
                f"{value:.3f}",
                (
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(std, 0.0),
                ),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=90,
            )

    lower_is_better = metric_name == "Generalized Loss"
    direction = "Lower is better" if lower_is_better else "Higher is better"
    ax.set_title(
        f"{metric_name} comparison across datasets",
        fontweight="bold",
        pad=40,
    )
    ax.text(
        0.5,
        1.015,
        (
            f"Mean ± std over outer CV folds | selective models use "
            f"c={float(report_cost):.2f} | {direction}"
        ),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#475569",
    )
    ax.set_xlabel("Datasets", fontweight="bold")
    ax.set_ylabel(f"{metric_name} (mean ± std)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [dataset.upper() for dataset in datasets],
        rotation=30,
        ha="right",
        fontweight="bold",
    )
    upper = max(plotted_values) if plotted_values else 1.0
    ax.set_ylim(0, min(1.05, max(upper * 1.28, 0.1)))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, ncol=min(4, len(models)), loc="upper left")
    fig.tight_layout()

    filenames = {
        "Selective Macro-F1": "selective_macro_f1_comparison.png",
        "Selective Micro-F1": "selective_micro_f1_comparison.png",
        "Generalized Loss": "generalized_loss_comparison.png",
    }
    output_path = os.path.join(output_dir, filenames[metric_name])
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)
    return output_path


def export_pa_results_table(all_results, output_csv_path):
    """Export schema-v2 full and cost-specific summaries for plot auditing."""
    records = []
    for dataset, dataset_results in all_results.items():
        for model, result in dataset_results.items():
            for scope, cost, summary in [
                ("full", None, result["full"]),
                *[
                    ("rejection", float(cost_key), cost_summary)
                    for cost_key, cost_summary in result.get("costs", {}).items()
                ],
            ]:
                record = {
                    "Dataset": dataset,
                    "Model": model,
                    "Scope": scope,
                    "Abstention_Cost": cost,
                }
                for metric, value in summary["mean"].items():
                    record[f"{metric}_Mean"] = value
                    record[f"{metric}_Std"] = summary["std"].get(metric, np.nan)
                records.append(record)
    frame = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    frame.to_csv(output_csv_path, index=False)
    return frame


def generate_pa_plots(
    all_results,
    abstention_costs,
    report_cost=0.3,
    output_dir="results_pa/plots_pa",
    tables_dir="results_pa/tables",
):
    """Generate the four requested partial-abstention comparison figures."""
    if not all_results:
        raise ValueError("all_results is empty; no partial-abstention plots to draw.")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    generated = [
        plot_rejection_cost_comparison(
            all_results, abstention_costs=abstention_costs, output_dir=output_dir
        )
    ]
    for metric in (
        "Selective Macro-F1",
        "Selective Micro-F1",
        "Generalized Loss",
    ):
        generated.append(
            plot_pa_metric_comparison(
                all_results,
                metric_name=metric,
                report_cost=report_cost,
                output_dir=output_dir,
            )
        )

    csv_path = os.path.join(tables_dir, "summary_results_pa.csv")
    export_pa_results_table(all_results, csv_path)
    return generated, csv_path


BSS_DATASET_FIGURES = {
    "Hamming Accuracy": "dataset_hamming_accuracy_comparison.png",
    "Subset Accuracy": "dataset_subset_accuracy_comparison.png",
    "Macro-F1": "dataset_macro_f1_comparison.png",
    "Micro-F1": "dataset_micro_f1_comparison.png",
    "Instance-F1": "dataset_instance_f1_comparison.png",
    "Generalized Loss": "dataset_generalized_loss_comparison.png",
    "Selective Macro-F1": "dataset_selective_macro_f1_comparison.png",
    "Selective Micro-F1": "dataset_selective_micro_f1_comparison.png",
    "Selective Instance-F1": "dataset_selective_instance_f1_comparison.png",
    "Selective Hamming Accuracy": "dataset_selective_hamming_accuracy_comparison.png",
}

BSS_REJECTION_FIGURES = {
    "Generalized Loss": "rejection_cost_generalized_loss.png",
    "Selective Macro-F1": "rejection_cost_selective_macro_f1.png",
    "Selective Micro-F1": "rejection_cost_selective_micro_f1.png",
    "Selective Instance-F1": "rejection_cost_selective_instance_f1.png",
    "Selective Hamming Accuracy": "rejection_cost_selective_hamming_accuracy.png",
}


def _atomic_save_figure(figure, target):
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}_", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    try:
        figure.savefig(temporary, dpi=300, facecolor="white", bbox_inches="tight")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    finally:
        plt.close(figure)
    return str(path)


def _completed_bss_datasets(run_document):
    requested_models = list(run_document.get("Settings", {}).get("models", []))
    completed = []
    for dataset, models in run_document.get("Results", {}).items():
        if not requested_models or not all(model in models for model in requested_models):
            continue
        valid = True
        for model in requested_models:
            summary = models[model]
            expected = int(summary.get("Expected Fold Count", 0))
            if (
                summary.get("Status") != "complete"
                or expected <= 0
                or summary.get("Completed Folds") != list(range(1, expected + 1))
            ):
                valid = False
                break
        if valid:
            completed.append(dataset)
    return completed, requested_models


def _bss_plot_models(run_document, bss_models):
    comparison = run_document.get("Comparison", {})
    ordered = [
        model
        for model in comparison.get("Model Order", [])
        if model in comparison.get("Models", {})
    ]
    return ordered + [model for model in bss_models if model not in ordered]


def _bss_model_record(run_document, dataset, model):
    if model in run_document.get("Results", {}).get(dataset, {}):
        return {
            "Kind": "selective",
            "Data": run_document["Results"][dataset][model],
        }
    source = run_document.get("Comparison", {}).get("Models", {}).get(model)
    if source is None or dataset not in source.get("Datasets", {}):
        return None
    return {"Kind": source.get("Kind"), "Data": source["Datasets"][dataset]}


def _bss_model_kind(run_document, model):
    if model in run_document.get("Settings", {}).get("models", []):
        return "selective"
    return (
        run_document.get("Comparison", {})
        .get("Models", {})
        .get(model, {})
        .get("Kind")
    )


def _full_counterpart_summary(full, selective_metric):
    mapping = {
        "Selective Macro-F1": "Macro-F1",
        "Selective Micro-F1": "Micro-F1",
        "Selective Instance-F1": "Instance-F1",
        "Selective Hamming Accuracy": "Hamming Accuracy",
    }
    source_metric = (
        "Hamming Accuracy"
        if selective_metric == "Generalized Loss"
        else mapping.get(selective_metric)
    )
    if source_metric is None or source_metric not in full.get("Mean", {}):
        return None
    transform = (
        (lambda value: 1.0 - float(value))
        if selective_metric == "Generalized Loss"
        else (lambda value: float(value))
    )
    rows = []
    for source in full.get("Raw Folds", []):
        if source.get(source_metric) is None:
            continue
        rows.append({
            "Fold": source.get("Fold"),
            selective_metric: transform(source[source_metric]),
        })
    return {
        "Mean": {selective_metric: transform(full["Mean"][source_metric])},
        # std(1-X) == std(X), so this complement is exact.
        "Std": {
            selective_metric: float(full.get("Std", {}).get(source_metric, 0.0))
        },
        "Raw Folds": rows,
    }


def _bss_summary(run_document, dataset, model, metric, report_cost):
    record = _bss_model_record(run_document, dataset, model)
    if record is None:
        return None
    data = record["Data"]
    full = data.get("Full", {})
    if metric in full.get("Mean", {}):
        return full
    if record["Kind"] == "complete_only":
        return _full_counterpart_summary(full, metric)
    cost_key = f"{float(report_cost):.2f}"
    selective = data.get("Costs", {}).get(cost_key, {}).get("Selective", {})
    if metric not in selective.get("Mean", {}):
        return None
    return selective


def _plot_bss_dataset_metric(
    run_document, completed, models, metric, report_cost, target
):
    x = np.arange(len(completed), dtype=np.float64)
    width = 0.78 / max(len(models), 1)
    figure, axis = plt.subplots(figsize=(max(10.5, len(completed) * 1.25), 6.2))
    values_for_limits = []
    for model_index, model in enumerate(models):
        positions, means, stds = [], [], []
        for dataset_index, dataset in enumerate(completed):
            summary = _bss_summary(run_document, dataset, model, metric, report_cost)
            if summary is None or summary.get("Mean", {}).get(metric) is None:
                continue
            positions.append(dataset_index)
            means.append(float(summary["Mean"][metric]))
            stds.append(float(summary.get("Std", {}).get(metric, 0.0)))
        offset = (model_index - (len(models) - 1) / 2.0) * width
        if not means:
            axis.bar(
                [],
                [],
                color=_get_model_color(model, model_index),
                edgecolor="#1F2937",
                linewidth=0.7,
                label=f"{_get_model_label(model)} (N/A)",
            )
            for dataset_index in range(len(completed)):
                axis.text(
                    dataset_index + offset,
                    0.015,
                    "N/A",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#64748B",
                )
            continue
        values_for_limits.extend(
            value + max(error, 0.0) for value, error in zip(means, stds)
        )
        axis.bar(
            np.asarray(positions, dtype=np.float64) + offset,
            means,
            width * 0.92,
            yerr=stds,
            capsize=3,
            color=_get_model_color(model, model_index),
            edgecolor="#1F2937",
            linewidth=0.7,
            label=_get_model_label(model),
        )
        for dataset_index in sorted(set(range(len(completed))) - set(positions)):
            axis.text(
                dataset_index + offset,
                0.015,
                "N/A",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=7,
                color="#64748B",
            )
    lower = metric == "Generalized Loss"
    scope_note = (
        "full prediction"
        if metric in ("Hamming Accuracy", "Subset Accuracy", "Macro-F1", "Micro-F1", "Instance-F1")
        else f"selective prediction at c={float(report_cost):.2f}"
    )
    axis.set_title(
        f"{metric} across completed datasets\n"
        f"{scope_note}; mean ± std over outer folds; "
        f"{'lower' if lower else 'higher'} is better; "
        f"updated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    axis.set_ylabel(metric)
    axis.set_xlabel(f"Completed datasets (n={len(completed)})")
    axis.set_xticks(x)
    axis.set_xticklabels([name.upper() for name in completed], rotation=30, ha="right")
    upper = max(values_for_limits) if values_for_limits else 1.0
    axis.set_ylim(0.0, max(0.1, upper * 1.18))
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.legend(frameon=False)
    figure.tight_layout()
    return _atomic_save_figure(figure, target)


def _full_counterpart_row(row, selective_metric):
    if selective_metric == "Generalized Loss":
        return 1.0 - float(row["Hamming Accuracy"])
    mapping = {
        "Selective Macro-F1": "Macro-F1",
        "Selective Micro-F1": "Micro-F1",
        "Selective Instance-F1": "Instance-F1",
        "Selective Hamming Accuracy": "Hamming Accuracy",
    }
    return float(row[mapping[selective_metric]])


def _plot_bss_rejection_cost(
    run_document, completed, models, metric, costs, target
):
    def finite_values(rows, name):
        values = []
        for row in rows:
            value = row.get(name)
            if value is None:
                continue
            numeric = float(value)
            if np.isfinite(numeric):
                values.append(numeric)
        return values

    def mean_std(values):
        return (
            (float(np.mean(values)), float(np.std(values)))
            if values
            else (float("nan"), 0.0)
        )

    figure, axes = plt.subplots(
        len(costs), 2, figsize=(13.0, max(3.1 * len(costs), 5.8)), squeeze=False
    )
    lower = metric == "Generalized Loss"
    selective_models = [
        model for model in models if _bss_model_kind(run_document, model) == "selective"
    ]
    for row_index, cost in enumerate(costs):
        cost_key = f"{float(cost):.2f}"
        left, right = axes[row_index]
        full_legend_added = False
        selective_legend_added = False
        for model_index, model in enumerate(models):
            full_values, selective_values = [], []
            for dataset in completed:
                record = _bss_model_record(run_document, dataset, model)
                if record is None:
                    continue
                for fold_row in record["Data"].get("Full", {}).get("Raw Folds", []):
                    try:
                        full_values.append(_full_counterpart_row(fold_row, metric))
                    except (KeyError, TypeError, ValueError):
                        continue
                if record["Kind"] == "selective":
                    selective_rows = (
                        record["Data"]
                        .get("Costs", {})
                        .get(cost_key, {})
                        .get("Selective", {})
                        .get("Raw Folds", [])
                    )
                    selective_values.extend(finite_values(selective_rows, metric))
            is_selective = model in selective_models
            full_mean, full_std = mean_std(full_values)
            if np.isfinite(full_mean):
                left.bar(
                    model_index - (0.16 if is_selective else 0.0),
                    full_mean,
                    width=0.50,
                    yerr=full_std,
                    color="#D7E5EE",
                    edgecolor="#475569",
                    capsize=2,
                    label="Full prediction" if not full_legend_added else None,
                )
                full_legend_added = True
            if not is_selective:
                continue
            selective_mean, selective_std = mean_std(selective_values)
            if np.isfinite(selective_mean):
                left.bar(
                    model_index + 0.16,
                    selective_mean,
                    width=0.50,
                    yerr=selective_std,
                    color=_get_model_color(model, model_index),
                    edgecolor="#1F2937",
                    hatch="//",
                    capsize=2,
                    label="With rejection"
                    if not selective_legend_added
                    else None,
                )
                selective_legend_added = True
            else:
                left.text(
                    model_index + 0.16,
                    0.02,
                    "N/A",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#64748B",
                )
        left.set_xticks(np.arange(len(models)))
        left.set_xticklabels(
            [_get_model_label(model) for model in models], rotation=12
        )
        left.set_ylabel(f"{metric}\nc={float(cost):.2f}")
        left.set_ylim(0.0, 1.05)
        left.grid(axis="y", linestyle="--", alpha=0.3)
        for selective_index, model in enumerate(selective_models):
            abs_values, aabs_values = [], []
            for dataset in completed:
                record = _bss_model_record(run_document, dataset, model)
                if record is None:
                    continue
                rows = (
                    record["Data"]
                    .get("Costs", {})
                    .get(cost_key, {})
                    .get("Selective", {})
                    .get("Raw Folds", [])
                )
                abs_values.extend(finite_values(rows, "ABS"))
                aabs_values.extend(finite_values(rows, "AABS"))
            abs_mean, _ = mean_std(abs_values)
            aabs_mean, _ = mean_std(aabs_values)
            if np.isfinite(abs_mean):
                right.bar(
                    selective_index,
                    abs_mean,
                    width=0.55,
                    color=_get_model_color(model, selective_index),
                    edgecolor="#1F2937",
                    hatch="//",
                    label="ABS" if selective_index == 0 else None,
                )
            if np.isfinite(aabs_mean):
                right.scatter(
                    selective_index,
                    aabs_mean,
                    color="white",
                    edgecolor="#1F2937",
                    linewidth=1.2,
                    s=55,
                    zorder=3,
                    label="AABS" if selective_index == 0 else None,
                )
        right.set_xticks(np.arange(len(selective_models)))
        right.set_xticklabels(
            [_get_model_label(model) for model in selective_models], rotation=12
        )
        right.set_ylim(0.0, 1.05)
        right.grid(axis="y", linestyle="--", alpha=0.3)
        if row_index == 0:
            left.set_title(f"Full vs selective {metric}")
            right.set_title("ABS and AABS")
            left.legend(frameon=False, loc="upper left")
            right.legend(frameon=False, loc="upper right")
    figure.suptitle(
        f"Rejection-cost profile: {metric}\n"
        f"{len(completed)} completed datasets; costs={list(costs)}; means across "
        f"outer-fold observations by model; "
        f"{'lower' if lower else 'higher'} is better; "
        f"updated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        y=1.002,
    )
    figure.tight_layout()
    return _atomic_save_figure(figure, target)


def generate_bss_spcc_plots(run_document, output_dir):
    """Atomically rebuild the locked 10 dataset and 5 rejection-cost figures."""

    completed, bss_models = _completed_bss_datasets(run_document)
    if not completed:
        return {}
    models = _bss_plot_models(run_document, bss_models)
    settings = run_document.get("Settings", {})
    report_cost = float(settings.get("report_cost", 0.3))
    costs = tuple(float(cost) for cost in settings.get("abstention_costs", ()))
    output_path = Path(output_dir)
    artifacts = {}
    for metric, filename in BSS_DATASET_FIGURES.items():
        artifacts[filename] = _plot_bss_dataset_metric(
            run_document,
            completed,
            models,
            metric,
            report_cost,
            output_path / filename,
        )
    for metric, filename in BSS_REJECTION_FIGURES.items():
        artifacts[filename] = _plot_bss_rejection_cost(
            run_document,
            completed,
            models,
            metric,
            costs,
            output_path / filename,
        )
    return artifacts
