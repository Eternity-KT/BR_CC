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
