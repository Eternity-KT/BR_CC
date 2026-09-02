"""Robust deployment trade-off figures for schema-v3 summaries."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _deployment_rows(run_document):
    for dataset_name, models in run_document.get("Results", {}).items():
        for model_name, summary in models.items():
            for cost, cost_summary in summary.get("Costs", {}).items():
                for row in cost_summary.get("Deployment", {}).get("Raw Folds", []):
                    yield {
                        "Dataset": dataset_name,
                        "Model": model_name,
                        "Cost": float(cost),
                        **row,
                    }


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _tradeoff_figure(run_document, path, x_name, y_name, title, x_label, y_label):
    grouped = {}
    for row in _deployment_rows(run_document):
        if not _finite(row.get(x_name)) or not _finite(row.get(y_name)):
            continue
        grouped.setdefault(
            (row["Dataset"], row["Model"], row["Cost"]), []
        ).append(
            (float(row[x_name]), float(row[y_name]))
        )
    figure, axis = plt.subplots(figsize=(8, 5))
    series = sorted({(dataset, model) for dataset, model, _ in grouped})
    for dataset, model in series:
        points = []
        for (candidate_dataset, candidate_model, cost), values in grouped.items():
            if (candidate_dataset, candidate_model) != (dataset, model):
                continue
            points.append((
                float(np.mean([value[0] for value in values])),
                float(np.mean([value[1] for value in values])),
                cost,
            ))
        points.sort(key=lambda value: (value[0], value[2]))
        if points:
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                label=f"{dataset}/{model}",
            )
    if not series:
        axis.text(
            0.5,
            0.5,
            "No complete deployment records",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    else:
        axis.legend(fontsize=8)
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _critical_figure(run_document, path):
    values = {}
    for _, models in run_document.get("Results", {}).items():
        for model_name, summary in models.items():
            for cost_summary in summary.get("Costs", {}).values():
                for record in cost_summary.get("Deployment", {}).get(
                    "Critical Labels", []
                ):
                    metric = record.get("Metrics", {}).get("Critical Selective F1")
                    if record.get("Status") == "Available" and _finite(metric):
                        values.setdefault(model_name, []).append(float(metric))
    figure, axis = plt.subplots(figsize=(8, 5))
    models = sorted(values)
    if models:
        axis.bar(models, [float(np.mean(values[model])) for model in models])
        axis.tick_params(axis="x", labelrotation=30)
    else:
        axis.text(
            0.5,
            0.5,
            "Critical-label policy is N/A",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_title("Critical-label selective F1")
    axis.set_ylabel("F1")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _il_dl_figure(run_document, path):
    values = {}
    for dataset_name, models in run_document.get("Results", {}).items():
        for model_name, summary in models.items():
            for record in summary.get("Groups", {}).get("Raw Records", []):
                group = record.get("Group")
                metric = record.get(f"{group} Full Macro-F1")
                if group in ("IL", "DL") and _finite(metric):
                    values.setdefault(
                        (dataset_name, model_name, group), []
                    ).append(float(metric))
    figure, axis = plt.subplots(figsize=(9, 5))
    keys = sorted(values)
    if keys:
        labels = [f"{dataset}\n{model}\n{group}" for dataset, model, group in keys]
        axis.bar(
            labels,
            [float(np.mean(values[key])) for key in keys],
            color=["#4c78a8" if key[2] == "IL" else "#f58518" for key in keys],
        )
        axis.tick_params(axis="x", labelrotation=30, labelsize=8)
    else:
        axis.text(
            0.5, 0.5, "IL/DL records are unavailable", ha="center", va="center",
            transform=axis.transAxes,
        )
    axis.set_title("IL/DL full Macro-F1")
    axis.set_ylabel("Full Macro-F1")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _objective_figure(run_document, path):
    values = {}
    for dataset_name, models in run_document.get("Results", {}).items():
        for model_name, summary in models.items():
            full_macro_f1 = summary.get("Full", {}).get("Mean", {}).get(
                "Macro-F1"
            )
            if not _finite(full_macro_f1):
                continue
            objectives = {
                record.get("Selection Objective")
                for record in summary.get("Model Metadata", [])
                if record.get("Selection Objective")
            }
            for objective in objectives:
                values[(dataset_name, model_name, objective)] = float(full_macro_f1)
    figure, axis = plt.subplots(figsize=(9, 5))
    keys = sorted(values)
    if keys:
        labels = [
            f"{dataset}\n{model}\n{objective}"
            for dataset, model, objective in keys
        ]
        axis.bar(labels, [values[key] for key in keys], color="#54a24b")
        axis.tick_params(axis="x", labelrotation=30, labelsize=8)
    else:
        axis.text(
            0.5, 0.5, "Objective-ablation records are unavailable",
            ha="center", va="center", transform=axis.transAxes,
        )
    axis.set_title("Selection-objective comparison")
    axis.set_ylabel("Full Macro-F1 (outer test)")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _calibration_figure(run_document, path):
    values = {}
    for dataset_name, models in run_document.get("Results", {}).items():
        for model_name, summary in models.items():
            for record in summary.get("Calibration", {}).get(
                "Reliability", {}
            ).get("Raw Records", []):
                predicted = record.get("Mean Predicted Probability")
                observed = record.get("Observed Positive Rate")
                if _finite(predicted) and _finite(observed):
                    values.setdefault((dataset_name, model_name), []).append(
                        (float(predicted), float(observed))
                    )
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect")
    for (dataset, model), rows in sorted(values.items()):
        rows.sort(key=lambda row: row[0])
        axis.plot(
            [row[0] for row in rows],
            [row[1] for row in rows],
            marker="o",
            label=f"{dataset}/{model}",
        )
    if values:
        axis.legend(fontsize=7)
    else:
        axis.text(
            0.5, 0.4, "Reliability records are unavailable",
            ha="center", va="center", transform=axis.transAxes,
        )
    axis.set_title("Calibration reliability")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed positive rate")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def generate_deployment_plots(run_document, output_dir):
    """Generate Q10 figures even for partial/missing/NaN-heavy result sets."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "risk_coverage_plot": output_path / "risk_coverage.png",
        "optimistic_gain_review_plot": (
            output_path / "optimistic_gain_vs_review_load.png"
        ),
        "error_capture_review_plot": (
            output_path / "error_capture_vs_review_load.png"
        ),
        "critical_label_plot": output_path / "per_label_critical_metrics.png",
        "il_dl_ablation_plot": output_path / "il_dl_ablation.png",
        "objective_comparison_plot": output_path / "objective_comparison.png",
        "calibration_reliability_plot": (
            output_path / "calibration_reliability.png"
        ),
    }
    _tradeoff_figure(
        run_document,
        paths["risk_coverage_plot"],
        "Coverage",
        "Selective Risk",
        "Risk–coverage trade-off",
        "Coverage",
        "Selective risk",
    )
    _tradeoff_figure(
        run_document,
        paths["optimistic_gain_review_plot"],
        "Review Load Position",
        "Optimistic Gain Macro-F1",
        "Optimistic gain vs review load",
        "Review load (label positions)",
        "Optimistic Macro-F1 gain",
    )
    _tradeoff_figure(
        run_document,
        paths["error_capture_review_plot"],
        "Review Load Position",
        "Error Capture Rate",
        "Error capture vs review load",
        "Review load (label positions)",
        "Error capture rate",
    )
    _critical_figure(run_document, paths["critical_label_plot"])
    _il_dl_figure(run_document, paths["il_dl_ablation_plot"])
    _objective_figure(run_document, paths["objective_comparison_plot"])
    _calibration_figure(run_document, paths["calibration_reliability_plot"])
    return {name: str(path) for name, path in paths.items()}
