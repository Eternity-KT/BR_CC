"""Generate the 16 locked v6.3 figures from result tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


MODEL_ORDER = [
    "BR",
    "CC",
    "MLC_PA",
    "GSI_v6_2",
    "GSI_v6_3_1L_ResidualBR",
    "GSI_v6_3_CC_X",
    "GSI_v6_3_PCC_X",
    "GSI_v6_3_CC_X_IL",
    "GSI_v6_3_PCC_X_IL",
]

NON_ABSTAINING_BASELINES = ("BR", "CC")

COLORS = {
    "BR": "#b7dbe8",
    "CC": "#9ecae1",
    "MLC_PA": "#7dbbd4",
    "GSI_v6_2": "#5aa6c4",
    "GSI_v6_3_1L_ResidualBR": "#222222",
    "GSI_v6_3_CC_X": "#d62728",
    "GSI_v6_3_PCC_X": "#1f77b4",
    "GSI_v6_3_CC_X_IL": "#2ca02c",
    "GSI_v6_3_PCC_X_IL": "#9467bd",
}

COMPLETE_FIGURES = {
    "hamming_accuracy": "complete_hamming_accuracy.png",
    "subset_accuracy": "complete_subset_accuracy.png",
    "micro_f1": "complete_micro_f1.png",
    "macro_f1": "complete_macro_f1.png",
    "instance_f1": "complete_instance_f1.png",
}

SELECTIVE_FIGURES = {
    "selective_hamming_accuracy": "selective_hamming_accuracy.png",
    "selective_subset_accuracy": "selective_subset_accuracy.png",
    "selective_micro_f1": "selective_micro_f1.png",
    "selective_macro_f1": "selective_macro_f1.png",
    "selective_instance_f1": "selective_instance_f1.png",
    "generalized_loss": "selective_generalized_loss.png",
}

REJECTION_FIGURES = {
    "generalized_loss": "rejection_cost_generalized_loss.png",
    "selective_hamming_accuracy": "rejection_cost_hamming_accuracy.png",
    "selective_micro_f1": "rejection_cost_micro_f1.png",
    "selective_macro_f1": "rejection_cost_macro_f1.png",
    "selective_instance_f1": "rejection_cost_instance_f1.png",
}


def _read(path):
    frame = pd.read_csv(path, na_values=["-"], keep_default_na=True)
    return frame


def _style_axis(axis, title, ylabel):
    axis.set_title(title, fontsize=11, weight="semibold")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _dataset_order(frame):
    metadata = (
        frame[["dataset", "mean_ir"]]
        .drop_duplicates("dataset")
        .sort_values(["mean_ir", "dataset"], kind="stable")
    )
    return metadata["dataset"].tolist()


def _bar_panel(axis, mean_frame, std_frame, metric, learner_label):
    datasets = _dataset_order(mean_frame)
    models = [model for model in MODEL_ORDER if model in set(mean_frame["model"])]
    x_values = np.arange(len(datasets), dtype=np.float64)
    width = 0.82 / max(len(models), 1)
    for index, model in enumerate(models):
        selected = mean_frame[mean_frame["model"] == model].set_index("dataset")
        selected_std = std_frame[std_frame["model"] == model].set_index("dataset")
        values = pd.to_numeric(selected.reindex(datasets)[metric], errors="coerce")
        errors = pd.to_numeric(
            selected_std.reindex(datasets)[metric], errors="coerce"
        )
        positions = x_values - 0.41 + width / 2 + index * width
        valid = values.notna().to_numpy()
        axis.bar(
            positions[valid],
            values.to_numpy()[valid],
            width=width * 0.94,
            color=COLORS[model],
            edgecolor="white",
            linewidth=0.35,
            label=model,
            yerr=errors.fillna(0.0).to_numpy()[valid],
            capsize=1.5,
        )
    axis.set_xticks(x_values)
    axis.set_xticklabels(datasets, rotation=38, ha="right", fontsize=8)
    _style_axis(axis, learner_label, metric.replace("_", " ").title())


def _plot_bars(
    table,
    metric,
    output,
    *,
    report_cost=None,
    all_scope=False,
):
    frame = _read(table)
    if report_cost is not None:
        costs = pd.to_numeric(frame["cost"], errors="coerce")
        frame = frame[np.isclose(costs, report_cost, equal_nan=False)]
    mean = frame[frame["aggregation"] == "mean"].copy()
    std = frame[frame["aggregation"] == "std"].copy()
    learners = list(dict.fromkeys(mean["base_learner"].dropna().tolist()))
    if not all_scope:
        learners = learners[:1]
    if not learners:
        return
    figure, axes = plt.subplots(
        len(learners),
        1,
        figsize=(max(10, len(_dataset_order(mean)) * 1.15), 4.8 * len(learners)),
        squeeze=False,
    )
    for row, learner in enumerate(learners):
        _bar_panel(
            axes[row, 0],
            mean[mean["base_learner"] == learner],
            std[std["base_learner"] == learner],
            metric,
            str(learner),
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), fontsize=8)
    figure.suptitle(
        f"v6.3 — {metric.replace('_', ' ').title()} (datasets ordered by MeanIR)",
        y=0.995,
        fontsize=13,
    )
    note = (
        "BR and CC are complete-metric references (coverage=1; no label abstention)."
        if report_cost is not None
        else "Missing inherited metrics are omitted (never plotted as zero)."
    )
    figure.text(
        0.995,
        0.006,
        note,
        ha="right",
        fontsize=7,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.025, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_rejection_curve(table, metric, output, *, all_scope=False):
    frame = _read(table)
    frame = frame[frame["aggregation"] == "fold"].copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame["cost"] = pd.to_numeric(frame["cost"], errors="coerce")
    learners = list(dict.fromkeys(frame["base_learner"].dropna().tolist()))
    if not all_scope:
        learners = learners[:1]
    if not learners:
        return
    figure, axes = plt.subplots(
        len(learners), 1, figsize=(9.5, 4.6 * len(learners)), squeeze=False
    )
    for row, learner in enumerate(learners):
        axis = axes[row, 0]
        learner_frame = frame[frame["base_learner"] == learner]
        for model in MODEL_ORDER:
            selected = learner_frame[learner_frame["model"] == model].dropna(
                subset=[metric, "cost"]
            )
            if selected.empty:
                continue
            grouped = selected.groupby("cost", sort=True)[metric].mean()
            dataset_count = int(selected["dataset"].nunique())
            axis.plot(
                grouped.index,
                grouped.values,
                marker="o",
                markersize=3.5,
                linewidth=1.6,
                linestyle="--" if model in NON_ABSTAINING_BASELINES else "-",
                color=COLORS[model],
                label=(
                    f"{model} complete ref. (n={dataset_count})"
                    if model in NON_ABSTAINING_BASELINES
                    else f"{model} (n={dataset_count})"
                ),
            )
        axis.set_xlabel("Rejection cost")
        _style_axis(axis, str(learner), metric.replace("_", " ").title())
        axis.legend(fontsize=7, ncol=2, frameon=False)
    figure.suptitle(f"v6.3 — Rejection-cost {metric.replace('_', ' ').title()}")
    figure.text(
        0.995,
        0.006,
        "BR/CC are cost-invariant complete-metric references (coverage=1); missing costs are not imputed.",
        ha="right",
        fontsize=7,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.025, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def generate(config_path):
    with Path(config_path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    settings = config["settings"]
    root = WORKSPACE_ROOT / settings["output_dir"]
    scopes = [V63_DIR for V63_DIR in ("logistic", "svm", "mlp", "all") if (root / V63_DIR).exists()]
    for scope in scopes:
        base = root / scope / "hamming_bop"
        tables = base / "tables"
        figures = base / "figures"
        all_scope = scope == "all"
        complete_table = tables / "complete_metric.csv"
        selective_table = tables / "selective_metric.csv"
        if not complete_table.exists() or not selective_table.exists():
            continue
        for metric, filename in COMPLETE_FIGURES.items():
            _plot_bars(
                complete_table,
                metric,
                figures / filename,
                all_scope=all_scope,
            )
        for metric, filename in SELECTIVE_FIGURES.items():
            _plot_bars(
                selective_table,
                metric,
                figures / filename,
                report_cost=float(settings["report_cost"]),
                all_scope=all_scope,
            )
        for metric, filename in REJECTION_FIGURES.items():
            _plot_rejection_curve(
                selective_table,
                metric,
                figures / filename,
                all_scope=all_scope,
            )
        print(f"Generated figures: {figures}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=WORKSPACE_ROOT / "configs" / "v6_3_run.json",
    )
    args = parser.parse_args()
    generate(args.config)


if __name__ == "__main__":
    main()
