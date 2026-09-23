"""Generate the requested four-model comparison figures for every result root.

The script discovers ``results*`` directories next to the repository root,
excluding the legacy ``results`` and ``results_pa`` directories.  For each
result root it selects the most complete compatible schema-v3 run and writes::

    user_figs/
        logistic/
        svm/
        mlp/
        all_figs/

Each base-learner directory contains full-prediction comparisons, selective
comparisons at the report cost, and rejection-cost curves.  ``all_figs``
contains the corresponding three-panel figures for Logistic, SVM, and MLP.

Older schema-v3 exports do not contain ``Selective Instance-F1`` and do not
retain the predictions needed to reconstruct it.  Those missing values are
shown explicitly as N/A; they are never replaced by the semantically different
``Optimistic Instance-F1`` metric.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_LEARNERS = ("Logistic", "SVM", "MLP")
MODEL_FAMILIES = ("BR", "CC", "MLC_PA", "GSI_MLC_PA")
SELECTIVE_FAMILIES = ("MLC_PA", "GSI_MLC_PA")
MODEL_LABELS = {
    "BR": "BR",
    "CC": "CC",
    "MLC_PA": "MLC-PA",
    "GSI_MLC_PA": "GSI-MLC-PA",
}
BASE_LABELS = {
    "Logistic": "Logistic",
    "SVM": "SVM",
    "MLP": "MLP",
}
MODEL_COLORS = {
    "BR": "#64748B",
    "CC": "#2563EB",
    "MLC_PA": "#D97706",
    "GSI_MLC_PA": "#4D7C0F",
}
MODEL_HATCHES = {
    "BR": "",
    "CC": "//",
    "MLC_PA": "xx",
    "GSI_MLC_PA": "..",
}
MODEL_MARKERS = {
    "BR": "o",
    "CC": "s",
    "MLC_PA": "^",
    "GSI_MLC_PA": "D",
}
MODEL_LINESTYLES = {
    "BR": "--",
    "CC": ":",
    "MLC_PA": "-.",
    "GSI_MLC_PA": "-",
}


@dataclass(frozen=True)
class MetricSpec:
    column: str
    title: str
    slug: str
    higher_is_better: bool
    baseline_column: str | None = None


FULL_METRICS = (
    MetricSpec("Hamming Accuracy", "Hamming Accuracy", "hamming_accuracy", True),
    MetricSpec("Subset Accuracy", "Subset Accuracy", "subset_accuracy", True),
    MetricSpec("Macro-F1", "Macro-F1", "macro_f1", True),
    MetricSpec("Micro-F1", "Micro-F1", "micro_f1", True),
    MetricSpec("Instance-F1", "Example-F1 (Instance-F1)", "instance_f1", True),
)

SELECTIVE_METRICS = (
    MetricSpec(
        "Generalized Loss",
        "Generalized Loss",
        "generalized_loss",
        False,
        baseline_column="Hamming Loss",
    ),
    MetricSpec(
        "Selective Macro-F1",
        "Selective Macro-F1",
        "selective_macro_f1",
        True,
        baseline_column="Macro-F1",
    ),
    MetricSpec(
        "Selective Micro-F1",
        "Selective Micro-F1",
        "selective_micro_f1",
        True,
        baseline_column="Micro-F1",
    ),
    MetricSpec(
        "Selective Instance-F1",
        "Selective Instance-F1",
        "selective_instance_f1",
        True,
        baseline_column="Instance-F1",
    ),
    MetricSpec(
        "Selective Hamming Accuracy",
        "Selective Hamming Accuracy",
        "selective_hamming_accuracy",
        True,
        baseline_column="Hamming Accuracy",
    ),
)

REQUIRED_COMPLETE_COLUMNS = {
    "Dataset",
    "Model",
    "Fold",
    *(spec.column for spec in FULL_METRICS),
    "Hamming Loss",
}
REQUIRED_SELECTIVE_COLUMNS = {
    "Dataset",
    "Model",
    "Cost",
    "Fold",
    "Generalized Loss",
    "Selective Macro-F1",
    "Selective Micro-F1",
    "Selective Hamming Accuracy",
}


@dataclass
class RunData:
    result_root: Path
    run_dir: Path
    complete: pd.DataFrame
    selective: pd.DataFrame

    @property
    def run_hash(self) -> str:
        return self.run_dir.name

    @property
    def datasets(self) -> list[str]:
        return sorted(self.complete["Dataset"].dropna().astype(str).unique())

    @property
    def costs(self) -> list[float]:
        return sorted(float(value) for value in self.selective["Cost"].dropna().unique())

    @property
    def fold_count(self) -> int:
        return int(self.complete["Fold"].nunique())


def configure_style() -> None:
    """Apply a consistent publication-oriented style."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.titlesize": 13,
            "axes.edgecolor": "#334155",
            "axes.linewidth": 0.8,
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
            "grid.color": "#CBD5E1",
            "grid.linestyle": "--",
            "grid.alpha": 0.55,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
        }
    )


def _model_id(family: str, base_learner: str) -> str:
    return f"{family}_{base_learner}"


def _required_model_ids(base_learners: Sequence[str] = BASE_LEARNERS) -> set[str]:
    return {
        _model_id(family, base)
        for base in base_learners
        for family in MODEL_FAMILIES
    }


def _required_selective_model_ids(
    base_learners: Sequence[str] = BASE_LEARNERS,
) -> set[str]:
    return {
        _model_id(family, base)
        for base in base_learners
        for family in SELECTIVE_FAMILIES
    }


def _generated_timestamp(results_json: Path) -> float:
    """Read a small timestamp signal without parsing a potentially huge JSON."""
    try:
        with results_json.open("r", encoding="utf-8") as handle:
            prefix = handle.read(4096)
        match = re.search(r'"Generated At":"([^"]+)"', prefix)
        if match:
            return pd.Timestamp(match.group(1)).timestamp()
    except (OSError, ValueError):
        pass
    try:
        return results_json.stat().st_mtime
    except OSError:
        return 0.0


def _candidate_score(run_dir: Path) -> tuple[int, int, int, int, float]:
    """Rank runs by compatibility, dataset/fold coverage, rows, then recency."""
    complete_path = run_dir / "complete_metrics.csv"
    selective_path = run_dir / "selective_metrics.csv"
    try:
        complete = pd.read_csv(complete_path)
        selective = pd.read_csv(selective_path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return (-1, -1, -1, -1, -1.0)

    if not REQUIRED_COMPLETE_COLUMNS.issubset(complete.columns):
        return (-1, -1, -1, -1, -1.0)
    if not REQUIRED_SELECTIVE_COLUMNS.issubset(selective.columns):
        return (-1, -1, -1, -1, -1.0)

    complete_models = set(complete["Model"].dropna().astype(str))
    selective_models = set(selective["Model"].dropna().astype(str))
    compatible = int(
        _required_model_ids().issubset(complete_models)
        and _required_selective_model_ids().issubset(selective_models)
    )
    dataset_count = int(complete["Dataset"].nunique())
    fold_count = int(complete["Fold"].nunique())
    row_count = int(len(complete) + len(selective))
    recency = _generated_timestamp(run_dir / "results_v3.json")
    return compatible, dataset_count, fold_count, row_count, recency


def select_run(result_root: Path) -> RunData:
    """Select one coherent run instead of mixing hashes/configurations."""
    tables_dir = result_root / "tables"
    candidates = [
        path
        for path in tables_dir.iterdir()
        if path.is_dir()
        and (path / "complete_metrics.csv").is_file()
        and (path / "selective_metrics.csv").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No compatible table run found under {tables_dir}")

    selected = max(candidates, key=_candidate_score)
    score = _candidate_score(selected)
    if score[0] != 1:
        raise ValueError(
            f"No run under {tables_dir} contains all four model families for "
            "Logistic, SVM, and MLP."
        )

    complete = pd.read_csv(selected / "complete_metrics.csv")
    selective = pd.read_csv(selected / "selective_metrics.csv")
    for frame in (complete, selective):
        frame["Dataset"] = frame["Dataset"].astype(str)
        frame["Model"] = frame["Model"].astype(str)
    selective["Cost"] = pd.to_numeric(selective["Cost"], errors="coerce")
    return RunData(result_root, selected, complete, selective)


def discover_result_roots(repo_root: Path) -> list[Path]:
    """Return result directories requested by the user."""
    excluded = {"results", "results_pa"}
    roots = [
        path
        for path in repo_root.iterdir()
        if path.is_dir()
        and path.name.startswith("results")
        and path.name not in excluded
        and (path / "tables").is_dir()
    ]
    return sorted(roots, key=lambda path: path.name.lower())


def choose_report_cost(costs: Sequence[float], requested: float) -> float:
    if not costs:
        raise ValueError("Selective metrics contain no rejection costs.")
    return min(costs, key=lambda value: abs(value - requested))


def _summary(
    frame: pd.DataFrame,
    *,
    model: str,
    metric: str,
    datasets: Sequence[str],
    cost: float | None = None,
) -> pd.DataFrame:
    """Return per-dataset fold mean and sample SD for one model/metric."""
    if metric not in frame.columns:
        return pd.DataFrame(
            {"Dataset": list(datasets), "mean": np.nan, "std": np.nan}
        )
    selected = frame[frame["Model"] == model]
    if cost is not None:
        selected = selected[np.isclose(selected["Cost"], cost)]
    selected = selected[["Dataset", metric]].copy()
    selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
    grouped = (
        selected.groupby("Dataset", sort=False)[metric]
        .agg(mean="mean", std="std")
        .reindex(datasets)
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped


def _comparison_values(
    run: RunData,
    *,
    base_learner: str,
    metric: MetricSpec,
    selective: bool,
    report_cost: float,
) -> dict[str, pd.DataFrame]:
    values: dict[str, pd.DataFrame] = {}
    for family in MODEL_FAMILIES:
        model = _model_id(family, base_learner)
        if selective and family in SELECTIVE_FAMILIES:
            values[family] = _summary(
                run.selective,
                model=model,
                metric=metric.column,
                datasets=run.datasets,
                cost=report_cost,
            )
        else:
            source_metric = metric.baseline_column if selective else metric.column
            if source_metric is None:
                raise ValueError(f"No baseline mapping configured for {metric.title}")
            values[family] = _summary(
                run.complete,
                model=model,
                metric=source_metric,
                datasets=run.datasets,
            )
    return values


def _finite_upper(comparison_sets: Iterable[dict[str, pd.DataFrame]]) -> float:
    candidates: list[float] = []
    for values in comparison_sets:
        for frame in values.values():
            tops = frame["mean"].to_numpy(dtype=float) + frame["std"].to_numpy(
                dtype=float
            )
            candidates.extend(tops[np.isfinite(tops)].tolist())
    if not candidates:
        return 1.0
    return max(candidates)


def _axis_upper(metric: MetricSpec, observed_upper: float) -> float:
    # Leave enough headroom for the vertical three-decimal value labels drawn
    # above the error-bar caps.  Scores/losses remain on their native [0, 1]
    # scale even though the display domain extends slightly past 1.0.
    if metric.column == "Generalized Loss":
        return min(1.16, max(0.1, observed_upper * 1.30 + 0.025))
    return min(1.16, max(0.1, observed_upper * 1.22 + 0.025))


def _annotate_missing(ax: plt.Axes, missing: Sequence[str]) -> None:
    if not missing:
        return
    labels = ", ".join(MODEL_LABELS[family] for family in missing)
    ax.text(
        0.995,
        0.02,
        f"N/A in export: {labels}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#9A3412",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "#FFF7ED",
            "edgecolor": "#FDBA74",
            "linewidth": 0.7,
        },
    )


def _draw_comparison_axis(
    ax: plt.Axes,
    *,
    run: RunData,
    base_learner: str,
    metric: MetricSpec,
    selective: bool,
    report_cost: float,
    ylim_upper: float,
    show_legend: bool,
) -> None:
    values = _comparison_values(
        run,
        base_learner=base_learner,
        metric=metric,
        selective=selective,
        report_cost=report_cost,
    )
    x = np.arange(len(run.datasets), dtype=float)
    width = 0.195
    missing: list[str] = []

    for index, family in enumerate(MODEL_FAMILIES):
        frame = values[family]
        means = frame["mean"].to_numpy(dtype=float)
        stds = frame["std"].to_numpy(dtype=float)
        positions = x + (index - 1.5) * width
        finite = np.isfinite(means)
        if not np.any(finite):
            missing.append(family)
            continue
        bars = ax.bar(
            positions[finite],
            means[finite],
            width=width * 0.92,
            yerr=stds[finite],
            capsize=2.0,
            color=MODEL_COLORS[family],
            edgecolor="#1F2937",
            linewidth=0.55,
            hatch=MODEL_HATCHES[family],
            alpha=0.92,
            label=MODEL_LABELS[family],
            error_kw={"elinewidth": 0.75, "capthick": 0.75},
            zorder=3,
        )
        label_fontsize = 6.4 if len(run.datasets) >= 8 else 7.6
        for bar, mean, std in zip(bars, means[finite], stds[finite]):
            ax.annotate(
                f"{mean:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2.0, mean + max(std, 0.0)),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                rotation=90,
                rotation_mode="anchor",
                fontsize=label_fontsize,
                color="#111827",
                clip_on=False,
                zorder=6,
            )

    ax.set_ylim(0, ylim_upper)
    ax.set_xlim(-0.65, len(run.datasets) - 0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [dataset.upper() for dataset in run.datasets],
        rotation=38,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel(metric.title)
    ax.set_title(f"Base learner: {BASE_LABELS[base_learner]}", fontweight="bold")
    ax.grid(axis="y", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    if show_legend:
        ax.legend(
            ncol=4,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.12),
            frameon=False,
        )
    _annotate_missing(ax, missing)


def _comparison_title(metric: MetricSpec, selective: bool, report_cost: float) -> str:
    direction = "lower is better" if not metric.higher_is_better else "higher is better"
    if selective:
        return (
            f"{metric.title} by dataset at c = {report_cost:.2f}\n"
            f"PA models use selective predictions; BR/CC are full-prediction references; {direction}"
        )
    return f"{metric.title} by dataset (full prediction; {direction})"


def _figure_note(run: RunData, *, selective: bool, report_cost: float) -> str:
    scope = (
        f"PA operating point c={report_cost:.2f}; BR/CC full references"
        if selective
        else "full prediction"
    )
    return (
        f"Source run {run.run_hash} | {len(run.datasets)} dataset(s) | "
        f"{run.fold_count} fold(s) | bars show fold mean +/- sample SD | {scope}"
    )


def plot_individual_comparison(
    run: RunData,
    *,
    base_learner: str,
    metric: MetricSpec,
    selective: bool,
    report_cost: float,
    output_path: Path,
) -> None:
    values = _comparison_values(
        run,
        base_learner=base_learner,
        metric=metric,
        selective=selective,
        report_cost=report_cost,
    )
    ylim_upper = _axis_upper(metric, _finite_upper([values]))
    width = max(10.5, 1.05 * len(run.datasets))
    fig, ax = plt.subplots(figsize=(width, 5.8))
    _draw_comparison_axis(
        ax,
        run=run,
        base_learner=base_learner,
        metric=metric,
        selective=selective,
        report_cost=report_cost,
        ylim_upper=ylim_upper,
        show_legend=True,
    )
    fig.suptitle(_comparison_title(metric, selective, report_cost), y=0.995)
    fig.text(
        0.5,
        0.006,
        _figure_note(run, selective=selective, report_cost=report_cost),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)


def plot_combined_comparison(
    run: RunData,
    *,
    metric: MetricSpec,
    selective: bool,
    report_cost: float,
    output_path: Path,
) -> None:
    all_values = [
        _comparison_values(
            run,
            base_learner=base,
            metric=metric,
            selective=selective,
            report_cost=report_cost,
        )
        for base in BASE_LEARNERS
    ]
    ylim_upper = _axis_upper(metric, _finite_upper(all_values))
    figure_width = max(11.5, 1.08 * len(run.datasets))
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(figure_width, 15.0),
        sharex=False,
        sharey=True,
    )
    for index, (ax, base) in enumerate(zip(axes, BASE_LEARNERS)):
        _draw_comparison_axis(
            ax,
            run=run,
            base_learner=base,
            metric=metric,
            selective=selective,
            report_cost=report_cost,
            ylim_upper=ylim_upper,
            show_legend=index == 0,
        )
    fig.suptitle(
        _comparison_title(metric, selective, report_cost)
        + "\nCombined view: Logistic / SVM / MLP",
        y=0.998,
    )
    fig.text(
        0.5,
        0.004,
        _figure_note(run, selective=selective, report_cost=report_cost),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.955), h_pad=1.2)
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)


def _cost_series(
    run: RunData,
    *,
    family: str,
    base_learner: str,
    metric: MetricSpec,
) -> pd.DataFrame:
    model = _model_id(family, base_learner)
    if family in SELECTIVE_FAMILIES:
        if metric.column not in run.selective.columns:
            return pd.DataFrame(
                {"Cost": run.costs, "mean": np.nan, "std": np.nan}
            )
        subset = run.selective[run.selective["Model"] == model][
            ["Cost", metric.column]
        ].copy()
        subset[metric.column] = pd.to_numeric(subset[metric.column], errors="coerce")
        return (
            subset.groupby("Cost", sort=True)[metric.column]
            .agg(mean="mean", std="std")
            .reindex(run.costs)
            .reset_index()
        )

    if metric.baseline_column is None:
        raise ValueError(f"No baseline mapping configured for {metric.title}")
    subset = run.complete[run.complete["Model"] == model][metric.baseline_column]
    numeric = pd.to_numeric(subset, errors="coerce")
    mean = float(numeric.mean())
    std = float(numeric.std()) if len(numeric.dropna()) > 1 else 0.0
    return pd.DataFrame(
        {"Cost": run.costs, "mean": [mean] * len(run.costs), "std": [std] * len(run.costs)}
    )


def _cost_axis_upper(
    run: RunData,
    metric: MetricSpec,
    base_learners: Sequence[str],
) -> float:
    candidates: list[float] = []
    for base in base_learners:
        for family in MODEL_FAMILIES:
            series = _cost_series(
                run,
                family=family,
                base_learner=base,
                metric=metric,
            )
            tops = series["mean"].to_numpy(dtype=float) + series["std"].fillna(0).to_numpy(
                dtype=float
            )
            candidates.extend(tops[np.isfinite(tops)].tolist())
    observed = max(candidates) if candidates else 1.0
    return _axis_upper(metric, observed)


def _draw_cost_axis(
    ax: plt.Axes,
    *,
    run: RunData,
    base_learner: str,
    metric: MetricSpec,
    ylim_upper: float,
    show_legend: bool,
) -> None:
    missing: list[str] = []
    for family in MODEL_FAMILIES:
        series = _cost_series(
            run,
            family=family,
            base_learner=base_learner,
            metric=metric,
        )
        x = series["Cost"].to_numpy(dtype=float)
        mean = series["mean"].to_numpy(dtype=float)
        std = series["std"].fillna(0.0).to_numpy(dtype=float)
        finite = np.isfinite(mean)
        if not np.any(finite):
            missing.append(family)
            continue
        ax.plot(
            x[finite],
            mean[finite],
            color=MODEL_COLORS[family],
            marker=MODEL_MARKERS[family],
            linestyle=MODEL_LINESTYLES[family],
            linewidth=1.9,
            markersize=5.3,
            label=MODEL_LABELS[family],
            zorder=4,
        )
        lower = np.maximum(0.0, mean[finite] - std[finite])
        upper = np.minimum(1.0, mean[finite] + std[finite])
        ax.fill_between(
            x[finite],
            lower,
            upper,
            color=MODEL_COLORS[family],
            alpha=0.10,
            linewidth=0,
            zorder=2,
        )

    ax.set_ylim(0, ylim_upper)
    if run.costs:
        left = min(run.costs)
        right = max(run.costs)
        padding = max((right - left) * 0.04, 0.008)
        ax.set_xlim(left - padding, right + padding)
    ax.set_xticks(run.costs)
    ax.set_xticklabels([f"{cost:.2f}" for cost in run.costs])
    ax.set_xlabel("Rejection cost (c)")
    ax.set_ylabel(metric.title)
    ax.set_title(f"Base learner: {BASE_LABELS[base_learner]}", fontweight="bold")
    ax.grid(True, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    if show_legend:
        ax.legend(ncol=2, loc="best", frameon=False)
    _annotate_missing(ax, missing)


def _cost_title(metric: MetricSpec) -> str:
    direction = "lower is better" if not metric.higher_is_better else "higher is better"
    return (
        f"{metric.title} across rejection costs\n"
        f"Mean across all datasets and folds; BR/CC are full-prediction references; {direction}"
    )


def _cost_note(run: RunData) -> str:
    return (
        f"Source run {run.run_hash} | {len(run.datasets)} dataset(s) | "
        f"{run.fold_count} fold(s) | line = mean, band = +/- sample SD"
    )


def plot_individual_cost(
    run: RunData,
    *,
    base_learner: str,
    metric: MetricSpec,
    output_path: Path,
) -> None:
    ylim_upper = _cost_axis_upper(run, metric, [base_learner])
    fig, ax = plt.subplots(figsize=(8.6, 5.7))
    _draw_cost_axis(
        ax,
        run=run,
        base_learner=base_learner,
        metric=metric,
        ylim_upper=ylim_upper,
        show_legend=True,
    )
    fig.suptitle(_cost_title(metric), y=0.995)
    fig.text(
        0.5,
        0.008,
        _cost_note(run),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.91))
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)


def plot_combined_cost(
    run: RunData,
    *,
    metric: MetricSpec,
    output_path: Path,
) -> None:
    ylim_upper = _cost_axis_upper(run, metric, BASE_LEARNERS)
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.8), sharex=True, sharey=True)
    for index, (ax, base) in enumerate(zip(axes, BASE_LEARNERS)):
        _draw_cost_axis(
            ax,
            run=run,
            base_learner=base,
            metric=metric,
            ylim_upper=ylim_upper,
            show_legend=index == 0,
        )
    fig.suptitle(_cost_title(metric) + "\nCombined view: Logistic / SVM / MLP", y=0.995)
    fig.text(
        0.5,
        0.008,
        _cost_note(run),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.88), w_pad=1.1)
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)


def _ensure_output_dirs(result_root: Path) -> dict[str, Path]:
    user_figs = result_root / "user_figs"
    outputs = {
        base: user_figs / base.lower()
        for base in BASE_LEARNERS
    }
    outputs["all"] = user_figs / "all_figs"
    for path in outputs.values():
        path.mkdir(parents=True, exist_ok=True)
    return outputs


def generate_for_result_root(
    result_root: Path,
    *,
    requested_report_cost: float,
) -> tuple[RunData, int]:
    run = select_run(result_root)
    report_cost = choose_report_cost(run.costs, requested_report_cost)
    outputs = _ensure_output_dirs(result_root)
    generated = 0

    for metric in FULL_METRICS:
        filename = f"full_{metric.slug}_comparison.png"
        for base in BASE_LEARNERS:
            plot_individual_comparison(
                run,
                base_learner=base,
                metric=metric,
                selective=False,
                report_cost=report_cost,
                output_path=outputs[base] / filename,
            )
            generated += 1
        plot_combined_comparison(
            run,
            metric=metric,
            selective=False,
            report_cost=report_cost,
            output_path=outputs["all"] / filename,
        )
        generated += 1

    for metric in SELECTIVE_METRICS:
        filename = f"{metric.slug}_comparison.png"
        for base in BASE_LEARNERS:
            plot_individual_comparison(
                run,
                base_learner=base,
                metric=metric,
                selective=True,
                report_cost=report_cost,
                output_path=outputs[base] / filename,
            )
            generated += 1
        plot_combined_comparison(
            run,
            metric=metric,
            selective=True,
            report_cost=report_cost,
            output_path=outputs["all"] / filename,
        )
        generated += 1

    for metric in SELECTIVE_METRICS:
        filename = f"rejection_cost_{metric.slug}.png"
        for base in BASE_LEARNERS:
            plot_individual_cost(
                run,
                base_learner=base,
                metric=metric,
                output_path=outputs[base] / filename,
            )
            generated += 1
        plot_combined_cost(
            run,
            metric=metric,
            output_path=outputs["all"] / filename,
        )
        generated += 1

    print(
        f"{result_root.name}: run={run.run_hash}, datasets={len(run.datasets)}, "
        f"folds={run.fold_count}, report_cost={report_cost:.2f}, figures={generated}"
    )
    if "Selective Instance-F1" not in run.selective.columns:
        print(
            "  warning: Selective Instance-F1 is absent from this export; "
            "MLC-PA/GSI-MLC-PA are marked N/A in the corresponding figures."
        )
    return run, generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing the results* directories.",
    )
    parser.add_argument(
        "--result-root",
        action="append",
        type=Path,
        default=None,
        help="Generate only this result root (repeatable).",
    )
    parser.add_argument(
        "--report-cost",
        type=float,
        default=0.30,
        help="Cost used for selective comparison bars (default: 0.30).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_style()
    repo_root = args.repo_root.resolve()
    roots = (
        [path.resolve() for path in args.result_root]
        if args.result_root
        else discover_result_roots(repo_root)
    )
    if not roots:
        raise FileNotFoundError("No eligible results directories were discovered.")

    total = 0
    for result_root in roots:
        _, generated = generate_for_result_root(
            result_root,
            requested_report_cost=args.report_cost,
        )
        total += generated
    print(f"Generated {total} figures across {len(roots)} result folder(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
