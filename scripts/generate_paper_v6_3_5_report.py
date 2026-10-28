"""Generate the JAIR-style v6.3.5 comparative LaTeX report and figures."""

import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_v6_3" / "paper_v6_3_5"
FIGURES = OUT / "figures"
DETAIL = OUT / "unified_detailed_folds.csv"
DATASETS_CSV = OUT / "dataset_characteristics.csv"
TEX = OUT / "Bao_Cao_Khoa_Hoc_GSI_MLC_PA_v6_3_5.tex"
PDF = OUT / "Bao_Cao_Khoa_Hoc_GSI_MLC_PA_v6_3_5.pdf"

MODELS = ["BR", "CC", "MLC_PA", "GSI_v6_3_3", "GSI_v6_3_5"]
DISPLAY = {
    "BR": "BR",
    "CC": "CC",
    "MLC_PA": "MLC-PA",
    "GSI_v6_3_3": "GSI v6.3.3",
    "GSI_v6_3_5": "GSI v6.3.5",
}
DATASET_ORDER = [
    "emotions", "scene", "yeast", "plantpseaac", "humanpseaac",
    "chd49", "music", "gpositivepseaac", "genbase", "viruspseaac",
]
SHORT_DATASET = {
    "emotions": "emotions", "scene": "scene", "yeast": "yeast",
    "plantpseaac": "plant", "humanpseaac": "human", "chd49": "chd49",
    "music": "music", "gpositivepseaac": "gpositive",
    "genbase": "genbase", "viruspseaac": "virus",
}
COLORS = {
    "BR": "#4c78a8", "CC": "#f58518", "MLC_PA": "#54a24b",
    "GSI_v6_3_3": "#b279a2", "GSI_v6_3_5": "#e45756",
}
HATCHES = {
    "BR": "", "CC": "//", "MLC_PA": "xx",
    "GSI_v6_3_3": "..", "GSI_v6_3_5": "\\\\",
}


def latex_escape(value):
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
        "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
        "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fmt(value, digits=4):
    return "--" if pd.isna(value) else f"{float(value):.{digits}f}"


def bold_best(values, higher=True, digits=4):
    finite = [float(value) for value in values if not pd.isna(value)]
    target = (max(finite) if higher else min(finite)) if finite else np.nan
    cells = []
    for value in values:
        cell = fmt(value, digits)
        if not pd.isna(value) and abs(float(value) - target) <= 5e-7:
            cell = rf"\textbf{{{cell}}}"
        cells.append(cell)
    return cells


def holm_adjust(p_values):
    items = sorted(p_values.items(), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    total = len(items)
    for index, (name, p_value) in enumerate(items):
        candidate = min(1.0, (total - index) * p_value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def load_data():
    detailed = pd.read_csv(DETAIL)
    datasets = pd.read_csv(DATASETS_CSV)
    dataset_means = (
        detailed.groupby(["Dataset", "Model"], sort=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    overall = dataset_means.groupby("Model").mean(numeric_only=True).reindex(MODELS)
    overall_std = dataset_means.groupby("Model").std(numeric_only=True).reindex(MODELS)
    return detailed, datasets, dataset_means, overall, overall_std


def statistical_tests(dataset_means):
    pivot = dataset_means.pivot(
        index="Dataset", columns="Model", values="Selective_Macro_F1"
    ).reindex(DATASET_ORDER)
    friedman = friedmanchisquare(*[pivot[model] for model in MODELS])
    raw = {}
    rows = []
    proposed = pivot["GSI_v6_3_5"]
    for comparator in ["BR", "CC", "MLC_PA", "GSI_v6_3_3"]:
        comparison = pivot[comparator]
        test = wilcoxon(
            proposed, comparison, zero_method="wilcox", alternative="two-sided"
        )
        p_value = float(test.pvalue)
        raw[comparator] = p_value
        delta = proposed - comparison
        rows.append({
            "Comparator": comparator,
            "Mean_Delta": float(delta.mean()),
            "Wins": int(np.sum(delta > 1e-12)),
            "Ties": int(np.sum(np.abs(delta) <= 1e-12)),
            "Losses": int(np.sum(delta < -1e-12)),
            "Wilcoxon_W": float(test.statistic),
            "p_raw": p_value,
        })
    adjusted = holm_adjust(raw)
    for row in rows:
        row["p_Holm"] = adjusted[row["Comparator"]]
    stats = pd.DataFrame(rows)

    ranks = []
    for _, row in pivot.iterrows():
        ranks.append(
            rankdata(-row[MODELS].to_numpy(dtype=float), method="average")
        )
    average_ranks = pd.Series(np.mean(ranks, axis=0), index=MODELS)
    return friedman, stats, average_ranks


def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def grouped_dataset_plot(dataset_means, metric, ylabel, name, ylim=None):
    pivot = dataset_means.pivot(
        index="Dataset", columns="Model", values=metric
    ).reindex(DATASET_ORDER)
    x_pos = np.arange(len(DATASET_ORDER))
    width = 0.16
    fig, axis = plt.subplots(figsize=(11.8, 4.6))
    for index, model in enumerate(MODELS):
        axis.bar(
            x_pos + (index - 2) * width,
            pivot[model],
            width,
            label=DISPLAY[model],
            color=COLORS[model],
            hatch=HATCHES[model],
            edgecolor="black",
            linewidth=0.45,
        )
    axis.set_xticks(
        x_pos,
        [SHORT_DATASET[item] for item in DATASET_ORDER],
        rotation=30,
        ha="right",
    )
    axis.set_ylabel(ylabel)
    if ylim is not None:
        axis.set_ylim(*ylim)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(
        ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.17)
    )
    save_figure(fig, name)


def create_figures(dataset_means, overall):
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    grouped_dataset_plot(
        dataset_means,
        "Selective_Macro_F1",
        "Selective Macro-F1",
        "fig1_selective_macro_f1",
        (0.0, 0.9),
    )
    grouped_dataset_plot(
        dataset_means, "Coverage", "Coverage", "fig2_coverage", (0.0, 1.05)
    )

    panels = [
        ("Full_Macro_F1", "Full Macro-F1"),
        ("Selective_Macro_F1", "Selective Macro-F1"),
        ("Coverage", "Coverage"),
        ("Selective_Hamming_Loss", "Selective Hamming Loss"),
        ("Generalized_Loss", "Generalized Loss"),
        ("Subset_Accuracy", "Subset Accuracy"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2))
    for axis, (metric, title) in zip(axes.ravel(), panels):
        values = overall.loc[MODELS, metric]
        axis.bar(
            np.arange(len(MODELS)),
            values,
            color=[COLORS[model] for model in MODELS],
            edgecolor="black",
            linewidth=0.45,
        )
        axis.set_xticks(
            np.arange(len(MODELS)),
            [DISPLAY[model] for model in MODELS],
            rotation=32,
            ha="right",
        )
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    save_figure(fig, "fig3_overall_metrics")

    fig, axis = plt.subplots(figsize=(7.0, 5.0))
    for model in MODELS:
        axis.scatter(
            overall.loc[model, "Coverage"],
            overall.loc[model, "Selective_Macro_F1"],
            s=95,
            color=COLORS[model],
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        axis.annotate(
            DISPLAY[model],
            (
                overall.loc[model, "Coverage"],
                overall.loc[model, "Selective_Macro_F1"],
            ),
            xytext=(6, 5),
            textcoords="offset points",
        )
    axis.set_xlabel("Coverage")
    axis.set_ylabel("Selective Macro-F1")
    axis.set_xlim(0.68, 1.025)
    axis.set_ylim(0.43, 0.60)
    axis.grid(alpha=0.25)
    save_figure(fig, "fig4_f1_coverage_pareto")

    pivot_f1 = dataset_means.pivot(
        index="Dataset", columns="Model", values="Selective_Macro_F1"
    ).reindex(DATASET_ORDER)
    pivot_cov = dataset_means.pivot(
        index="Dataset", columns="Model", values="Coverage"
    ).reindex(DATASET_ORDER)
    delta_f1 = pivot_f1["GSI_v6_3_5"] - pivot_f1["GSI_v6_3_3"]
    delta_cov = pivot_cov["GSI_v6_3_5"] - pivot_cov["GSI_v6_3_3"]
    x_pos = np.arange(len(DATASET_ORDER))
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.8), sharex=True)
    axes[0].bar(
        x_pos,
        delta_f1,
        color=np.where(delta_f1 >= 0, "#54a24b", "#e45756"),
        edgecolor="black",
        linewidth=0.4,
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel(r"$\Delta$ Selective Macro-F1")
    axes[1].bar(
        x_pos,
        delta_cov,
        color="#4c78a8",
        edgecolor="black",
        linewidth=0.4,
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel(r"$\Delta$ Coverage")
    axes[1].set_xticks(
        x_pos,
        [SHORT_DATASET[item] for item in DATASET_ORDER],
        rotation=30,
        ha="right",
    )
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    save_figure(fig, "fig5_v635_delta")

    fig, axis = plt.subplots(figsize=(8.0, 4.5))
    x_pos = np.arange(len(MODELS))
    width = 0.34
    axis.bar(
        x_pos - width / 2,
        overall.loc[MODELS, "Selective_Positive_Macro_F1"],
        width,
        label="F1 lớp dương",
        color="#e45756",
        edgecolor="black",
    )
    axis.bar(
        x_pos + width / 2,
        overall.loc[MODELS, "Selective_Negative_Macro_F1"],
        width,
        label="F1 lớp âm",
        color="#4c78a8",
        edgecolor="black",
    )
    axis.set_xticks(
        x_pos, [DISPLAY[model] for model in MODELS], rotation=22, ha="right"
    )
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("F1 trung bình")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    save_figure(fig, "fig6_positive_negative_f1")

    selective_models = ["MLC_PA", "GSI_v6_3_3", "GSI_v6_3_5"]
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    x_pos = np.arange(len(selective_models))
    width = 0.34
    axis.bar(
        x_pos - width / 2,
        overall.loc[selective_models, "Error_Capture_Rate"],
        width,
        label="Error capture",
        color="#54a24b",
        edgecolor="black",
    )
    axis.bar(
        x_pos + width / 2,
        overall.loc[selective_models, "Rejected_Error_Rate"],
        width,
        label="Error rate vùng từ chối",
        color="#f58518",
        edgecolor="black",
    )
    axis.set_xticks(
        x_pos, [DISPLAY[model] for model in selective_models]
    )
    axis.set_ylim(0, 0.52)
    axis.set_ylabel("Tỷ lệ")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    save_figure(fig, "fig7_rejection_quality")


def dataset_table(datasets):
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Đặc trưng của 10 tập dữ liệu được đọc trực tiếp bởi chương trình.}",
        r"\label{tab:datasets}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Tập dữ liệu & $N$ & $d$ & $K$ & Card. & Density & $\pi$ range \\",
        r"\midrule",
    ]
    ordered = datasets.set_index("Dataset").loc[DATASET_ORDER].reset_index()
    for _, row in ordered.iterrows():
        lines.append(
            f"{latex_escape(row['Dataset'])} & {int(row['Samples'])} & "
            f"{int(row['Features'])} & {int(row['Labels'])} & "
            f"{row['Label_Cardinality']:.3f} & {row['Label_Density']:.3f} & "
            f"[{row['Min_Positive_Prior']:.3f}, "
            f"{row['Max_Positive_Prior']:.3f}] \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return lines


def overall_table(overall, overall_std, metrics, caption, label):
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * len(metrics) + "}",
        r"\toprule",
        "Mô hình & " + " & ".join(name for _, name, _ in metrics) + r" \\",
        r"\midrule",
    ]
    best = {}
    for metric, _, higher in metrics:
        values = overall.loc[MODELS, metric]
        best[metric] = values.max() if higher else values.min()
    for model in MODELS:
        cells = []
        for metric, _, _ in metrics:
            value = overall.loc[model, metric]
            std = overall_std.loc[model, metric]
            cell = (
                "--" if pd.isna(value) or pd.isna(std)
                else f"{value:.4f} $\\pm$ {std:.4f}"
            )
            if not pd.isna(value) and abs(value - best[metric]) <= 5e-7:
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(
            latex_escape(DISPLAY[model])
            + " & "
            + " & ".join(cells)
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    return lines


def per_dataset_metric_table(
    dataset_means, metric, caption, label, higher=True
):
    pivot = dataset_means.pivot(
        index="Dataset", columns="Model", values=metric
    ).reindex(DATASET_ORDER)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        "Tập dữ liệu & "
        + " & ".join(latex_escape(DISPLAY[model]) for model in MODELS)
        + r" \\",
        r"\midrule",
    ]
    for dataset in DATASET_ORDER:
        cells = bold_best(
            [pivot.loc[dataset, model] for model in MODELS], higher=higher
        )
        lines.append(
            latex_escape(dataset) + " & " + " & ".join(cells) + r" \\"
        )
    means = [pivot[model].mean() for model in MODELS]
    lines += [
        r"\midrule",
        "Trung bình & "
        + " & ".join(bold_best(means, higher=higher))
        + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return lines


def statistical_table(stats, average_ranks):
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{So sánh Wilcoxon ghép cặp ở mức tập dữ liệu: v6.3.5 trừ mô hình đối chứng. Giá trị $p$ được hiệu chỉnh Holm.}",
        r"\label{tab:wilcoxon}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Đối chứng & $\overline{\Delta F_1}$ & Thắng & Hòa & Thua & $W$ & $p_{Holm}$ \\",
        r"\midrule",
    ]
    for _, row in stats.iterrows():
        lines.append(
            f"{latex_escape(DISPLAY[row['Comparator']])} & "
            f"{row['Mean_Delta']:+.4f} & {int(row['Wins'])} & "
            f"{int(row['Ties'])} & {int(row['Losses'])} & "
            f"{row['Wilcoxon_W']:.1f} & {row['p_Holm']:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    lines += [
        r"\noindent Thứ hạng trung bình Selective Macro-F1 "
        r"(nhỏ hơn là tốt hơn): "
        + ", ".join(
            f"{latex_escape(DISPLAY[model])}={average_ranks[model]:.2f}"
            for model in MODELS
        )
        + "."
    ]
    return lines


def full_appendix_tables(dataset_means):
    sections = []
    table_specs = [
        (
            "Các metric dự đoán đầy đủ theo từng tập dữ liệu và mô hình",
            [
                "Full_Macro_F1", "Full_Micro_F1", "Full_Example_F1",
                "Subset_Accuracy", "Hamming_Loss",
            ],
            ["MaF1", "MiF1", "ExF1", "SubAcc", "HL"],
        ),
        (
            "Các metric dự đoán chọn lọc theo từng tập dữ liệu và mô hình",
            [
                "Selective_Macro_F1", "Selective_Micro_F1",
                "Selective_Instance_F1", "Selective_Hamming_Loss",
                "Generalized_Loss", "Coverage",
            ],
            ["SMaF1", "SMiF1", "SInsF1", "SHL", "GL", "Cov"],
        ),
        (
            "Các chỉ số lớp và chất lượng từ chối theo từng tập dữ liệu và mô hình",
            [
                "Selective_Positive_Macro_F1",
                "Selective_Negative_Macro_F1",
                "Positive_Coverage", "Negative_Coverage",
                "Error_Capture_Rate", "Rejected_Error_Rate",
                "Sample_Abstention_Rate_ABS",
            ],
            ["F1+", "F1-", "Cov+", "Cov-", "ECR", "RER", "ABS"],
        ),
    ]
    for table_index, (caption, metrics, names) in enumerate(
        table_specs, start=1
    ):
        sections += [
            r"\begin{landscape}",
            r"\begin{longtable}{ll" + "r" * len(metrics) + "}",
            rf"\caption{{{caption}.}}\label{{tab:appendix-{table_index}}}\\",
            r"\toprule",
            "Dataset & Model & " + " & ".join(names) + r" \\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            "Dataset & Model & " + " & ".join(names) + r" \\",
            r"\midrule",
            r"\endhead",
        ]
        for dataset in DATASET_ORDER:
            subset = dataset_means[
                dataset_means["Dataset"] == dataset
            ].set_index("Model")
            for model_index, model in enumerate(MODELS):
                dataset_cell = latex_escape(dataset) if model_index == 0 else ""
                values = " & ".join(
                    fmt(subset.loc[model, metric]) for metric in metrics
                )
                sections.append(
                    f"{dataset_cell} & {latex_escape(DISPLAY[model])} & "
                    f"{values} \\\\"
                )
            sections.append(r"\addlinespace[2pt]")
        sections += [
            r"\bottomrule",
            r"\end{longtable}",
            r"\end{landscape}",
        ]
    return sections


def build_report(
    detailed,
    datasets,
    dataset_means,
    overall,
    overall_std,
    friedman,
    stats,
    ranks,
):
    v633 = overall.loc["GSI_v6_3_3"]
    v635 = overall.loc["GSI_v6_3_5"]
    baselines = overall.loc[["BR", "CC", "MLC_PA"], "Selective_Macro_F1"]
    best_baseline_f1 = baselines.max()
    best_baseline_name = baselines.idxmax()
    chd = dataset_means[
        dataset_means["Dataset"] == "chd49"
    ].set_index("Model")

    lines = [
        r"\documentclass[11pt,a4paper,twoside]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[vietnamese]{babel}",
        r"\usepackage[T5]{fontenc}",
        r"\usepackage{amsmath,amssymb,amsfonts}",
        r"\usepackage{booktabs,longtable,multirow,tabularx}",
        r"\usepackage{graphicx,float,pdflscape}",
        r"\usepackage[font=small,labelfont=bf]{caption}",
        r"\usepackage{xcolor,hyperref,microtype,fancyhdr}",
        r"\usepackage[a4paper,left=27mm,right=27mm,top=23mm,bottom=25mm,headheight=16pt]{geometry}",
        r"\graphicspath{{figures/}}",
        r"\hypersetup{colorlinks=true,linkcolor=blue!55!black,citecolor=blue!55!black,urlcolor=blue!55!black}",
        r"\setlength{\parindent}{1.25em}",
        r"\setlength{\parskip}{0pt}",
        r"\sloppy",
        r"\pagestyle{fancy}",
        r"\fancyhf{}",
        r"\fancyhead[LE]{\small NHÓM NGHIÊN CỨU GSI-MLC-PA}",
        r"\fancyhead[RO]{\small SO SÁNH v6.3.3 VÀ v6.3.5}",
        r"\fancyfoot[C]{\thepage}",
        r"\renewcommand{\headrulewidth}{0pt}",
        r"\begin{document}",
        r"\thispagestyle{empty}",
        r"\noindent\small Báo cáo nghiên cứu thực nghiệm GSI-MLC-PA (2026) "
        r"\hfill Hoàn thành 10/2026",
        r"\vspace{2.5em}",
        r"\begin{center}",
        r"{\LARGE\bfseries Phân loại đa nhãn với cơ chế từ chối theo cấu trúc nhãn:\\[0.25em] So sánh thực nghiệm BR, CC, MLC-PA, GSI-MLC-PA v6.3.3 và v6.3.5}\\[1.5em]",
        r"{\large\bfseries Nhóm nghiên cứu Machine Learning và Khai phá dữ liệu}\\[0.3em]",
        r"{\itshape Báo cáo nội bộ có khả năng tái lập; trình bày theo bố cục JAIR của Nguyen và Hüllermeier (2021)}\\",
        r"\end{center}",
        r"\vspace{1.2em}",
        r"\begin{center}\bfseries Tóm tắt\end{center}",
        (
            "Báo cáo đánh giá năm phương pháp phân loại đa nhãn trên mười "
            "tập dữ liệu bằng cùng một quy trình kiểm định chéo phân tầng "
            "5-fold, cùng bộ phân loại Logistic Regression và cùng các fold. "
            "Hai mô hình BR và CC luôn dự đoán đầy đủ; MLC-PA sử dụng quyết "
            "định Bayes cho Hamming loss với chi phí từ chối $c=0.30$; "
            "GSI-MLC-PA v6.3.3 áp dụng suy diễn ba chế độ và từ chối trên "
            "mọi nhãn; v6.3.5 chỉ cho phép từ chối trên tập nhãn phụ thuộc "
            "DL, trong khi IL luôn trả quyết định. Kết quả cho thấy v6.3.3 "
            f"đạt Selective Macro-F1 trung bình cao nhất "
            f"({v633['Selective_Macro_F1']:.4f}) ở coverage "
            f"{v633['Coverage']:.4f}. v6.3.5 tăng coverage lên "
            f"{v635['Coverage']:.4f} nhưng Selective Macro-F1 giảm còn "
            f"{v635['Selective_Macro_F1']:.4f}. Dù vậy, v6.3.5 vẫn cao "
            f"hơn đối chứng phi GSI tốt nhất ({DISPLAY[best_baseline_name]}, "
            f"{best_baseline_f1:.4f}) về Selective Macro-F1. Kết quả được "
            "diễn giải như một đánh đổi coverage--chất lượng, không phải sự "
            "thống trị tuyệt đối của một mô hình."
        ),
        r"\vspace{0.5em}",
        r"\noindent\textbf{Từ khóa:} phân loại đa nhãn; từ chối từng phần; "
        r"selective prediction; Binary Relevance; Classifier Chains; "
        r"GSI-MLC-PA.",
        r"\section{Giới thiệu}",
        (
            "Trong phân loại đa nhãn, một mẫu có thể đồng thời gắn với nhiều "
            "nhãn. Khi xác suất dự đoán không đủ tin cậy, partial abstention "
            "cho phép hệ thống chỉ từ chối một số vị trí nhãn thay vì từ chối "
            "toàn bộ mẫu. Khung quyết định này đặc biệt hữu ích khi chất lượng "
            "dự đoán và khối lượng cần chuyển cho con người cùng quan trọng "
            "\\cite{nguyen2021,chow1970}. Tuy nhiên, F1 chọn lọc chỉ được tính "
            "trên các vị trí đã quyết định; vì thế mọi kết luận về F1 phải đi "
            "kèm coverage và chất lượng vùng bị từ chối."
        ),
        (
            "Mục tiêu của báo cáo là kiểm định cơ chế v6.3.5: bỏ từ chối cho "
            "nhãn độc lập IL và chỉ giữ từ chối cho nhãn phụ thuộc DL. Báo "
            "cáo không sử dụng số liệu từ các runner cũ có cấu hình khác. "
            "Toàn bộ bảng và hình được sinh từ tệp "
            "\\texttt{unified\\_detailed\\_folds.csv}, tạo bởi mã nguồn hiện "
            "tại ngày 14/10/2026."
        ),
        r"\section{Các phương pháp so sánh}",
        r"\subsection{BR và CC}",
        (
            "Binary Relevance (BR) học một bộ phân loại nhị phân độc lập cho "
            "mỗi nhãn. Classifier Chains (CC) nối các nhãn theo thứ tự tự "
            "nhiên và dùng các dự đoán trước làm đặc trưng cho nhãn sau "
            "\\cite{read2011}. Cả hai mô hình trong thực nghiệm này luôn xuất "
            "ma trận nhị phân đầy đủ, do đó coverage bằng 1."
        ),
        r"\subsection{MLC-PA}",
        (
            "MLC-PA dùng xác suất biên từ BR và quy tắc Bayes tối ưu cho "
            "generalized Hamming loss dưới giả định độc lập nhãn "
            "\\cite{nguyen2021}. Với phạt tuyến tính, mỗi lỗi đã quyết định "
            "có chi phí 1 và mỗi vị trí từ chối có chi phí $c=0.30$."
        ),
        r"\subsection{GSI-MLC-PA v6.3.3 và v6.3.5}",
        (
            "v6.3.3 thực hiện peeling để chia nhãn thành IL/DL, dùng BR theo "
            "tầng cho IL, OS-NMF cho DL, hiệu chuẩn theo mất cân bằng và hai "
            "ngưỡng $\\tau_0,\\tau_1$ để quyết định 0, từ chối hoặc 1. Cấu "
            "hình thực tế dùng $c=0.30$, $\\gamma_{min}=0.70$, "
            "$\\tau_{balance}=0.85$, $\\alpha=0.25$ và SAMG legacy."
        ),
        (
            "v6.3.5 kế thừa SAMG2 OOF-guarded của v6.3.4. Sau khi tạo dự "
            "đoán chọn lọc, các ô IL đang từ chối được điền bằng quyết định "
            "đầy đủ theo ngưỡng thích ứng; cột DL không bị sửa. Vì vậy cơ "
            "chế mới trực tiếp tăng IL coverage lên 1, nhưng có thể đưa các "
            "trường hợp IL khó trở lại tập đánh giá."
        ),
        r"\begin{table}[H]\centering",
        r"\caption{Tóm tắt cấu hình mô hình trong benchmark thống nhất.}",
        r"\label{tab:models}\small",
        r"\begin{tabularx}{\textwidth}{lXcc}\toprule",
        r"Mô hình & Cấu trúc & Có từ chối & Phạm vi từ chối \\\midrule",
        r"BR & Bộ phân loại nhị phân độc lập theo nhãn & Không & -- \\",
        r"CC & Chuỗi phân loại theo thứ tự nhãn tự nhiên & Không & -- \\",
        r"MLC-PA & BR xác suất + Hamming BOP, phạt tuyến tính & Có & Mọi nhãn \\",
        r"GSI v6.3.3 & Peeling IL/DL + OS-NMF + Tri-Regime + SAMG legacy & Có & Mọi nhãn \\",
        r"GSI v6.3.5 & v6.3.4/SAMG2 + quyết định đầy đủ cho IL & Có & Chỉ DL \\",
        r"\bottomrule\end{tabularx}\end{table}",
        r"\section{Thuật toán đề xuất: từ động cơ thiết kế đến triển khai}",
        r"\subsection{Bài toán mà thiết kế cần giải quyết}",
        (
            "Một ngưỡng $0.5$ chung cho mọi nhãn ngầm giả định rằng xác suất "
            "đã được hiệu chuẩn như nhau và hai lớp có mức độ khó tương đương. "
            "Giả định này không phù hợp với dữ liệu đa nhãn: prior của từng nhãn "
            "có thể chênh lệch lớn, một số nhãn còn dự đoán được từ đặc trưng "
            "đầu vào (IL), trong khi các nhãn khác vẫn chia sẻ sai số có cấu trúc "
            "(DL). Mặt khác, ép mô hình quyết định tại mọi vị trí làm các xác "
            "suất gần ranh giới trở thành lỗi; nhưng từ chối quá nhiều lại làm "
            "F1 chọn lọc tăng một cách không hữu dụng vì hệ thống chuyển phần lớn "
            "công việc cho khâu review. Do đó thuật toán phải đồng thời trả lời "
            "ba câu hỏi: xác suất được tạo ra thế nào, vị trí nào đủ tin cậy để "
            "quyết định, và nhóm nhãn nào thực sự được phép từ chối."
        ),
        (
            "GSI-MLC-PA tách ba câu hỏi này thành ba tầng. Tầng cấu trúc phân "
            "chia IL/DL và chỉ mô hình hóa phụ thuộc khi còn bằng chứng trong "
            "phần dư. Tầng quyết định biến xác suất thành $0/-1/1$ bằng hai "
            "ngưỡng riêng cho từng nhãn. Tầng chính sách của v6.3.5 loại bỏ "
            "trạng thái $-1$ trên IL nhưng giữ nguyên nó trên DL. Sự tách biệt "
            "này là chủ ý: chi phí $c$ kiểm soát giá tương đối của việc từ chối, "
            r"còn \texttt{rejection\_scope} kiểm soát nơi tổ chức cho phép từ "
            "chối; chúng không phải hai mô hình phân loại độc lập."
        ),
        r"\subsection{Học cấu trúc nhãn và xác suất dự đoán}",
        (
            "Đầu tiên, CV peeling đánh giá từng nhãn bằng dự đoán out-of-fold "
            "(OOF). Nhãn đạt tiêu chuẩn ở một tầng được đưa vào IL; xác suất OOF "
            "của IL đã chọn được chuẩn hóa rồi nối vào $X$ để tầng sau có thể sử "
            "dụng thông tin đã học mà không nhìn nhãn thật của chính mẫu đó. "
            "Cách dùng OOF ở đây nhằm ngăn rò rỉ dữ liệu khi chọn cấu trúc. Các "
            "nhãn còn lại tạo thành DL. Mỗi DL có một mô hình BR nền trên không "
            "gian đặc trưng sau peeling."
        ),
        (
            "Với ít nhất hai DL, chương trình tính phần dư giữa nhãn thật và "
            "xác suất OOF, sau đó tính tương quan Pearson có dấu giữa các cột "
            "phần dư. Chỉ các cạnh có độ lớn tương quan đạt "
            "$0.25$ mới được giữ. OS-NMF thực hiện đúng một bước hiệu chỉnh "
            r"logit với hệ số $\alpha=0.25$, chuẩn hóa theo bậc và chặn dịch "
            "logit để tránh khuếch đại vòng lặp. Một bước là lựa chọn có chủ "
            "đích: nó khai thác tín hiệu phụ thuộc còn lại nhưng không biến suy "
            "diễn thành quá trình lặp khó ổn định. Sau đó bộ hiệu chuẩn thích "
            "nghi được fit trên xác suất OOF và áp dụng nhất quán cho xác suất "
            "huấn luyện/test."
        ),
        r"\subsection{Từ chi phí $c$ đến hai ngưỡng thích nghi}",
        (
            r"Nếu xác suất hậu nghiệm đã hiệu chuẩn là $p=P(y=1\mid x)$, chi "
            "phí dự đoán 0 là $p$, chi phí dự đoán 1 là $1-p$, còn chi phí từ "
            "chối là $c$. Quy tắc Chow đối xứng vì vậy quyết định 0 khi "
            r"$p\le c$, quyết định 1 khi $p\ge 1-c$, và từ chối ở giữa. Với "
            "$c=0.30$, khoảng khởi đầu là $[0.30,0.70]$. $c$ nhỏ làm từ chối "
            "rẻ hơn và vùng từ chối rộng hơn; $c$ tiến về $0.5$ làm vùng này "
            r"hẹp lại. Con số $0.30$ không có nghĩa coverage bằng $70\%$."
        ),
        r"Để phản ánh mất cân bằng của nhãn $l$, đặt",
        r"\begin{equation}\pi_l=\frac{1}{N}\sum_i y_{il},\qquad "
        r"\beta_l=1-2|\pi_l-0.5|.\end{equation}",
        (
            r"$\beta_l=1$ biểu thị nhãn cân bằng và tiến về 0 khi một lớp trở "
            r"nên cực hiếm. Với $\pi_l\le0.5$, các ngưỡng Bayes bất đối xứng "
            "trước khi chặn được tính là"
        ),
        r"\begin{equation}\tau^{a}_{1}=\frac{(1-c)\pi_l}"
        r"{c(1-\pi_l)+(1-c)\pi_l},\qquad "
        r"\tau^{a}_{0}=\frac{c\pi_l}"
        r"{(1-c)(1-\pi_l)+c\pi_l}.\end{equation}",
        (
            r"Nhánh $\pi_l>0.5$ được tính đối xứng bằng cách thay "
            r"$\pi_l$ bởi $1-\pi_l$ rồi đảo hai ngưỡng về không gian lớp dương. "
            r"Mã hiện tại chặn $\tau_1^a$ vào phía trên bởi $0.55$ cho nhãn có "
            r"lớp dương thiểu số, và chặn $\tau_0^a$ từ phía dưới bởi $0.45$ "
            "cho nhãn có lớp âm thiểu số. Hai chặn này ngăn ngưỡng xác nhận lớp "
            "hiếm đi quá xa 0.5 và làm recall của lớp hiếm sụp đổ."
        ),
        (
            "Thay vì chuyển đột ngột giữa Chow và Bayes bất đối xứng, thuật "
            "toán dùng trọng số mềm"
        ),
        r"\begin{equation}s_l=\sigma\!\left(20(\beta_l-0.85)\right),\quad "
        r"\tau_{0l}=s_lc+(1-s_l)\tau^a_0,\quad "
        r"\tau_{1l}=s_l(1-c)+(1-s_l)\tau^a_1.\end{equation}",
        (
            "Nhờ đó, thay đổi nhỏ của prior giữa các fold không gây bước nhảy "
            "lớn của ngưỡng. Sau khi blend, chương trình bảo đảm "
            r"$\tau_{0l}<\tau_{1l}$ và khoảng cách tối thiểu 0.02."
        ),
        r"\subsection{Coverage guard và SAMG: vì sao cần hai lớp bảo vệ}",
        (
            "Hai ngưỡng lý thuyết vẫn có thể bao phủ quá ít mẫu nếu phân phối "
            "xác suất thực nghiệm tập trung trong vùng từ chối. Coverage guard "
            r"đo coverage trên OOF; nếu nhỏ hơn $\gamma_{min}=0.70$, nó co dần "
            r"khoảng $(\tau_0,\tau_1)$ về 0.5, tối đa 50 bước, đồng thời giữ "
            "precision guard theo phía lớp hiếm. Dùng OOF thay cho fitted "
            "probability giúp ước lượng coverage trên dữ liệu chưa được mô hình "
            r"thấy. $\gamma_{min}$ là sàn thực nghiệm dùng khi học ngưỡng, không "
            "phải bảo đảm tuyệt đối cho từng test fold."
        ),
        (
            r"SAMG chỉ xét nhãn cực lệch với $\pi_l\le0.20$ hoặc "
            r"$\pi_l\ge0.80$. Với lớp dương hiếm, ứng viên legacy tồn tại khi "
            "điểm dương nhỏ nhất còn lớn hơn percentile 99.5 của điểm âm; ngưỡng "
            r"$\tau_1$ ứng viên được đặt giữa khoảng trống đó. Ý tưởng là hạ "
            "ngưỡng xác nhận dương chỉ khi dữ liệu cho thấy một biên phân tách "
            "sạch. v6.3.3 chấp nhận ứng viên này trực tiếp. v6.3.5 kế thừa SAMG2 "
            "của v6.3.4: ứng viên vẫn được phát hiện từ fitted probability nhưng "
            r"chỉ được nhận nếu kiểm tra OOF không làm giảm $F_1^+$ và mức giảm "
            r"$F_1^-$ không vượt 0.01. Nếu hai phân phối chồng lấn, không có "
            "ngưỡng tốt hơn, hoặc guard OOF thất bại, ngưỡng sau coverage guard "
            "được giữ nguyên. Nhánh lớp âm hiếm dùng phép đối xứng và khoảng "
            "phân tách sạch trong triển khai hiện tại."
        ),
        (
            "Guard này giải thích vì sao SAMG không thể tự động chữa mọi nhãn "
            "hiếm. Nó chỉ thay đổi quy tắc quyết định khi xác suất đã chứa tín "
            "hiệu phân hạng đủ rõ; nó không tạo thêm mẫu dương và không cải "
            "thiện mô hình xác suất khi điểm của hai lớp chồng lấn."
        ),
        r"\begin{figure}[H]\centering",
        r"\fbox{\begin{minipage}{0.94\textwidth}\small",
        r"\textbf{Mã giả 1: Huấn luyện GSI-MLC-PA v6.3.5}\\[0.3em]",
        r"\textbf{Input:} $X,Y,c=0.30,\gamma_{min}=0.70$; tham số peeling, OS-NMF và SAMG2.\\",
        r"\textbf{Output:} mô hình xác suất, phân hoạch $(IL,DL)$ và bảng ngưỡng $T$.\\[0.2em]",
        r"1. Chạy CV peeling bằng dự đoán OOF để thu các tầng $IL_1,\ldots,IL_m$ và phần dư $DL$.\\",
        r"2. Với từng tầng IL: fit BR trên đặc trưng hiện tại; nối xác suất OOF đã chuẩn hóa của các IL trước vào $X$.\\",
        r"3. Fit BR nền cho từng DL và sinh xác suất OOF; nếu $|DL|\ge2$, tính tương quan phần dư và áp dụng một bước OS-NMF.\\",
        r"4. Fit bộ hiệu chuẩn thích nghi trên $(P^{OOF},Y)$; biến đổi $P^{OOF}$.\\",
        r"5. Với mỗi nhãn $l$: tính $(\pi_l,\beta_l)$; tạo ngưỡng Chow và Bayes bất đối xứng; blend mềm để có $(\tau_{0l},\tau_{1l})$.\\",
        r"6. Nếu coverage OOF $<\gamma_{min}$, co vùng từ chối với precision guard.\\",
        r"7. Nếu nhãn cực lệch và có khoảng phân tách sạch, tạo ứng viên SAMG; với lớp dương hiếm, chỉ nhận khi OOF guard đạt yêu cầu.\\",
        r"8. Lưu $T_l=(\tau_{0l},\tau_{1l},\pi_l)$ cùng metadata audit cho mọi nhãn.",
        r"\end{minipage}}",
        r"\caption{Mã giả huấn luyện. Mọi quyết định cấu trúc và ngưỡng đều chỉ dùng outer-training fold.}\label{alg:train}",
        r"\end{figure}",
        r"\subsection{Suy diễn và chính sách chỉ từ chối DL của v6.3.5}",
        (
            "Với mỗi xác suất test $p_{il}$, quyết định chọn lọc ban đầu là"
        ),
        r"\begin{equation}\tilde y_{il}=\begin{cases}"
        r"0,&p_{il}\le\tau_{0l},\\-1,&\tau_{0l}<p_{il}<\tau_{1l},\\"
        r"1,&p_{il}\ge\tau_{1l}.\end{cases}\end{equation}",
        (
            "v6.3.3 trả trực tiếp ma trận này nên cả IL lẫn DL đều có thể mang "
            "$-1$. v6.3.5 trước hết cũng tạo đúng ma trận trên, sau đó chỉ ghi "
            "đè các cột IL bằng quyết định đầy đủ. Ngưỡng đầy đủ $t_l$ không "
            r"được tùy ý chọn lại: $t_l=\tau_{1l}$ nếu $\pi_l\le0.20$, "
            r"$t_l=\tau_{0l}$ nếu $\pi_l\ge0.80$, và $t_l=0.5$ trong vùng còn "
            "lại. Dùng ngưỡng xác nhận lớp hiếm ở hai đuôi giúp quyết định đầy "
            "đủ không phá bỏ precision guard đã học; dùng 0.5 cho nhãn không "
            "cực lệch giữ cách diễn giải xác suất chuẩn."
        ),
        r"\begin{figure}[H]\centering",
        r"\fbox{\begin{minipage}{0.94\textwidth}\small",
        r"\textbf{Mã giả 2: Dự đoán của v6.3.5}\\[0.3em]",
        r"\textbf{Input:} mẫu $X_*$, $(IL,DL)$ và bảng ngưỡng $T$.\\",
        r"1. Sinh xác suất $P$ bằng BR theo tầng cho IL, BR+OS-NMF cho DL, rồi áp dụng bộ hiệu chuẩn.\\",
        r"2. Cho mọi $(i,l)$: gán 0 nếu $p_{il}\le\tau_{0l}$; gán 1 nếu $p_{il}\ge\tau_{1l}$; ngược lại gán $-1$.\\",
        r"3. Với mỗi $l\in IL$: chọn $t_l=\tau_{1l}$ nếu $\pi_l\le0.20$; $t_l=\tau_{0l}$ nếu $\pi_l\ge0.80$; nếu không thì $t_l=0.5$.\\",
        r"4. Ghi đè toàn bộ cột IL bằng $\mathbf{1}[p_{il}\ge t_l]$. Không sửa các cột DL.\\",
        r"\textbf{Return:} $\hat Y\in\{0,1\}^{IL}\times\{0,-1,1\}^{DL}$.",
        r"\end{minipage}}",
        r"\caption{Mã giả suy diễn và vị trí chính xác của thay đổi v6.3.5.}\label{alg:infer}",
        r"\end{figure}",
        (
            "Lý do chỉ bỏ từ chối trên IL là một giả thuyết vận hành, không "
            "phải định lý rằng IL luôn dễ. Peeling đã tìm thấy các nhãn có thể "
            "tách khỏi mô hình phụ thuộc, nên việc buộc chúng trả kết quả làm "
            "giảm tải review mà không can thiệp vào vùng bất định của DL. Đổi "
            "lại, các ca IL từng bị loại khỏi selective metric nay quay lại và "
            "có thể làm F1 giảm. Chính đánh đổi có thể kiểm chứng này là lý do "
            "báo cáo đồng thời coverage, F1, generalized loss và chất lượng vùng "
            "từ chối, thay vì chỉ báo cáo một chỉ số."
        ),
        r"\section{Thiết kế thực nghiệm}",
        r"\subsection{Dữ liệu và giao thức}",
        (
            "Mười tập dữ liệu được nạp qua \\texttt{src.data.loader}. Mỗi tập "
            "dùng iterative multilabel stratification 5-fold với seed 42. "
            "MaxAbsScaler chỉ được fit trên outer-training fold. Mọi mô hình "
            "dùng Logistic Regression và seed fold tương ứng. Không có mẫu "
            "test nào tham gia chọn IL/DL, hiệu chuẩn hoặc chọn ngưỡng. "
            "Bảng~\\ref{tab:datasets} được tính trực tiếp từ ma trận nhãn, "
            "tránh các mô tả cũ không còn khớp dữ liệu."
        ),
    ]
    lines += dataset_table(datasets)
    lines += [
        r"\subsection{Định nghĩa metric}",
        r"Với tập vị trí được quyết định $D$, coverage và selective Hamming "
        r"loss là",
        r"\begin{equation}\mathrm{Cov}=\frac{|D|}{NK},\qquad "
        r"L_{H}^{sel}=\frac{1}{|D|}\sum_{(i,l)\in D}"
        r"\mathbf{1}[y_{il}\ne\hat y_{il}].\end{equation}",
        r"Selective Macro-F1 tính F1 riêng cho từng nhãn trên các mẫu đã "
        r"quyết định của nhãn đó rồi lấy trung bình. Generalized loss tuyến "
        r"tính là",
        r"\begin{equation}L_g=\frac{E_D+cA}{NK},\end{equation}",
        r"trong đó $E_D$ là số lỗi trên vùng quyết định, $A$ là số vị trí "
        r"từ chối và $c=0.30$. ABS là tỷ lệ mẫu có ít nhất một nhãn bị từ "
        r"chối; AABS chính là $1-\mathrm{Cov}$. Error-capture rate là tỷ lệ "
        r"lỗi của quyết định đầy đủ nằm trong vùng bị từ chối. Các metric "
        r"full gồm Macro-F1, Micro-F1, Example-F1, Subset Accuracy và "
        r"Hamming Loss. F1 âm được tính bằng cách xem lớp 0 là lớp đích; "
        r"nó được báo cáo để tránh diễn giải sai trên nhãn dương hiếm.",
        r"\section{Kết quả}",
        r"\subsection{Tổng quan toàn bộ metric}",
    ]
    lines += overall_table(
        overall,
        overall_std,
        [
            ("Full_Macro_F1", "Macro-F1", True),
            ("Full_Micro_F1", "Micro-F1", True),
            ("Full_Example_F1", "Example-F1", True),
            ("Subset_Accuracy", "Subset Acc.", True),
            ("Hamming_Loss", "Hamming", False),
        ],
        r"Trung bình $\pm$ độ lệch chuẩn giữa 10 tập dữ liệu cho dự đoán "
        r"đầy đủ. In đậm là tốt nhất theo từng cột.",
        "tab:overall-full",
    )
    lines += overall_table(
        overall,
        overall_std,
        [
            ("Selective_Macro_F1", "Macro-F1", True),
            ("Selective_Micro_F1", "Micro-F1", True),
            ("Selective_Instance_F1", "Instance-F1", True),
            ("Selective_Hamming_Loss", "Hamming", False),
            ("Coverage", "Coverage", True),
            ("Generalized_Loss", "$L_g$", False),
        ],
        r"Trung bình $\pm$ độ lệch chuẩn giữa 10 tập dữ liệu cho dự đoán "
        r"chọn lọc.",
        "tab:overall-selective",
    )
    lines += [
        (
            f"v6.3.3 đạt Selective Macro-F1 cao nhất "
            f"({v633['Selective_Macro_F1']:.4f}) nhưng coverage thấp nhất "
            f"({v633['Coverage']:.4f}). v6.3.5 tăng coverage thêm "
            f"{v635['Coverage']-v633['Coverage']:+.4f} và giảm generalized "
            f"loss từ {v633['Generalized_Loss']:.4f} xuống "
            f"{v635['Generalized_Loss']:.4f}; đổi lại Selective Macro-F1 "
            f"giảm {v635['Selective_Macro_F1']-v633['Selective_Macro_F1']:+.4f}. "
            "MLC-PA có selective Hamming và generalized loss thấp nhất, cho "
            "thấy vùng từ chối của nó nhắm vào lỗi tốt hơn; tuy nhiên "
            "Macro-F1 dương thấp hơn đáng kể. Do các mục tiêu này khác nhau, "
            "không thể kết luận một mô hình thắng trên mọi tiêu chí."
        ),
        r"\begin{figure}[H]\centering",
        r"\includegraphics[width=\textwidth]{fig3_overall_metrics.pdf}",
        r"\caption{So sánh các metric trung bình. Với Hamming và generalized "
        r"loss, giá trị thấp hơn tốt hơn.}\label{fig:overall}\end{figure}",
        r"\subsection{Kết quả theo từng tập dữ liệu}",
    ]
    lines += per_dataset_metric_table(
        dataset_means,
        "Selective_Macro_F1",
        "Selective Macro-F1 trên từng tập dữ liệu.",
        "tab:dataset-f1",
        True,
    )
    lines += per_dataset_metric_table(
        dataset_means,
        "Coverage",
        "Coverage trên từng tập dữ liệu; giá trị lớn không tự thân đồng nghĩa "
        "chất lượng tốt hơn.",
        "tab:dataset-coverage",
        True,
    )
    lines += [
        r"\begin{figure}[H]\centering",
        r"\includegraphics[width=\textwidth]{fig1_selective_macro_f1.pdf}",
        r"\caption{Selective Macro-F1 theo tập dữ liệu.}"
        r"\label{fig:dataset-f1}\end{figure}",
        r"\begin{figure}[H]\centering",
        r"\includegraphics[width=\textwidth]{fig2_coverage.pdf}",
        r"\caption{Coverage theo tập dữ liệu. BR và CC luôn quyết định đầy đủ.}"
        r"\label{fig:coverage}\end{figure}",
        (
            f"Trên chd49, v6.3.5 tăng Selective Macro-F1 từ "
            f"{chd.loc['GSI_v6_3_3','Selective_Macro_F1']:.4f} lên "
            f"{chd.loc['GSI_v6_3_5','Selective_Macro_F1']:.4f} và coverage "
            f"từ {chd.loc['GSI_v6_3_3','Coverage']:.4f} lên "
            f"{chd.loc['GSI_v6_3_5','Coverage']:.4f}. Tuy nhiên, audit theo "
            "nhãn cho thấy nhãn hiếm dương số 3 vẫn thuộc DL và F1 dương "
            "vẫn bằng 0; cải thiện toàn tập đến từ IL. Ngược lại, emotions, "
            "scene, music và gpositivepseaac giảm F1 khi các trường hợp IL "
            "khó bị buộc phải quyết định."
        ),
        r"\subsection{Đánh đổi coverage--chất lượng}",
        r"\begin{figure}[H]\centering",
        r"\includegraphics[width=0.78\textwidth]{fig4_f1_coverage_pareto.pdf}",
        r"\caption{Vị trí trung bình trên mặt phẳng coverage--Selective "
        r"Macro-F1.}\label{fig:pareto}\end{figure}",
        r"\begin{figure}[H]\centering",
        r"\includegraphics[width=0.92\textwidth]{fig5_v635_delta.pdf}",
        r"\caption{Chênh lệch v6.3.5 trừ v6.3.3 theo tập dữ liệu.}"
        r"\label{fig:delta}\end{figure}",
        (
            "v6.3.5 nằm gần MLC-PA về coverage nhưng đạt Selective Macro-F1 "
            "cao hơn. v6.3.3 ưu tiên chất lượng dương trên vùng đã quyết định, "
            "trong khi v6.3.5 giảm tải từ chối cho IL. Lựa chọn giữa hai phiên "
            "bản vì vậy phụ thuộc chi phí review thực tế; chỉ nhìn F1 hoặc chỉ "
            "nhìn coverage đều không đủ."
        ),
        r"\subsection{F1 dương, F1 âm và chất lượng vùng từ chối}",
    ]
    lines += overall_table(
        overall,
        overall_std,
        [
            ("Selective_Positive_Macro_F1", "$F_1^+$", True),
            ("Selective_Negative_Macro_F1", "$F_1^-$", True),
            ("Positive_Coverage", "Cov$^+$", True),
            ("Negative_Coverage", "Cov$^-$", True),
            ("Error_Capture_Rate", "ECR", True),
            ("Rejected_Error_Rate", "RER", True),
        ],
        "Chỉ số theo lớp và chất lượng vùng từ chối. RER không xác định đối "
        "với mô hình không từ chối.",
        "tab:class-reject",
    )
    lines += [
        r"\begin{figure}[H]\centering",
        r"\includegraphics[width=0.82\textwidth]{fig6_positive_negative_f1.pdf}",
        r"\caption{Khoảng cách giữa F1 dương và F1 âm. F1 âm cao trên dữ liệu "
        r"thưa có thể chủ yếu phản ánh prior lớp âm.}"
        r"\label{fig:class-f1}\end{figure}",
        r"\begin{figure}[H]\centering",
        r"\includegraphics[width=0.72\textwidth]{fig7_rejection_quality.pdf}",
        r"\caption{Khả năng đưa lỗi full prediction vào vùng từ chối.}"
        r"\label{fig:rejection}\end{figure}",
        (
            "Mọi mô hình đều có $F_1^-$ cao hơn $F_1^+$, đặc biệt trên dữ "
            "liệu thưa. Đây không phải bằng chứng rằng mô hình học tốt lớp "
            "thiểu số: một bộ phân loại thiên về 0 có thể đạt F1 âm rất cao. "
            "MLC-PA có ECR và RER cao nhất, nghĩa là vùng từ chối tập trung "
            "nhiều lỗi full prediction hơn; v6.3.3/v6.3.5 đổi một phần hiệu "
            "quả này lấy F1 dương cao hơn."
        ),
        r"\subsection{Kiểm định thống kê}",
        (
            f"Kiểm định Friedman trên Selective Macro-F1 trung bình của 10 "
            f"tập cho kết quả $\\chi^2={friedman.statistic:.3f}$, "
            f"$p={friedman.pvalue:.3g}$, bác bỏ giả thuyết các mô hình có "
            "thứ hạng như nhau. Wilcoxon được thực hiện trên 10 giá trị "
            "trung bình theo dataset và hiệu chỉnh Holm; các fold không được "
            "coi là mẫu độc lập."
        ),
    ]
    lines += statistical_table(stats, ranks)
    lines += [
        (
            "v6.3.5 cao hơn BR, CC và MLC-PA với $p_{Holm}<0.05$. Chênh "
            "lệch v6.3.5--v6.3.3 không đạt ý nghĩa ở mức 0.05; vì vậy không "
            "đủ bằng chứng thống kê để tuyên bố v6.3.3 vượt v6.3.5 về "
            "Selective Macro-F1 trên quần thể dataset, dù trung bình quan "
            "sát của v6.3.3 cao hơn."
        ),
        r"\section{Thảo luận và giới hạn}",
        r"\begin{enumerate}",
        r"\item Báo cáo chỉ dùng Logistic Regression. Kết luận không tự động "
        r"mở rộng sang SVM hoặc MLP.",
        r"\item Chỉ có 10 dataset; kiểm định thống kê có công suất hạn chế. "
        r"Dataset, không phải fold, được dùng làm đơn vị kiểm định.",
        r"\item BR/CC có coverage 1 nên selective metric bằng full metric; "
        r"so sánh trực tiếp với mô hình từ chối phải luôn xét đồng thời "
        r"coverage hoặc generalized loss.",
        r"\item v6.3.5 đồng thời kế thừa SAMG2 của v6.3.4 và thay đổi phạm "
        r"vi từ chối. Do đó chênh lệch v6.3.3--v6.3.5 không phải ablation "
        r"đơn yếu tố hoàn toàn; ảnh hưởng thuần của phạm vi từ chối được "
        r"kiểm tra bằng việc DL prediction giữ nguyên trong v6.3.5 so với "
        r"v6.3.4.",
        r"\item F1 của nhãn cực hiếm có phương sai lớn. chd49 nhãn 3 chỉ có "
        r"16 mẫu dương; kết quả F1 bằng 0 phản ánh thiếu tín hiệu xếp hạng, "
        r"không thể giải quyết chỉ bằng thay đổi phạm vi từ chối.",
        r"\item Thời gian fit được ghi nhận để audit nhưng không dùng kết "
        r"luận tốc độ do benchmark chạy tuần tự trên một môi trường Windows "
        r"không cô lập tải nền.",
        r"\end{enumerate}",
        r"\section{Kết luận}",
        (
            "v6.3.3 là lựa chọn tốt hơn khi ưu tiên Selective Macro-F1; "
            "v6.3.5 phù hợp hơn khi yêu cầu IL luôn có kết quả và coverage "
            "cao hơn. v6.3.5 không nên thay mặc định v6.3.3 chỉ dựa trên "
            "thử nghiệm hiện tại. Cách triển khai hợp lý là giữ "
            "\\texttt{rejection\\_scope=all\\_labels} làm mặc định và cung "
            "cấp \\texttt{dl\\_only} như một operating mode có đánh đổi "
            "được mô tả rõ."
        ),
        r"\appendix",
        r"\section{Ma trận kết quả đầy đủ}",
        (
            "Các bảng sau chứa trung bình 5 fold cho mọi metric được dùng "
            "trong báo cáo. Ký hiệu: MaF1/MiF1/ExF1 là full "
            "Macro/Micro/Example-F1; S là selective; GL là generalized loss; "
            "ECR là error-capture rate; RER là rejected-region error rate."
        ),
    ]
    lines += full_appendix_tables(dataset_means)
    lines += [
        r"\section{Khả năng tái lập}",
        r"\begin{verbatim}",
        r"python scripts/run_paper_v6_3_5_benchmark.py",
        r"python scripts/generate_paper_v6_3_5_report.py",
        r"\end{verbatim}",
        (
            "Môi trường ghi nhận: Python 3.13.14; NumPy 2.3.4; pandas 2.3.3; "
            "scikit-learn 1.7.2; SciPy 1.16.3; Matplotlib 3.10.7. Base commit "
            "của repository: \\texttt{3fba3ec}; các thay đổi v6.3.4/v6.3.5 "
            "thuộc working tree của thực nghiệm này."
        ),
        r"\begin{thebibliography}{9}",
        r"\bibitem{nguyen2021} V.-L. Nguyen and E. Hüllermeier. Multilabel "
        r"Classification with Partial Abstention: Bayes-Optimal Prediction "
        r"under Label Independence. \textit{Journal of Artificial "
        r"Intelligence Research}, 72:613--665, 2021.",
        r"\bibitem{read2011} J. Read, B. Pfahringer, G. Holmes, and E. Frank. "
        r"Classifier Chains for Multi-label Classification. \textit{Machine "
        r"Learning}, 85:333--359, 2011.",
        r"\bibitem{chow1970} C. K. Chow. On Optimum Recognition Error and "
        r"Reject Tradeoff. \textit{IEEE Transactions on Information Theory}, "
        r"16(1):41--46, 1970.",
        r"\bibitem{sechidis2011} K. Sechidis, G. Tsoumakas, and I. Vlahavas. "
        r"On the Stratification of Multi-label Data. In \textit{ECML PKDD}, "
        r"2011.",
        r"\end{thebibliography}",
        r"\end{document}",
    ]
    TEX.write_text("\n".join(lines), encoding="utf-8")


def write_supporting_outputs(
    detailed, dataset_means, overall, overall_std, friedman, stats, ranks
):
    overall_export = overall.copy()
    overall_export.columns = [
        f"{column}_mean" for column in overall_export.columns
    ]
    std_export = overall_std.copy()
    std_export.columns = [
        f"{column}_dataset_std" for column in std_export.columns
    ]
    overall_export.join(std_export).to_csv(
        OUT / "overall_metric_summary.csv"
    )
    stats.to_csv(OUT / "statistical_tests.csv", index=False)
    pd.DataFrame({
        "Model": ranks.index,
        "Average_Rank": ranks.values,
    }).to_csv(OUT / "average_ranks.csv", index=False)
    digest = hashlib.sha256(DETAIL.read_bytes()).hexdigest()
    manifest = {
        "source_csv": DETAIL.name,
        "source_csv_sha256": digest,
        "n_fold_records": int(len(detailed)),
        "n_dataset_model_records": int(len(dataset_means)),
        "models": MODELS,
        "datasets": DATASET_ORDER,
        "friedman_statistic": float(friedman.statistic),
        "friedman_pvalue": float(friedman.pvalue),
    }
    (OUT / "report_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def compile_pdf():
    pdflatex = Path(
        r"C:\Users\ADMIN\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
    )
    command = [
        str(pdflatex),
        "-interaction=nonstopmode",
        "-halt-on-error",
        TEX.name,
    ]
    for _ in range(2):
        result = subprocess.run(
            command,
            cwd=OUT,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            (OUT / "latex_compile_error.log").write_text(
                result.stdout + "\n" + result.stderr,
                encoding="utf-8",
            )
            raise RuntimeError(
                "LaTeX compilation failed; see latex_compile_error.log"
            )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    detailed, datasets, dataset_means, overall, overall_std = load_data()
    friedman, stats, ranks = statistical_tests(dataset_means)
    create_figures(dataset_means, overall)
    build_report(
        detailed,
        datasets,
        dataset_means,
        overall,
        overall_std,
        friedman,
        stats,
        ranks,
    )
    write_supporting_outputs(
        detailed,
        dataset_means,
        overall,
        overall_std,
        friedman,
        stats,
        ranks,
    )
    compile_pdf()
    print(f"Generated {TEX}")
    print(f"Generated {PDF}")


if __name__ == "__main__":
    main()
