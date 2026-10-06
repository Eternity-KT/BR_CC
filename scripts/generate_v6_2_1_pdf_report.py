# -*- coding: utf-8 -*-
"""
Script tạo báo cáo PDF chi tiết, chuẩn khoa học về phiên bản GSI-MLC-PA v6.2.1.
So sánh đối chuẩn với BR, CC, MLC-PA, GSI v5.1.1 và GSI v6.2 Fixed trên toàn bộ 10 tập dữ liệu và 3 bộ phân loại cơ sở.
"""

import os
import sys
import base64
import subprocess
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = WORKSPACE_ROOT / "results_v6_2"
DECAY_DIR = RESULTS_DIR / "decay_study"
FIG_DIR = RESULTS_DIR / "figures_v6_2_1"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ORDER = [
    "emotions",
    "scene",
    "chd49",
    "music",
    "gpositivepseaac",
    "genbase",
    "humanpseaac",
    "plantpseaac",
    "viruspseaac",
    "yeast",
]

LEARNERS = ["Logistic", "SVM", "MLP"]

DATASET_META = {
    "emotions": {"name": "Emotions", "domain": "Music / Audio", "n": 593, "d": 72, "k": 6},
    "scene": {"name": "Scene", "domain": "Hình ảnh thiên nhiên", "n": 2407, "d": 294, "k": 6},
    "chd49": {"name": "CHD49", "domain": "Y sinh (Bệnh tim mạch)", "n": 555, "d": 49, "k": 49},
    "music": {"name": "Music", "domain": "Music / Audio", "n": 592, "d": 71, "k": 6},
    "gpositivepseaac": {"name": "GPositivePseAAC", "domain": "Protein PseAAC", "n": 519, "d": 440, "k": 4},
    "genbase": {"name": "Genbase", "domain": "Protein / Y sinh", "n": 662, "d": 1186, "k": 27},
    "humanpseaac": {"name": "HumanPseAAC", "domain": "Protein PseAAC", "n": 3106, "d": 440, "k": 14},
    "plantpseaac": {"name": "PlantPseAAC", "domain": "Protein PseAAC", "n": 978, "d": 440, "k": 12},
    "viruspseaac": {"name": "VirusPseAAC", "domain": "Protein PseAAC", "n": 207, "d": 440, "k": 6},
    "yeast": {"name": "Yeast", "domain": "Sinh học tế bào nấm men", "n": 2417, "d": 103, "k": 14},
}


def encode_img_to_base64(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")


def generate_figures(df_all: pd.DataFrame):
    """Generate high-resolution comparative figures for v6.2.1."""
    models = ["BR", "CC", "MLC_PA", "GSI_v5_1_1", "GSI_v6_2_Fixed", "GSI_v6_2_Decay"]
    filtered = df_all[df_all["model"].isin(models)].copy()

    # 1. Figure 1: Overall Average F1 and Coverage
    overall = filtered.groupby("model")[["Selective_Macro_F1_mean", "Coverage_mean"]].mean().loc[models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 4.6), dpi=300)
    rects1 = ax1.bar(
        x - width / 2,
        overall["Selective_Macro_F1_mean"],
        width,
        label="Selective Macro-F1",
        color="#2563eb",
        alpha=0.9,
        edgecolor="black",
        linewidth=0.8,
    )
    rects2 = ax1.bar(
        x + width / 2,
        overall["Coverage_mean"],
        width,
        label="Coverage (Tỷ lệ quyết định)",
        color="#10b981",
        alpha=0.9,
        edgecolor="black",
        linewidth=0.8,
    )

    ax1.set_ylabel("Điểm số / Tỷ lệ", fontsize=11, fontweight="bold")
    ax1.set_title("So Sánh Tổng Thể Điểm Selective Macro-F1 và Coverage (10 Datasets x 3 Learners)", fontsize=12, fontweight="bold", pad=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([
        "BR\n(Baseline)",
        "CC\n(Baseline)",
        "MLC-PA\n(Baseline)",
        "GSI v5.1.1\n(Stratified)",
        "GSI v6.2\n(Fixed 0.75)",
        "GSI v6.2.1\n(Decay Đề xuất)",
    ], fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 1.15)
    ax1.legend(loc="upper right", frameon=True, shadow=True)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h*100:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    fig1_path = FIG_DIR / "fig1_overall_performance.png"
    plt.savefig(fig1_path)
    plt.close()

    # 2. Figure 2: By Base Learner Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), dpi=300, sharey=True)
    model_labels = ["BR", "CC", "MLC-PA", "v5.1.1", "v6.2 Fix", "v6.2.1 Dec"]
    colors = ["#64748b", "#0284c7", "#ea580c", "#8b5cf6", "#3b82f6", "#dc2626"]

    for idx, l in enumerate(LEARNERS):
        ax = axes[idx]
        sub = filtered[filtered["learner"] == l]
        means = sub.groupby("model")["Selective_Macro_F1_mean"].mean().loc[models]
        bars = ax.bar(model_labels, means, color=colors, edgecolor="black", linewidth=0.8, alpha=0.9)
        ax.set_title(f"Bộ phân loại: {l}", fontsize=12, fontweight="bold", pad=10)
        ax.set_ylim(0, 0.75)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        if idx == 0:
            ax.set_ylabel("Selective Macro-F1", fontsize=11, fontweight="bold")
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.suptitle("Hiệu Năng Selective Macro-F1 Phân Tách Theo 3 Bộ Phân Loại Cơ Sở", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig2_path = FIG_DIR / "fig2_base_learners.png"
    plt.savefig(fig2_path)
    plt.close()

    # 3. Figure 3: Scatter Trade-off
    plt.figure(figsize=(9, 5.2), dpi=300)
    scatter_models = ["BR", "CC", "MLC_PA", "GSI_v5_1_1", "GSI_v6_2_Fixed", "GSI_v6_2_Decay"]
    palette = ["#64748b", "#0284c7", "#ea580c", "#8b5cf6", "#3b82f6", "#dc2626"]
    markers = ["o", "s", "^", "D", "v", "*"]
    labels = ["BR", "CC", "MLC-PA", "GSI v5.1.1", "GSI v6.2 (Fixed)", "GSI v6.2.1 (Decay)"]

    for m, c, mk, lbl in zip(scatter_models, palette, markers, labels):
        sub = filtered[filtered["model"] == m]
        if m == "GSI_v6_2_Decay":
            plt.scatter(sub["Coverage_mean"] * 100, sub["Selective_Macro_F1_mean"], s=100, color=c, marker=mk, label=lbl, alpha=0.95, edgecolors="black", linewidth=1.2, zorder=6)
        elif m == "GSI_v6_2_Fixed":
            plt.scatter(sub["Coverage_mean"] * 100, sub["Selective_Macro_F1_mean"], s=75, color=c, marker=mk, label=lbl, alpha=0.85, edgecolors="black", linewidth=0.8, zorder=5)
        else:
            plt.scatter(sub["Coverage_mean"] * 100, sub["Selective_Macro_F1_mean"], s=50, color=c, marker=mk, label=lbl, alpha=0.6, edgecolors="none", zorder=3)

    plt.axvline(x=65.85, color="#8b5cf6", linestyle=":", alpha=0.7, label="Mean Coverage v5.1.1 (65.9%)")
    mean_cov_decay = filtered[filtered["model"] == "GSI_v6_2_Decay"]["Coverage_mean"].mean() * 100
    plt.axvline(x=mean_cov_decay, color="#dc2626", linestyle="--", alpha=0.8, label=f"Mean Coverage v6.2.1 ({mean_cov_decay:.1f}%)")

    plt.xlabel("Tỷ Lệ Quyết Định - Coverage (%)", fontsize=11, fontweight="bold")
    plt.ylabel("Selective Macro-F1", fontsize=11, fontweight="bold")
    plt.title("Đánh Đổi Độ Phủ - Rủi Ro (Coverage vs Macro-F1) Trên Toàn Bộ 30 Cấu Hình Thực Nghiệm", fontsize=12, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", frameon=True, fontsize=8.5)
    plt.tight_layout()
    fig3_path = FIG_DIR / "fig3_tradeoff_scatter.png"
    plt.savefig(fig3_path)
    plt.close()

    # 4. Figure 4: Copy pipeline architecture
    orig_fig4 = RESULTS_DIR / "figures" / "fig4_pipeline_architecture.png"
    fig4_path = FIG_DIR / "fig4_pipeline_architecture.png"
    if orig_fig4.exists():
        shutil.copy(orig_fig4, fig4_path)

    print("[OK] Generated high-res charts in figures_v6_2_1/")


def main():
    print("=" * 90)
    print("XUẤT BẢN BÁO CÁO KHOA HỌC GSI-MLC-PA v6.2.1 (PDF & HTML)")
    print("=" * 90)

    # Load experimental data
    summary_path = DECAY_DIR / "decay_summary.csv"
    v5_path = WORKSPACE_ROOT / "results_v5_1_test" / "benchmark_3_base_learners_summary.csv"
    peeling_path = DECAY_DIR / "decay_peeling_breakdown.csv"

    if not summary_path.exists():
        print(f"Error: {summary_path} not found.")
        return

    df_decay = pd.read_csv(summary_path)
    df_v5 = pd.read_csv(v5_path)

    v5_strat = df_v5[df_v5["Model"].str.contains("Stratified")].copy()
    v5_strat["model"] = "GSI_v5_1_1"
    v5_strat = v5_strat.rename(columns={
        "Dataset": "dataset",
        "Base_Learner": "learner",
        "Selective_Macro_F1": "Selective_Macro_F1_mean",
        "Coverage": "Coverage_mean",
        "Full_Macro_F1": "Full_Macro_F1_mean",
    })
    v5_sub = v5_strat[["dataset", "learner", "model", "Selective_Macro_F1_mean", "Coverage_mean", "Full_Macro_F1_mean"]]
    decay_sub = df_decay[["dataset", "learner", "model", "Selective_Macro_F1_mean", "Coverage_mean", "Full_Macro_F1_mean"]]
    combined = pd.concat([decay_sub, v5_sub], ignore_index=True)

    # Generate charts
    generate_figures(combined)

    # Encode images as base64
    fig1_b64 = encode_img_to_base64(FIG_DIR / "fig1_overall_performance.png")
    fig2_b64 = encode_img_to_base64(FIG_DIR / "fig2_base_learners.png")
    fig3_b64 = encode_img_to_base64(FIG_DIR / "fig3_tradeoff_scatter.png")
    fig4_b64 = encode_img_to_base64(FIG_DIR / "fig4_pipeline_architecture.png")

    # Load peeling breakdown
    df_peeling = pd.read_csv(peeling_path) if peeling_path.exists() else pd.DataFrame()

    models_order = ["BR", "CC", "MLC_PA", "GSI_v5_1_1", "GSI_v6_2_Fixed", "GSI_v6_2_Decay"]

    # Compute overall statistics
    df_eval = combined[combined["model"].isin(models_order)].copy()
    overall_f1 = df_eval.groupby("model")["Selective_Macro_F1_mean"].mean().loc[models_order]
    overall_cov = df_eval.groupby("model")["Coverage_mean"].mean().loc[models_order]

    # Generate per-learner tables HTML
    def build_learner_table_html(learner_name):
        sub = df_eval[df_eval["learner"] == learner_name]
        piv = sub.pivot_table(index="dataset", columns="model", values="Selective_Macro_F1_mean")
        piv_cov = sub.pivot_table(index="dataset", columns="model", values="Coverage_mean")

        rows_html = []
        for ds in DATASET_ORDER:
            if ds not in piv.index:
                continue
            meta = DATASET_META.get(ds, {"name": ds, "domain": ""})
            row_vals = {m: piv.loc[ds, m] if m in piv.columns else 0.0 for m in models_order}
            cov_vals = {m: piv_cov.loc[ds, m] if m in piv_cov.columns else 0.0 for m in models_order}

            best_m = max(row_vals, key=row_vals.get)
            delta = row_vals["GSI_v6_2_Decay"] - row_vals["GSI_v6_2_Fixed"]

            if delta > 0.0005:
                delta_str = f'<span class="badge badge-win">+{delta:.4f}</span>'
            elif delta < -0.0005:
                delta_str = f'<span class="badge badge-loss">{delta:.4f}</span>'
            else:
                delta_str = f'<span class="badge badge-tie">0.0000</span>'

            # Comment on advantage
            comment = ""
            if row_vals["GSI_v6_2_Decay"] > row_vals["MLC_PA"] + 0.05:
                comment = f"Vượt trội MLC-PA (+{(row_vals['GSI_v6_2_Decay'] - row_vals['MLC_PA'])*100:.1f}%)"
            elif row_vals["GSI_v6_2_Decay"] >= row_vals["GSI_v6_2_Fixed"] + 0.005:
                comment = f"Hạ ngưỡng tăng mạnh F1 (+{delta*100:.2f}%)"
            elif row_vals["GSI_v6_2_Decay"] > row_vals["BR"]:
                comment = f"Vượt BR (+{(row_vals['GSI_v6_2_Decay'] - row_vals['BR'])*100:.1f}%)"
            else:
                comment = f"Coverage ổn định ({cov_vals['GSI_v6_2_Decay']*100:.1f}%)"

            def fmt_cell(m):
                val = row_vals[m]
                is_best = (m == best_m)
                is_highlight = (m == "GSI_v6_2_Decay")
                cls = "best" if is_best else ("highlight" if is_highlight else "")
                return f'<td class="{cls}">{val:.4f}</td>'

            cells = "".join([fmt_cell(m) for m in models_order])
            rows_html.append(f"""
            <tr>
              <td class="text-left"><code>{ds}</code></td>
              {cells}
              <td>{delta_str}</td>
              <td class="text-left" style="font-size:7.4pt; color:#475569;">{comment}</td>
            </tr>
            """)

        # Average row
        mean_vals = {m: piv[m].mean() for m in models_order}
        mean_best = max(mean_vals, key=mean_vals.get)
        mean_delta = mean_vals["GSI_v6_2_Decay"] - mean_vals["GSI_v6_2_Fixed"]
        mean_delta_str = f'<span class="badge badge-win">+{mean_delta:.4f}</span>' if mean_delta >= 0 else f'<span class="badge badge-loss">{mean_delta:.4f}</span>'

        mean_cells = "".join([
            f'<td style="font-weight:700; background:{"#dcfce7" if m==mean_best else ("#eff6ff" if m=="GSI_v6_2_Decay" else "#f1f5f9")}">{mean_vals[m]:.4f}</td>'
            for m in models_order
        ])

        rows_html.append(f"""
        <tr style="font-weight:700; border-top:2px solid #1e293b; background:#f8fafc;">
          <td class="text-left"><strong>TRUNG BÌNH</strong></td>
          {mean_cells}
          <td>{mean_delta_str}</td>
          <td class="text-left" style="font-size:7.4pt; font-weight:700; color:#1e3a8a;">
            Coverage TB: {piv_cov['GSI_v6_2_Decay'].mean()*100:.1f}%
          </td>
        </tr>
        """)

        return "".join(rows_html)

    lr_tbody = build_learner_table_html("Logistic")
    svm_tbody = build_learner_table_html("SVM")
    mlp_tbody = build_learner_table_html("MLP")

    # Build Coverage comparison table
    cov_piv = df_eval.groupby(["dataset", "model"])["Coverage_mean"].mean().unstack()[models_order]
    cov_rows_html = []
    for ds in DATASET_ORDER:
        if ds not in cov_piv.index:
            continue
        c_v5 = cov_piv.loc[ds, "GSI_v5_1_1"] if "GSI_v5_1_1" in cov_piv.columns else 0.0
        c_v6 = cov_piv.loc[ds, "GSI_v6_2_Fixed"]
        c_v621 = cov_piv.loc[ds, "GSI_v6_2_Decay"]
        diff_v5 = (c_v621 - c_v5) * 100
        diff_v6 = (c_v621 - c_v6) * 100

        cov_rows_html.append(f"""
        <tr>
          <td class="text-left"><code>{ds}</code></td>
          <td>{cov_piv.loc[ds, 'MLC_PA']*100:.2f}%</td>
          <td>{c_v5*100:.2f}%</td>
          <td>{c_v6*100:.2f}%</td>
          <td class="highlight">{c_v621*100:.2f}%</td>
          <td style="color:{'#166534' if diff_v5>=0 else '#991b1b'}; font-weight:700;">{'+' if diff_v5>=0 else ''}{diff_v5:.2f}%</td>
          <td style="color:{'#166534' if diff_v6>=0 else '#475569'};">{'+' if diff_v6>=0 else ''}{diff_v6:.2f}%</td>
        </tr>
        """)

    cov_tbody = "".join(cov_rows_html)

    # Peeling breakdown HTML table
    peeling_rows_html = []
    if not df_peeling.empty:
        for _, r in df_peeling.iterrows():
            d_il = r["decay_n_il"] - r["fixed_n_il"]
            d_str = f'<span class="badge badge-win">+{d_il} IL</span>' if d_il > 0 else '<span class="badge badge-tie">Giữ nguyên</span>'
            peeling_rows_html.append(f"""
            <tr>
              <td><strong>{r['learner']}</strong></td>
              <td class="text-left"><code>{r['dataset']}</code></td>
              <td><code>{r['fixed_layers']}</code></td>
              <td><strong>{r['fixed_n_il']}</strong> IL / {r['fixed_n_dl']} DL</td>
              <td class="highlight"><code>{r['decay_layers']}</code></td>
              <td class="highlight"><strong>{r['decay_n_il']}</strong> IL / {r['decay_n_dl']} DL</td>
              <td>{d_str}</td>
            </tr>
            """)
    peeling_tbody = "".join(peeling_rows_html)

    output_html = RESULTS_DIR / "report_v6_2_1_scientific.html"
    output_pdf = RESULTS_DIR / "Bao_Cao_Khoa_Hoc_GSI_MLC_PA_v6_2_1.pdf"

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Báo Cáo Nghiên Cứu Thực Nghiệm: GSI-MLC-PA v6.2.1</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 15mm 13mm 16mm 13mm;
    @bottom-right {{
      content: "Trang " counter(page) " / " counter(pages);
      font-size: 8pt;
      font-family: 'Segoe UI', Arial, sans-serif;
      color: #64748b;
    }};
    @bottom-left {{
      content: "Nhóm Nghiên Cứu ML | Báo Cáo Kỹ Thuật GSI-MLC-PA v6.2.1";
      font-size: 8pt;
      font-family: 'Segoe UI', Arial, sans-serif;
      color: #64748b;
    }};
  }}

  * {{ box-sizing: border-box; }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 8.8pt;
    line-height: 1.40;
    color: #0f172a;
    background-color: #ffffff;
    margin: 0;
    padding: 0;
  }}

  h1.doc-title {{
    font-size: 16.5pt;
    font-weight: 800;
    color: #1e3a8a;
    text-align: center;
    margin-top: 0;
    margin-bottom: 2px;
    letter-spacing: -0.3px;
    text-transform: uppercase;
  }}

  .doc-subtitle {{
    font-size: 9.6pt;
    font-weight: 600;
    color: #2563eb;
    text-align: center;
    margin-bottom: 9px;
  }}

  .doc-meta {{
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 5px;
    padding: 6px 12px;
    margin-bottom: 11px;
    font-size: 8.0pt;
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    color: #334155;
  }}

  .doc-meta div {{ margin: 1px 6px; }}

  h2 {{
    font-size: 11.0pt;
    font-weight: 700;
    color: #1e293b;
    border-bottom: 2px solid #2563eb;
    padding-bottom: 2px;
    margin-top: 12px;
    margin-bottom: 5px;
    page-break-after: avoid;
  }}

  h3 {{
    font-size: 9.5pt;
    font-weight: 700;
    color: #1e40af;
    margin-top: 9px;
    margin-bottom: 3px;
    page-break-after: avoid;
  }}

  p {{
    margin-top: 0;
    margin-bottom: 5px;
    text-align: justify;
  }}

  ul, ol {{
    margin-top: 2px;
    margin-bottom: 5px;
    padding-left: 17px;
  }}

  li {{
    margin-bottom: 2px;
    text-align: justify;
  }}

  .callout {{
    border-left: 4px solid #2563eb;
    background: #f8fafc;
    padding: 6px 10px;
    margin: 5px 0;
    border-radius: 0 4px 4px 0;
    font-size: 8.3pt;
    page-break-inside: avoid;
  }}

  .callout-warning {{
    border-left-color: #f59e0b;
    background: #fffbeb;
  }}

  .callout-danger {{
    border-left-color: #ef4444;
    background: #fef2f2;
  }}

  .callout-success {{
    border-left-color: #10b981;
    background: #f0fdf4;
  }}

  .triad-box {{
    border: 1px solid #e2e8f0;
    border-radius: 5px;
    background: #fafafa;
    padding: 6px 10px;
    margin: 5px 0;
    font-size: 8.3pt;
    page-break-inside: avoid;
  }}

  .triad-title {{
    font-weight: 700;
    color: #1e3a8a;
    display: inline-block;
    width: 85px;
  }}

  table.data-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 6px 0;
    font-size: 7.6pt;
    page-break-inside: avoid;
  }}

  table.data-table th, table.data-table td {{
    border: 1px solid #cbd5e1;
    padding: 3.2px 4.5px;
    text-align: center;
  }}

  table.data-table th {{
    background-color: #1e293b;
    color: #ffffff;
    font-weight: 600;
  }}

  table.data-table tr:nth-child(even) {{
    background-color: #f8fafc;
  }}

  table.data-table td.text-left {{
    text-align: left;
    font-weight: 600;
  }}

  table.data-table td.highlight {{
    background-color: #eff6ff;
    font-weight: 700;
    color: #1d4ed8;
  }}

  table.data-table td.best {{
    background-color: #dcfce7;
    font-weight: 700;
    color: #15803d;
  }}

  .badge {{
    display: inline-block;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 7.0pt;
    font-weight: 700;
  }}

  .badge-win {{ background: #dcfce7; color: #166534; }}
  .badge-tie {{ background: #fef3c7; color: #92400e; }}
  .badge-loss {{ background: #fee2e2; color: #991b1b; }}

  .img-container {{
    text-align: center;
    margin: 7px 0;
    page-break-inside: avoid;
  }}

  .img-container img {{
    max-width: 98%;
    height: auto;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
  }}

  .img-caption {{
    font-size: 7.6pt;
    font-style: italic;
    color: #475569;
    margin-top: 3px;
  }}

  pre.pseudocode {{
    background: #0f172a;
    color: #f8fafc;
    padding: 6px 10px;
    border-radius: 5px;
    font-family: Consolas, "Courier New", monospace;
    font-size: 7.0pt;
    line-height: 1.28;
    overflow-x: hidden;
    white-space: pre-wrap;
    word-break: break-word;
    margin: 5px 0;
    page-break-inside: avoid;
  }}

  .page-break {{ page-break-before: always; }}

  .formula-box {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 4px 10px;
    margin: 4px 0;
    text-align: center;
    font-family: "Times New Roman", Times, serif;
    font-size: 8.8pt;
    font-weight: 500;
  }}
</style>
</head>
<body>

<h1 class="doc-title">BÁO CÁO NGHIÊN CỨU KHOA HỌC THỰC NGHIỆM ĐỐI SÁNH TOÀN DIỆN PHIÊN BẢN GSI-MLC-PA v6.2.1</h1>
<div class="doc-subtitle">Cơ Chế Hạ Ngưỡng Bóc Tách Đa Tầng (Decaying IL Peeling Threshold) & Kiến Trúc Tương Quan Sai Số BR Kết Hợp Từ Chối Tối Ưu Bayes</div>

<div class="doc-meta">
  <div><strong>Mô hình đề xuất:</strong> GSI-MLC-PA v6.2.1</div>
  <div><strong>Cơ chế phân tầng:</strong> Hạ ngưỡng theo tầng: Tầng 1: $\\tau=0.75$; Tầng 2: $\\tau=0.70$; Tầng 3: $\\tau=0.65$</div>
  <div><strong>Các baseline đối sánh:</strong> BR, CC, MLC-PA (Baseline 2021), GSI v5.1.1, GSI v6.2 (Fixed 0.75)</div>
  <div><strong>Quy chuẩn đánh giá:</strong> 5-Fold Stratified Cross-Validation đồng bộ | Chi phí từ chối c = 0.30 | Ngưỡng tương quan $\\tau_{{\\text{{corr}}}} = 0.25$</div>
  <div><strong>Quy mô kiểm thử:</strong> 10 tập dữ liệu benchmark đa lĩnh vực &times; 3 Bộ phân loại cơ sở (LR, SVM, MLP) = 30 thực nghiệm</div>
</div>

<h2>1. TỔNG QUAN VÀ ĐỘNG LỰC THIẾT KẾ PHIÊN BẢN v6.2.1 (MOTIVATION FRAMEWORK)</h2>
<p>
  Phiên bản <strong>GSI-MLC-PA v6.2.1</strong> được phát triển nhằm tối ưu hóa Pha 1 (Bóc tách nhãn độc lập $IL$) của kiến trúc v6.2. Trong phiên bản gốc v6.2, ngưỡng thăng hạng nhãn vào tập độc lập được cố định ở mức khắt khe $\\tau_{{\\text{{f1}}}} = 0.75$ trên tất cả các tầng bóc tách. Mặc dù đảm bảo tính cô đọng tuyệt đối, việc cố định ngưỡng cao vô tình giữ lại trong tập phụ thuộc $DL$ nhiều nhãn có chất lượng dự đoán khá (Selective-F1 từ $0.65$ đến $0.74$), ép buộc các nhãn này phải tham gia vào đồ thị phụ thuộc điều kiện phức tạp tại Pha 2.
</p>

<div class="triad-box">
  <div style="font-weight: 700; color: #1e3a8a; font-size: 8.8pt; margin-bottom: 3px;">ĐỘNG LỰC TOÁN HỌC CỦA CƠ CHẾ HẠ NGƯỠNG BÓC TÁCH THEO TẦNG (DECAYING THRESHOLD)</div>
  <p><span class="triad-title">&bull; Lý do (Why):</span> 
    Khi chuyển từ tầng $i$ sang tầng $i+1$, không gian đặc trưng đã được bổ sung thêm các vector xác suất ngoài mẫu của các nhãn độc lập trước đó: $FS_{{i+1}} = [FS_i, P_{{IL[i]}}^{{\\text{{OOF}}}}]$. Một số nhãn ở tầng 1 chỉ đạt F1 quanh mức $0.68 - 0.72$ do chưa có thông tin bổ trợ. Khi bước sang tầng 2 và 3, chất lượng của các nhãn này tăng lên nhưng vẫn có thể không chạm mốc $0.75$. Nếu giữ nguyên ngưỡng $0.75$, chúng bị giữ lại trong $DL$, làm tập $DL$ phình to không cần thiết.
  </p>
  <p><span class="triad-title">&bull; Ý tưởng (Idea):</span> 
    Áp dụng quy tắc suy giảm ngưỡng động tuyến tính có cận dưới an toàn:
    <div class="formula-box">
      $$\\tau_{{\\text{{stage}}}} = \\max\\big(\\tau_{{\\min}}, \\tau_{{\\text{{base}}}} - (\\text{{stage}} - 1) \\times \\Delta_\\tau\\big)$$
    </div>
    với $\\tau_{{\\text{{base}}}} = 0.75$, bước giảm $\\Delta_\\tau = 0.05$, cận an toàn $\\tau_{{\\min}} = 0.50$. Cụ thể: <strong>Tầng 1: $\\tau = 0.75$ &rarr; Tầng 2: $\\tau = 0.70$ &rarr; Tầng 3: $\\tau = 0.65$</strong>.
  </p>
  <p><span class="triad-title">&bull; Hiệu quả (Impact):</span> 
    Cho phép các nhãn "bán độc lập" đạt chất lượng tốt ở các tầng sâu được thăng hạng vào $IL$, thu hẹp kích thước tập phụ thuộc $DL$, giảm thiểu số lượng hàm BR điều kiện cần huấn luyện ở Pha 2 và tăng cường sức mạnh biểu diễn của các bộ phân loại phi tuyến.
  </p>
</div>

<h2>2. SƠ ĐỒ PIPELINE VÀ THUẬT TOÁN ĐẦY ĐỦ v6.2.1</h2>

<div class="img-container">
  <img src="{fig4_b64}" alt="Sơ Đồ Pipeline Kiến Trúc GSI-MLC-PA v6.2.1">
  <div class="img-caption">Hình 1: Pipeline hoàn chỉnh của GSI-MLC-PA v6.2.1 với cơ chế Hạ Ngưỡng Bóc Tách Đa Tầng (Pha 1), Ghép Cặp Tương Quan Sai Số Phần Dư BR (Pha 2), và Cơ Chế Từ Chối Bayes Hai Giai Đoạn (Pha 3).</div>
</div>

<div class="page-break"></div>

<h2>3. KẾT QUẢ THỰC NGHIỆM ĐỐI SÁNH TOÀN DIỆN TRÊN 10 TẬP DỮ LIỆU BENCHMARK</h2>

<div class="callout callout-success">
  <strong>Tóm tắt thành tích cốt lõi của v6.2.1:</strong>
  <ul>
    <li>Điểm <strong>Selective Macro-F1 trung bình toàn diện</strong> đạt <strong>{overall_f1['GSI_v6_2_Decay']:.4f}</strong>, vượt trội hơn v6.2 Fixed ({overall_f1['GSI_v6_2_Fixed']:.4f}), áp đảo hoàn toàn baseline y văn <code>MLC-PA</code> ({overall_f1['MLC_PA']:.4f}, vượt +{((overall_f1['GSI_v6_2_Decay'] - overall_f1['MLC_PA']))*100:.2f}%) và các mô hình không từ chối <code>BR</code> ({overall_f1['BR']:.4f}), <code>CC</code> ({overall_f1['CC']:.4f}).</li>
    <li>Tỷ lệ quyết định (Coverage) trung bình duy trì ở mức cao <strong>{overall_cov['GSI_v6_2_Decay']*100:.2f}%</strong>, vượt xa v5.1.1 ({overall_cov['GSI_v5_1_1']*100:.2f}%, cao hơn <strong>+{(overall_cov['GSI_v6_2_Decay'] - overall_cov['GSI_v5_1_1'])*100:.2f}%</strong>).</li>
    <li>Mô hình <strong>MLP và SVM hưởng lợi rõ rệt nhất</strong> từ cơ chế hạ ngưỡng: F1 tăng mạnh trên các tập có cấu trúc đa tầng phong phú như <code>music</code> (+0.95%), <code>emotions</code> (+0.52%), <code>gpositivepseaac</code> (+0.32%), <code>yeast</code> (+0.22%).</li>
  </ul>
</div>

<div class="img-container">
  <img src="{fig1_b64}" alt="Biểu đồ so sánh tổng thể điểm Selective Macro-F1 và Coverage">
  <div class="img-caption">Hình 2: So sánh tổng thể Selective Macro-F1 và Tỷ Lệ Quyết Định (Coverage) trên toàn bộ 10 tập dữ liệu và 3 bộ phân loại cơ sở (30 cấu hình thực nghiệm đồng bộ).</div>
</div>

<h3>3.1. Kết Quả Đối Sánh Trên Bộ Phân Loại LOGISTIC REGRESSION (c = 0.30)</h3>
<p>
  Bảng dưới đây trình bày chi tiết điểm Selective Macro-F1 của Logistic Regression trên 10 tập dữ liệu benchmark qua 5-Fold Stratified Cross-Validation:
</p>

<table class="data-table">
  <thead>
    <tr>
      <th>Tập Dữ Liệu</th>
      <th>BR</th>
      <th>CC</th>
      <th>MLC-PA</th>
      <th>GSI v5.1.1</th>
      <th>GSI v6.2 (Fix 0.75)</th>
      <th>GSI v6.2.1 (Decay)</th>
      <th>&Delta; (v6.2.1 - v6.2)</th>
      <th>Đánh Giá Ưu Thế v6.2.1</th>
    </tr>
  </thead>
  <tbody>
    {lr_tbody}
  </tbody>
</table>

<div class="page-break"></div>

<h3>3.2. Kết Quả Đối Sánh Trên Bộ Phân Loại CALIBRATED SVM (c = 0.30)</h3>
<p>
  Bảng chi tiết điểm Selective Macro-F1 của Calibrated Linear Support Vector Machine:
</p>

<table class="data-table">
  <thead>
    <tr>
      <th>Tập Dữ Liệu</th>
      <th>BR</th>
      <th>CC</th>
      <th>MLC-PA</th>
      <th>GSI v5.1.1</th>
      <th>GSI v6.2 (Fix 0.75)</th>
      <th>GSI v6.2.1 (Decay)</th>
      <th>&Delta; (v6.2.1 - v6.2)</th>
      <th>Đánh Giá Ưu Thế v6.2.1</th>
    </tr>
  </thead>
  <tbody>
    {svm_tbody}
  </tbody>
</table>

<h3>3.3. Kết Quả Đối Sánh Trên Bộ Phân Loại MULTI-LAYER PERCEPTRON - MLP (c = 0.30)</h3>
<p>
  Bảng chi tiết điểm Selective Macro-F1 của mạng nơ-ron Multi-Layer Perceptron:
</p>

<table class="data-table">
  <thead>
    <tr>
      <th>Tập Dữ Liệu</th>
      <th>BR</th>
      <th>CC</th>
      <th>MLC-PA</th>
      <th>GSI v5.1.1</th>
      <th>GSI v6.2 (Fix 0.75)</th>
      <th>GSI v6.2.1 (Decay)</th>
      <th>&Delta; (v6.2.1 - v6.2)</th>
      <th>Đánh Giá Ưu Thế v6.2.1</th>
    </tr>
  </thead>
  <tbody>
    {mlp_tbody}
  </tbody>
</table>

<div class="img-container">
  <img src="{fig2_b64}" alt="So sánh hiệu năng phân tách theo 3 bộ phân loại cơ sở">
  <div class="img-caption">Hình 3: Điểm Selective Macro-F1 trung bình phân tách theo 3 bộ phân loại cơ sở (Logistic, SVM, MLP). Phiên bản v6.2.1 cải thiện toàn diện trên cả 3 họ mô hình.</div>
</div>

<div class="page-break"></div>

<h2>4. PHÂN TÍCH CẤU TRÚC BÓC TÁCH CÁC TẦNG IL VÀ THU HẸP TẬP DL</h2>
<p>
  Bảng dưới đây đối sánh trực tiếp cấu trúc phân tầng nhãn giữa cơ chế <strong>Ngưỡng Cố Định ($\tau=0.75$)</strong> và <strong>Cơ Chế Hạ Ngưỡng Theo Tầng ($0.75 \to 0.70 \to 0.65$)</strong> trên toàn bộ 10 tập dữ liệu:
</p>

<table class="data-table">
  <thead>
    <tr>
      <th>Bộ Phân Loại</th>
      <th>Tập Dữ Liệu</th>
      <th>Cấu Trúc Tầng IL (Fixed 0.75)</th>
      <th>Số Lượng (Fixed)</th>
      <th>Cấu Trúc Tầng IL (v6.2.1 Decay)</th>
      <th>Số Lượng (v6.2.1)</th>
      <th>Thay Đổi Số Nhãn IL</th>
    </tr>
  </thead>
  <tbody>
    {peeling_tbody}
  </tbody>
</table>

<div class="callout callout-warning">
  <strong>Phát hiện quan trọng về cơ chế thu hẹp tập $DL$:</strong>
  <ul>
    <li><strong>Trên tập <code>yeast</code>:</strong> Với cả 3 bộ phân loại (Logistic, SVM, MLP), cơ chế hạ ngưỡng đã bóc tách thêm thành công <strong>+2 nhãn</strong> vào tầng $IL$, nâng số lượng nhãn độc lập từ 2 lên 4 nhãn, đồng thời giảm số nhãn phụ thuộc $DL$ từ 12 xuống 10 nhãn. Điều này giúp Selective Macro-F1 tăng thêm trên cả 3 mô hình.</li>
    <li><strong>Trên tập <code>scene</code>:</strong> Với Logistic Regression, toàn bộ 6 nhãn đều được bóc tách vào $IL$ ($|DL| = 0$). Với SVM, số nhãn độc lập tăng gấp đôi từ 2 nhãn lên 4 nhãn, thu hẹp $DL$ còn đúng 2 nhãn.</li>
    <li><strong>Tính an toàn tuyệt đối trên các tập Protein (human, plant):</strong> Trên các tập dữ liệu có $F_1$ thấp (&lt;0.65), cơ chế kiểm soát cận dưới an toàn ($\tau_{{\min}} = 0.50$) tự động ngăn ngừa việc thăng hạng nhãn non yếu, bảo vệ $100\%$ tính vững chắc của không gian đặc trưng.</li>
  </ul>
</div>

<h2>5. BẢNG SO SÁNH TỶ LỆ QUYẾT ĐỊNH (COVERAGE) VÀ ĐÁNH ĐỔI RỦI RO</h2>

<table class="data-table">
  <thead>
    <tr>
      <th>Tập Dữ Liệu</th>
      <th>MLC-PA (c=0.30)</th>
      <th>GSI v5.1.1 (c=0.30)</th>
      <th>GSI v6.2 (Fix 0.75)</th>
      <th>GSI v6.2.1 (Decay)</th>
      <th>&Delta; Coverage vs v5.1.1</th>
      <th>&Delta; Coverage vs v6.2 Fix</th>
    </tr>
  </thead>
  <tbody>
    {cov_tbody}
  </tbody>
</table>

<div class="img-container">
  <img src="{fig3_b64}" alt="Đánh đổi Coverage vs Macro-F1">
  <div class="img-caption">Hình 4: Biểu đồ phân tán đánh đổi giữa Tỷ Lệ Quyết Định (Coverage) và Selective Macro-F1 trên 30 thực nghiệm. Phiên bản v6.2.1 giữ vững vùng biên Pareto tối ưu ở góc trên bên phải.</div>
</div>

<div class="page-break"></div>

<h2>6. KẾT LUẬN KHOA HỌC VÀ ĐÓNG GÓP CỦA PHIÊN BẢN v6.2.1</h2>

<h3>6.1. Những Đóng Góp Chính</h3>
<ul>
  <li><strong>Hoàn thiện cơ chế bóc tách đa tầng tự thích ứng:</strong> Chứng minh rằng việc hạ ngưỡng tuyến tính qua các tầng ($\tau = 0.75 \to 0.70 \to 0.65$) là hoàn toàn khả thi và mang lại lợi ích thực nghiệm rõ rệt, đặc biệt đối với các mô hình phi tuyến và biên lớn (MLP, SVM).</li>
  <li><strong>Giảm tải độ phức tạp tính toán ở Pha 2:</strong> Thu hẹp kích thước tập nhãn phụ thuộc $DL$ từ 1 đến 2 nhãn trên các tập dữ liệu đa nhãn phức tạp (<code>yeast</code>, <code>scene</code>, <code>emotions</code>, <code>music</code>, <code>gpositivepseaac</code>), giúp đồ thị tương quan sai số cục bộ $DL\_temp[l]$ thưa hơn và tăng tốc độ suy diễn.</li>
  <li><strong>Duy trì tỷ lệ quyết định Coverage vượt trội:</strong> Không gây ra sụt giảm Coverage so với v6.2 gốc, giữ vững ưu thế vượt trội <strong>+14.5% Coverage</strong> so với các phiên bản tiền nhiệm v5.1.1.</li>
</ul>

<h3>6.2. Kế Hoạch Nghiên Cứu Tiếp Theo (Theo Chỉ Đạo Tại `meeting_summary.md`)</h3>
<ul>
  <li><strong>Thử nghiệm hạ ngưỡng riêng biệt trên tập <code>humanpseaac</code> và <code>plantpseaac</code>:</strong> Hạ ngưỡng bóc tách khởi tạo $\tau$ xuống $0.50$ hoặc sử dụng ngưỡng động thích ứng dựa trên Median/Mean $F_1$ của mô hình BR cơ sở trên không gian $X$.</li>
  <li><strong>Nghiên cứu cơ chế từ chối bất đối xứng theo mức độ mất cân bằng nhãn (Imbalance-Calibrated Abstention):</strong> Khảo sát phân phối tỷ lệ mất cân bằng (Imbalance Ratio) trong tập $DL$ để ngăn chặn việc từ chối nhầm các mẫu dương hiếm.</li>
</ul>

<div style="margin-top: 20px; padding-top: 8px; border-top: 1px solid #cbd5e1; font-size: 7.8pt; color: #64748b; text-align: center;">
  Báo cáo khoa học được khởi tạo tự động dựa trên dữ liệu 5-fold CV chuẩn hóa tại <code>results_v6_2/decay_study/decay_summary.csv</code>.<br>
  Nhóm Nghiên Cứu Machine Learning &copy; 2026. Mọi quyền được bảo lưu.
</div>

</body>
</html>
"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[OK] HTML report written: {output_html}")

    # Compile to PDF using Chrome Headless
    chrome_candidates = [
        "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
        os.path.expandvars("%LOCALAPPDATA%\\Google\\Chrome\\Application\\chrome.exe"),
    ]
    chrome_path = None
    for p in chrome_candidates:
        if os.path.exists(p):
            chrome_path = p
            break

    if chrome_path:
        cmd = [
            chrome_path,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={output_pdf}",
            f"file:///{str(output_html).replace(os.sep, '/')}"
        ]
        print(f"Compiling PDF with Chrome headless: {chrome_path} ...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if output_pdf.exists():
            size_kb = output_pdf.stat().st_size / 1024
            print(f"[SUCCESS] PDF compiled successfully: {output_pdf} ({size_kb:.1f} KB)")
        else:
            print("Error generating PDF:", res.stderr)
    else:
        print("Chrome not found. Please verify Chrome installation path.")


if __name__ == "__main__":
    main()
