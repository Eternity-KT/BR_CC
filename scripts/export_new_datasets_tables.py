"""Script to export Coverage and Macro-F1 tables for MLC-PA and GSI-MLC-PA on the 5 new datasets."""

import glob
import os
import sys
import pandas as pd
import numpy as np


def find_latest_selective_metrics(output_dir="results_pa_v3_new_datasets"):
    csv_paths = glob.glob(os.path.join(output_dir, "tables", "*", "selective_metrics.csv"))
    if not csv_paths:
        return None
    csv_paths.sort(key=os.path.getmtime, reverse=True)
    return csv_paths[0]


def generate_markdown(output_dir="results_pa_v3_new_datasets"):
    csv_path = find_latest_selective_metrics(output_dir)
    if not csv_path:
        print(f"No selective_metrics.csv found in {output_dir}")
        return ""

    df_raw = pd.read_csv(csv_path)
    # Aggregate over 5 folds (mean and std)
    df = df_raw.groupby(["Dataset", "Model", "Cost"]).agg({
        "Coverage": ["mean", "std"],
        "Selective Macro-F1": ["mean", "std"],
        "Optimistic Macro-F1": ["mean", "std"],
    }).reset_index()

    # Flatten column names
    df.columns = [
        "Dataset", "Model", "Cost",
        "Coverage_Mean", "Coverage_Std",
        "Selective_F1_Mean", "Selective_F1_Std",
        "Optimistic_F1_Mean", "Optimistic_F1_Std"
    ]

    out = []
    w = out.append

    w("# Bảng Kết Quả Đánh Giá Trên 5 Tập Dữ Liệu Mới: CHD49 và 4 Tập *PseAAC\n")
    w("> **Mô hình thực nghiệm:**")
    w("> 1. **MLC-PA:** Multi-Label Classification with Partial Abstention (Nguyen & Hüllermeier, 2021).")
    w("> 2. **GSI-MLC-PA:** Group-Sensitive Information Multi-Label Classification with Partial Abstention.")
    w("> **5 Tập dữ liệu mới:** `chd49`, `viruspseaac`, `gpositivepseaac`, `plantpseaac`, `humanpseaac` (Đánh giá qua 5-Fold Cross-Validation).\n")
    w("---")
    w("### Định nghĩa các chỉ số:")
    w("- **Coverage (Độ bao phủ $\\Gamma$):** Tỷ lệ phần trăm các vị trí nhãn được chấp nhận dự đoán (không từ chối).")
    w("- **Selective Macro-F1:** Macro-F1 tính trên các vị trí nhãn được chấp nhận.")
    w("- **Optimistic Macro-F1:** Macro-F1 giả định chuyên gia con người can thiệp gán đúng các nhãn bị từ chối.")
    w("- **Full Macro-F1 ($c = 0.50$):** Macro-F1 khi mô hình dự đoán toàn bộ (Coverage = 100%).\n")
    w("---\n")

    datasets = ["chd49", "viruspseaac", "gpositivepseaac", "plantpseaac", "humanpseaac"]
    base_learners = ["Logistic", "SVM", "MLP"]

    for idx, base in enumerate(base_learners, start=1):
        mlc_model = f"MLC_PA_{base}"
        gsi_model = f"GSI_MLC_PA_{base}"

        sub_df = df[df["Model"].isin([mlc_model, gsi_model])]
        if sub_df.empty:
            continue

        w(f"## {idx}. Lớp Phân Loại Cơ Sở: {base}\n")
        w(f"### {idx}.1. So sánh Tổng quan theo Mức Chi phí Từ chối ($c$) (Trung bình 5 Tập Dữ Liệu Mới)\n")
        w(f"| Chi phí $c$ | {mlc_model} Coverage | {mlc_model} Sel. F1 | {gsi_model} Coverage | {gsi_model} Sel. F1 | $\\Delta$ Coverage | $\\Delta$ Sel. F1 |")
        w("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

        costs = sorted(sub_df["Cost"].unique())
        for c in costs:
            m_c = sub_df[(sub_df["Model"] == mlc_model) & (np.isclose(sub_df["Cost"], c))]
            g_c = sub_df[(sub_df["Model"] == gsi_model) & (np.isclose(sub_df["Cost"], c))]

            cov_m = m_c["Coverage_Mean"].mean()
            f1_m = m_c["Selective_F1_Mean"].mean()
            cov_g = g_c["Coverage_Mean"].mean()
            f1_g = g_c["Selective_F1_Mean"].mean()

            d_cov = (cov_g - cov_m) * 100
            d_f1 = f1_g - f1_m

            tag_c = f"$c = {c:.2f}$" if c < 0.50 else "Full ($c = 0.50$)"
            w(f"| **{tag_c}** | {cov_m:.4f} | {f1_m:.4f} | **{cov_g:.4f}** | **{f1_g:.4f}** | {'+' if d_cov >= 0 else ''}{d_cov:.2f}% | {'+' if d_f1 >= 0 else ''}{d_f1:.4f} |")

        w("\n")
        w(f"### {idx}.2. Chi tiết 5 Tập Dữ Liệu Mới tại Ngưỡng Chi Phí Chuẩn $c = 0.30$\n")
        w("| Tập dữ liệu | MLC-PA Coverage | MLC-PA Sel. F1 | GSI Coverage | GSI Sel. F1 | $\\Delta$ Coverage | $\\Delta$ Sel. F1 | Mô hình tốt hơn |")
        w("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

        sub03 = sub_df[np.isclose(sub_df["Cost"], 0.30)]
        for ds in datasets:
            m_rows = sub03[(sub03["Dataset"] == ds) & (sub03["Model"] == mlc_model)]
            g_rows = sub03[(sub03["Dataset"] == ds) & (sub03["Model"] == gsi_model)]
            if m_rows.empty or g_rows.empty:
                continue
            m_r = m_rows.iloc[0]
            g_r = g_rows.iloc[0]

            cov_m, f1_m = m_r["Coverage_Mean"], m_r["Selective_F1_Mean"]
            cov_g, f1_g = g_r["Coverage_Mean"], g_r["Selective_F1_Mean"]

            d_cov = (cov_g - cov_m) * 100
            d_f1 = f1_g - f1_m

            if d_cov > 0.05 and d_f1 >= -0.005:
                winner = "**GSI 🏆** (+Cov)"
            elif d_f1 > 0.005:
                winner = "**GSI 🏆** (+F1)"
            elif d_cov < -0.05 and d_f1 < -0.005:
                winner = "MLC-PA"
            else:
                winner = "Tương đương"

            w(f"| **{ds.upper()}** | {cov_m:.4f} | {f1_m:.4f} | **{cov_g:.4f}** | **{f1_g:.4f}** | {'+' if d_cov >= 0 else ''}{d_cov:.2f}% | {'+' if d_f1 >= 0 else ''}{d_f1:.4f} | {winner} |")

        m_cov_mean = sub03[sub03["Model"] == mlc_model]["Coverage_Mean"].mean()
        m_f1_mean = sub03[sub03["Model"] == mlc_model]["Selective_F1_Mean"].mean()
        g_cov_mean = sub03[sub03["Model"] == gsi_model]["Coverage_Mean"].mean()
        g_f1_mean = sub03[sub03["Model"] == gsi_model]["Selective_F1_Mean"].mean()
        d_cov_mean = (g_cov_mean - m_cov_mean) * 100
        d_f1_mean = g_f1_mean - m_f1_mean

        winner_mean = "**GSI 🏆**" if (d_cov_mean >= 0 and d_f1_mean >= -0.002) else "Tương đương"
        w(f"| **TRUNG BÌNH** | **{m_cov_mean:.4f}** | **{m_f1_mean:.4f}** | **{g_cov_mean:.4f}** | **{g_f1_mean:.4f}** | **{'+' if d_cov_mean >= 0 else ''}{d_cov_mean:.2f}%** | **{'+' if d_f1_mean >= 0 else ''}{d_f1_mean:.4f}** | {winner_mean} |\n")

    # Add Section 4: Ma trận so sánh tổng hợp
    w("## 4. Ma Trận So Sánh Toàn Diện: Coverage & Selective Macro-F1 tại $c = 0.30$\n")
    w("| Tập dữ liệu | MLC-PA Logistic | GSI Logistic | MLC-PA SVM | GSI SVM | MLC-PA MLP | GSI MLP |")
    w("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")

    sub03_all = df[np.isclose(df["Cost"], 0.30)]
    for ds in datasets:
        row_str = f"| **{ds.upper()}** |"
        for base in base_learners:
            m_r = sub03_all[(sub03_all["Dataset"] == ds) & (sub03_all["Model"] == f"MLC_PA_{base}")]
            g_r = sub03_all[(sub03_all["Dataset"] == ds) & (sub03_all["Model"] == f"GSI_MLC_PA_{base}")]
            m_text = f"{m_r['Coverage_Mean'].iloc[0]:.3f} / {m_r['Selective_F1_Mean'].iloc[0]:.3f}" if not m_r.empty else "N/A"
            g_text = f"{g_r['Coverage_Mean'].iloc[0]:.3f} / {g_r['Selective_F1_Mean'].iloc[0]:.3f}" if not g_r.empty else "N/A"
            row_str += f" {m_text} | **{g_text}** |"
        w(row_str)

    # Average row
    avg_row = "| **TRUNG BÌNH** |"
    for base in base_learners:
        m_r = sub03_all[sub03_all["Model"] == f"MLC_PA_{base}"]
        g_r = sub03_all[sub03_all["Model"] == f"GSI_MLC_PA_{base}"]
        m_text = f"{m_r['Coverage_Mean'].mean():.4f} / {m_r['Selective_F1_Mean'].mean():.4f}"
        g_text = f"{g_r['Coverage_Mean'].mean():.4f} / {g_r['Selective_F1_Mean'].mean():.4f}"
        avg_row += f" {m_text} | **{g_text}** |"
    w(avg_row)
    w("\n")

    return "\n".join(out)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "results_pa_v3_new_datasets"
    md = generate_markdown(out_dir)
    print(md)
    with open("results_new_datasets.md", "w", encoding="utf-8") as fp:
        fp.write(md)
    print("\nSaved to results_new_datasets.md")
