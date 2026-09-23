import pandas as pd
import numpy as np

csv_path = "results_pa_v4/tables/434d53dc787206d6/selective_metrics.csv"
df = pd.read_csv(csv_path)
df_30 = df[np.isclose(df["Cost"], 0.30)]

grouped = df_30.groupby(["Dataset", "Model"])[["Coverage", "Selective Macro-F1"]].mean().reset_index()

datasets = sorted(df_30["Dataset"].unique())

def format_cell(cov1, f1_1, cov2, f1_2):
    # returns (cell1_str, cell2_str)
    # bold whichever is higher
    c1_str = f"**{cov1:.3f}**" if cov1 > cov2 else f"{cov1:.3f}"
    c2_str = f"**{cov2:.3f}**" if cov2 > cov1 else f"{cov2:.3f}"
    if np.isclose(cov1, cov2, atol=1e-4):
        c1_str = f"{cov1:.3f}"
        c2_str = f"{cov2:.3f}"

    f1_str = f"**{f1_1:.3f}**" if f1_1 > f1_2 else f"{f1_1:.3f}"
    f2_str = f"**{f1_2:.3f}**" if f1_2 > f1_1 else f"{f1_2:.3f}"
    if np.isclose(f1_1, f1_2, atol=1e-4):
        f1_str = f"{f1_1:.3f}"
        f2_str = f"{f1_2:.3f}"

    return f"{c1_str} / {f1_str}", f"{c2_str} / {f2_str}"

rows = []
for d in datasets:
    d_df = grouped[grouped["Dataset"] == d]
    def get_val(model):
        sub = d_df[d_df["Model"] == model]
        if sub.empty:
            return 0.0, 0.0
        return sub["Coverage"].values[0], sub["Selective Macro-F1"].values[0]

    cov_mlc_log, f1_mlc_log = get_val("MLC_PA_Logistic")
    cov_gsi_log, f1_gsi_log = get_val("GSI_MLC_PA_Logistic")
    cell_mlc_log, cell_gsi_log = format_cell(cov_mlc_log, f1_mlc_log, cov_gsi_log, f1_gsi_log)

    cov_mlc_svm, f1_mlc_svm = get_val("MLC_PA_SVM")
    cov_gsi_svm, f1_gsi_svm = get_val("GSI_MLC_PA_SVM")
    cell_mlc_svm, cell_gsi_svm = format_cell(cov_mlc_svm, f1_mlc_svm, cov_gsi_svm, f1_gsi_svm)

    cov_mlc_mlp, f1_mlc_mlp = get_val("MLC_PA_MLP")
    cov_gsi_mlp, f1_gsi_mlp = get_val("GSI_MLC_PA_MLP")
    cell_mlc_mlp, cell_gsi_mlp = format_cell(cov_mlc_mlp, f1_mlc_mlp, cov_gsi_mlp, f1_gsi_mlp)

    rows.append(f"| **{d.upper()}** | {cell_mlc_log} | {cell_gsi_log} | {cell_mlc_svm} | {cell_gsi_svm} | {cell_mlc_mlp} | {cell_gsi_mlp} |")

# Mean row
mean_df = grouped.groupby("Model")[["Coverage", "Selective Macro-F1"]].mean()
cov_mlc_log, f1_mlc_log = mean_df.loc["MLC_PA_Logistic"]
cov_gsi_log, f1_gsi_log = mean_df.loc["GSI_MLC_PA_Logistic"]
cell_mlc_log, cell_gsi_log = format_cell(cov_mlc_log, f1_mlc_log, cov_gsi_log, f1_gsi_log)

cov_mlc_svm, f1_mlc_svm = mean_df.loc["MLC_PA_SVM"]
cov_gsi_svm, f1_gsi_svm = mean_df.loc["GSI_MLC_PA_SVM"]
cell_mlc_svm, cell_gsi_svm = format_cell(cov_mlc_svm, f1_mlc_svm, cov_gsi_svm, f1_gsi_svm)

cov_mlc_mlp, f1_mlc_mlp = mean_df.loc["MLC_PA_MLP"]
cov_gsi_mlp, f1_gsi_mlp = mean_df.loc["GSI_MLC_PA_MLP"]
cell_mlc_mlp, cell_gsi_mlp = format_cell(cov_mlc_mlp, f1_mlc_mlp, cov_gsi_mlp, f1_gsi_mlp)

mean_row = f"| **TRUNG BÌNH** | {cell_mlc_log} | {cell_gsi_log} | {cell_mlc_svm} | {cell_gsi_svm} | {cell_mlc_mlp} | {cell_gsi_mlp} |"

print("\n".join(rows))
print(mean_row)
