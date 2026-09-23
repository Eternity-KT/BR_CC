import pandas as pd
import numpy as np

csv_path = "results_pa_v4/tables/434d53dc787206d6/selective_metrics.csv"
df = pd.read_csv(csv_path)

# Filter for Cost == 0.30
df_30 = df[np.isclose(df["Cost"], 0.30)]

# Group by Dataset and Model to get mean Coverage and Selective Macro-F1
summary = df_30.groupby(["Dataset", "Model"])[["Coverage", "Selective Macro-F1", "Generalized Loss", "Optimistic Macro-F1"]].mean().reset_index()

print("="*90)
print(f"{'Dataset':<16} | {'Model':<20} | {'Coverage':<10} | {'Selective F1':<14} | {'Gen Loss':<10}")
print("="*90)

models_of_interest = [
    "MLC_PA_MLP", "GSI_MLC_PA_MLP",
    "MLC_PA_Logistic", "GSI_MLC_PA_Logistic",
    "MLC_PA_SVM", "GSI_MLC_PA_SVM"
]

for d in summary["Dataset"].unique():
    d_df = summary[summary["Dataset"] == d]
    print(f"\n--- {d.upper()} ---")
    for m in models_of_interest:
        row = d_df[d_df["Model"] == m]
        if not row.empty:
            cov = row["Coverage"].values[0]
            f1 = row["Selective Macro-F1"].values[0]
            loss = row["Generalized Loss"].values[0]
            print(f"{d:<16} | {m:<20} | {cov*100:6.2f}%    | {f1:14.4f} | {loss:10.4f}")

print("\n" + "="*90)
print("COMPLETE METRICS (Full Macro-F1, Full Hamming Loss):")
print("="*90)

comp_path = "results_pa_v4/tables/434d53dc787206d6/complete_metrics.csv"
comp_df = pd.read_csv(comp_path)
comp_summary = comp_df.groupby(["Dataset", "Model"])[["Macro-F1", "Hamming Loss", "Subset Accuracy", "Micro-F1"]].mean().reset_index()

for m in models_of_interest:
    m_comp = comp_summary[comp_summary["Model"] == m]
    if not m_comp.empty:
        f1_mean = m_comp["Macro-F1"].mean()
        ham_mean = m_comp["Hamming Loss"].mean()
        sub_mean = m_comp["Subset Accuracy"].mean()
        mic_mean = m_comp["Micro-F1"].mean()
        print(f"{m:<20} | Full Macro-F1: {f1_mean:8.4f} | Hamming Loss: {ham_mean:8.4f} | Subset Acc: {sub_mean:8.4f} | Micro-F1: {mic_mean:8.4f}")

