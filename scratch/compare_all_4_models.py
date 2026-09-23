import pandas as pd
import numpy as np

comp_path = "results_pa_v4/tables/434d53dc787206d6/complete_metrics.csv"
sel_path = "results_pa_v4/tables/434d53dc787206d6/selective_metrics.csv"

comp_df = pd.read_csv(comp_path)
sel_df = pd.read_csv(sel_path)
sel_30 = sel_df[np.isclose(sel_df["Cost"], 0.30)]

comp_mean = comp_df.groupby(["Dataset", "Model"])[["Macro-F1", "Hamming Loss", "Subset Accuracy", "Micro-F1"]].mean().reset_index()
sel_mean = sel_30.groupby(["Dataset", "Model"])[["Coverage", "Selective Macro-F1"]].mean().reset_index()

datasets = sorted(comp_df["Dataset"].unique())

def get_metrics(dataset, model):
    c_row = comp_mean[(comp_mean["Dataset"] == dataset) & (comp_mean["Model"] == model)]
    s_row = sel_mean[(sel_mean["Dataset"] == dataset) & (sel_mean["Model"] == model)]
    
    if s_row.empty:
        # For BR / CC: Coverage is 1.0, Selective F1 is Full Macro-F1
        cov = 1.0
        sel_f1 = c_row["Macro-F1"].values[0] if not c_row.empty else 0.0
    else:
        cov = s_row["Coverage"].values[0]
        sel_f1 = s_row["Selective Macro-F1"].values[0]
        
    full_f1 = c_row["Macro-F1"].values[0] if not c_row.empty else 0.0
    sub_acc = c_row["Subset Accuracy"].values[0] if not c_row.empty else 0.0
    ham_loss = c_row["Hamming Loss"].values[0] if not c_row.empty else 0.0
    return cov, sel_f1, full_f1, sub_acc, ham_loss

print("="*110)
print("COMPARISON: BR vs CC vs MLC-PA vs GSI-MLC-PA (Selective: Cov / F1 at c=0.30, where BR/CC has Cov=1.0)")
print("="*110)

base_learners = ["Logistic", "SVM", "MLP"]

for bl in base_learners:
    print(f"\n### BASE LEARNER: {bl.upper()}")
    models = [f"BR_{bl}", f"CC_{bl}", f"MLC_PA_{bl}", f"GSI_MLC_PA_{bl}"]
    header = f"| {'Dataset':<16} | {models[0]:<18} | {models[1]:<18} | {models[2]:<18} | {models[3]:<18} |"
    print(header)
    print("|:" + "-"*16 + "-|-:" + "-"*16 + ":|-:" + "-"*16 + ":|-:" + "-"*16 + ":|-:" + "-"*16 + ":|")
    
    cov_list = {m: [] for m in models}
    f1_list = {m: [] for m in models}
    
    for d in datasets:
        vals = {m: get_metrics(d, m) for m in models}
        for m in models:
            cov_list[m].append(vals[m][0])
            f1_list[m].append(vals[m][1])
            
        # Find best cov and best f1 among the 4 models
        max_cov = max(vals[m][0] for m in models)
        max_f1 = max(vals[m][1] for m in models)
        
        cells = []
        for m in models:
            cov, sel_f1 = vals[m][0], vals[m][1]
            c_str = f"**{cov:.3f}**" if np.isclose(cov, max_cov, atol=1e-3) else f"{cov:.3f}"
            f_str = f"**{sel_f1:.3f}**" if np.isclose(sel_f1, max_f1, atol=1e-3) else f"{sel_f1:.3f}"
            cells.append(f"{c_str} / {f_str}")
        print(f"| {d.upper():<16} | {cells[0]:<18} | {cells[1]:<18} | {cells[2]:<18} | {cells[3]:<18} |")
        
    # Mean row
    mean_cov = {m: np.mean(cov_list[m]) for m in models}
    mean_f1 = {m: np.mean(f1_list[m]) for m in models}
    max_mcov = max(mean_cov.values())
    max_mf1 = max(mean_f1.values())
    mean_cells = []
    for m in models:
        c_str = f"**{mean_cov[m]:.3f}**" if np.isclose(mean_cov[m], max_mcov, atol=1e-3) else f"{mean_cov[m]:.3f}"
        f_str = f"**{mean_f1[m]:.3f}**" if np.isclose(mean_f1[m], max_mf1, atol=1e-3) else f"{mean_f1[m]:.3f}"
        mean_cells.append(f"{c_str} / {f_str}")
    print(f"| {'TRUNG BÌNH':<16} | {mean_cells[0]:<18} | {mean_cells[1]:<18} | {mean_cells[2]:<18} | {mean_cells[3]:<18} |")
