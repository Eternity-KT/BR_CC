import pandas as pd
import numpy as np

comp_path = "results_pa_v4/tables/434d53dc787206d6/complete_metrics.csv"
comp_df = pd.read_csv(comp_path)

mean_comp = comp_df.groupby("Model")[["Macro-F1", "Hamming Loss", "Subset Accuracy", "Micro-F1"]].mean()

base_learners = ["Logistic", "SVM", "MLP"]
for bl in base_learners:
    models = [f"BR_{bl}", f"CC_{bl}", f"MLC_PA_{bl}", f"GSI_MLC_PA_{bl}"]
    print(f"\n--- {bl.upper()} ---")
    sub = mean_comp.loc[models]
    print(sub.to_string())
