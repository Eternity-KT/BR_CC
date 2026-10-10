import pandas as pd
import numpy as np

# Rows from task-276 (emotions, scene, plantpseaac, humanpseaac)
rows_batch1 = [
    {
        "dataset": "emotions", "learner": "Logistic", "model": "GSI_v6_2",
        "Full_Macro_F1_mean": 0.639894, "Full_Macro_F1_std": 0.023275,
        "Selective_Macro_F1_mean": 0.688263, "Selective_Macro_F1_std": 0.018366,
        "Coverage_mean": 0.691088, "Coverage_std": 0.017610,
        "Full_Micro_F1_mean": 0.657085, "Full_Micro_F1_std": 0.012952,
        "Selective_Micro_F1_mean": 0.729297, "Selective_Micro_F1_std": 0.016457,
        "Hamming_Loss_mean": 0.194331, "Hamming_Loss_std": 0.006175,
        "Selective_Hamming_Loss_mean": 0.111303, "Selective_Hamming_Loss_std": 0.010796,
        "Subset_Accuracy_mean": 0.290646, "Subset_Accuracy_std": 0.027305
    },
    {
        "dataset": "emotions", "learner": "Logistic", "model": "GSI_v6_3",
        "Full_Macro_F1_mean": 0.676650, "Full_Macro_F1_std": 0.018132,
        "Selective_Macro_F1_mean": 0.722436, "Selective_Macro_F1_std": 0.023400,
        "Coverage_mean": 0.735484, "Coverage_std": 0.013016,
        "Full_Micro_F1_mean": 0.676637, "Full_Micro_F1_std": 0.018415,
        "Selective_Micro_F1_mean": 0.716111, "Selective_Micro_F1_std": 0.025458,
        "Hamming_Loss_mean": 0.234517, "Hamming_Loss_std": 0.008403,
        "Selective_Hamming_Loss_mean": 0.262998, "Selective_Hamming_Loss_std": 0.019792,
        "Subset_Accuracy_mean": 0.229010, "Subset_Accuracy_std": 0.043140
    },
    {
        "dataset": "emotions", "learner": "Logistic", "model": "GSI_v6_3_1",
        "Full_Macro_F1_mean": 0.678489, "Full_Macro_F1_std": 0.012101,
        "Selective_Macro_F1_mean": 0.755206, "Selective_Macro_F1_std": 0.022315,
        "Coverage_mean": 0.740560, "Coverage_std": 0.029168,
        "Full_Micro_F1_mean": 0.686562, "Full_Micro_F1_std": 0.007784,
        "Selective_Micro_F1_mean": 0.763073, "Selective_Micro_F1_std": 0.017202,
        "Hamming_Loss_mean": 0.199064, "Hamming_Loss_std": 0.004580,
        "Selective_Hamming_Loss_mean": 0.174580, "Selective_Hamming_Loss_std": 0.014109,
        "Subset_Accuracy_mean": 0.302917, "Subset_Accuracy_std": 0.028021
    },
    {
        "dataset": "scene", "learner": "Logistic", "model": "GSI_v6_2",
        "Full_Macro_F1_mean": 0.714033, "Full_Macro_F1_std": 0.022352,
        "Selective_Macro_F1_mean": 0.764225, "Selective_Macro_F1_std": 0.015087,
        "Coverage_mean": 0.888405, "Coverage_std": 0.007204,
        "Full_Micro_F1_mean": 0.708304, "Full_Micro_F1_std": 0.022545,
        "Selective_Micro_F1_mean": 0.778898, "Selective_Micro_F1_std": 0.018940,
        "Hamming_Loss_mean": 0.095392, "Hamming_Loss_std": 0.005976,
        "Selective_Hamming_Loss_mean": 0.057199, "Selective_Hamming_Loss_std": 0.004608,
        "Subset_Accuracy_mean": 0.570590, "Subset_Accuracy_std": 0.032588
    },
    {
        "dataset": "scene", "learner": "Logistic", "model": "GSI_v6_3",
        "Full_Macro_F1_mean": 0.701035, "Full_Macro_F1_std": 0.016805,
        "Selective_Macro_F1_mean": 0.726599, "Selective_Macro_F1_std": 0.014745,
        "Coverage_mean": 0.736905, "Coverage_std": 0.011452,
        "Full_Micro_F1_mean": 0.685341, "Full_Micro_F1_std": 0.015050,
        "Selective_Micro_F1_mean": 0.709503, "Selective_Micro_F1_std": 0.013022,
        "Hamming_Loss_mean": 0.142471, "Hamming_Loss_std": 0.007314,
        "Selective_Hamming_Loss_mean": 0.173721, "Selective_Hamming_Loss_std": 0.008205,
        "Subset_Accuracy_mean": 0.396448, "Subset_Accuracy_std": 0.018909
    },
    {
        "dataset": "scene", "learner": "Logistic", "model": "GSI_v6_3_1",
        "Full_Macro_F1_mean": 0.729202, "Full_Macro_F1_std": 0.020541,
        "Selective_Macro_F1_mean": 0.797442, "Selective_Macro_F1_std": 0.017821,
        "Coverage_mean": 0.760355, "Coverage_std": 0.010736,
        "Full_Micro_F1_mean": 0.718850, "Full_Micro_F1_std": 0.018530,
        "Selective_Micro_F1_mean": 0.790386, "Selective_Micro_F1_std": 0.017482,
        "Hamming_Loss_mean": 0.106478, "Hamming_Loss_std": 0.006871,
        "Selective_Hamming_Loss_mean": 0.094802, "Selective_Hamming_Loss_std": 0.007626,
        "Subset_Accuracy_mean": 0.556395, "Subset_Accuracy_std": 0.023607
    },
    {
        "dataset": "plantpseaac", "learner": "Logistic", "model": "GSI_v6_2",
        "Full_Macro_F1_mean": 0.143489, "Full_Macro_F1_std": 0.017106,
        "Selective_Macro_F1_mean": 0.095125, "Selective_Macro_F1_std": 0.030369,
        "Coverage_mean": 0.929295, "Coverage_std": 0.003690,
        "Full_Micro_F1_mean": 0.256718, "Full_Micro_F1_std": 0.029259,
        "Selective_Micro_F1_mean": 0.196405, "Selective_Micro_F1_std": 0.029698,
        "Hamming_Loss_mean": 0.094756, "Hamming_Loss_std": 0.001294,
        "Selective_Hamming_Loss_mean": 0.070696, "Selective_Hamming_Loss_std": 0.001304,
        "Subset_Accuracy_mean": 0.137976, "Subset_Accuracy_std": 0.013690
    },
    {
        "dataset": "plantpseaac", "learner": "Logistic", "model": "GSI_v6_3",
        "Full_Macro_F1_mean": 0.247231, "Full_Macro_F1_std": 0.009228,
        "Selective_Macro_F1_mean": 0.222108, "Selective_Macro_F1_std": 0.005552,
        "Coverage_mean": 0.702946, "Coverage_std": 0.014031,
        "Full_Micro_F1_mean": 0.271949, "Full_Micro_F1_std": 0.008587,
        "Selective_Micro_F1_mean": 0.235473, "Selective_Micro_F1_std": 0.003531,
        "Hamming_Loss_mean": 0.311635, "Hamming_Loss_std": 0.007460,
        "Selective_Hamming_Loss_mean": 0.688238, "Selective_Hamming_Loss_std": 0.014516,
        "Subset_Accuracy_mean": 0.010284, "Subset_Accuracy_std": 0.006591
    },
    {
        "dataset": "plantpseaac", "learner": "Logistic", "model": "GSI_v6_3_1",
        "Full_Macro_F1_mean": 0.204183, "Full_Macro_F1_std": 0.021250,
        "Selective_Macro_F1_mean": 0.271776, "Selective_Macro_F1_std": 0.026968,
        "Coverage_mean": 0.690667, "Coverage_std": 0.011100,
        "Full_Micro_F1_mean": 0.303235, "Full_Micro_F1_std": 0.027832,
        "Selective_Micro_F1_mean": 0.429855, "Selective_Micro_F1_std": 0.025849,
        "Hamming_Loss_mean": 0.102744, "Hamming_Loss_std": 0.002943,
        "Selective_Hamming_Loss_mean": 0.085802, "Selective_Hamming_Loss_std": 0.005421,
        "Subset_Accuracy_mean": 0.148245, "Subset_Accuracy_std": 0.015197
    },
    {
        "dataset": "humanpseaac", "learner": "Logistic", "model": "GSI_v6_2",
        "Full_Macro_F1_mean": 0.112385, "Full_Macro_F1_std": 0.011044,
        "Selective_Macro_F1_mean": 0.086887, "Selective_Macro_F1_std": 0.007694,
        "Coverage_mean": 0.922448, "Coverage_std": 0.001791,
        "Full_Micro_F1_mean": 0.285116, "Full_Micro_F1_std": 0.027386,
        "Selective_Micro_F1_mean": 0.191837, "Selective_Micro_F1_std": 0.011868,
        "Hamming_Loss_mean": 0.086046, "Hamming_Loss_std": 0.002926,
        "Selective_Hamming_Loss_mean": 0.057747, "Selective_Hamming_Loss_std": 0.002230,
        "Subset_Accuracy_mean": 0.152639, "Subset_Accuracy_std": 0.015710
    },
    {
        "dataset": "humanpseaac", "learner": "Logistic", "model": "GSI_v6_3",
        "Full_Macro_F1_mean": 0.203612, "Full_Macro_F1_std": 0.003131,
        "Selective_Macro_F1_mean": 0.183577, "Selective_Macro_F1_std": 0.003965,
        "Coverage_mean": 0.703262, "Coverage_std": 0.004994,
        "Full_Micro_F1_mean": 0.246349, "Full_Micro_F1_std": 0.010200,
        "Selective_Micro_F1_mean": 0.204428, "Selective_Micro_F1_std": 0.009203,
        "Hamming_Loss_mean": 0.338987, "Hamming_Loss_std": 0.013503,
        "Selective_Hamming_Loss_mean": 0.768132, "Selective_Hamming_Loss_std": 0.038988,
        "Subset_Accuracy_mean": 0.003539, "Subset_Accuracy_std": 0.000624
    },
    {
        "dataset": "humanpseaac", "learner": "Logistic", "model": "GSI_v6_3_1",
        "Full_Macro_F1_mean": 0.171564, "Full_Macro_F1_std": 0.016437,
        "Selective_Macro_F1_mean": 0.232530, "Selective_Macro_F1_std": 0.021006,
        "Coverage_mean": 0.702060, "Coverage_std": 0.005754,
        "Full_Micro_F1_mean": 0.340688, "Full_Micro_F1_std": 0.026493,
        "Selective_Micro_F1_mean": 0.470337, "Selective_Micro_F1_std": 0.027896,
        "Hamming_Loss_mean": 0.091733, "Hamming_Loss_std": 0.003036,
        "Selective_Hamming_Loss_mean": 0.075922, "Selective_Hamming_Loss_std": 0.003111,
        "Subset_Accuracy_mean": 0.185107, "Subset_Accuracy_std": 0.017548
    }
]

# Yeast rows
df_yeast = pd.read_csv("results_v6_3/v6_3_1_comparison_summary.csv")
rows_yeast = df_yeast.to_dict(orient="records")

all_rows = rows_batch1 + rows_yeast
df_master = pd.DataFrame(all_rows)
df_master.to_csv("results_v6_3/v6_3_1_master_summary.csv", index=False)
print("Successfully generated master summary with all 5 datasets!")
