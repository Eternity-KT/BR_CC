"""
Benchmark evaluation for GSI-MLC-PA v6.1.1 on 5 datasets.
Compares:
1. v5.1 Baseline (Peeling + Dense CC)
2. v5.1.1 (Peeling + Sparse CC theta=0.75)
3. v6.1 (5-Fold CV Peeling + ECC)
4. v6.1.1 (v5.1 Peeling + ECC on DL)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.data.loader import load_dataset
from src.evaluation.cv import get_multilabel_cv
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_selective_macro_f1,
)
from src.models.gsi_v6_1_1 import GSIMLCPAv6_1_1Classifier

FIRST_5_DATASETS = ["emotions", "scene", "chd49", "music", "gpositivepseaac"]
BASE_LEARNER = "logistic"
N_FOLDS = 5
RANDOM_STATE = 42
COST = 0.30

# Load v5.1 baseline results
v5_df = pd.read_csv(WORKSPACE_ROOT / "results_v5_1_test/benchmark_3_base_learners_summary.csv")
v5_df = v5_df[v5_df["Model"].str.contains("GSI_v5_1_Stratified_Logistic")].copy()
v5_map = {r["Dataset"].strip(): r["Selective_Macro_F1"] for _, r in v5_df.iterrows()}
v5_cov_map = {r["Dataset"].strip(): r["Coverage"] for _, r in v5_df.iterrows()}
v5_full_map = {r["Dataset"].strip(): r["Full_Macro_F1"] for _, r in v5_df.iterrows()}

# Load v5.1.1 baseline results from layer_ablation_logistic.csv
v511_df = pd.read_csv(WORKSPACE_ROOT / "results_v5_1_test/layer_ablation_logistic.csv")
v511_full = v511_df[v511_df["Regime"] == "FULL_SYSTEM"].copy()
v511_map = {r["Dataset"].strip(): r["selective_macro_f1"] for _, r in v511_full.iterrows()}
v511_cov_map = {r["Dataset"].strip(): r["coverage"] for _, r in v511_full.iterrows()}
v511_full_map = {r["Dataset"].strip(): r["full_macro_f1"] for _, r in v511_full.iterrows()}

# Load v6.1 results
v61_df = pd.read_csv(WORKSPACE_ROOT / "results_v6_1/v6_1_summary.csv")
v61_df = v61_df[(v61_df["model"] == "GSI_v6_1") & (v61_df["base_learner"] == "Logistic")].copy()
v61_map = {r["dataset"].strip(): r["selective_macro_f1"] for _, r in v61_df.iterrows()}
v61_cov_map = {r["dataset"].strip(): r["coverage"] for _, r in v61_df.iterrows()}
v61_full_map = {r["dataset"].strip(): r["full_macro_f1"] for _, r in v61_df.iterrows()}

results = []

print("=" * 90)
print("CHẠY THỰC NGHIỆM GSI-MLC-PA v6.1.1 (v5.1 PEELING + ECC ON DL) TRÊN 5 TẬP DỮ LIỆU")
print("=" * 90)

for d_idx, dataset_name in enumerate(FIRST_5_DATASETS, 1):
    print(f"\n[{d_idx}/5] Đang xử lý tập dữ liệu: {dataset_name} ...", flush=True)
    X, Y, _, _ = load_dataset(dataset_name, base_dir=str(WORKSPACE_ROOT))
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int32)
    n_samples, n_labels = Y.shape

    cv = get_multilabel_cv(n_splits=N_FOLDS, random_state=RANDOM_STATE)

    fold_sel_f1 = []
    fold_full_f1 = []
    fold_cov = []
    fold_il_counts = []
    fold_subset_acc = []
    fold_hamm_loss = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, Y)):
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]

        model = GSIMLCPAv6_1_1Classifier(
            base_learner=BASE_LEARNER,
            stratified_threshold=0.70,
            validation_size=0.20,
            max_peeling_depth=3,
            aug_normalization="matching",
            n_chains=10,
            cost=COST,
            random_state=RANDOM_STATE + fold_idx * 17,
        )
        model.fit(X_train, Y_train)

        # Full predictions
        full_preds = model.predict_full(X_test)
        full_metrics = compute_all_metrics(Y_test, full_preds)
        fold_full_f1.append(full_metrics["Macro-F1"])

        # Selective predictions with abstention at cost c = 0.30
        sel_preds = model.predict(X_test, cost=COST)
        sel_f1 = compute_selective_macro_f1(Y_test, sel_preds, abstain_value=-1)
        cov = float(np.mean(sel_preds != -1))

        fold_sel_f1.append(sel_f1)
        fold_cov.append(cov)
        fold_il_counts.append(len(model.independent_labels_))
        fold_subset_acc.append(full_metrics["Subset Accuracy"])
        fold_hamm_loss.append(full_metrics["Hamming Loss"])

    row = {
        "dataset": dataset_name,
        "n_samples": n_samples,
        "n_labels": n_labels,
        "mean_il": np.mean(fold_il_counts),
        # v5.1
        "v5_1_sel_f1": v5_map.get(dataset_name, np.nan),
        "v5_1_full_f1": v5_full_map.get(dataset_name, np.nan),
        "v5_1_cov": v5_cov_map.get(dataset_name, np.nan),
        # v5.1.1
        "v5_1_1_sel_f1": v511_map.get(dataset_name, np.nan),
        "v5_1_1_full_f1": v511_full_map.get(dataset_name, np.nan),
        "v5_1_1_cov": v511_cov_map.get(dataset_name, np.nan),
        # v6.1
        "v6_1_sel_f1": v61_map.get(dataset_name, np.nan),
        "v6_1_full_f1": v61_full_map.get(dataset_name, np.nan),
        "v6_1_cov": v61_cov_map.get(dataset_name, np.nan),
        # v6.1.1
        "v6_1_1_sel_f1": np.mean(fold_sel_f1),
        "v6_1_1_full_f1": np.mean(fold_full_f1),
        "v6_1_1_cov": np.mean(fold_cov),
        "v6_1_1_subset_acc": np.mean(fold_subset_acc),
        "v6_1_1_hamming_loss": np.mean(fold_hamm_loss),
    }
    results.append(row)

    print(f"  -> v5.1.1: Sel-F1 = {row['v5_1_1_sel_f1']:.4f} | Full-F1 = {row['v5_1_1_full_f1']:.4f} | Cov = {row['v5_1_1_cov']:.2%}")
    print(f"  -> v6.1.1: Sel-F1 = {row['v6_1_1_sel_f1']:.4f} | Full-F1 = {row['v6_1_1_full_f1']:.4f} | Cov = {row['v6_1_1_cov']:.2%} | IL={row['mean_il']:.1f}/{n_labels}")

df_res = pd.DataFrame(results)
print("\n" + "=" * 90)
print("BẢNG ĐỐI SÁNH SELECTIVE MACRO-F1: v6.1.1 vs. v5.1.1 và v5.1")
print("=" * 90)
cols_sel = ["dataset", "v5_1_sel_f1", "v5_1_1_sel_f1", "v6_1_sel_f1", "v6_1_1_sel_f1"]
print(df_res[cols_sel].to_string(index=False))

print("\n" + "=" * 90)
print("BẢNG ĐỐI SÁNH FULL MACRO-F1: v6.1.1 vs. v5.1.1 và v5.1")
print("=" * 90)
cols_full = ["dataset", "v5_1_full_f1", "v5_1_1_full_f1", "v6_1_full_f1", "v6_1_1_full_f1"]
print(df_res[cols_full].to_string(index=False))

print("\n" + "=" * 90)
print("BẢNG ĐỐI SÁNH COVERAGE: v6.1.1 vs. v5.1.1 và v5.1")
print("=" * 90)
cols_cov = ["dataset", "v5_1_cov", "v5_1_1_cov", "v6_1_cov", "v6_1_1_cov"]
print(df_res[cols_cov].to_string(index=False))

print("\n" + "=" * 90)
print("TRUNG BÌNH TRÊN 5 TẬP DỮ LIỆU:")
print("=" * 90)
print(f"v5.1:   Selective F1 = {df_res['v5_1_sel_f1'].mean():.4f} | Full F1 = {df_res['v5_1_full_f1'].mean():.4f} | Coverage = {df_res['v5_1_cov'].mean():.2%}")
print(f"v5.1.1: Selective F1 = {df_res['v5_1_1_sel_f1'].mean():.4f} | Full F1 = {df_res['v5_1_1_full_f1'].mean():.4f} | Coverage = {df_res['v5_1_1_cov'].mean():.2%}")
print(f"v6.1:   Selective F1 = {df_res['v6_1_sel_f1'].mean():.4f} | Full F1 = {df_res['v6_1_full_f1'].mean():.4f} | Coverage = {df_res['v6_1_cov'].mean():.2%}")
print(f"v6.1.1: Selective F1 = {df_res['v6_1_1_sel_f1'].mean():.4f} | Full F1 = {df_res['v6_1_1_full_f1'].mean():.4f} | Coverage = {df_res['v6_1_1_cov'].mean():.2%}")

output_csv = WORKSPACE_ROOT / "scratch/v6_1_1_benchmark_5datasets.csv"
df_res.to_csv(output_csv, index=False)
print(f"\nĐã lưu kết quả chi tiết tại: {output_csv}")
