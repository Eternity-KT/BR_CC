"""Benchmark Experiment for GSI-MLC-PA v5.1 on the Remaining 5 Datasets and Full 10-Dataset Evaluation.

Evaluates the new Data-Driven Stratified Peeling Partitioning and Ascending-Correlation
Classifier Chain (without complexity penalty) against GSI v5 (Greedy), standard BR,
and standard CC across the remaining 5 datasets:
1. genbase (662 samples, 1186 features, 27 labels)
2. humanpseaac (3106 samples, 440 features, 14 labels)
3. plantpseaac (978 samples, 440 features, 12 labels)
4. viruspseaac (207 samples, 440 features, 6 labels)
5. yeast (2417 samples, 103 features, 14 labels)

Then combines with the first 5 datasets (emotions, scene, chd49, music, gpositivepseaac)
to produce the complete 10-dataset benchmark report.

Protocol:
---------
- 5-Fold Multilabel Stratified Cross-Validation (random_state=42).
- MaxAbsScaler fit strictly on training fold, applied to test fold.
- Evaluation at rejection cost c = 0.30 (Selective) and c = 0.50 (Full).
- Outputs saved to results_v5_1_test/
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.data.loader import load_dataset
from src.evaluation.cv import get_multilabel_cv
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_selective_macro_f1,
    compute_selective_micro_f1,
)
from src.models.binary_relevance import BinaryRelevanceClassifier
from src.models.classifier_chain import ClassifierChainClassifier
from src.models.gsi_mlc_pa import GSIMLCPartialAbstentionClassifier

REMAINING_DATASETS = [
    "genbase",
    "humanpseaac",
    "plantpseaac",
    "viruspseaac",
    "yeast",
]

FIRST_5_DATASETS = [
    "emotions",
    "scene",
    "chd49",
    "music",
    "gpositivepseaac",
]

ALL_10_DATASETS = FIRST_5_DATASETS + REMAINING_DATASETS

OUTPUT_DIR = Path("results_v5_1_test")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def evaluate_fold_predictions(
    y_test: np.ndarray,
    model,
    X_test: np.ndarray,
    cost: float = 0.30,
) -> Dict[str, float]:
    """Compute all full and selective metrics for one model on test data."""
    if hasattr(model, "predict_full"):
        full_preds = model.predict_full(X_test)
        sel_preds = model.predict(X_test, cost=cost)
    elif hasattr(model, "predict"):
        full_preds = model.predict(X_test)
        sel_preds = full_preds
    else:
        raise ValueError("Model does not expose predict()")

    full_metrics = compute_all_metrics(y_test, full_preds)

    if isinstance(model, GSIMLCPartialAbstentionClassifier):
        sel_macro_f1 = compute_selective_macro_f1(y_test, sel_preds)
        sel_micro_f1 = compute_selective_micro_f1(y_test, sel_preds)
        coverage = float(np.mean(sel_preds != -1))
    else:
        sel_macro_f1 = full_metrics["Macro-F1"]
        sel_micro_f1 = full_metrics["Micro-F1"]
        coverage = 1.0

    return {
        "Full_Macro_F1": float(full_metrics["Macro-F1"]),
        "Selective_Macro_F1": float(sel_macro_f1),
        "Coverage": float(coverage),
        "Full_Micro_F1": float(full_metrics["Micro-F1"]),
        "Selective_Micro_F1": float(sel_micro_f1),
        "Hamming_Loss": float(full_metrics["Hamming Loss"]),
        "Subset_Accuracy": float(full_metrics["Subset Accuracy"]),
    }


def run_remaining_5_benchmark(
    threshold: float = 0.70,
    base_learner: str = "logistic",
    n_splits: int = 5,
    random_state: int = 42,
):
    print("=" * 80)
    print("STARTING GSI-MLC-PA v5.1 BENCHMARK ON REMAINING 5 DATASETS (NO PENALTY)")
    print(f"Base Learner: {base_learner} | IL Threshold: {threshold} | CV: {n_splits}-Fold")
    print(f"Datasets: {REMAINING_DATASETS}")
    print("=" * 80)

    dataset_summaries = []

    for dataset_name in REMAINING_DATASETS:
        print(f"\n---> Loading dataset: {dataset_name} ...", flush=True)
        X, Y, feat_names, label_names = load_dataset(dataset_name)
        n_samples, n_labels = Y.shape
        print(f"     Samples: {n_samples}, Features: {X.shape[1]}, Labels: {n_labels}", flush=True)

        cv = get_multilabel_cv(n_splits=n_splits, random_state=random_state)

        model_names = [
            "BR",
            "CC",
            "GSI_v5_Greedy",
            "GSI_v5_1_Stratified",
        ]

        fold_results = {m: [] for m in model_names}
        fold_times = {m: [] for m in model_names}
        fold_il_counts = {m: [] for m in model_names}
        fold_stages = {m: [] for m in model_names}
        fold_reasons = {m: [] for m in model_names}

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, Y)):
            print(f"       Fold {fold_idx + 1}/{n_splits} running...", flush=True)
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            # Fit scaler strictly on train fold
            scaler = MaxAbsScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # 1. Baseline: BR
            t0 = time.perf_counter()
            model_br = BinaryRelevanceClassifier(base_estimator=base_learner, random_state=random_state)
            model_br.fit(X_train, Y_train)
            t_br = time.perf_counter() - t0
            res_br = evaluate_fold_predictions(Y_test, model_br, X_test)
            fold_results["BR"].append(res_br)
            fold_times["BR"].append(t_br)
            fold_il_counts["BR"].append(n_labels)
            fold_stages["BR"].append(1)
            fold_reasons["BR"].append("fixed_br")

            # 2. Baseline: CC
            t0 = time.perf_counter()
            model_cc = ClassifierChainClassifier(base_estimator=base_learner, random_state=random_state)
            model_cc.fit(X_train, Y_train)
            t_cc = time.perf_counter() - t0
            res_cc = evaluate_fold_predictions(Y_test, model_cc, X_test)
            fold_results["CC"].append(res_cc)
            fold_times["CC"].append(t_cc)
            fold_il_counts["CC"].append(0)
            fold_stages["CC"].append(1)
            fold_reasons["CC"].append("fixed_cc")

            # 3. GSI v5 (Greedy Macro-F1 + Descending Tree Order)
            t0 = time.perf_counter()
            model_v5 = GSIMLCPartialAbstentionClassifier(
                base_learner=base_learner,
                partition_mode="learned",
                final_order="correlation",
                selection_objective="full_macro_f1",
                decision_policy="macro_f1",
                cost=0.30,
                random_state=random_state,
            )
            model_v5.fit(X_train, Y_train)
            t_v5 = time.perf_counter() - t0
            res_v5 = evaluate_fold_predictions(Y_test, model_v5, X_test)
            fold_results["GSI_v5_Greedy"].append(res_v5)
            fold_times["GSI_v5_Greedy"].append(t_v5)
            fold_il_counts["GSI_v5_Greedy"].append(len(getattr(model_v5, "independent_labels_", [])))
            fold_stages["GSI_v5_Greedy"].append(n_labels)
            fold_reasons["GSI_v5_Greedy"].append("greedy_forward")

            # 4. GSI v5.1 Stratified (Stratified Peeling + Ascending Correlation CC, NO PENALTY)
            t0 = time.perf_counter()
            model_v5_1 = GSIMLCPartialAbstentionClassifier(
                base_learner=base_learner,
                partition_mode="stratified_peeling",
                stratified_threshold=threshold,
                dl_order_direction="ascending",
                final_order="ascending_correlation",
                decision_policy="macro_f1",
                cost=0.30,
                random_state=random_state,
                use_complexity_penalty=False,
            )
            model_v5_1.fit(X_train, Y_train)
            t_v5_1 = time.perf_counter() - t0
            res_v5_1 = evaluate_fold_predictions(Y_test, model_v5_1, X_test)
            fold_results["GSI_v5_1_Stratified"].append(res_v5_1)
            fold_times["GSI_v5_1_Stratified"].append(t_v5_1)
            fold_il_counts["GSI_v5_1_Stratified"].append(
                len(getattr(model_v5_1, "independent_labels_", []))
            )
            audit_v5_1 = getattr(model_v5_1, "stratified_peeling_audit_", {}) or {}
            fold_stages["GSI_v5_1_Stratified"].append(audit_v5_1.get("num_stages_executed", 1))
            fold_reasons["GSI_v5_1_Stratified"].append(audit_v5_1.get("stopping_reason", "n/a"))

        # Summarize folds for this dataset
        print(f"     Results for {dataset_name} across 5 folds:", flush=True)
        for m in model_names:
            m_res = fold_results[m]
            mean_full_f1 = np.mean([r["Full_Macro_F1"] for r in m_res])
            mean_sel_f1 = np.mean([r["Selective_Macro_F1"] for r in m_res])
            mean_cov = np.mean([r["Coverage"] for r in m_res])
            mean_time = np.mean(fold_times[m])
            mean_il = np.mean(fold_il_counts[m])
            mean_st = np.mean(fold_stages[m])
            mode_reason = max(set(fold_reasons[m]), key=fold_reasons[m].count)

            summary_row = {
                "Dataset": dataset_name,
                "Model": m,
                "Full_Macro_F1": round(float(mean_full_f1), 4),
                "Selective_Macro_F1": round(float(mean_sel_f1), 4),
                "Coverage": round(float(mean_cov), 4),
                "Mean_IL_Count": round(float(mean_il), 1),
                "Total_Labels": int(n_labels),
                "Mean_Stages": round(float(mean_st), 1),
                "Stopping_Reason": str(mode_reason),
                "Time_Seconds": round(float(mean_time), 3),
            }
            dataset_summaries.append(summary_row)
            print(
                f"       {m:<24} | Sel F1: {mean_sel_f1:.4f} | Cov: {mean_cov:.3f} | "
                f"Full F1: {mean_full_f1:.4f} | IL: {mean_il:.1f}/{n_labels} | Stages: {mean_st:.1f} | Time: {mean_time:.3f}s",
                flush=True,
            )

    df_remaining = pd.DataFrame(dataset_summaries)
    rem_csv_path = OUTPUT_DIR / "benchmark_remaining_5datasets_summary.csv"
    df_remaining.to_csv(rem_csv_path, index=False)
    print(f"\n[OK] Remaining 5 datasets summary saved to: {rem_csv_path}", flush=True)

    # Load first 5 datasets summary if exists
    first5_csv = OUTPUT_DIR / "benchmark_5datasets_summary.csv"
    if first5_csv.exists():
        df_first5 = pd.read_csv(first5_csv)
        # Filter out WithPenalty and rename GSI_v5_1_NoPenalty to GSI_v5_1_Stratified
        df_first5_clean = df_first5[df_first5["Model"] != "GSI_v5_1_WithPenalty"].copy()
        df_first5_clean["Model"] = df_first5_clean["Model"].replace(
            {"GSI_v5_1_NoPenalty": "GSI_v5_1_Stratified"}
        )
        df_all_10 = pd.concat([df_first5_clean, df_remaining], ignore_index=True)
    else:
        df_all_10 = df_remaining.copy()

    all10_csv_path = OUTPUT_DIR / "benchmark_all_10datasets_summary.csv"
    df_all_10.to_csv(all10_csv_path, index=False)
    print(f"[OK] Full 10-Dataset summary saved to: {all10_csv_path}", flush=True)

    # 1. Report for Remaining 5 Datasets
    _generate_and_save_report(
        df_remaining,
        REMAINING_DATASETS,
        OUTPUT_DIR / "benchmark_remaining_5datasets_report.md",
        title="BÁO CÁO THỰC NGHIỆM GSI-MLC-PA V5.1 TRÊN 5 TẬP DỮ LIỆU CÒN LẠI (NO PENALTY)",
        base_learner=base_learner,
        threshold=threshold,
    )

    # 2. Report for All 10 Datasets
    _generate_and_save_report(
        df_all_10,
        ALL_10_DATASETS,
        OUTPUT_DIR / "benchmark_all_10datasets_report.md",
        title="BÁO CÁO THỰC NGHIỆM TOÀN DIỆN GSI-MLC-PA V5.1 TRÊN TOÀN BỘ 10 TẬP DỮ LIỆU (NO PENALTY)",
        base_learner=base_learner,
        threshold=threshold,
    )


def _generate_and_save_report(
    df: pd.DataFrame,
    dataset_list: List[str],
    output_path: Path,
    title: str,
    base_learner: str,
    threshold: float,
):
    md = []
    md.append(f"# {title}\n")
    md.append(f"**Cấu hình:** Base Learner = `{base_learner}`, Ngưỡng phân định độc lập $\\tau = {threshold}$, 5-Fold Stratified CV, Chi phí từ chối $c = 0.30$, Không dùng hàm phạt (`use_complexity_penalty=False`).\n")
    md.append("## 1. Bảng Tổng Hợp Selective Macro-F1 và Coverage (c = 0.30)\n")
    md.append("| STT | Tập dữ liệu | Số mẫu / Nhãn | BR (F1 / Cov) | CC (F1 / Cov) | GSI v5 Greedy (F1 / Cov) | GSI v5.1 Stratified (F1 / Cov) | Tăng trưởng vs. v5 Greedy |")
    md.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    models_needed = ["BR", "CC", "GSI_v5_Greedy", "GSI_v5_1_Stratified"]

    for idx, ds in enumerate(dataset_list, 1):
        sub = df[df["Dataset"] == ds].set_index("Model")
        if not all(m in sub.index for m in models_needed):
            continue
        tot_l = int(sub.loc["BR", "Total_Labels"])
        br_str = f"{sub.loc['BR', 'Selective_Macro_F1']:.4f} / 1.00"
        cc_str = f"{sub.loc['CC', 'Selective_Macro_F1']:.4f} / 1.00"
        v5_str = f"{sub.loc['GSI_v5_Greedy', 'Selective_Macro_F1']:.4f} / {sub.loc['GSI_v5_Greedy', 'Coverage']:.3f}"
        v5_1_f1 = sub.loc['GSI_v5_1_Stratified', 'Selective_Macro_F1']
        v5_1_cov = sub.loc['GSI_v5_1_Stratified', 'Coverage']
        v5_1_str = f"**{v5_1_f1:.4f}** / {v5_1_cov:.3f}"

        diff = v5_1_f1 - sub.loc['GSI_v5_Greedy', 'Selective_Macro_F1']
        diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
        if diff > 0.01:
            diff_str = f"**{diff_str} 🏆**"

        md.append(f"| {idx} | **{ds}** | {tot_l} nhãn | {br_str} | {cc_str} | {v5_str} | {v5_1_str} | {diff_str} |")

    # Add Mean row
    means = df[df["Dataset"].isin(dataset_list)].groupby("Model").mean(numeric_only=True)
    if all(m in means.index for m in models_needed):
        mean_br_str = f"{means.loc['BR', 'Selective_Macro_F1']:.4f} / 1.00"
        mean_cc_str = f"{means.loc['CC', 'Selective_Macro_F1']:.4f} / 1.00"
        mean_v5_str = f"{means.loc['GSI_v5_Greedy', 'Selective_Macro_F1']:.4f} / {means.loc['GSI_v5_Greedy', 'Coverage']:.3f}"
        mean_v5_1_f1 = means.loc['GSI_v5_1_Stratified', 'Selective_Macro_F1']
        mean_v5_1_cov = means.loc['GSI_v5_1_Stratified', 'Coverage']
        mean_v5_1_str = f"**{mean_v5_1_f1:.4f}** / {mean_v5_1_cov:.3f}"
        mean_diff = mean_v5_1_f1 - means.loc['GSI_v5_Greedy', 'Selective_Macro_F1']
        mean_diff_str = f"**+{mean_diff:.4f} 🏆**" if mean_diff >= 0 else f"{mean_diff:.4f}"

        md.append(f"| — | **TRUNG BÌNH** | — | {mean_br_str} | {mean_cc_str} | {mean_v5_str} | {mean_v5_1_str} | {mean_diff_str} |\n")

    md.append("## 2. Chi Tiết Cấu Trúc Phân Tầng IL / DL và Thời Gian Huấn Luyện")
    md.append("| Tập dữ liệu | Mô hình | Số Tầng Bóc Tách (Stages) | Số Lượng IL Trung Bình | Tỷ Lệ IL (%) | Lý Do Dừng | Thời Gian (s) |")
    md.append("| :--- | :--- | :---: | :---: | :---: | :--- | :---: |")

    for ds in dataset_list:
        sub = df[df["Dataset"] == ds].set_index("Model")
        for m in ["GSI_v5_Greedy", "GSI_v5_1_Stratified"]:
            if m not in sub.index:
                continue
            r = sub.loc[m]
            il_pct = (r['Mean_IL_Count'] / r['Total_Labels']) * 100.0
            md.append(
                f"| **{ds}** | `{m}` | {r['Mean_Stages']:.1f} | {r['Mean_IL_Count']:.1f} / {r['Total_Labels']} | {il_pct:.1f}% | `{r['Stopping_Reason']}` | {r['Time_Seconds']:.3f}s |"
            )

    text = "\n".join(md)
    output_path.write_text(text, encoding="utf-8")
    print(f"\n[OK] Report saved to: {output_path}", flush=True)
    print("\n" + text, flush=True)


if __name__ == "__main__":
    run_remaining_5_benchmark(threshold=0.70, base_learner="logistic", n_splits=5)
