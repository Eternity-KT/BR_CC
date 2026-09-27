"""5-Dataset Benchmark Experiment for GSI-MLC-PA v5.1.

Evaluates the new Data-Driven Stratified Peeling Partitioning and Ascending-Correlation
Classifier Chain against GSI v5 (Greedy), standard BR, and standard CC across 5 datasets:
1. emotions (593 samples, 6 labels)
2. scene (2407 samples, 6 labels)
3. chd49 (555 samples, 6 labels)
4. music (593 samples, 6 labels)
5. gpositivepseaac (519 samples, 4 labels)

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
from sklearn.linear_model import LogisticRegression
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

TARGET_DATASETS = [
    "emotions",
    "scene",
    "chd49",
    "music",
    "gpositivepseaac",
]

OUTPUT_DIR = Path("results_v5_1_test")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def evaluate_fold_predictions(
    y_test: np.ndarray,
    model,
    X_test: np.ndarray,
    cost: float = 0.30,
) -> Dict[str, float]:
    """Compute all full and selective metrics for one model on test data."""
    # Full metrics at c = 0.50 (no abstention)
    if hasattr(model, "predict_full"):
        full_preds = model.predict_full(X_test)
        sel_preds = model.predict(X_test, cost=cost)
    elif hasattr(model, "predict"):
        full_preds = model.predict(X_test)
        sel_preds = full_preds  # standard non-selective baselines have Coverage = 1.0
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


def run_5_datasets_benchmark(
    threshold: float = 0.70,
    base_learner: str = "logistic",
    n_splits: int = 5,
    random_state: int = 42,
):
    print("=" * 80)
    print(f"STARTING GSI-MLC-PA v5.1 BENCHMARK ON 5 DATASETS")
    print(f"Base Learner: {base_learner} | IL Threshold: {threshold} | CV: {n_splits}-Fold")
    print("=" * 80)

    all_records = []
    dataset_summaries = []

    for dataset_name in TARGET_DATASETS:
        print(f"\n---> Loading dataset: {dataset_name} ...", flush=True)
        X, Y, feat_names, label_names = load_dataset(dataset_name)
        n_samples, n_labels = Y.shape
        print(f"     Samples: {n_samples}, Features: {X.shape[1]}, Labels: {n_labels}", flush=True)

        cv = get_multilabel_cv(n_splits=n_splits, random_state=random_state)

        # Models to evaluate
        model_names = [
            "BR",
            "CC",
            "GSI_v5_Greedy",
            "GSI_v5_1_NoPenalty",
            "GSI_v5_1_WithPenalty",
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

            # 4. GSI v5.1 No Penalty (Stratified Peeling + Ascending Correlation CC)
            t0 = time.perf_counter()
            model_v5_1_no_pen = GSIMLCPartialAbstentionClassifier(
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
            model_v5_1_no_pen.fit(X_train, Y_train)
            t_v5_1_no_pen = time.perf_counter() - t0
            res_v5_1_no_pen = evaluate_fold_predictions(Y_test, model_v5_1_no_pen, X_test)
            fold_results["GSI_v5_1_NoPenalty"].append(res_v5_1_no_pen)
            fold_times["GSI_v5_1_NoPenalty"].append(t_v5_1_no_pen)
            fold_il_counts["GSI_v5_1_NoPenalty"].append(
                len(getattr(model_v5_1_no_pen, "independent_labels_", []))
            )
            audit_no_pen = getattr(model_v5_1_no_pen, "stratified_peeling_audit_", {}) or {}
            fold_stages["GSI_v5_1_NoPenalty"].append(audit_no_pen.get("num_stages_executed", 1))
            fold_reasons["GSI_v5_1_NoPenalty"].append(audit_no_pen.get("stopping_reason", "n/a"))

            # 5. GSI v5.1 With Penalty (Stratified Peeling + Ascending Correlation CC + Complexity Penalty)
            t0 = time.perf_counter()
            model_v5_1_with_pen = GSIMLCPartialAbstentionClassifier(
                base_learner=base_learner,
                partition_mode="stratified_peeling",
                stratified_threshold=threshold,
                dl_order_direction="ascending",
                final_order="ascending_correlation",
                decision_policy="macro_f1",
                cost=0.30,
                random_state=random_state,
                use_complexity_penalty=True,
                complexity_penalty_lambda=0.01,
            )
            model_v5_1_with_pen.fit(X_train, Y_train)
            t_v5_1_with_pen = time.perf_counter() - t0
            res_v5_1_with_pen = evaluate_fold_predictions(Y_test, model_v5_1_with_pen, X_test)
            fold_results["GSI_v5_1_WithPenalty"].append(res_v5_1_with_pen)
            fold_times["GSI_v5_1_WithPenalty"].append(t_v5_1_with_pen)
            fold_il_counts["GSI_v5_1_WithPenalty"].append(
                len(getattr(model_v5_1_with_pen, "independent_labels_", []))
            )
            audit_with_pen = getattr(model_v5_1_with_pen, "stratified_peeling_audit_", {}) or {}
            fold_stages["GSI_v5_1_WithPenalty"].append(audit_with_pen.get("num_stages_executed", 1))
            fold_reasons["GSI_v5_1_WithPenalty"].append(audit_with_pen.get("stopping_reason", "n/a"))

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
                f"       {m:<32} | Sel F1: {mean_sel_f1:.4f} | Cov: {mean_cov:.3f} | "
                f"Full F1: {mean_full_f1:.4f} | IL: {mean_il:.1f}/{n_labels} | Stages: {mean_st:.1f} | Time: {mean_time:.3f}s",
                flush=True,
            )

    df_summary = pd.DataFrame(dataset_summaries)
    csv_path = OUTPUT_DIR / "benchmark_5datasets_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\n[OK] Summary table saved to: {csv_path}", flush=True)

    # Build and print Markdown Report
    md_report = []
    md_report.append("# BÁO CÁO THỰC NGHIỆM ĐỐI CHUẨN GSI-MLC-PA V5.1 (NO PENALTY VS. WITH PENALTY) TRÊN 5 TẬP DỮ LIỆU\n")
    md_report.append(f"**Cấu hình:** Base Learner = `{base_learner}`, Ngưỡng phân định độc lập $\\tau = {threshold}$, 5-Fold Stratified CV, Phạt chi phí tính toán $\\lambda = 0.01$.\n")
    md_report.append("## 1. Bảng Tổng Hợp Kết Quả Selective Macro-F1 và Coverage (c = 0.30)\n")
    md_report.append("| Tập dữ liệu | BR (F1/Cov) | CC (F1/Cov) | GSI v5 Greedy | GSI v5.1 (No Penalty) | GSI v5.1 (With Penalty) | Thời gian NoPen vs WithPen |")
    md_report.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for ds in TARGET_DATASETS:
        sub = df_summary[df_summary["Dataset"] == ds].set_index("Model")
        br_str = f"{sub.loc['BR', 'Selective_Macro_F1']:.4f} / 1.00"
        cc_str = f"{sub.loc['CC', 'Selective_Macro_F1']:.4f} / 1.00"
        v5_str = f"{sub.loc['GSI_v5_Greedy', 'Selective_Macro_F1']:.4f} / {sub.loc['GSI_v5_Greedy', 'Coverage']:.3f}"
        v5_1_no_str = f"{sub.loc['GSI_v5_1_NoPenalty', 'Selective_Macro_F1']:.4f} / {sub.loc['GSI_v5_1_NoPenalty', 'Coverage']:.3f}"
        v5_1_pen_str = f"**{sub.loc['GSI_v5_1_WithPenalty', 'Selective_Macro_F1']:.4f}** / {sub.loc['GSI_v5_1_WithPenalty', 'Coverage']:.3f}"
        time_str = f"{sub.loc['GSI_v5_1_NoPenalty', 'Time_Seconds']:.2f}s vs **{sub.loc['GSI_v5_1_WithPenalty', 'Time_Seconds']:.2f}s**"

        md_report.append(f"| **{ds}** | {br_str} | {cc_str} | {v5_str} | {v5_1_no_str} | {v5_1_pen_str} | {time_str} |")

    # Add Mean row
    means = df_summary.groupby("Model").mean(numeric_only=True)
    mean_br_str = f"{means.loc['BR', 'Selective_Macro_F1']:.4f} / 1.00"
    mean_cc_str = f"{means.loc['CC', 'Selective_Macro_F1']:.4f} / 1.00"
    mean_v5_str = f"{means.loc['GSI_v5_Greedy', 'Selective_Macro_F1']:.4f} / {means.loc['GSI_v5_Greedy', 'Coverage']:.3f}"
    mean_v5_1_no_str = f"{means.loc['GSI_v5_1_NoPenalty', 'Selective_Macro_F1']:.4f} / {means.loc['GSI_v5_1_NoPenalty', 'Coverage']:.3f}"
    mean_v5_1_pen_str = f"**{means.loc['GSI_v5_1_WithPenalty', 'Selective_Macro_F1']:.4f}** / {means.loc['GSI_v5_1_WithPenalty', 'Coverage']:.3f}"
    mean_time_str = f"{means.loc['GSI_v5_1_NoPenalty', 'Time_Seconds']:.2f}s vs **{means.loc['GSI_v5_1_WithPenalty', 'Time_Seconds']:.2f}s**"

    md_report.append(f"| **TRUNG BÌNH** | {mean_br_str} | {mean_cc_str} | {mean_v5_str} | {mean_v5_1_no_str} | {mean_v5_1_pen_str} | {mean_time_str} |\n")

    md_report.append("## 2. Chi Tiết Ảnh Hưởng Của Hàm Phạt Phức Tạp (Complexity Penalty Diagnostics)")
    md_report.append("| Tập dữ liệu | Model | Số Tầng Bóc Tách (Stages) | Số Lượng IL | Lý Do Dừng | Thời Gian (s) |")
    md_report.append("| :--- | :--- | :---: | :---: | :--- | :---: |")

    for ds in TARGET_DATASETS:
        sub = df_summary[df_summary["Dataset"] == ds].set_index("Model")
        for m in ["GSI_v5_Greedy", "GSI_v5_1_NoPenalty", "GSI_v5_1_WithPenalty"]:
            r = sub.loc[m]
            md_report.append(
                f"| **{ds}** | `{m}` | {r['Mean_Stages']:.1f} | {r['Mean_IL_Count']:.1f}/{r['Total_Labels']} | `{r['Stopping_Reason']}` | {r['Time_Seconds']:.3f}s |"
            )

    report_text = "\n".join(md_report)
    report_file = OUTPUT_DIR / "benchmark_5datasets_report.md"
    report_file.write_text(report_text, encoding="utf-8")
    print(f"\n[OK] Markdown report saved to: {report_file}", flush=True)
    print("\n" + report_text, flush=True)


if __name__ == "__main__":
    run_5_datasets_benchmark(threshold=0.70, base_learner="logistic", n_splits=5)
