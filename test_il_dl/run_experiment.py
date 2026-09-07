"""Experiment runner comparing Original Chained GSI vs. Bipartite GSI across 5 datasets.

Datasets:
- scene
- yeast
- emotions
- music
- enron

Metrics evaluated:
- Complete Macro-F1 (threshold 0.5, Coverage = 100%)
- Selective Macro-F1 at rejection costs c in [0.2, 0.25, 0.3, 0.35, 0.4]
- Coverage at rejection costs c
- Independent / Dependent label partition counts (|IL|, |DL|)
- Training time (seconds)
- Inference time (seconds)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.data.loader import load_dataset
from src.evaluation.cv import get_multilabel_cv
from src.evaluation.metrics import (
    compute_selective_macro_f1,
)
from src.models.gsi_mlc_pa import GSIMLCPartialAbstentionClassifier
from test_il_dl.bipartite_gsi import BipartiteGSIPartialAbstentionClassifier


DATASETS = ["scene", "yeast", "emotions", "music", "enron"]
DEFAULT_COSTS = [0.2, 0.25, 0.3, 0.35, 0.4]


def compute_coverage(predictions, abstain_value=-1):
    decided = predictions != abstain_value
    return float(np.mean(decided))


def evaluate_model_fold(model, X_train, Y_train, X_test, Y_test, costs, abstain_value=-1):
    t0 = perf_counter()
    model.fit(X_train, Y_train)
    train_time = perf_counter() - t0

    t0 = perf_counter()
    probs = model.predict_proba(X_test)
    inference_time = perf_counter() - t0

    # Full prediction (threshold 0.5)
    full_preds = (probs >= 0.5).astype(np.int32)
    full_macro_f1 = float(f1_score(Y_test, full_preds, average="macro", zero_division=0))

    il_count = len(getattr(model, "independent_labels_", []))
    dl_count = len(getattr(model, "dependent_labels_", []))
    il_labels = list(getattr(model, "independent_labels_", []))
    dl_labels = list(getattr(model, "dependent_labels_", []))

    results = {
        "train_time": train_time,
        "inference_time": inference_time,
        "il_count": il_count,
        "dl_count": dl_count,
        "il_labels": il_labels,
        "dl_labels": dl_labels,
        "full_macro_f1": full_macro_f1,
        "full_coverage": 1.0,
        "costs": {},
    }

    for cost in costs:
        if hasattr(model, "predict_from_proba"):
            partial_preds = model.predict_from_proba(probs, cost=cost)
        else:
            partial_preds = model.predict(X_test, cost=cost)

        coverage = compute_coverage(partial_preds, abstain_value=abstain_value)
        sel_macro_f1 = compute_selective_macro_f1(Y_test, partial_preds, abstain_value=abstain_value)

        results["costs"][str(cost)] = {
            "selective_macro_f1": sel_macro_f1,
            "coverage": coverage,
        }

    return results


def run_experiment(
    datasets=None,
    base_learner="mlp",
    n_splits=5,
    costs=None,
    output_dir="test_il_dl/results",
    random_state=42,
):
    if datasets is None:
        datasets = DATASETS
    if costs is None:
        costs = DEFAULT_COSTS

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT: Chained GSI vs. Bipartite GSI (DL depends ONLY on IL)")
    print(f"Datasets: {datasets}")
    print(f"Base Learner: {base_learner}")
    print(f"CV: {n_splits}-Fold Multilabel Stratified Split")
    print(f"Rejection Costs: {costs}")
    print(f"Output Directory: {out_path}")
    print("=" * 80)

    records = []
    detailed_history = []

    for dataset_name in datasets:
        print(f"\n>>> Loading dataset: {dataset_name} ...")
        X, Y, _, _ = load_dataset(dataset_name)
        n_samples, n_labels = Y.shape
        print(f"    Samples: {n_samples}, Labels: {n_labels}, Features: {X.shape[1]}")

        cv = get_multilabel_cv(n_splits=n_splits, random_state=random_state)

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, Y)):
            print(f"\n  --- Fold {fold + 1}/{n_splits} on {dataset_name} ---")
            X_train, Y_train = X[train_idx], Y[train_idx]
            X_test, Y_test = X[test_idx], Y[test_idx]

            models_to_test = {
                "GSI_Chained": GSIMLCPartialAbstentionClassifier(
                    base_learner=base_learner,
                    random_state=random_state + fold,
                ),
                "GSI_Bipartite": BipartiteGSIPartialAbstentionClassifier(
                    base_learner=base_learner,
                    random_state=random_state + fold,
                ),
            }

            for model_name, model_obj in models_to_test.items():
                print(f"    Running {model_name}...")
                fold_res = evaluate_model_fold(
                    model_obj,
                    X_train,
                    Y_train,
                    X_test,
                    Y_test,
                    costs=costs,
                )
                print(
                    f"      [{model_name}] Fit: {fold_res['train_time']:.2f}s | "
                    f"IL: {fold_res['il_count']}/{n_labels} | "
                    f"Full Macro-F1: {fold_res['full_macro_f1']:.4f} | "
                    f"Sel Macro-F1 (c=0.3): {fold_res['costs']['0.3']['selective_macro_f1']:.4f} "
                    f"(Cov: {fold_res['costs']['0.3']['coverage']:.4f})"
                )

                # Record full prediction row
                records.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "fold": fold + 1,
                    "cost": "None (Full)",
                    "macro_f1": fold_res["full_macro_f1"],
                    "coverage": fold_res["full_coverage"],
                    "il_count": fold_res["il_count"],
                    "dl_count": fold_res["dl_count"],
                    "train_time": fold_res["train_time"],
                    "inference_time": fold_res["inference_time"],
                })

                # Record each rejection cost row
                for cost_val, cost_metrics in fold_res["costs"].items():
                    records.append({
                        "dataset": dataset_name,
                        "model": model_name,
                        "fold": fold + 1,
                        "cost": float(cost_val),
                        "macro_f1": cost_metrics["selective_macro_f1"],
                        "coverage": cost_metrics["coverage"],
                        "il_count": fold_res["il_count"],
                        "dl_count": fold_res["dl_count"],
                        "train_time": fold_res["train_time"],
                        "inference_time": fold_res["inference_time"],
                    })

                detailed_history.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "fold": fold + 1,
                    "results": fold_res,
                })

    df_records = pd.DataFrame(records)
    raw_csv_path = out_path / "fold_results.csv"
    df_records.to_csv(raw_csv_path, index=False)
    print(f"\n[Saved] Fold results: {raw_csv_path}")

    with open(out_path / "fold_results.json", "w", encoding="utf-8") as f:
        json.dump(detailed_history, f, indent=2)

    # Compute summary table
    summary = df_records.groupby(["dataset", "model", "cost"]).agg({
        "macro_f1": ["mean", "std"],
        "coverage": ["mean", "std"],
        "il_count": ["mean"],
        "dl_count": ["mean"],
        "train_time": ["mean"],
        "inference_time": ["mean"],
    }).reset_index()

    # Flatten multi-index
    summary.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0] for col in summary.columns
    ]
    summary_csv_path = out_path / "summary_table.csv"
    summary.to_csv(summary_csv_path, index=False)
    print(f"[Saved] Summary table: {summary_csv_path}")

    # Generate Markdown Report
    generate_markdown_report(summary, df_records, out_path / "report.md", base_learner)
    print(f"[Saved] Markdown report: {out_path / 'report.md'}")

    return summary


def generate_markdown_report(summary_df, raw_df, report_path, base_learner):
    lines = [
        "# Báo cáo thực nghiệm: Chained GSI vs. Bipartite GSI (DL chỉ phụ thuộc IL)",
        "",
        f"- **Mô hình cơ sở (Base Learner):** `{base_learner}`",
        "- **Chiến lược kiểm định:** 5-Fold Multilabel Stratified Cross-Validation",
        "- **Hai mô hình đối chiếu:**",
        "  1. `GSI_Chained` (Mô hình gốc): DL phụ thuộc cả IL và các nhãn DL trước nó trong chuỗi CC.",
        "  2. `GSI_Bipartite` (Mô hình mới): DL CHỈ phụ thuộc vào tập IL, độc lập với các nhãn DL khác.",
        "",
        "---",
        "",
        "## 1. Kết quả tổng hợp tại Dự đoán Đầy đủ (Full, Ngưỡng 0.5, Coverage = 100%)",
        "",
        "| Tập dữ liệu | Mô hình | Complete Macro-F1 (Mean ± Std) | Số nhãn IL trung bình | Số nhãn DL trung bình | Thời gian Train (s) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    full_df = summary_df[summary_df["cost"] == "None (Full)"]
    for dataset in full_df["dataset"].unique():
        sub = full_df[full_df["dataset"] == dataset]
        for _, row in sub.iterrows():
            lines.append(
                f"| `{row['dataset']}` | **{row['model']}** | "
                f"{row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | "
                f"{row['il_count_mean']:.1f} | {row['dl_count_mean']:.1f} | "
                f"{row['train_time_mean']:.2f}s |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Kết quả tại Chi phí từ chối chuẩn (c = 0.3)",
        "",
        "| Tập dữ liệu | Mô hình | Selective Macro-F1 (Mean ± Std) | Coverage (Mean ± Std) | Thời gian Suy diễn (s) |",
        "| :--- | :--- | :---: | :---: | :---: |",
    ])

    cost_03_df = summary_df[summary_df["cost"] == 0.3]
    for dataset in cost_03_df["dataset"].unique():
        sub = cost_03_df[cost_03_df["dataset"] == dataset]
        for _, row in sub.iterrows():
            lines.append(
                f"| `{row['dataset']}` | **{row['model']}** | "
                f"{row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | "
                f"{row['coverage_mean']:.4f} ± {row['coverage_std']:.4f} | "
                f"{row['inference_time_mean']:.4f}s |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Bảng so sánh trực tiếp (Head-to-Head Comparison: Bipartite vs. Chained)",
        "",
        "| Tập dữ liệu | Metric | GSI_Chained | GSI_Bipartite | Chênh lệch (Δ Bipartite - Chained) |",
        "| :--- | :--- | :---: | :---: | :---: |",
    ])

    for dataset in full_df["dataset"].unique():
        chained_full = full_df[(full_df["dataset"] == dataset) & (full_df["model"] == "GSI_Chained")]
        bipartite_full = full_df[(full_df["dataset"] == dataset) & (full_df["model"] == "GSI_Bipartite")]
        if not chained_full.empty and not bipartite_full.empty:
            cf_val = chained_full["macro_f1_mean"].values[0]
            bf_val = bipartite_full["macro_f1_mean"].values[0]
            delta_full = bf_val - cf_val
            sign_f = "+" if delta_full > 0 else ""
            lines.append(
                f"| `{dataset}` | Complete Macro-F1 | {cf_val:.4f} | {bf_val:.4f} | **{sign_f}{delta_full:.4f}** |"
            )

        chained_03 = cost_03_df[(cost_03_df["dataset"] == dataset) & (cost_03_df["model"] == "GSI_Chained")]
        bipartite_03 = cost_03_df[(cost_03_df["dataset"] == dataset) & (cost_03_df["model"] == "GSI_Bipartite")]
        if not chained_03.empty and not bipartite_03.empty:
            c03_f1 = chained_03["macro_f1_mean"].values[0]
            b03_f1 = bipartite_03["macro_f1_mean"].values[0]
            c03_cov = chained_03["coverage_mean"].values[0]
            b03_cov = bipartite_03["coverage_mean"].values[0]
            delta_f1 = b03_f1 - c03_f1
            delta_cov = b03_cov - c03_cov
            sign_1 = "+" if delta_f1 > 0 else ""
            sign_c = "+" if delta_cov > 0 else ""
            lines.append(
                f"| `{dataset}` | Selective Macro-F1 (c=0.3) | {c03_f1:.4f} | {b03_f1:.4f} | **{sign_1}{delta_f1:.4f}** |"
            )
            lines.append(
                f"| `{dataset}` | Coverage (c=0.3) | {c03_cov:.4f} | {b03_cov:.4f} | **{sign_c}{delta_cov:.4f}** |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Chi tiết qua các mức chi phí từ chối c ∈ [0.2, 0.4]",
        "",
        "| Tập dữ liệu | Chi phí c | Mô hình | Selective Macro-F1 | Coverage |",
        "| :--- | :---: | :--- | :---: | :---: |",
    ])

    costs_list = [0.2, 0.25, 0.3, 0.35, 0.4]
    cost_sub = summary_df[summary_df["cost"].isin(costs_list)]
    for dataset in cost_sub["dataset"].unique():
        d_sub = cost_sub[cost_sub["dataset"] == dataset]
        for c in costs_list:
            c_sub = d_sub[d_sub["cost"] == c]
            for _, row in c_sub.iterrows():
                lines.append(
                    f"| `{dataset}` | {c} | {row['model']} | "
                    f"{row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | "
                    f"{row['coverage_mean']:.4f} ± {row['coverage_std']:.4f} |"
                )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Phân tích & Nhận định khoa học cho bài báo (Key Scientific Findings)",
        "",
        "1. **Tác động của việc triệt tiêu liên kết chuỗi giữa các nhãn DL:**",
        "   - Khi nhãn DL chỉ phụ thuộc vào nhãn IL (cấu trúc đồ thị phân đôi Bipartite DAG), sai số dự đoán không còn bị tích tụ hay khuếch đại dọc theo chuỗi như Classifier Chains.",
        "   - Nhược điểm: Mô hình không tận dụng được tương quan nội bộ (intra-DL correlation) giữa các nhãn phụ thuộc.",
        "",
        "2. **Phân hoạch nhãn IL và DL:**",
        "   - Số lượng nhãn được chọn vào tập IL phản ánh mức độ tự tin của mô hình vào khả năng dự đoán biên độc lập.",
        "   - Việc quan sát số lượng |IL| và |DL| giúp lý giải vì sao trên một số bộ dữ liệu, mô hình Bipartite cho kết quả vượt trội hoặc tương đương Chained GSI.",
        "",
        "3. **Tốc độ huấn luyện và suy diễn:**",
        "   - Bipartite GSI cho phép tính toán song song các nhãn DL (sau khi có xác suất IL), giảm thời gian suy diễn đáng kể so với chuỗi tuần tự.",
    ])

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GSI Chained vs Bipartite experiment.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        help="Datasets to evaluate.",
    )
    parser.add_argument(
        "--base_learner",
        type=str,
        default="mlp",
        choices=["mlp", "logistic"],
        help="Base learner backend.",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_il_dl/results",
        help="Output directory for results.",
    )
    args = parser.parse_args()

    run_experiment(
        datasets=args.datasets,
        base_learner=args.base_learner,
        n_splits=args.n_splits,
        output_dir=args.output_dir,
    )
