"""
End-to-End Benchmark Experiment for GSI-MLC-PA v6.2 across Benchmark Datasets.

Reference:
    - meeting_summary.md (Core Section, v6.2)
    - spec/spec_v6_2.md

Evaluates and compares:
1. BR: Binary Relevance
2. CC: Classifier Chains (natural order)
3. ECC: Standalone Ensemble of Classifier Chains (10 random chains)
4. GSI_v6: GSI Core (5-Fold CV Peeling + Single CC)
5. GSI_v6_2: GSI v6.2 (5-Fold CV Peeling + Singleton Rule + Residual Error Correlation BR for DL)

Outputs written strictly isolated to `results_v6_2/`:
- `results_v6_2/v6_2_summary.csv`: Macro/Micro-F1, Coverage, Hamming Loss averages across outer folds.
- `results_v6_2/v6_2_detailed_folds.csv`: Per-fold evaluation metrics.
- `results_v6_2/v6_2_benchmark_report.md`: Markdown summary report with comparative tables.
- `results_v6_2/residual_correlation_matrices/`: Detailed error correlation matrices and dependency graphs.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from src.models.base_learners import create_binary_estimator, create_multilabel_estimator
from src.models.binary_relevance import BinaryRelevanceClassifier
from src.models.classifier_chain import ClassifierChainClassifier
from src.models.ensemble_classifier_chain import EnsembleClassifierChainClassifier
from src.models.gsi_mlc_pa import GSIMLCPartialAbstentionClassifier
from src.models.gsi_v6_1 import GSIMLCPAv6_1Classifier
from src.models.gsi_v6_2 import GSIMLCPAv6_2Classifier
from src.models.mlc_pa import MLCPartialAbstentionClassifier

ALL_10_DATASETS = [
    "emotions",
    "scene",
    "chd49",
    "music",
    "gpositivepseaac",
    "genbase",
    "humanpseaac",
    "plantpseaac",
    "viruspseaac",
    "yeast",
]

DEFAULT_BASE_LEARNERS = ["logistic", "svm_calibrated", "mlp"]


def evaluate_model_fold(
    y_test: np.ndarray,
    model: Any,
    X_test: np.ndarray,
    cost: float = 0.30,
) -> Dict[str, float]:
    """Compute complete set of full and selective metrics for one model on test fold."""
    if hasattr(model, "predict_full"):
        full_preds = model.predict_full(X_test)
        try:
            sel_preds = model.predict(X_test, cost=cost)
        except TypeError:
            sel_preds = model.predict(X_test)
    elif hasattr(model, "predict"):
        full_preds = model.predict(X_test)
        sel_preds = full_preds
    else:
        raise ValueError("Model does not expose predict()")

    full_metrics = compute_all_metrics(y_test, full_preds)

    if isinstance(
        model,
        (
            MLCPartialAbstentionClassifier,
            GSIMLCPartialAbstentionClassifier,
            GSIMLCPAv6_1Classifier,
            GSIMLCPAv6_2Classifier,
        ),
    ):
        sel_macro_f1 = compute_selective_macro_f1(y_test, sel_preds)
        sel_micro_f1 = compute_selective_micro_f1(y_test, sel_preds)
        coverage = float(np.mean(sel_preds != -1))
        decided = sel_preds != -1
        sel_hamming = (
            float(np.mean(y_test[decided] != sel_preds[decided]))
            if np.any(decided)
            else 0.0
        )
    else:
        sel_macro_f1 = full_metrics["Macro-F1"]
        sel_micro_f1 = full_metrics["Micro-F1"]
        coverage = 1.0
        sel_hamming = full_metrics["Hamming Loss"]

    return {
        "Full_Macro_F1": float(full_metrics["Macro-F1"]),
        "Selective_Macro_F1": float(sel_macro_f1),
        "Coverage": float(coverage),
        "Full_Micro_F1": float(full_metrics["Micro-F1"]),
        "Selective_Micro_F1": float(sel_micro_f1),
        "Hamming_Loss": float(full_metrics["Hamming Loss"]),
        "Selective_Hamming_Loss": float(sel_hamming),
        "Subset_Accuracy": float(full_metrics["Subset Accuracy"]),
    }


def run_v6_2_benchmark(
    datasets: List[str],
    base_learners: List[str],
    threshold: float = 0.75,
    corr_threshold: float = 0.25,
    n_splits: int = 5,
    cost: float = 0.30,
    output_dir: Path = WORKSPACE_ROOT / "results_v6_2",
    random_state: int = 42,
) -> pd.DataFrame:
    """Run outer CV benchmark comparing v6.2 against baselines across datasets and base learners."""
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir = output_dir / "residual_correlation_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    summary_csv_path = output_dir / "v6_2_summary.csv"
    detailed_csv_path = output_dir / "v6_2_detailed_folds.csv"
    report_md_path = output_dir / "v6_2_benchmark_report.md"

    detailed_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    learner_display_names = [
        {"logistic": "Logistic", "svm_calibrated": "SVM", "svm": "SVM", "mlp": "MLP"}.get(l, l.upper())
        for l in base_learners
    ]

    if detailed_csv_path.exists():
        try:
            df_old_det = pd.read_csv(detailed_csv_path)
            for d in datasets:
                df_old_det = df_old_det[
                    ~((df_old_det["dataset"] == d) & (df_old_det["learner"].isin(learner_display_names)))
                ]
            detailed_rows = df_old_det.to_dict("records")
        except Exception:
            detailed_rows = []

    if summary_csv_path.exists():
        try:
            df_old_sum = pd.read_csv(summary_csv_path)
            for d in datasets:
                df_old_sum = df_old_sum[
                    ~((df_old_sum["dataset"] == d) & (df_old_sum["learner"].isin(learner_display_names)))
                ]
            summary_rows = df_old_sum.to_dict("records")
        except Exception:
            summary_rows = []

    print("\n" + "=" * 90, flush=True)
    print(f"BENCHMARK GSI-MLC-PA v6.2 (Residual Error Correlation Conditional BR on DL)", flush=True)
    print(f"Datasets: {len(datasets)} | Base Learners: {base_learners}", flush=True)
    print(f"Outer CV: {n_splits}-Fold | Peeling tau: {threshold} | Corr tau: {corr_threshold} | Cost c: {cost}", flush=True)
    print("=" * 90, flush=True)

    for learner in base_learners:
        display_learner = {
            "logistic": "Logistic",
            "svm_calibrated": "SVM",
            "svm": "SVM",
            "mlp": "MLP",
        }.get(learner, learner.upper())

        print(f"\n=======================================================", flush=True)
        print(f" >>> BASE LEARNER: {display_learner} ({learner})", flush=True)
        print(f"=======================================================", flush=True)

        for dataset_name in datasets:
            print(f"\n---> Loading dataset: {dataset_name} ...", flush=True)
            X, Y, _, _ = load_dataset(dataset_name, base_dir=str(WORKSPACE_ROOT))
            if hasattr(X, "toarray"):
                X = X.toarray()
            X = np.asarray(X, dtype=np.float32)
            Y = np.asarray(Y, dtype=np.int32)
            n_samples, n_labels = Y.shape
            n_features = X.shape[1]
            print(f"     Samples: {n_samples}, Labels: {n_labels}, Features: {n_features}", flush=True)

            cv = get_multilabel_cv(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            )

            models_to_test = {
                "BR": lambda: create_multilabel_estimator(learner, random_state=random_state),
                "CC": lambda: ClassifierChainClassifier(
                    base_estimator=create_binary_estimator(learner, random_state=random_state),
                    random_state=random_state,
                ),
                "MLC_PA": lambda: MLCPartialAbstentionClassifier(
                    base_estimator=learner,
                    cost=cost,
                    random_state=random_state,
                ),
                "ECC": lambda: EnsembleClassifierChainClassifier(
                    base_estimator=create_binary_estimator(learner, random_state=random_state),
                    n_chains=10,
                    random_state=random_state,
                ),
                "GSI_v6": lambda: GSIMLCPartialAbstentionClassifier(
                    base_learner=learner,
                    partition_mode="cv_stratified_peeling",
                    stratified_threshold=threshold,
                    cost=cost,
                    random_state=random_state,
                ),
                "GSI_v6_2": lambda: GSIMLCPAv6_2Classifier(
                    base_learner=learner,
                    stratified_threshold=threshold,
                    residual_corr_threshold=corr_threshold,
                    cv_folds=5,
                    max_peeling_depth=3,
                    cost=cost,
                    random_state=random_state,
                ),
            }

            model_fold_metrics: Dict[str, List[Dict[str, float]]] = {
                m: [] for m in models_to_test
            }

            # Export residual correlation matrix on full dataset once for audit report
            v6_2_audit_saved = False

            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, Y)):
                X_tr, y_tr = X[train_idx], Y[train_idx]
                X_te, y_te = X[test_idx], Y[test_idx]

                scaler = MaxAbsScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_te = scaler.transform(X_te)

                for model_name, factory in models_to_test.items():
                    t0 = time.perf_counter()
                    clf = factory()
                    clf.fit(X_tr, y_tr)
                    train_time = time.perf_counter() - t0

                    t1 = time.perf_counter()
                    metrics = evaluate_model_fold(y_te, clf, X_te, cost=cost)
                    infer_time = time.perf_counter() - t1

                    # Save residual correlation matrix from Fold 0 of GSI_v6_2
                    if model_name == "GSI_v6_2" and not v6_2_audit_saved:
                        if clf.audit_table_ is not None and not clf.audit_table_.empty:
                            audit_csv = matrix_dir / f"{dataset_name}_{learner}_audit.csv"
                            clf.audit_table_.to_csv(audit_csv, index=False)
                        if clf.residual_corr_matrix_ is not None:
                            corr_csv = matrix_dir / f"{dataset_name}_{learner}_residual_corr_matrix.csv"
                            np.savetxt(corr_csv, clf.residual_corr_matrix_, delimiter=",", fmt="%.4f")
                        v6_2_audit_saved = True

                    fold_record = {
                        "dataset": dataset_name,
                        "learner": display_learner,
                        "model": model_name,
                        "fold": fold_idx + 1,
                        "train_time_sec": train_time,
                        "infer_time_sec": infer_time,
                        **metrics,
                    }
                    detailed_rows.append(fold_record)
                    model_fold_metrics[model_name].append(metrics)

            # Print summary for this dataset
            print(f"     Results on {dataset_name} ({display_learner}):", flush=True)
            for m in models_to_test:
                m_list = model_fold_metrics[m]
                mean_sel_f1 = float(np.mean([x["Selective_Macro_F1"] for x in m_list]))
                std_sel_f1 = float(np.std([x["Selective_Macro_F1"] for x in m_list]))
                mean_full_f1 = float(np.mean([x["Full_Macro_F1"] for x in m_list]))
                mean_cov = float(np.mean([x["Coverage"] for x in m_list]))
                mean_ham = float(np.mean([x["Hamming_Loss"] for x in m_list]))

                print(
                    f"       [{m:<8}] Sel-Macro-F1: {mean_sel_f1:.4f} ± {std_sel_f1:.4f} | "
                    f"Full-Macro-F1: {mean_full_f1:.4f} | Cov: {mean_cov:.4f} | Ham: {mean_ham:.4f}",
                    flush=True,
                )

                summary_rows.append({
                    "dataset": dataset_name,
                    "learner": display_learner,
                    "model": m,
                    "Selective_Macro_F1_mean": mean_sel_f1,
                    "Selective_Macro_F1_std": std_sel_f1,
                    "Full_Macro_F1_mean": mean_full_f1,
                    "Coverage_mean": mean_cov,
                    "Hamming_Loss_mean": mean_ham,
                })

            # Checkpoint
            pd.DataFrame(detailed_rows).to_csv(detailed_csv_path, index=False)
            pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)

    df_summary = pd.DataFrame(summary_rows)
    df_detailed = pd.DataFrame(detailed_rows)

    # Generate Markdown Report
    _write_markdown_report(report_md_path, df_summary, base_learners, threshold, corr_threshold, cost)
    print(f"\n[DONE] Benchmark complete. Outputs saved in {output_dir}", flush=True)
    return df_summary


def _write_markdown_report(
    report_path: Path,
    df_summary: pd.DataFrame,
    base_learners: List[str],
    threshold: float,
    corr_threshold: float,
    cost: float,
):
    """Write comprehensive markdown benchmark report."""
    lines = [
        "# BÁO CÁO THỰC NGHIỆM ĐỐI SÁNH GSI-MLC-PA v6.2 (RESIDUAL ERROR CORRELATION)",
        "",
        "**Tài liệu tham chiếu:** `meeting_summary.md`, `spec/spec_v6_2.md`  ",
        f"**Cấu hình thực nghiệm:** Ngưỡng bóc tách $\\tau = {threshold}$, Ngưỡng tương quan $\\tau_{{\\text{{corr}}}} = {corr_threshold}$, Chi phí từ chối $c = {cost}$, 5-Fold Cross Validation.  ",
        "",
        "## 1. Giới Thiệu Cốt Lõi",
        "Phiên bản **v6.2** triển khai thuật toán được chỉ đạo trong `meeting_summary.md`:",
        "1. **Bóc tách tầng độc lập (IL):** Vòng lặp `Do...While(1)` kết hợp quy tắc biên Singleton DL.",
        "2. **Xử lý tập DL bằng tương quan sai số:** Đo lường $PCC(l - f(l), p - f(p))$ để ghép đặc trưng có điều kiện $FS[l] = FS(X, IL, DL\\_temp[l])$.",
        "3. **Mô hình BR tinh chỉnh:** Huấn luyện bộ phân loại nhị phân BR riêng biệt cho từng nhãn $DL$ trên không gian đặc trưng điều kiện tương ứng.",
        "",
        "## 2. Bảng Tổng Hợp Kết Quả Selective Macro-F1",
        "",
    ]

    piv = df_summary.pivot(index=["learner", "dataset"], columns="model", values="Selective_Macro_F1_mean")
    lines.append(piv.to_markdown())
    lines.append("\n\n## 3. Bảng Tổng Hợp Coverage (Tỷ Lệ Quyết Định)")
    piv_cov = df_summary.pivot(index=["learner", "dataset"], columns="model", values="Coverage_mean")
    lines.append(piv_cov.to_markdown())
    lines.append("\n\n## 4. Ma Trận Tương Quan Sai Số Chi Tiết")
    lines.append("Toàn bộ ma trận Pearson Correlation và đồ thị phụ thuộc cục bộ $DL\\_temp[l]$ được lưu trữ tại `results_v6_2/residual_correlation_matrices/`.")
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Run GSI-MLC-PA v6.2 Benchmark")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["emotions", "scene", "genbase"],
        help="Datasets to benchmark. Use 'all' for all 10 datasets.",
    )
    parser.add_argument(
        "--base-learners",
        nargs="+",
        default=["logistic"],
        choices=["logistic", "svm_calibrated", "mlp"],
        help="Base learners to evaluate.",
    )
    parser.add_argument("--threshold", type=float, default=0.75, help="Selective-F1 threshold")
    parser.add_argument("--corr-threshold", type=float, default=0.25, help="PCC residual error threshold")
    parser.add_argument("--cv", type=int, default=5, help="Outer CV folds")
    parser.add_argument("--cost", type=float, default=0.30, help="Rejection cost c")
    args = parser.parse_args()

    if args.datasets == ["all"]:
        datasets = ALL_10_DATASETS
    else:
        datasets = args.datasets

    run_v6_2_benchmark(
        datasets=datasets,
        base_learners=args.base_learners,
        threshold=args.threshold,
        corr_threshold=args.corr_threshold,
        n_splits=args.cv,
        cost=args.cost,
    )


if __name__ == "__main__":
    main()
