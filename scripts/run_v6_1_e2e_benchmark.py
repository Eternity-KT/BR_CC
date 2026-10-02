"""
End-to-End Benchmark Experiment for GSI-MLC-PA v6.1 across 10 Datasets.

Evaluates and compares:
1. BR: Binary Relevance
2. CC: Classifier Chains (natural order)
3. ECC: Standalone Ensemble of Classifier Chains (10 random chains)
4. GSI_v6: GSI Core (5-Fold CV Peeling + Single CC)
5. GSI_v6_1: GSI v6.1 (5-Fold CV Peeling + Singleton DL Promotion + DL ECC)

Across:
- 3 Base Learners: Logistic Regression (logistic), Calibrated LinearSVC (svm_calibrated), and MLP (mlp)
- 10 Benchmark Datasets:
  emotions, scene, chd49, music, gpositivepseaac, genbase, humanpseaac, plantpseaac, viruspseaac, yeast
- Evaluation Protocol: 5-Fold Multilabel Stratified Cross-Validation
- Partial Abstention: Rejection Cost c = 0.30
- Peeling Threshold: tau = 0.75

Outputs written to `results_v6_1/`.
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
    model,
    X_test: np.ndarray,
    cost: float = 0.30,
) -> Dict[str, float]:
    """Compute complete set of full and selective metrics for one model on test fold."""
    if hasattr(model, "predict_full"):
        full_preds = model.predict_full(X_test)
        sel_preds = model.predict(X_test, cost=cost)
    elif hasattr(model, "predict"):
        full_preds = model.predict(X_test)
        sel_preds = full_preds
    else:
        raise ValueError("Model does not expose predict()")

    full_metrics = compute_all_metrics(y_test, full_preds)

    if isinstance(model, (GSIMLCPartialAbstentionClassifier, GSIMLCPAv6_1Classifier)):
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


def run_benchmark(
    datasets: List[str],
    base_learners: List[str],
    threshold: float = 0.75,
    n_splits: int = 5,
    n_chains: int = 10,
    cost: float = 0.30,
    output_dir: Path = WORKSPACE_ROOT / "results_v6_1",
    random_state: int = 42,
) -> pd.DataFrame:
    """Run full outer CV benchmark across datasets and base learners for v6.1."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = output_dir / "v6_1_summary.csv"
    detailed_csv_path = output_dir / "v6_1_detailed_folds.csv"
    report_md_path = output_dir / "v6_1_benchmark_report.md"

    detailed_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 90, flush=True)
    print(f"BENCHMARK GSI-MLC-PA v6.1 (ECC on DL + Singleton Rule)", flush=True)
    print(f"Datasets: {len(datasets)} | Base Learners: {base_learners}", flush=True)
    print(f"Outer CV: {n_splits}-Fold | Threshold tau: {threshold} | Cost c: {cost} | Chains: {n_chains}", flush=True)
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
            print(f"     N = {n_samples}, d = {n_features}, K = {n_labels}", flush=True)

            cv = get_multilabel_cv(n_splits=n_splits, random_state=random_state)

            model_keys = ["BR", "CC", "ECC", "GSI_v6", "GSI_v6_1"]
            fold_metrics: Dict[str, List[Dict[str, float]]] = {m: [] for m in model_keys}
            fold_times: Dict[str, List[float]] = {m: [] for m in model_keys}
            fold_il_counts: Dict[str, List[int]] = {m: [] for m in model_keys}

            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, Y)):
                print(f"       Fold {fold_idx + 1}/{n_splits} running ...", flush=True)
                X_train, X_test = X[train_idx], X[test_idx]
                Y_train, Y_test = Y[train_idx], Y[test_idx]

                # Standard scaling per train fold
                scaler = MaxAbsScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # 1. BR
                t0 = time.perf_counter()
                model_br = create_multilabel_estimator(learner, random_state=random_state)
                model_br.fit(X_train, Y_train)
                t_br = time.perf_counter() - t0
                res_br = evaluate_model_fold(Y_test, model_br, X_test, cost=cost)
                fold_metrics["BR"].append(res_br)
                fold_times["BR"].append(t_br)
                fold_il_counts["BR"].append(n_labels)

                # 2. CC
                t0 = time.perf_counter()
                model_cc = ClassifierChainClassifier(
                    base_estimator=create_binary_estimator(learner, random_state=random_state),
                    random_state=random_state,
                )
                model_cc.fit(X_train, Y_train)
                t_cc = time.perf_counter() - t0
                res_cc = evaluate_model_fold(Y_test, model_cc, X_test, cost=cost)
                fold_metrics["CC"].append(res_cc)
                fold_times["CC"].append(t_cc)
                fold_il_counts["CC"].append(0)

                # 3. ECC (Standalone Ensemble of Classifier Chains)
                t0 = time.perf_counter()
                model_ecc = EnsembleClassifierChainClassifier(
                    base_estimator=create_binary_estimator(learner, random_state=random_state),
                    n_chains=n_chains,
                    random_state=random_state,
                )
                model_ecc.fit(X_train, Y_train)
                t_ecc = time.perf_counter() - t0
                res_ecc = evaluate_model_fold(Y_test, model_ecc, X_test, cost=cost)
                fold_metrics["ECC"].append(res_ecc)
                fold_times["ECC"].append(t_ecc)
                fold_il_counts["ECC"].append(0)

                # 4. GSI_v6 Core (Single CC)
                t0 = time.perf_counter()
                model_v6 = GSIMLCPartialAbstentionClassifier(
                    base_learner=learner,
                    partition_mode="v6",
                    stratified_threshold=threshold,
                    cv_folds=n_splits,
                    peeling_metric="selective_f1",
                    dl_order_direction="ascending",
                    final_order="ascending_correlation",
                    cost=cost,
                    random_state=random_state,
                )
                model_v6.fit(X_train, Y_train)
                t_v6 = time.perf_counter() - t0
                res_v6 = evaluate_model_fold(Y_test, model_v6, X_test, cost=cost)
                fold_metrics["GSI_v6"].append(res_v6)
                fold_times["GSI_v6"].append(t_v6)
                fold_il_counts["GSI_v6"].append(len(getattr(model_v6, "independent_labels_", [])))

                # 5. GSI_v6_1 (5-Fold Peeling + Singleton DL Promotion + DL ECC)
                t0 = time.perf_counter()
                model_v6_1 = GSIMLCPAv6_1Classifier(
                    base_learner=learner,
                    stratified_threshold=threshold,
                    cv_folds=n_splits,
                    n_chains=n_chains,
                    cost=cost,
                    random_state=random_state,
                )
                model_v6_1.fit(X_train, Y_train)
                t_v6_1 = time.perf_counter() - t0
                res_v6_1 = evaluate_model_fold(Y_test, model_v6_1, X_test, cost=cost)
                fold_metrics["GSI_v6_1"].append(res_v6_1)
                fold_times["GSI_v6_1"].append(t_v6_1)
                fold_il_counts["GSI_v6_1"].append(len(getattr(model_v6_1, "independent_labels_", [])))

                # Record per-fold detailed record
                for m in model_keys:
                    rec = {
                        "dataset": dataset_name,
                        "base_learner": display_learner,
                        "fold": fold_idx + 1,
                        "model": m,
                        **fold_metrics[m][-1],
                        "il_count": fold_il_counts[m][-1],
                        "fit_time_sec": fold_times[m][-1],
                    }
                    detailed_rows.append(rec)

            # Summarize dataset performance
            print(f"\n   >>> Summary for {dataset_name} [{display_learner}]:", flush=True)
            for m in model_keys:
                m_res = fold_metrics[m]
                mean_sel_f1 = float(np.mean([r["Selective_Macro_F1"] for r in m_res]))
                std_sel_f1 = float(np.std([r["Selective_Macro_F1"] for r in m_res]))
                mean_cov = float(np.mean([r["Coverage"] for r in m_res]))
                mean_full_f1 = float(np.mean([r["Full_Macro_F1"] for r in m_res]))
                mean_subset = float(np.mean([r["Subset_Accuracy"] for r in m_res]))
                mean_hamming = float(np.mean([r["Hamming_Loss"] for r in m_res]))
                mean_time = float(np.mean(fold_times[m]))
                mean_il = float(np.mean(fold_il_counts[m]))

                sum_row = {
                    "dataset": dataset_name,
                    "base_learner": display_learner,
                    "model": m,
                    "n_samples": n_samples,
                    "n_labels": n_labels,
                    "selective_macro_f1": mean_sel_f1,
                    "selective_macro_f1_std": std_sel_f1,
                    "coverage": mean_cov,
                    "full_macro_f1": mean_full_f1,
                    "subset_accuracy": mean_subset,
                    "hamming_loss": mean_hamming,
                    "mean_il_labels": mean_il,
                    "fit_time_sec": mean_time,
                }
                summary_rows.append(sum_row)

                print(
                    f"       {m:10s} | Sel-F1: {mean_sel_f1:.4f}±{std_sel_f1:.4f} | "
                    f"Cov: {mean_cov*100:5.1f}% | Full-F1: {mean_full_f1:.4f} | Subset: {mean_subset:.4f} | Time: {mean_time:.2f}s",
                    flush=True,
                )

            # Checkpoint intermediate results to CSV
            pd.DataFrame(detailed_rows).to_csv(detailed_csv_path, index=False)
            pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"\n[Success] Benchmark results saved to: {summary_csv_path}")

    # Generate Markdown report
    _generate_markdown_report(df_summary, report_md_path)
    print(f"[Success] Markdown report generated: {report_md_path}")
    return df_summary


def _generate_markdown_report(df: pd.DataFrame, out_path: Path):
    """Generate Markdown report summarizing v6.1 benchmark vs baselines."""
    lines = [
        "# BÁO CÁO KẾT QUẢ THỰC NGHIỆM GSI-MLC-PA v6.1 (ECC on DL + Singleton Rule)\n",
        f"**Thời gian xuất:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        "**Tài liệu tham chiếu:** `meeting_summary.md`, `spec/spec_v6_1.md`  ",
        "**Các mô hình so sánh:** `BR`, `CC`, `ECC`, `GSI_v6`, `GSI_v6_1`\n",
        "---\n",
        "## 1. Bảng Tổng Hợp Trung Bình Toàn Cầu (Grand Mean across Datasets)\n",
    ]

    # Grand mean table grouped by base_learner and model
    grand_mean = df.groupby(["base_learner", "model"]).agg({
        "selective_macro_f1": "mean",
        "coverage": "mean",
        "full_macro_f1": "mean",
        "subset_accuracy": "mean",
        "hamming_loss": "mean",
        "mean_il_labels": "mean",
        "fit_time_sec": "mean",
    }).reset_index()

    lines.append(grand_mean.to_markdown(index=False))
    lines.append("\n---\n")
    lines.append("## 2. Bảng Chi Tiết Từng Tập Dữ Liệu\n")

    for ds in df["dataset"].unique():
        lines.append(f"### Tập dữ liệu: `{ds}`\n")
        sub_df = df[df["dataset"] == ds][
            ["base_learner", "model", "selective_macro_f1", "coverage", "full_macro_f1", "subset_accuracy", "hamming_loss", "mean_il_labels", "fit_time_sec"]
        ]
        lines.append(sub_df.to_markdown(index=False))
        lines.append("\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Run GSI-MLC-PA v6.1 End-to-End Benchmark.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_10_DATASETS,
        help="Datasets to evaluate (default: all 10).",
    )
    parser.add_argument(
        "--base-learners",
        nargs="+",
        default=DEFAULT_BASE_LEARNERS,
        help="Base learners to evaluate: logistic, svm_calibrated, mlp.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Peeling Selective-F1 threshold (default: 0.75).",
    )
    parser.add_argument(
        "--cost",
        type=float,
        default=0.30,
        help="Partial abstention rejection cost c (default: 0.30).",
    )
    parser.add_argument(
        "--n-chains",
        type=int,
        default=10,
        help="Number of Classifier Chains in ECC (default: 10).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of outer CV folds (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(WORKSPACE_ROOT / "results_v6_1"),
        help="Directory to save benchmark results.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )

    args = parser.parse_args()
    run_benchmark(
        datasets=args.datasets,
        base_learners=args.base_learners,
        threshold=args.threshold,
        n_splits=args.cv_folds,
        n_chains=args.n_chains,
        cost=args.cost,
        output_dir=Path(args.output_dir),
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
