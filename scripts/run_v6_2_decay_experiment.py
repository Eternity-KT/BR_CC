"""
Benchmark Experiment for GSI-MLC-PA v6.2 with Decaying IL Peeling Threshold.

Reference:
    - meeting_summary.md (Core Section, Decaying Threshold Option)
    - spec/spec_v6_2.md

This script evaluates the effect of decaying the independent label (IL) peeling
threshold across stages:
    - Stage 1: tau_f1 = 0.75
    - Stage 2: tau_f1 = 0.70
    - Stage 3: tau_f1 = 0.65
versus the fixed threshold baseline (tau_f1 = 0.75 across all stages).

Models compared on the exact same 5-Fold Stratified CV splits:
1. BR: Binary Relevance
2. CC: Classifier Chains
3. MLC_PA: Multi-Label Partial Abstention (cost c=0.30)
4. GSI_v6_2_Fixed: Fixed peeling threshold tau=0.75 (default v6.2)
5. GSI_v6_2_Decay: Decaying peeling threshold (0.75 -> 0.70 -> 0.65)

Outputs written strictly isolated to `results_v6_2/decay_study/`:
- `decay_summary.csv`: Macro/Micro F1, Coverage, Hamming Loss averages across folds.
- `decay_detailed_folds.csv`: Per-fold evaluation metrics.
- `decay_peeling_breakdown.md`: Layer-by-layer IL promotions and remaining DL sets.
- `report_decaying_threshold_study.md`: Comparative scientific report.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from src.models.classifier_chain import ClassifierChainClassifier
from src.models.gsi_v6_2 import GSIMLCPAv6_2Classifier
from src.models.mlc_pa import MLCPartialAbstentionClassifier

FIRST_5_DATASETS = [
    "emotions",
    "scene",
    "chd49",
    "music",
    "gpositivepseaac",
]

DEFAULT_BASE_LEARNERS = ["logistic", "svm_calibrated", "mlp"]


def evaluate_model_fold(
    y_test: np.ndarray,
    model: Any,
    X_test: np.ndarray,
    cost: float = 0.30,
) -> Dict[str, float]:
    """Evaluate model fold predictions with both full and selective metrics."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        full_preds = (probs >= 0.5).astype(int)
    elif hasattr(model, "predict"):
        full_preds = model.predict(X_test)
    else:
        raise ValueError("Model does not expose predict or predict_proba")

    if hasattr(model, "predict_selective"):
        sel_preds = model.predict_selective(X_test, cost=cost)
    elif hasattr(model, "predict"):
        try:
            sel_preds = model.predict(X_test, cost=cost)
        except TypeError:
            sel_preds = model.predict(X_test)
    else:
        sel_preds = full_preds

    full_metrics = compute_all_metrics(y_test, full_preds)

    if isinstance(model, (MLCPartialAbstentionClassifier, GSIMLCPAv6_2Classifier)):
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


def run_decay_benchmark(
    datasets: List[str],
    base_learners: List[str],
    init_threshold: float = 0.75,
    decay_step: float = 0.05,
    corr_threshold: float = 0.25,
    n_splits: int = 5,
    cost: float = 0.30,
    output_dir: Path = WORKSPACE_ROOT / "results_v6_2" / "decay_study",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute outer CV benchmark comparing Fixed vs Decaying threshold."""
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_csv_path = output_dir / "decay_detailed_folds.csv"
    summary_csv_path = output_dir / "decay_summary.csv"
    peeling_breakdown_path = output_dir / "decay_peeling_breakdown.md"
    report_path = output_dir / "report_decaying_threshold_study.md"

    detailed_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    peeling_breakdown_rows: List[Dict[str, Any]] = []

    learner_display_names = [
        {"logistic": "Logistic", "svm_calibrated": "SVM", "svm": "SVM", "mlp": "MLP"}.get(l, l.upper())
        for l in base_learners
    ]

    peeling_breakdown_csv = output_dir / "decay_peeling_breakdown.csv"

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

    if peeling_breakdown_csv.exists():
        try:
            df_old_pb = pd.read_csv(peeling_breakdown_csv)
            for d in datasets:
                df_old_pb = df_old_pb[
                    ~((df_old_pb["dataset"] == d) & (df_old_pb["learner"].isin(learner_display_names)))
                ]
            peeling_breakdown_rows = df_old_pb.to_dict("records")
        except Exception:
            peeling_breakdown_rows = []

    print("\n" + "=" * 90, flush=True)
    print("BENCHMARK GSI-MLC-PA v6.2: DECAYING PEELING THRESHOLD STUDY", flush=True)
    print(f"Datasets ({len(datasets)}): {datasets}", flush=True)
    print(f"Base Learners: {base_learners}", flush=True)
    print(f"Outer CV: {n_splits}-Fold | Cost c: {cost} | Corr tau: {corr_threshold}", flush=True)
    print(f"Peeling Thresholds: Fixed={init_threshold} vs Decay={init_threshold}->{init_threshold - decay_step}->{init_threshold - 2 * decay_step}", flush=True)
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

            # Analyze full dataset peeling structure
            scaler_full = MaxAbsScaler()
            X_full_scaled = scaler_full.fit_transform(X)

            clf_fixed_full = GSIMLCPAv6_2Classifier(
                base_learner=learner,
                stratified_threshold=init_threshold,
                residual_corr_threshold=corr_threshold,
                cv_folds=5,
                max_peeling_depth=3,
                decaying_threshold=False,
                cost=cost,
                random_state=random_state,
            )
            clf_fixed_full.fit(X_full_scaled, Y)

            clf_decay_full = GSIMLCPAv6_2Classifier(
                base_learner=learner,
                stratified_threshold=init_threshold,
                residual_corr_threshold=corr_threshold,
                cv_folds=5,
                max_peeling_depth=3,
                decaying_threshold=True,
                threshold_decay_step=decay_step,
                cost=cost,
                random_state=random_state,
            )
            clf_decay_full.fit(X_full_scaled, Y)

            peeling_breakdown_rows.append({
                "learner": display_learner,
                "dataset": dataset_name,
                "fixed_layers": str(clf_fixed_full.independent_layers_),
                "fixed_n_il": len(clf_fixed_full.independent_labels_),
                "fixed_dl": str(clf_fixed_full.dependent_labels_),
                "fixed_n_dl": len(clf_fixed_full.dependent_labels_),
                "decay_layers": str(clf_decay_full.independent_layers_),
                "decay_n_il": len(clf_decay_full.independent_labels_),
                "decay_dl": str(clf_decay_full.dependent_labels_),
                "decay_n_dl": len(clf_decay_full.dependent_labels_),
            })

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
                "GSI_v6_2_Fixed": lambda: GSIMLCPAv6_2Classifier(
                    base_learner=learner,
                    stratified_threshold=init_threshold,
                    residual_corr_threshold=corr_threshold,
                    cv_folds=5,
                    max_peeling_depth=3,
                    decaying_threshold=False,
                    cost=cost,
                    random_state=random_state,
                ),
                "GSI_v6_2_Decay": lambda: GSIMLCPAv6_2Classifier(
                    base_learner=learner,
                    stratified_threshold=init_threshold,
                    residual_corr_threshold=corr_threshold,
                    cv_folds=5,
                    max_peeling_depth=3,
                    decaying_threshold=True,
                    threshold_decay_step=decay_step,
                    cost=cost,
                    random_state=random_state,
                ),
            }

            model_fold_metrics: Dict[str, List[Dict[str, float]]] = {
                m: [] for m in models_to_test
            }

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
                    test_time = time.perf_counter() - t1

                    metrics["train_time"] = train_time
                    metrics["test_time"] = test_time
                    model_fold_metrics[model_name].append(metrics)

                    detailed_rows.append({
                        "dataset": dataset_name,
                        "learner": display_learner,
                        "model": model_name,
                        "fold": fold_idx + 1,
                        "train_time": train_time,
                        "test_time": test_time,
                        **metrics,
                    })

                fixed_f1 = model_fold_metrics["GSI_v6_2_Fixed"][-1]["Selective_Macro_F1"]
                decay_f1 = model_fold_metrics["GSI_v6_2_Decay"][-1]["Selective_Macro_F1"]
                diff = decay_f1 - fixed_f1
                diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
                print(
                    f"     [Fold {fold_idx + 1}/{n_splits}] "
                    f"Fixed_F1: {fixed_f1:.4f} | Decay_F1: {decay_f1:.4f} ({diff_str}) | "
                    f"Fixed_Cov: {model_fold_metrics['GSI_v6_2_Fixed'][-1]['Coverage']:.3f} | "
                    f"Decay_Cov: {model_fold_metrics['GSI_v6_2_Decay'][-1]['Coverage']:.3f}",
                    flush=True,
                )

            # Aggregate fold metrics for summary table
            for model_name in models_to_test:
                f_list = model_fold_metrics[model_name]
                summary_rows.append({
                    "dataset": dataset_name,
                    "learner": display_learner,
                    "model": model_name,
                    "Selective_Macro_F1_mean": float(np.mean([m["Selective_Macro_F1"] for m in f_list])),
                    "Selective_Macro_F1_std": float(np.std([m["Selective_Macro_F1"] for m in f_list])),
                    "Full_Macro_F1_mean": float(np.mean([m["Full_Macro_F1"] for m in f_list])),
                    "Coverage_mean": float(np.mean([m["Coverage"] for m in f_list])),
                    "Hamming_Loss_mean": float(np.mean([m["Hamming_Loss"] for m in f_list])),
                    "Selective_Hamming_Loss_mean": float(np.mean([m["Selective_Hamming_Loss"] for m in f_list])),
                    "train_time_mean": float(np.mean([m["train_time"] for m in f_list])),
                    "test_time_mean": float(np.mean([m["test_time"] for m in f_list])),
                })

            # Check difference between Fixed and Decay
            mean_fixed = [r for r in summary_rows if r["dataset"] == dataset_name and r["learner"] == display_learner and r["model"] == "GSI_v6_2_Fixed"][-1]["Selective_Macro_F1_mean"]
            mean_decay = [r for r in summary_rows if r["dataset"] == dataset_name and r["learner"] == display_learner and r["model"] == "GSI_v6_2_Decay"][-1]["Selective_Macro_F1_mean"]
            cov_fixed = [r for r in summary_rows if r["dataset"] == dataset_name and r["learner"] == display_learner and r["model"] == "GSI_v6_2_Fixed"][-1]["Coverage_mean"]
            cov_decay = [r for r in summary_rows if r["dataset"] == dataset_name and r["learner"] == display_learner and r["model"] == "GSI_v6_2_Decay"][-1]["Coverage_mean"]
            d_f1 = mean_decay - mean_fixed
            d_cov = cov_decay - cov_fixed
            print(
                f"   ==> MEAN F1: Fixed={mean_fixed:.4f} vs Decay={mean_decay:.4f} "
                f"(Diff: {'+' if d_f1>=0 else ''}{d_f1:.4f}) | "
                f"Coverage: Fixed={cov_fixed*100:.1f}% vs Decay={cov_decay*100:.1f}% "
                f"(Diff: {'+' if d_cov>=0 else ''}{d_cov*100:.1f}%)",
                flush=True,
            )

    df_detailed = pd.DataFrame(detailed_rows)
    df_detailed.to_csv(detailed_csv_path, index=False)
    print(f"\n[Saved] Detailed fold results: {detailed_csv_path}", flush=True)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"[Saved] Summary results: {summary_csv_path}", flush=True)

    # Save peeling breakdown
    df_peeling = pd.DataFrame(peeling_breakdown_rows)
    df_peeling.to_csv(peeling_breakdown_csv, index=False)
    with open(peeling_breakdown_path, "w", encoding="utf-8") as f:
        f.write("# BẢNG PHÂN TÁCH CÁC TẦNG IL & TẬP DL (CỐ ĐỊNH vs HẠ NGƯỠNG THEO TẦNG)\n\n")
        f.write("So sánh cấu trúc bóc tách giữa ngưỡng cố định (tau=0.75) và cơ chế hạ ngưỡng (0.75 -> 0.70 -> 0.65):\n\n")
        f.write(df_peeling.to_markdown(index=False))
        f.write("\n")
    print(f"[Saved] Peeling breakdown: {peeling_breakdown_path} & {peeling_breakdown_csv}", flush=True)

    # Generate Markdown Report
    generate_markdown_report(
        df_summary=df_summary,
        df_peeling=df_peeling,
        report_path=report_path,
        datasets=datasets,
        base_learners=base_learners,
    )
    print(f"[Saved] Comparative study report: {report_path}", flush=True)

    return df_summary, df_detailed


def generate_markdown_report(
    df_summary: pd.DataFrame,
    df_peeling: pd.DataFrame,
    report_path: Path,
    datasets: List[str],
    base_learners: List[str],
) -> None:
    """Generate professional scientific markdown report."""
    unique_datasets = list(df_summary["dataset"].unique())
    lines: List[str] = [
        "# BÁO CÁO THỰC NGHIỆM: HẠ NGƯỠNG BÓC TÁCH NHÃN IL THEO TẦNG TRÊN GSI-MLC-PA v6.2 (PHIÊN BẢN v6.2.1)",
        "",
        "**Tài liệu tham chiếu:** `meeting_summary.md` (Ưu tiên làm trước), `spec/spec_v6_2.md`",
        "**Cấu hình thực nghiệm:**",
        f"- **Tập dữ liệu ({len(unique_datasets)} datasets):** {', '.join(unique_datasets)}",
        "- **Bộ phân loại cơ sở (3 learners):** Logistic Regression (`Logistic`), Calibrated SVM (`SVM`), Multi-Layer Perceptron (`MLP`)",
        "- **Quy tắc phân tầng:**",
        "  - **GSI_v6_2_Fixed:** $\\tau_{\\text{f1}} = 0.75$ cố định cho tất cả các tầng.",
        "  - **GSI_v6_2_Decay (v6.2.1):** Hạ ngưỡng theo tầng: Tầng 1: $\\tau = 0.75$; Tầng 2: $\\tau = 0.70$; Tầng 3: $\\tau = 0.65$.",
        "- **Các tham số khác:** $\\tau_{\\text{corr}} = 0.25$, Chi phí từ chối $c = 0.30$, 5-Fold Stratified Cross-Validation.",
        "",
        "---",
        "",
        "## 1. Bảng So Sánh Selective Macro-F1 (Mean ± Std)",
        "",
    ]

    piv_f1 = df_summary.pivot(
        index=["learner", "dataset"],
        columns="model",
        values="Selective_Macro_F1_mean",
    )
    lines.append(piv_f1.to_markdown())

    lines.append("\n\n## 2. Bảng Chênh Lệch Hiệu Năng: GSI_v6_2_Decay vs GSI_v6_2_Fixed\n")
    piv_compare = df_summary[df_summary["model"].isin(["GSI_v6_2_Fixed", "GSI_v6_2_Decay"])].pivot(
        index=["learner", "dataset"],
        columns="model",
        values=["Selective_Macro_F1_mean", "Coverage_mean", "Hamming_Loss_mean"],
    )

    diff_df = pd.DataFrame(index=piv_compare.index)
    diff_df["Fixed_Macro_F1"] = piv_compare[("Selective_Macro_F1_mean", "GSI_v6_2_Fixed")].round(4)
    diff_df["Decay_Macro_F1"] = piv_compare[("Selective_Macro_F1_mean", "GSI_v6_2_Decay")].round(4)
    diff_df["Δ F1"] = (diff_df["Decay_Macro_F1"] - diff_df["Fixed_Macro_F1"]).round(4)

    diff_df["Fixed_Coverage"] = (piv_compare[("Coverage_mean", "GSI_v6_2_Fixed")] * 100).round(2).astype(str) + "%"
    diff_df["Decay_Coverage"] = (piv_compare[("Coverage_mean", "GSI_v6_2_Decay")] * 100).round(2).astype(str) + "%"
    diff_df["Δ Coverage"] = ((piv_compare[("Coverage_mean", "GSI_v6_2_Decay")] - piv_compare[("Coverage_mean", "GSI_v6_2_Fixed")]) * 100).round(2).astype(str) + "%"

    diff_df["Fixed_Hamming"] = piv_compare[("Hamming_Loss_mean", "GSI_v6_2_Fixed")].round(4)
    diff_df["Decay_Hamming"] = piv_compare[("Hamming_Loss_mean", "GSI_v6_2_Decay")].round(4)
    diff_df["Δ Hamming"] = (diff_df["Decay_Hamming"] - diff_df["Fixed_Hamming"]).round(4)

    lines.append(diff_df.to_markdown())

    lines.append("\n\n## 3. Bảng Cấu Trúc Bóc Tách Nhãn IL & DL (Toàn Bộ Tập Dữ Liệu)\n")
    lines.append(df_peeling.to_markdown(index=False))

    lines.append("\n\n## 4. Bảng Tỷ Lệ Quyết Định (Coverage) Toàn Bộ Mô Hình\n")
    piv_cov = df_summary.pivot(
        index=["learner", "dataset"],
        columns="model",
        values="Coverage_mean",
    )
    lines.append((piv_cov * 100).round(2).astype(str).add("%").to_markdown())

    lines.append("\n\n## 5. Bảng Hamming Loss Toàn Bộ Mô Hình\n")
    piv_ham = df_summary.pivot(
        index=["learner", "dataset"],
        columns="model",
        values="Hamming_Loss_mean",
    )
    lines.append(piv_ham.round(4).to_markdown())

    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Run GSI-MLC-PA v6.2 Decaying Peeling Threshold Study")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=FIRST_5_DATASETS,
        help="Datasets to benchmark. Can be list of names, or 'remaining', or 'all'.",
    )
    parser.add_argument(
        "--base-learners",
        nargs="+",
        default=DEFAULT_BASE_LEARNERS,
        choices=["logistic", "svm_calibrated", "mlp"],
        help="Base learners to evaluate.",
    )
    parser.add_argument("--threshold", type=float, default=0.75, help="Initial Selective-F1 threshold")
    parser.add_argument("--decay-step", type=float, default=0.05, help="Threshold decay step per stage")
    parser.add_argument("--corr-threshold", type=float, default=0.25, help="PCC residual error threshold")
    parser.add_argument("--cv", type=int, default=5, help="Outer CV folds")
    parser.add_argument("--cost", type=float, default=0.30, help="Rejection cost c")
    args = parser.parse_args()

    REMAINING_5_DATASETS = ["genbase", "humanpseaac", "plantpseaac", "viruspseaac", "yeast"]
    ALL_10_DATASETS = FIRST_5_DATASETS + REMAINING_5_DATASETS

    if args.datasets == ["all"]:
        datasets = ALL_10_DATASETS
    elif args.datasets == ["remaining"]:
        datasets = REMAINING_5_DATASETS
    else:
        datasets = args.datasets

    run_decay_benchmark(
        datasets=datasets,
        base_learners=args.base_learners,
        init_threshold=args.threshold,
        decay_step=args.decay_step,
        corr_threshold=args.corr_threshold,
        n_splits=args.cv,
        cost=args.cost,
    )


if __name__ == "__main__":
    main()
