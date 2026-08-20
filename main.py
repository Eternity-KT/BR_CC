"""
Main Experiment Pipeline for Multi-Label Classification:
- Binary Relevance with LinearSVC (BR)
- Binary Relevance with Logistic Regression (BR_Logistic)
- Classifier Chains with LinearSVC (CC)

Evaluates models across benchmark multi-label datasets using 5-Fold Cross Validation.
Generates all comparison charts, heatmaps, and summary tables.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import load_dataset, DATASET_CONFIG
from src.models.binary_relevance import BinaryRelevanceClassifier, BinaryRelevanceLogisticRegression
from src.models.classifier_chain import ClassifierChainClassifier
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.cv import get_multilabel_cv
from src.visualization.plots import generate_all_plots


def _create_model(model_name, random_state=42):
    """
    Factory function to create model instance by name.
    """
    m = model_name.upper()
    if m in ("BR", "BR_SVC", "BR_LINEARSVC"):
        return BinaryRelevanceClassifier(base_estimator="svm", random_state=random_state)
    elif m in ("BR_LOGISTIC", "BR_LR", "BR_LOGREG"):
        return BinaryRelevanceLogisticRegression(random_state=random_state)
    elif m in ("CC", "CC_SVC", "CC_LINEARSVC"):
        return ClassifierChainClassifier(base_estimator="svm", random_state=random_state)
    elif m in ("CC_LOGISTIC", "CC_LR", "CC_LOGREG"):
        return ClassifierChainClassifier(base_estimator="logistic", random_state=random_state)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported: BR, BR_Logistic, CC, CC_Logistic")


def _standardize_model_name(name):
    m = name.upper()
    if m in ("BR", "BR_SVC", "BR_LINEARSVC"):
        return "BR"
    elif m in ("BR_LOGISTIC", "BR_LR", "BR_LOGREG"):
        return "BR_Logistic"
    elif m in ("CC", "CC_SVC", "CC_LINEARSVC"):
        return "CC"
    elif m in ("CC_LOGISTIC", "CC_LR", "CC_LOGREG"):
        return "CC_Logistic"
    return name


def run_experiment(datasets=None, models=None, n_splits=5, random_state=42, output_dir="results"):
    """
    Run complete experiment pipeline across specified datasets and models.

    Parameters:
        datasets (list, optional): List of dataset names to evaluate. Defaults to all 10 datasets.
        models (list, optional): List of models to evaluate ('BR', 'BR_Logistic', 'CC').
        n_splits (int, default=5): Number of cross-validation folds.
        random_state (int, default=42): Seed for reproducibility.
        output_dir (str, default='results'): Directory to save figures and tables.
    """
    if datasets is None:
        datasets = list(DATASET_CONFIG.keys())

    if models is None:
        models = ["BR", "BR_Logistic", "CC"]

    std_models = [_standardize_model_name(m) for m in models]

    print("=" * 90, flush=True)
    print(" MULTI-LABEL CLASSIFICATION BENCHMARK EXPERIMENT", flush=True)
    print(f" Models ({len(std_models)}):    {', '.join(std_models)}", flush=True)
    print(f" Datasets ({len(datasets)}):  {', '.join(datasets)}", flush=True)
    print(f" Evaluation:    {n_splits}-Fold Multilabel Stratified Cross-Validation", flush=True)
    print(f" Output Dir:    {output_dir}", flush=True)
    print(f" Start Time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 90, flush=True)

    all_results = {}
    total_start_time = time.time()

    for idx, d_name in enumerate(datasets, 1):
        d_start = time.time()
        print(f"\n[{idx}/{len(datasets)}] Processing dataset: {d_name.upper()}...", flush=True)

        # 1. Load data
        X, Y, f_names, l_names = load_dataset(d_name)
        n_samples, n_features = X.shape
        n_labels = Y.shape[1]
        cardinality = float(Y.sum(axis=1).mean())
        density = float(Y.mean())

        print(f"  -> Samples: {n_samples:,} | Features: {n_features:,} | Labels: {n_labels} | "
              f"Cardinality: {cardinality:.2f} | Density: {density:.4f}", flush=True)

        # 2. Setup 5-fold cross-validation
        cv = get_multilabel_cv(n_splits=n_splits, random_state=random_state)
        splits = list(cv.split(X, Y))

        fold_metrics = {m: [] for m in std_models}

        # 3. Cross-validation loop
        for fold_i, (train_idx, test_idx) in enumerate(splits, 1):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            for m_name in std_models:
                t0 = time.time()
                clf = _create_model(m_name, random_state=random_state)
                clf.fit(X_train, Y_train)
                Y_pred = clf.predict(X_test)
                metrics = compute_all_metrics(Y_test, Y_pred)
                metrics["train_time"] = time.time() - t0
                fold_metrics[m_name].append(metrics)

        # 4. Aggregate results across folds
        all_results[d_name] = {}
        for m_name in std_models:
            df_m = pd.DataFrame(fold_metrics[m_name])
            all_results[d_name][m_name] = {
                "mean": df_m.mean().to_dict(),
                "std": df_m.std().to_dict(),
                "raw_folds": fold_metrics[m_name]
            }

        d_elapsed = time.time() - d_start
        f1_summary = " | ".join([
            f"{m} Macro-F1: {all_results[d_name][m]['mean']['Macro-F1']:.4f}±{all_results[d_name][m]['std']['Macro-F1']:.3f}"
            for m in std_models
        ])
        print(f"  -> Done in {d_elapsed:.2f}s | {f1_summary}", flush=True)

    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 90, flush=True)
    print(f" ALL EXPERIMENTS COMPLETED in {total_elapsed:.2f} seconds ({total_elapsed/60:.1f} mins)", flush=True)
    print("=" * 90, flush=True)

    # 5. Generate plots and tables
    figures_dir = os.path.join(output_dir, "figures")
    tables_dir = os.path.join(output_dir, "tables")
    print(f"\nGenerating figures and summary tables in '{output_dir}'...", flush=True)

    gen_files, csv_path = generate_all_plots(all_results, output_dir=figures_dir, tables_dir=tables_dir)

    # Save raw json results
    json_path = os.path.join(tables_dir, "raw_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved CSV Results: {csv_path}", flush=True)
    print(f"Saved Raw JSON:    {json_path}", flush=True)
    print(f"Generated Figures ({len(gen_files)} files):", flush=True)
    for f in gen_files:
        print(f"  - {f}", flush=True)

    # Print summary table in console
    print("\n" + "=" * 110, flush=True)
    print(f"{'DATASET':<15} {'MODEL':<14} {'MACRO-F1':<18} {'MICRO-F1':<18} {'HAMMING LOSS':<18} {'SUBSET ACC':<18}", flush=True)
    print("-" * 110, flush=True)
    for d_name in datasets:
        for m in std_models:
            macro = f"{all_results[d_name][m]['mean']['Macro-F1']:.3f} ± {all_results[d_name][m]['std']['Macro-F1']:.3f}"
            micro = f"{all_results[d_name][m]['mean']['Micro-F1']:.3f} ± {all_results[d_name][m]['std']['Micro-F1']:.3f}"
            hl = f"{all_results[d_name][m]['mean']['Hamming Loss']:.3f} ± {all_results[d_name][m]['std']['Hamming Loss']:.3f}"
            acc = f"{all_results[d_name][m]['mean']['Subset Accuracy']:.3f} ± {all_results[d_name][m]['std']['Subset Accuracy']:.3f}"
            print(f"{d_name.upper():<15} {m:<14} {macro:<18} {micro:<18} {hl:<18} {acc:<18}", flush=True)
        print("-" * 110, flush=True)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Multi-Label Classification Benchmark Pipeline (BR, BR_Logistic, CC)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="List of datasets to evaluate (e.g., emotions music scene). Defaults to all 10 datasets.")
    parser.add_argument("--models", nargs="+", default=["BR", "BR_Logistic", "CC"],
                        help="List of models to evaluate (e.g., BR BR_Logistic CC). Defaults to BR, BR_Logistic, CC.")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="Number of cross-validation folds (default: 5).")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for figures and tables (default: results).")

    args = parser.parse_args()

    run_experiment(
        datasets=args.datasets,
        models=args.models,
        n_splits=args.n_splits,
        random_state=args.random_state,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

