import itertools
import json
import tempfile
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator

from src.evaluation.cache import (
    backup_model_cache,
    import_legacy_model_results,
    load_model_cache,
    save_model_cache,
)
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_partial_abstention_metrics,
)
from src.models import (
    BinaryRelevanceClassifier,
    BinaryRelevanceLogisticRegression,
    BinaryRelevanceMLP,
    ClassifierChainClassifier,
    GSIMLCPartialAbstentionClassifier,
    MLCPartialAbstentionClassifier,
)
from src.visualization.plots import generate_pa_plots


class _FixedProbabilityEstimator(BaseEstimator):
    """Cloneable test estimator whose input contains its probabilities."""

    def fit(self, X, Y):
        self.n_labels_ = np.asarray(Y).shape[1]
        return self

    def predict_proba(self, X):
        self.predict_calls_ = getattr(self, "predict_calls_", 0) + 1
        return np.asarray(X, dtype=np.float64)[:, :self.n_labels_]


class _MeanFieldConditionalClassifier:
    """P(Y=1)=0.1+0.8*mean(parent features), for exact test oracles."""

    def __init__(self, n_parents):
        self.n_parents = n_parents

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.n_parents:
            parent_mean = X[:, -self.n_parents:].mean(axis=1)
        else:
            parent_mean = np.full(X.shape[0], 0.5)
        positive = 0.1 + 0.8 * parent_mean
        return np.column_stack((1.0 - positive, positive))


class _MeanFieldChainEstimator(BaseEstimator):
    def fit(self, X, Y):
        self.fit_size_ = len(X)
        self.order_ = list(range(np.asarray(Y).shape[1]))
        self.classifiers_ = [
            _MeanFieldConditionalClassifier(position)
            for position in range(len(self.order_))
        ]
        return self

def run_tests():
    # 1. Synthetic data
    np.random.seed(42)
    X = np.random.randn(60, 12)
    Y = np.random.randint(0, 2, size=(60, 5))
    # Make label 4 degenerate (all 0)
    Y[:, 4] = 0

    print("1. Testing BinaryRelevanceLogisticRegression...")
    clf_lr = BinaryRelevanceLogisticRegression()
    clf_lr.fit(X, Y)
    preds_lr = clf_lr.predict(X)
    probs_lr = clf_lr.predict_proba(X)
    scores_lr = clf_lr.decision_function(X)

    assert preds_lr.shape == (60, 5), f"Shape mismatch: {preds_lr.shape}"
    assert probs_lr.shape == (60, 5), f"Probs shape mismatch: {probs_lr.shape}"
    assert scores_lr.shape == (60, 5), f"Scores shape mismatch: {scores_lr.shape}"
    assert np.all((probs_lr >= 0.0) & (probs_lr <= 1.0)), "Probs out of [0, 1] range"
    assert np.all(preds_lr[:, 4] == 0), "Degenerate label not 0"
    assert np.all(probs_lr[:, 4] == 0.0), "Degenerate label prob not 0.0"
    print("   -> Passed BinaryRelevanceLogisticRegression!")

    print("2. Testing BinaryRelevanceClassifier with base_estimator='logistic'...")
    clf_br_str = BinaryRelevanceClassifier(base_estimator="logistic")
    clf_br_str.fit(X, Y)
    preds_br = clf_br_str.predict(X)
    probs_br = clf_br_str.predict_proba(X)
    assert preds_br.shape == (60, 5)
    assert probs_br.shape == (60, 5)
    assert np.all((probs_br >= 0.0) & (probs_br <= 1.0))
    print("   -> Passed BinaryRelevanceClassifier(base_estimator='logistic')!")

    print("3. Testing BinaryRelevanceClassifier with base_estimator='svm' (LinearSVC)...")
    clf_br_svm = BinaryRelevanceClassifier(base_estimator="svm")
    clf_br_svm.fit(X, Y)
    preds_svm = clf_br_svm.predict(X)
    probs_svm = clf_br_svm.predict_proba(X)
    assert preds_svm.shape == (60, 5)
    assert probs_svm.shape == (60, 5)
    assert np.all((probs_svm >= 0.0) & (probs_svm <= 1.0))
    print("   -> Passed BinaryRelevanceClassifier(base_estimator='svm')!")

    print("4. Testing BinaryRelevanceMLP...")
    clf_br_mlp = BinaryRelevanceMLP(hidden_layer_sizes=(32,), max_iter=100)
    clf_br_mlp.fit(X, Y)
    preds_mlp = clf_br_mlp.predict(X)
    probs_mlp = clf_br_mlp.predict_proba(X)
    assert preds_mlp.shape == (60, 5)
    assert probs_mlp.shape == (60, 5)
    assert np.all((probs_mlp >= 0.0) & (probs_mlp <= 1.0))
    assert np.all(preds_mlp[:, 4] == 0)
    print("   -> Passed BinaryRelevanceMLP!")

    print("5. Testing ClassifierChain with base_estimator='logistic'...")
    clf_cc_lr = ClassifierChainClassifier(base_estimator="logistic")
    clf_cc_lr.fit(X, Y)
    preds_cc = clf_cc_lr.predict(X)
    probs_cc = clf_cc_lr.predict_proba(X)
    assert preds_cc.shape == (60, 5)
    assert probs_cc.shape == (60, 5)
    assert np.all((probs_cc >= 0.0) & (probs_cc <= 1.0))
    print("   -> Passed ClassifierChainClassifier(base_estimator='logistic')!")

    print("6. Testing ClassifierChain with base_estimator='mlp'...")
    clf_cc_mlp = ClassifierChainClassifier(base_estimator="mlp")
    clf_cc_mlp.fit(X, Y)
    preds_cc_mlp = clf_cc_mlp.predict(X)
    probs_cc_mlp = clf_cc_mlp.predict_proba(X)
    assert preds_cc_mlp.shape == (60, 5)
    assert probs_cc_mlp.shape == (60, 5)
    assert np.all((probs_cc_mlp >= 0.0) & (probs_cc_mlp <= 1.0))
    print("   -> Passed ClassifierChainClassifier(base_estimator='mlp')!")

    print("7. Testing MLC-PA linear/SEP Bayes decision rule...")
    probabilities = np.array([[0.10, 0.20, 0.21, 0.79, 0.80, 0.90]])
    y_dummy = np.zeros_like(probabilities, dtype=np.int32)
    mlc_sep = MLCPartialAbstentionClassifier(
        base_estimator=_FixedProbabilityEstimator(),
        cost=0.20,
        penalty="linear",
    ).fit(probabilities, y_dummy)
    np.testing.assert_array_equal(
        mlc_sep.predict(probabilities),
        np.array([[0, 0, -1, -1, 1, 1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        mlc_sep.predict_full(probabilities),
        np.array([[0, 0, 0, 1, 1, 1]], dtype=np.int32),
    )
    mlc_no_abstention = MLCPartialAbstentionClassifier(
        base_estimator=_FixedProbabilityEstimator(),
        cost=0.50,
        penalty="linear",
    ).fit(probabilities, y_dummy)
    assert np.all(mlc_no_abstention.decision_mask(probabilities))
    print("   -> Passed MLC-PA SEP rule!")

    print("8. Testing MLC-PA concave/PAR global optimum...")
    par_probabilities = np.array([[0.05, 0.24, 0.39, 0.52]])
    par_cost = 0.20
    mlc_par = MLCPartialAbstentionClassifier(
        base_estimator=_FixedProbabilityEstimator(),
        cost=par_cost,
        penalty="concave",
    ).fit(par_probabilities, np.zeros_like(par_probabilities, dtype=np.int32))
    actual_mask = mlc_par.decision_mask(par_probabilities)[0]
    label_risks = np.minimum(par_probabilities[0], 1.0 - par_probabilities[0])
    n_labels = label_risks.size
    candidates = []
    for bits in itertools.product((False, True), repeat=n_labels):
        mask = np.asarray(bits, dtype=bool)
        abstentions = n_labels - int(mask.sum())
        risk = float(label_risks[mask].sum())
        risk += abstentions * n_labels * par_cost / (n_labels + abstentions)
        candidates.append((risk, int(mask.sum()), mask))
    minimum_risk = min(item[0] for item in candidates)
    maximum_decisions_at_minimum = max(
        item[1] for item in candidates
        if np.isclose(item[0], minimum_risk)
    )
    actual_abstentions = n_labels - int(actual_mask.sum())
    actual_risk = float(label_risks[actual_mask].sum())
    actual_risk += (
        actual_abstentions
        * n_labels
        * par_cost
        / (n_labels + actual_abstentions)
    )
    assert np.isclose(actual_risk, minimum_risk)
    assert int(actual_mask.sum()) == maximum_decisions_at_minimum
    decided_risks = label_risks[actual_mask]
    abstained_risks = label_risks[~actual_mask]
    if decided_risks.size and abstained_risks.size:
        assert decided_risks.max() <= abstained_risks.min()
    print("   -> Passed MLC-PA PAR global-optimum rule!")

    print("9. Testing partial-abstention metrics for SEP and PAR...")
    y_partial_true = np.array([[0, 1, 1, 0]], dtype=np.int32)
    y_partial_pred = np.array([[0, -1, 0, -1]], dtype=np.int32)
    complete_metrics = compute_all_metrics(y_partial_true, y_partial_true)
    assert "Macro Precision" not in complete_metrics
    assert "Macro Recall" not in complete_metrics
    sep_metrics = compute_partial_abstention_metrics(
        y_partial_true, y_partial_pred, cost=0.25, penalty="linear"
    )
    assert np.isclose(sep_metrics["Generalized Hamming Loss"], 0.375)
    assert np.isclose(sep_metrics["Selective Hamming Loss"], 0.5)
    assert np.isclose(sep_metrics["Coverage"], 0.5)
    assert np.isclose(sep_metrics["Abstention Rate"], 0.5)
    assert np.isclose(sep_metrics["Abstentions-as-Zero Macro-F1"], 0.0)
    assert np.isclose(sep_metrics["Abstentions-as-Zero Micro-F1"], 0.0)
    assert np.isclose(sep_metrics["ABS"], 1.0)
    assert np.isclose(sep_metrics["AABS"], 0.5)
    par_metrics = compute_partial_abstention_metrics(
        y_partial_true, y_partial_pred, cost=0.25, penalty="concave"
    )
    assert np.isclose(par_metrics["Generalized Hamming Loss"], 1.0 / 3.0)

    f1_true = np.array(
        [[1, 1], [1, 0], [0, 1], [0, 0]], dtype=np.int32
    )
    f1_partial = np.array(
        [[1, -1], [-1, 0], [0, 1], [1, -1]], dtype=np.int32
    )
    f1_metrics = compute_partial_abstention_metrics(
        f1_true, f1_partial, cost=0.25, penalty="linear"
    )
    assert np.isclose(
        f1_metrics["Abstentions-as-Zero Macro-F1"], 7.0 / 12.0
    )
    assert np.isclose(
        f1_metrics["Abstentions-as-Zero Micro-F1"], 4.0 / 7.0
    )
    assert np.isclose(f1_metrics["Selective Macro-F1"], 5.0 / 6.0)
    assert np.isclose(f1_metrics["Selective Micro-F1"], 4.0 / 5.0)

    all_abstained = compute_partial_abstention_metrics(
        f1_true, np.full_like(f1_true, -1), cost=0.25, penalty="linear"
    )
    assert all_abstained["Selective Macro-F1"] == 0.0
    assert all_abstained["Selective Micro-F1"] == 0.0
    print("   -> Passed MLC-PA partial metrics!")

    print("10. Testing per-model JSON cache and legacy import...")
    with tempfile.TemporaryDirectory() as temporary_directory:
        cached_datasets = {
            "emotions": {
                "mean": {"Macro-F1": 0.5},
                "std": {"Macro-F1": 0.1},
                "raw_folds": [],
            }
        }
        cache_path = save_model_cache(
            temporary_directory,
            "MLC_PA",
            {
                **cached_datasets,
                "scene": {
                    "full": {
                        "mean": {
                            "Macro-F1": 0.5,
                            "Macro Precision": 0.6,
                            "Macro Recall": 0.4,
                        },
                        "std": {},
                        "raw_folds": [],
                    },
                    "costs": {
                        "0.30": {
                            "mean": {
                                "Selective Macro-F1": 0.7,
                                "Selective Micro-F1": 0.8,
                            },
                            "std": {},
                            "raw_folds": [],
                        }
                    },
                },
            },
            settings={"abstention_cost": 0.25},
        )
        assert cache_path.name == "MLC_PA.json"
        persisted_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        persisted_text = cache_path.read_text(encoding="utf-8")
        assert "Macro Precision" not in persisted_text
        assert "Macro Recall" not in persisted_text
        selective_mean = (
            persisted_cache["datasets"]["scene"]["costs"]["0.30"]["mean"]
        )
        assert selective_mean["Selective Macro-F1"] == 0.7
        assert selective_mean["Selective Micro-F1"] == 0.8
        loaded_cache = load_model_cache(temporary_directory, "MLC_PA")
        assert loaded_cache["datasets"]["emotions"] == cached_datasets["emotions"]
        assert loaded_cache["settings"]["abstention_cost"] == 0.25
        backup_path = backup_model_cache(
            temporary_directory, "MLC_PA", reason="schema_test"
        )
        assert backup_path.exists()

        legacy = {"emotions": {"BR_MLP": cached_datasets["emotions"]}}
        imported = import_legacy_model_results(legacy, "BR_MLP")
        assert imported == cached_datasets
    print("   -> Passed resumable per-model cache!")

    print("11. Testing GSI decision-time BOP cost sweep...")
    gsi = GSIMLCPartialAbstentionClassifier(
        br_estimator=_FixedProbabilityEstimator(),
        cc_estimator=_MeanFieldChainEstimator(),
    )
    boundary_probabilities = np.array([[0.29, 0.30, 0.50, 0.70, 0.71]])
    np.testing.assert_array_equal(
        gsi._apply_bop(boundary_probabilities),
        np.array([[0, 0, -1, 1, 1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        gsi._apply_bop(boundary_probabilities, cost=0.2),
        np.array([[-1, -1, -1, -1, -1]], dtype=np.int32),
    )
    GSIMLCPartialAbstentionClassifier(cost=0.2)._validate_parameters()
    print("   -> Passed decision-time cost sweep boundaries!")

    print("12. Testing two-state and mean-field marginalization...")
    gsi.n_labels_ = 3
    gsi.order_ = [0, 1, 2]
    inference_x = np.array([[0.2, 0.8], [0.7, 0.1]], dtype=np.float32)
    direct_probabilities = np.array(
        [[0.25, 0.60, 0.40], [0.75, 0.20, 0.80]], dtype=np.float64
    )
    fake_chain = _MeanFieldChainEstimator().fit(
        inference_x, np.zeros((2, 3), dtype=np.int32)
    )
    one_parent_probabilities = gsi._configured_probabilities(
        inference_x, direct_probabilities, fake_chain, independent_labels=[0]
    )
    # One parent: (1-p)*0.1 + p*0.9 = 0.1 + 0.8*p.
    np.testing.assert_allclose(
        one_parent_probabilities[:, 1],
        0.1 + 0.8 * direct_probabilities[:, 0],
    )
    mean_field_probabilities = gsi._configured_probabilities(
        inference_x,
        direct_probabilities,
        fake_chain,
        independent_labels=[0, 1],
    )
    # Two parents: f(x, E[Y0], E[Y1]) under the mean-field plug-in.
    np.testing.assert_allclose(
        mean_field_probabilities[:, 2],
        0.1 + 0.8 * direct_probabilities[:, :2].mean(axis=1),
    )
    all_dependent_probabilities = gsi._configured_probabilities(
        inference_x, direct_probabilities, fake_chain, independent_labels=[]
    )
    np.testing.assert_allclose(all_dependent_probabilities[:, 0], 0.5)
    prefix_reused = gsi._configured_probabilities(
        inference_x,
        direct_probabilities,
        fake_chain,
        independent_labels=[1],
        initial_probabilities=all_dependent_probabilities,
        start_position=1,
    )
    full_recomputation = gsi._configured_probabilities(
        inference_x, direct_probabilities, fake_chain, independent_labels=[1]
    )
    np.testing.assert_allclose(prefix_reused, full_recomputation)
    print("   -> Passed marginalization rules!")

    print("13. Testing rejection-free sequential IL/DL selection...")
    gsi.n_labels_ = 3
    gsi.order_ = [0, 1, 2]
    selection_y = np.array(
        [[0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1]],
        dtype=np.int32,
    )
    direct = np.array(
        [
            [0.1, 0.1, 0.9],
            [0.9, 0.9, 0.1],
            [0.1, 0.1, 0.9],
            [0.9, 0.9, 0.1],
        ]
    )
    selection_x = np.zeros((4, 2), dtype=np.float32)
    selection_chain = _MeanFieldChainEstimator().fit(selection_x, selection_y)
    # Any rejection call during partition selection must fail this test.
    original_apply_bop = gsi._apply_bop
    gsi._apply_bop = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("rejection was used during IL/DL selection")
    )
    independent, dependent_labels, score, history = gsi._select_partition(
        selection_x, selection_y, direct, selection_chain
    )
    gsi._apply_bop = original_apply_bop
    assert independent == [0]
    assert dependent_labels == [1, 2]
    assert score > history[0]["score"]
    assert history[1]["tested_label"] == 0
    assert history[1]["accepted"] is True
    assert history[2]["accepted"] is False
    assert gsi.evaluated_configurations_ == 4
    print("   -> Passed rejection-free complete Macro-F1 objective!")

    print("14. Testing GSI internal validation, frozen partition, and refit...")
    gsi_x = np.linspace(0.02, 0.98, 120, dtype=np.float32).reshape(40, 3)
    gsi_y = (gsi_x > np.array([0.45, 0.50, 0.55])).astype(np.int32)
    fitted_gsi = GSIMLCPartialAbstentionClassifier(
        validation_size=0.25,
        br_estimator=_FixedProbabilityEstimator(),
        cc_estimator=_MeanFieldChainEstimator(),
        random_state=7,
    ).fit(gsi_x, gsi_y)
    assert fitted_gsi.selection_train_size_ + fitted_gsi.validation_size_ == 40
    assert fitted_gsi.selection_train_size_ < 40
    assert fitted_gsi.refit_train_size_ == 40
    assert fitted_gsi.br_model_.n_labels_ == 3
    assert fitted_gsi.cc_model_.fit_size_ == 40
    assert sorted(
        fitted_gsi.independent_labels_ + fitted_gsi.dependent_labels_
    ) == [0, 1, 2]
    assert not set(fitted_gsi.independent_labels_) & set(
        fitted_gsi.dependent_labels_
    )
    assert fitted_gsi.label_correlation_.shape == (3, 3)
    assert sorted(fitted_gsi.correlation_order_) == [0, 1, 2]
    assert set(fitted_gsi.dependent_parent_map_) == set(
        fitted_gsi.dependent_labels_
    )
    final_probabilities = fitted_gsi.predict_proba(gsi_x[:5])
    final_partial = fitted_gsi.predict(gsi_x[:5])
    assert final_probabilities.shape == (5, 3)
    assert np.all((final_probabilities >= 0) & (final_probabilities <= 1))
    assert np.all(np.isin(final_partial, (-1, 0, 1)))
    from main import (
        _cache_settings,
        _create_model,
        _evaluate_model,
        _standardize_model_name,
    )
    calls_before_test_evaluation = fitted_gsi.br_model_.predict_calls_
    evaluation_metrics = _evaluate_model(
        "GSI_MLC_PA",
        fitted_gsi,
        gsi_x[:5],
        gsi_y[:5],
        abstention_costs=[0.2, 0.25, 0.3, 0.35, 0.4],
    )
    assert fitted_gsi.br_model_.predict_calls_ == calls_before_test_evaluation + 1
    assert "Macro-F1" in evaluation_metrics["full"]
    assert sorted(evaluation_metrics["costs"]) == [
        "0.20", "0.25", "0.30", "0.35", "0.40"
    ]
    assert "ABS" in evaluation_metrics["costs"]["0.30"]
    assert _standardize_model_name("gsimlcpa") == "GSI_MLC_PA"
    factory_model = _create_model("GSI_MLC_PA", gsi_validation_size=0.25)
    assert isinstance(factory_model, GSIMLCPartialAbstentionClassifier)
    assert factory_model.cost == 0.3
    assert MLCPartialAbstentionClassifier().cost == 0.3
    gsi_settings = _cache_settings(
        "GSI_MLC_PA",
        5,
        42,
        [0.2, 0.25, 0.3, 0.35, 0.4],
        0.3,
        "linear",
        "mlp",
        0.25,
    )
    assert gsi_settings["abstention_costs"] == [0.2, 0.25, 0.3, 0.35, 0.4]
    assert gsi_settings["validation_size"] == 0.25
    assert gsi_settings["base_estimators"] == "BR_MLP+CC_MLP"
    assert (
        gsi_settings["f1_abstention_policy"]
        == "full_separate_selective_ignore"
    )
    print("   -> Passed leakage-safe selection/refit flow!")

    print("15. Testing schema-v2 partial-abstention plots...")
    costs = [0.2, 0.25, 0.3, 0.35, 0.4]
    synthetic_results = {}
    for dataset_index, dataset_name in enumerate(("demo_a", "demo_b")):
        synthetic_results[dataset_name] = {}
        for model_index, model_name in enumerate(
            ("BR_MLP", "CC_MLP", "MLC_PA", "GSI_MLC_PA")
        ):
            base = 0.55 + 0.03 * model_index + 0.01 * dataset_index
            full_summary = {
                "mean": {
                    "Macro-F1": base,
                    "Micro-F1": base + 0.02,
                    "Hamming Loss": 0.25 - 0.02 * model_index,
                },
                "std": {
                    "Macro-F1": 0.02,
                    "Micro-F1": 0.015,
                    "Hamming Loss": 0.01,
                },
                "raw_folds": [],
            }
            cost_summaries = {}
            if model_name in ("MLC_PA", "GSI_MLC_PA"):
                for cost in costs:
                    key = f"{cost:.2f}"
                    cost_summaries[key] = {
                        "mean": {
                            "Selective Macro-F1": min(base + 0.08, 0.99),
                            "Selective Micro-F1": min(base + 0.09, 0.99),
                            "Generalized Loss": 0.18,
                            "ABS": 0.42,
                            "AABS": 0.21,
                        },
                        "std": {
                            "Selective Macro-F1": 0.025,
                            "Selective Micro-F1": 0.02,
                            "Generalized Loss": 0.012,
                            "ABS": 0.03,
                            "AABS": 0.02,
                        },
                        "raw_folds": [],
                    }
            synthetic_results[dataset_name][model_name] = {
                "full": full_summary,
                "costs": cost_summaries,
            }
    with tempfile.TemporaryDirectory() as temporary_directory:
        plot_directory = Path(temporary_directory) / "plots_pa"
        table_directory = Path(temporary_directory) / "tables"
        generated, csv_path = generate_pa_plots(
            synthetic_results,
            abstention_costs=costs,
            report_cost=0.3,
            output_dir=plot_directory,
            tables_dir=table_directory,
        )
        assert len(generated) == 4
        assert Path(csv_path).exists()
        assert {Path(path).name for path in generated} == {
            "rejection_cost_comparison.png",
            "selective_macro_f1_comparison.png",
            "selective_micro_f1_comparison.png",
            "generalized_loss_comparison.png",
        }
        assert all(Path(path).stat().st_size > 10_000 for path in generated)
    print("   -> Passed schema-v2 plot generation!")

    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
