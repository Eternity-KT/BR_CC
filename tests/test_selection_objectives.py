"""Phase-Q6 tests for objective injection into GSI partition selection."""

import unittest
from unittest import mock

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, precision_score, recall_score

from main import _cache_settings, _create_model, _evaluate_model
from src.decision import create_configured_policy
from src.decision.base import abstention_penalty
from src.models.gsi_mlc_pa import GSIMLCPartialAbstentionClassifier
from src.selection import (
    available_selection_objectives,
    canonical_selection_objective,
    evaluate_selection_objective,
)


class _FixedProbabilityEstimator(BaseEstimator):
    def fit(self, x_data, y_data):
        self.n_labels_ = np.asarray(y_data).shape[1]
        return self

    def predict_proba(self, x_data):
        return np.asarray(x_data, dtype=np.float64)[:, : self.n_labels_]


class _ConditionalClassifier:
    def __init__(self, n_parents):
        self.n_parents = n_parents

    def predict_proba(self, x_data):
        x_data = np.asarray(x_data, dtype=np.float64)
        if self.n_parents:
            parent_mean = x_data[:, -self.n_parents :].mean(axis=1)
        else:
            parent_mean = np.full(x_data.shape[0], 0.5)
        positive = 0.1 + 0.8 * parent_mean
        return np.column_stack((1.0 - positive, positive))


class _MeanFieldChainEstimator(BaseEstimator):
    def __init__(self, order=None):
        self.order = order

    def fit(self, x_data, y_data):
        n_labels = np.asarray(y_data).shape[1]
        self.order_ = (
            list(range(n_labels)) if self.order is None else list(self.order)
        )
        self.classifiers_ = [
            _ConditionalClassifier(position) for position in range(n_labels)
        ]
        return self


def _selection_fixture():
    truth = np.array(
        [[0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1]],
        dtype=np.int32,
    )
    direct = np.array(
        [
            [0.1, 0.1, 0.9],
            [0.9, 0.9, 0.1],
            [0.1, 0.1, 0.9],
            [0.9, 0.9, 0.1],
        ],
        dtype=np.float64,
    )
    features = np.zeros((4, 2), dtype=np.float32)
    chain = _MeanFieldChainEstimator().fit(features, truth)
    return features, truth, direct, chain


def _instance_fbeta(truth, predictions, beta=1.0, abstain_value=-1):
    decided = predictions != abstain_value
    tp = np.sum((truth == 1) & (predictions == 1) & decided, axis=1)
    fp = np.sum((truth == 0) & (predictions == 1) & decided, axis=1)
    fn = np.sum((truth == 1) & (predictions == 0) & decided, axis=1)
    beta_squared = beta * beta
    denominator = (1.0 + beta_squared) * tp + beta_squared * fn + fp
    scores = np.ones(len(truth), dtype=np.float64)
    nonempty = denominator > 0
    scores[nonempty] = (
        (1.0 + beta_squared) * tp[nonempty] / denominator[nonempty]
    )
    return scores


def _instance_jaccard(truth, predictions, abstain_value=-1):
    decided = predictions != abstain_value
    tp = np.sum((truth == 1) & (predictions == 1) & decided, axis=1)
    fp = np.sum((truth == 0) & (predictions == 1) & decided, axis=1)
    fn = np.sum((truth == 1) & (predictions == 0) & decided, axis=1)
    union = tp + fp + fn
    scores = np.ones(len(truth), dtype=np.float64)
    nonempty = union > 0
    scores[nonempty] = tp[nonempty] / union[nonempty]
    return scores


class SelectionObjectiveRegistryTests(unittest.TestCase):
    def test_registry_contains_required_objectives_and_alias(self):
        self.assertEqual(
            available_selection_objectives(),
            (
                "full_macro_f1",
                "immediate_instance_f1",
                "bop_instance_f1",
                "bop_jaccard",
                "macro_precision",
                "macro_recall",
                "f_beta_0_5",
                "f_beta_2",
            ),
        )
        self.assertEqual(
            canonical_selection_objective("complete_macro_f1"),
            "full_macro_f1",
        )
        with self.assertRaises(ValueError):
            canonical_selection_objective("outer_test_accuracy")

    def test_default_macro_f1_is_the_frozen_threshold_objective(self):
        truth = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int32)
        probabilities = np.array(
            [[0.9, 0.4], [0.6, 0.8], [0.49, 0.51]], dtype=np.float64
        )
        result = evaluate_selection_objective(
            "full_macro_f1", truth, probabilities, cost=0.01
        )
        expected_predictions = (probabilities >= 0.5).astype(np.int32)
        expected_score = f1_score(
            truth, expected_predictions, average="macro", zero_division=0
        )
        np.testing.assert_array_equal(result.predictions, expected_predictions)
        self.assertEqual(result.score, expected_score)
        self.assertEqual(result.policy_config["name"], "hamming_bop")
        self.assertEqual(result.policy_config["cost"], 0.5)
        self.assertEqual(result.diagnostics["coverage"], 1.0)

    def test_precision_recall_and_predicted_positive_diagnostics(self):
        truth = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int32)
        probabilities = np.array(
            [[0.9, 0.4], [0.6, 0.8], [0.49, 0.51]], dtype=np.float64
        )
        predictions = (probabilities >= 0.5).astype(np.int32)
        precision = evaluate_selection_objective(
            "macro_precision", truth, probabilities
        )
        recall = evaluate_selection_objective(
            "macro_recall", truth, probabilities
        )
        self.assertEqual(
            precision.score,
            precision_score(
                truth, predictions, average="macro", zero_division=0
            ),
        )
        self.assertEqual(
            recall.score,
            recall_score(
                truth, predictions, average="macro", zero_division=0
            ),
        )
        self.assertEqual(
            precision.diagnostics["predicted_positive_rate"],
            float(np.mean(predictions)),
        )

    def test_bop_objectives_apply_corresponding_policy_and_realized_penalty(self):
        truth = np.array([[1, 1], [0, 0], [1, 0]], dtype=np.int32)
        probabilities = np.array(
            [[0.4, 0.4], [0.5, 0.5], [0.8, 0.45]], dtype=np.float64
        )
        cases = (
            ("immediate_instance_f1", "fbeta_bop", 1.0, False),
            ("bop_instance_f1", "fbeta_bop", 1.0, True),
            ("bop_jaccard", "jaccard_bop", None, True),
            ("f_beta_0_5", "fbeta_bop", 0.5, True),
            ("f_beta_2", "fbeta_bop", 2.0, True),
        )
        for objective, policy_name, beta, allows_abstention in cases:
            with self.subTest(objective=objective):
                result = evaluate_selection_objective(
                    objective,
                    truth,
                    probabilities,
                    cost=0.1,
                    penalty="concave",
                )
                self.assertEqual(result.policy_config["name"], policy_name)
                self.assertEqual(result.beta, beta)
                if policy_name != "hamming_bop":
                    self.assertEqual(
                        result.policy_config["allow_abstention"],
                        allows_abstention,
                    )
                abstentions = np.sum(result.predictions == -1, axis=1)
                penalties = (
                    abstention_penalty(
                        abstentions, 2, 0.1, "concave"
                    )
                    if allows_abstention
                    else np.zeros(len(truth))
                )
                if objective == "bop_jaccard":
                    base = _instance_jaccard(truth, result.predictions)
                else:
                    base = _instance_fbeta(
                        truth, result.predictions, beta=beta
                    )
                self.assertAlmostEqual(
                    result.score, float(np.mean(base - penalties)), places=12
                )

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            evaluate_selection_objective(
                "full_macro_f1", np.array([[2]]), np.array([[0.5]])
            )
        with self.assertRaises(ValueError):
            evaluate_selection_objective(
                "full_macro_f1", np.array([[1]]), np.array([[np.nan]])
            )


class GSIObjectiveInjectionTests(unittest.TestCase):
    def _select(self, objective="full_macro_f1", **parameters):
        features, truth, direct, chain = _selection_fixture()
        classifier = GSIMLCPartialAbstentionClassifier(
            selection_objective=objective, **parameters
        )
        classifier.n_labels_ = truth.shape[1]
        classifier.order_ = list(range(truth.shape[1]))
        output = classifier._select_partition(
            features, truth, direct, chain
        )
        return classifier, output

    def test_default_partition_and_scores_match_pre_q6_fixture(self):
        classifier, (independent, dependent, score, history) = self._select()
        self.assertEqual(independent, [0])
        self.assertEqual(dependent, [1, 2])
        self.assertGreater(score, history[0]["score"])
        self.assertTrue(history[1]["accepted"])
        self.assertFalse(history[2]["accepted"])
        self.assertEqual(classifier.evaluated_configurations_, 4)
        for record in history:
            self.assertEqual(record["selection_objective"], "full_macro_f1")
            self.assertEqual(record["selection_policy"], "hamming_bop")
            self.assertEqual(record["score"], record["full_macro_f1"])
            self.assertEqual(record["inner_split_seed"], 42)
        self.assertEqual(
            classifier.selection_config_["final_decision_policy"]["name"],
            "hamming_bop",
        )

    def test_each_objective_updates_history_with_its_own_policy(self):
        expected_policies = {
            "immediate_instance_f1": "fbeta_bop",
            "bop_instance_f1": "fbeta_bop",
            "bop_jaccard": "jaccard_bop",
            "macro_precision": "hamming_bop",
            "macro_recall": "hamming_bop",
            "f_beta_0_5": "fbeta_bop",
            "f_beta_2": "fbeta_bop",
        }
        for objective, expected_policy in expected_policies.items():
            with self.subTest(objective=objective):
                classifier, (_, _, score, history) = self._select(
                    objective, cost=0.1
                )
                self.assertTrue(np.isfinite(score))
                self.assertTrue(history)
                self.assertTrue(
                    all(
                        row["selection_objective"] == objective
                        and row["selection_policy"] == expected_policy
                        for row in history
                    )
                )
                self.assertEqual(
                    classifier.selection_policy_config_["name"],
                    expected_policy,
                )

    def test_final_decision_policy_is_injected_without_formula_in_selector(self):
        probabilities = np.array([[0.4, 0.4]], dtype=np.float64)
        classifier = GSIMLCPartialAbstentionClassifier(
            decision_policy="fbeta", beta=1.0, cost=0.1
        )
        expected = create_configured_policy(
            "fbeta", beta=1.0, cost=0.1
        ).predict(probabilities)
        np.testing.assert_array_equal(
            classifier._apply_bop(probabilities), expected
        )
        with self.assertRaises(ValueError):
            GSIMLCPartialAbstentionClassifier(
                selection_objective="invalid"
            )._validate_parameters()

    def test_factory_and_cache_settings_expose_nondefault_configuration(self):
        classifier = _create_model(
            "GSI_MLC_PA",
            abstention_cost=0.12,
            gsi_selection_objective="bop_jaccard",
            gsi_decision_policy="jaccard",
            gsi_beta=2.0,
            gsi_penalty="concave",
        )
        self.assertEqual(classifier.selection_objective, "bop_jaccard")
        self.assertEqual(classifier.decision_policy, "jaccard")
        self.assertEqual(classifier.beta, 2.0)
        self.assertEqual(classifier.penalty, "concave")
        default_settings = _cache_settings(
            "GSI_MLC_PA", 5, 42, [0.2], 0.2, "linear", "mlp", 0.2
        )
        configured_settings = _cache_settings(
            "GSI_MLC_PA",
            5,
            42,
            [0.2],
            0.2,
            "linear",
            "mlp",
            0.2,
            "bop_jaccard",
            "jaccard",
            2.0,
            "concave",
        )
        self.assertEqual(
            default_settings["selection_objective"], "complete_macro_f1"
        )
        self.assertEqual(
            configured_settings["selection_objective_canonical"],
            "bop_jaccard",
        )
        self.assertEqual(configured_settings["decision_policy"], "jaccard")
        self.assertNotEqual(default_settings, configured_settings)

    def test_spy_confirms_only_inner_validation_targets_reach_objective(self):
        train_x = np.linspace(0.02, 0.98, 120, dtype=np.float32).reshape(40, 3)
        train_y = (
            train_x > np.array([0.45, 0.50, 0.55], dtype=np.float32)
        ).astype(np.int32)
        outer_x = np.full((2, 3), 0.999, dtype=np.float32)
        outer_y = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
        observed_targets = []
        real_objective = evaluate_selection_objective

        def objective_spy(name, y_true, probabilities, **settings):
            observed_targets.append(np.asarray(y_true).copy())
            return real_objective(
                name, y_true, probabilities, **settings
            )

        classifier = GSIMLCPartialAbstentionClassifier(
            validation_size=0.25,
            br_estimator=_FixedProbabilityEstimator(),
            cc_estimator=_MeanFieldChainEstimator(),
            random_state=7,
        )
        with mock.patch(
            "src.models.gsi_mlc_pa.evaluate_selection_objective",
            side_effect=objective_spy,
        ):
            classifier.fit(train_x, train_y)
        history_before_outer_evaluation = list(classifier.selection_history_)
        classifier.predict_proba(outer_x)
        self.assertEqual(classifier.selection_history_, history_before_outer_evaluation)
        self.assertTrue(observed_targets)
        self.assertTrue(
            all(target.shape == (10, 3) for target in observed_targets)
        )
        self.assertFalse(
            any(np.array_equal(target, outer_y) for target in observed_targets)
        )

    def test_schema_v3_metadata_persists_selection_audit(self):
        classifier, (_, _, score, history) = self._select(
            "bop_jaccard", cost=0.1
        )
        classifier.independent_labels_ = [0]
        classifier.dependent_labels_ = [1, 2]
        classifier.validation_objective_ = score
        classifier.selection_history_ = history
        classifier.predict_proba = lambda x_data: np.array(
            [[0.8, 0.2, 0.4], [0.1, 0.7, 0.6]], dtype=np.float64
        )[: len(x_data)]
        result = _evaluate_model(
            "GSI_MLC_PA",
            classifier,
            np.zeros((2, 1)),
            np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32),
            [0.1],
            metric_schema=3,
            label_names=("a", "b", "c"),
        )
        metadata = result["Model Metadata"]
        self.assertEqual(metadata["Selection Objective"], "bop_jaccard")
        self.assertEqual(
            metadata["Selection Config"]["selection_policy"]["name"],
            "jaccard_bop",
        )
        self.assertEqual(
            len(metadata["Selection History"]),
            len(classifier.selection_history_),
        )


if __name__ == "__main__":
    unittest.main()
