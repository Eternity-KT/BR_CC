"""Greedy Selection of Independent labels for MLC with partial abstention.

GSI-MLC-PA combines the two MLP mechanisms already used by this project:

* BR-MLP estimates direct marginal probabilities for every label.
* CC-MLP estimates a label conditionally on its predecessors.

The independent/dependent partition is selected on an internal validation
split.  The partition is frozen before the two probability models are refit
on all outer-training data, so an outer test fold is never used for model
selection.
"""

from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
except ImportError:  # pragma: no cover - project requirements normally provide it
    MultilabelStratifiedShuffleSplit = None

from .binary_relevance import BinaryRelevanceMLP
from .classifier_chain import ClassifierChainClassifier


def _clone_or_copy(estimator):
    try:
        return clone(estimator)
    except (TypeError, RuntimeError):
        return deepcopy(estimator)


def _as_dense_float(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix.")
    return X


def _positive_probability(classifier, X):
    """Return P(Y=1) from a binary classifier with a tolerant API adapter."""
    if hasattr(classifier, "predict_proba"):
        probabilities = np.asarray(classifier.predict_proba(X), dtype=np.float64)
        if probabilities.ndim == 1:
            result = probabilities
        elif probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            result = probabilities[:, 1]
        elif probabilities.ndim == 2 and probabilities.shape[1] == 1:
            result = probabilities[:, 0]
        else:
            raise ValueError("Binary predict_proba() returned an invalid shape.")
    elif hasattr(classifier, "decision_function"):
        scores = np.asarray(classifier.decision_function(X), dtype=np.float64)
        result = 1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0)))
    else:
        result = np.asarray(classifier.predict(X), dtype=np.float64)
    return np.clip(np.ravel(result), 0.0, 1.0)


class GSIMLCPartialAbstentionClassifier(BaseEstimator, ClassifierMixin):
    """GSI-MLC-PA with cost-independent IL/DL selection.

    Parameters
    ----------
    cost : float, default=0.3
        Default BOP rejection cost used by :meth:`predict`.  It is only a
        decision-time setting and never participates in training or IL/DL
        selection.
    validation_size : float, default=0.2
        Fraction of the outer-training fold used only for IL/DL selection.
    refit : bool, default=True
        Refit BR-MLP and CC-MLP on the complete outer-training fold after the
        partition has been frozen.
    order : sequence of int or None, default=None
        CC predecessor order.  None uses natural label order.
    br_estimator, cc_estimator : estimator or None
        Test/extensibility hooks.  Defaults use the same BR-MLP and CC-MLP
        implementations as the benchmark.
    abstain_value : int, default=-1
        Integer sentinel representing the abstention symbol.
    random_state : int, default=42
        Seed for the internal split and MLP estimators.

    Notes
    -----
    For one predecessor, dependent inference computes the exact two-state
    marginal ``(1-p) f(x,0) + p f(x,1)``.  With multiple predecessors, it uses
    the standard mean-field plug-in approximation: each binary predecessor is
    replaced by its finalized soft marginal under the current IL/DL
    configuration before evaluating the conditional CC classifier.
    """

    def __init__(
        self,
        cost=0.3,
        validation_size=0.2,
        refit=True,
        order=None,
        br_estimator=None,
        cc_estimator=None,
        abstain_value=-1,
        random_state=42,
    ):
        self.cost = cost
        self.validation_size = validation_size
        self.refit = refit
        self.order = order
        self.br_estimator = br_estimator
        self.cc_estimator = cc_estimator
        self.abstain_value = abstain_value
        self.random_state = random_state

    def _validate_parameters(self):
        if not 0.0 <= float(self.cost) <= 1.0:
            raise ValueError("cost must lie in [0, 1].")
        if not 0.0 < float(self.validation_size) < 1.0:
            raise ValueError("validation_size must lie strictly between 0 and 1.")
        if self.abstain_value in (0, 1):
            raise ValueError("abstain_value must differ from 0 and 1.")

    def _validated_order(self, n_labels):
        if self.order is None:
            return list(range(n_labels))
        order = [int(label) for label in self.order]
        if sorted(order) != list(range(n_labels)):
            raise ValueError("order must be a permutation of all label indices.")
        return order

    def _make_br_model(self):
        if self.br_estimator is None:
            return BinaryRelevanceMLP(random_state=self.random_state)
        return _clone_or_copy(self.br_estimator)

    def _make_cc_model(self, order=None):
        requested_order = self.order_ if order is None else list(order)
        if self.cc_estimator is None:
            return ClassifierChainClassifier(
                base_estimator="mlp",
                order=requested_order,
                random_state=self.random_state,
            )
        estimator = _clone_or_copy(self.cc_estimator)
        if hasattr(estimator, "set_params"):
            try:
                estimator.set_params(order=requested_order)
            except (TypeError, ValueError):
                pass
        return estimator

    def _split_train_validation(self, X, Y):
        indices = np.arange(X.shape[0])
        if MultilabelStratifiedShuffleSplit is not None:
            try:
                splitter = MultilabelStratifiedShuffleSplit(
                    n_splits=1,
                    test_size=float(self.validation_size),
                    random_state=self.random_state,
                )
                train_indices, validation_indices = next(splitter.split(X, Y))
                return train_indices, validation_indices
            except (TypeError, ValueError):
                pass
        return train_test_split(
            indices,
            test_size=float(self.validation_size),
            random_state=self.random_state,
            shuffle=True,
        )

    @staticmethod
    def _validate_probability_matrix(probabilities, n_samples, n_labels, name):
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.shape != (n_samples, n_labels):
            raise ValueError(
                f"{name}.predict_proba() must return shape "
                f"({n_samples}, {n_labels}), got {probabilities.shape}."
            )
        return np.clip(probabilities, 0.0, 1.0)

    def _direct_probabilities(self, model, X):
        return self._validate_probability_matrix(
            model.predict_proba(X), X.shape[0], self.n_labels_, "BR model"
        )

    def _configured_probabilities(
        self,
        X,
        direct_probabilities,
        cc_model,
        independent_labels,
        initial_probabilities=None,
        start_position=0,
    ):
        """Infer probabilities sequentially under one fixed IL/DL partition."""
        if not hasattr(cc_model, "classifiers_") or not hasattr(cc_model, "order_"):
            raise ValueError(
                "cc_estimator must expose fitted classifiers_ and order_ attributes."
            )
        if list(cc_model.order_) != self.order_:
            raise ValueError("The fitted CC order does not match the GSI order.")
        if len(cc_model.classifiers_) != self.n_labels_:
            raise ValueError("The fitted CC must contain one classifier per label.")

        if not 0 <= int(start_position) < self.n_labels_:
            raise ValueError("start_position must index a label in the chain.")
        independent = set(int(label) for label in independent_labels)
        if initial_probabilities is None:
            probabilities = np.zeros_like(direct_probabilities, dtype=np.float64)
            start_position = 0
        else:
            probabilities = self._validate_probability_matrix(
                initial_probabilities,
                X.shape[0],
                self.n_labels_,
                "initial probabilities",
            ).copy()
        for position, label_index in enumerate(self.order_):
            if position < start_position:
                continue
            if label_index in independent:
                probabilities[:, label_index] = direct_probabilities[:, label_index]
                continue

            predecessors = self.order_[:position]
            classifier = cc_model.classifiers_[position]
            if not predecessors:
                # A DL chain root is the unconditional first CC classifier.
                probabilities[:, label_index] = _positive_probability(
                    classifier, X
                )
                continue

            # These are the already-finalized probabilities under the current
            # configuration: BR for preceding IL labels and marginalized CC
            # probabilities for preceding DL labels.
            predecessor_probabilities = probabilities[:, predecessors]
            if len(predecessors) == 1:
                zeros = np.zeros((X.shape[0], 1), dtype=np.float32)
                ones = np.ones((X.shape[0], 1), dtype=np.float32)
                probability_given_zero = _positive_probability(
                    classifier, np.hstack((X, zeros))
                )
                probability_given_one = _positive_probability(
                    classifier, np.hstack((X, ones))
                )
                parent_probability = predecessor_probabilities[:, 0]
                probabilities[:, label_index] = (
                    (1.0 - parent_probability) * probability_given_zero
                    + parent_probability * probability_given_one
                )
            else:
                # Mean-field marginalization: q(Y_parents) is represented by
                # its factorized means, avoiding an exponential 2**m sum.
                mean_field_features = np.hstack(
                    (X, predecessor_probabilities.astype(np.float32))
                )
                probabilities[:, label_index] = _positive_probability(
                    classifier, mean_field_features
                )
        return np.clip(probabilities, 0.0, 1.0)

    @staticmethod
    def _compute_label_correlation(Y):
        """Return a finite Phi/Pearson matrix for binary label columns."""
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim != 2:
            raise ValueError("Y must be a two-dimensional label matrix.")
        if Y.shape[1] == 1:
            return np.ones((1, 1), dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation = np.corrcoef(Y, rowvar=False)
        correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
        correlation = np.clip(correlation, -1.0, 1.0)
        np.fill_diagonal(correlation, 1.0)
        return correlation

    @staticmethod
    def _correlation_order(correlation, independent_labels, dependent_labels):
        """Order labels after IL/DL is frozen using absolute correlations.

        Independent labels form deterministic roots.  Remaining dependent
        labels are appended by strongest connection to an already placed
        label.  Ties are resolved by label index.
        """
        correlation = np.asarray(correlation, dtype=np.float64)
        independent = sorted(int(label) for label in independent_labels)
        remaining = set(int(label) for label in dependent_labels)
        order = list(independent)
        if not order and remaining:
            root = min(
                remaining,
                key=lambda label: (
                    -float(np.sum(np.abs(correlation[label]))),
                    label,
                ),
            )
            order.append(root)
            remaining.remove(root)
        while remaining:
            best = min(
                remaining,
                key=lambda label: (
                    -max(
                        (abs(float(correlation[label, parent])) for parent in order),
                        default=0.0,
                    ),
                    label,
                ),
            )
            order.append(best)
            remaining.remove(best)
        return order

    @staticmethod
    def _dependent_parent_map(correlation, order, dependent_labels):
        """Record the strongest preceding parent for every dependent label."""
        correlation = np.asarray(correlation, dtype=np.float64)
        dependent = set(int(label) for label in dependent_labels)
        parent_map = {}
        for position, label in enumerate(order):
            if label not in dependent:
                continue
            predecessors = order[:position]
            if not predecessors:
                parent_map[int(label)] = None
                continue
            parent_map[int(label)] = int(
                min(
                    predecessors,
                    key=lambda parent: (
                        -abs(float(correlation[label, parent])),
                        parent,
                    ),
                )
            )
        return parent_map

    def _apply_bop(self, probabilities, cost=None):
        """Apply BOP once, after all final probabilities have been produced."""
        cost = float(self.cost if cost is None else cost)
        if not 0.0 <= cost <= 1.0:
            raise ValueError("cost must lie in [0, 1].")
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if cost >= 0.5:
            return (probabilities >= 0.5).astype(np.int32)
        predictions = np.full(
            probabilities.shape, self.abstain_value, dtype=np.int32
        )
        predictions[probabilities <= cost] = 0
        predictions[probabilities >= 1.0 - cost] = 1
        return predictions

    def _configuration_objective(self, Y, probabilities):
        """Score a candidate partition without any rejection mechanism."""
        full_predictions = (np.asarray(probabilities) >= 0.5).astype(np.int32)
        return float(
            f1_score(Y, full_predictions, average="macro", zero_division=0)
        )

    def _select_partition(self, X, Y, direct_probabilities, cc_model):
        """Move labels from DL to IL sequentially using complete Macro-F1."""
        independent = []
        dependent = set(range(self.n_labels_))
        probability_cache = {}

        def evaluate(label_set, initial_probabilities=None, start_position=0):
            key = frozenset(label_set)
            if key not in probability_cache:
                candidate_probabilities = self._configured_probabilities(
                    X,
                    direct_probabilities,
                    cc_model,
                    key,
                    initial_probabilities=initial_probabilities,
                    start_position=start_position,
                )
                probability_cache[key] = (
                    candidate_probabilities,
                    self._configuration_objective(Y, candidate_probabilities),
                )
            return probability_cache[key]

        current_probabilities, current_score = evaluate(independent)
        history = [{
            "step": 0,
            "tested_label": None,
            "accepted": True,
            "score": current_score,
            "full_macro_f1": current_score,
            "improvement": 0.0,
        }]

        for label in list(self.order_):
            candidate_set = independent + [label]
            candidate_probabilities, candidate_score = evaluate(
                candidate_set,
                initial_probabilities=current_probabilities,
                start_position=self.order_.index(label),
            )
            improvement = float(candidate_score - current_score)
            accepted = improvement > 1e-12
            if accepted:
                independent.append(label)
                dependent.remove(label)
                current_probabilities = candidate_probabilities
                current_score = float(candidate_score)
            history.append({
                "step": len(history),
                "tested_label": int(label),
                "accepted": bool(accepted),
                "score": float(candidate_score),
                "full_macro_f1": float(candidate_score),
                "improvement": improvement,
            })

        self.evaluated_configurations_ = int(len(probability_cache))
        return sorted(independent), sorted(dependent), current_score, history

    def fit(self, X, Y):
        """Select IL/DL on validation, freeze it, then optionally refit."""
        self._validate_parameters()
        X_array = _as_dense_float(X)
        Y_array = np.asarray(Y, dtype=np.int32)
        if Y_array.ndim != 2 or Y_array.shape[0] != X_array.shape[0]:
            raise ValueError("Y must have shape (n_samples, n_labels).")
        if Y_array.shape[1] == 0 or not np.all(np.isin(Y_array, (0, 1))):
            raise ValueError("Y must contain at least one binary label column.")
        if X_array.shape[0] < 4:
            raise ValueError("GSI_MLC_PA requires at least four training samples.")

        self.n_labels_ = Y_array.shape[1]
        self.selection_order_ = self._validated_order(self.n_labels_)
        self.order_ = list(self.selection_order_)
        selection_train, validation = self._split_train_validation(X_array, Y_array)
        if len(selection_train) == 0 or len(validation) == 0:
            raise ValueError("The internal train/validation split is empty.")
        self.selection_train_size_ = int(len(selection_train))
        self.validation_size_ = int(len(validation))

        selection_br = self._make_br_model()
        selection_cc = self._make_cc_model(order=self.selection_order_)
        selection_br.fit(X_array[selection_train], Y_array[selection_train])
        selection_cc.fit(X_array[selection_train], Y_array[selection_train])
        if hasattr(selection_cc, "order_"):
            self.order_ = [int(label) for label in selection_cc.order_]
            self.selection_order_ = list(self.order_)

        validation_direct = self._direct_probabilities(
            selection_br, X_array[validation]
        )
        (
            self.independent_labels_,
            self.dependent_labels_,
            self.validation_objective_,
            self.selection_history_,
        ) = self._select_partition(
            X_array[validation],
            Y_array[validation],
            validation_direct,
            selection_cc,
        )

        # IL/DL is frozen before correlation is calculated.  Thus neither
        # correlation nor a rejection cost can feed back into label selection.
        self.label_correlation_ = self._compute_label_correlation(Y_array)
        if self.order is None:
            desired_final_order = self._correlation_order(
                self.label_correlation_,
                self.independent_labels_,
                self.dependent_labels_,
            )
        else:
            desired_final_order = self._validated_order(self.n_labels_)
        self.correlation_order_ = [int(label) for label in desired_final_order]

        # Final fitting uses no validation score and never sees the outer test
        # fold supplied later to predict().
        if self.refit:
            self.br_model_ = self._make_br_model()
            self.br_model_.fit(X_array, Y_array)
            final_x = X_array
            final_y = Y_array
            self.refit_train_size_ = int(X_array.shape[0])
        else:
            self.br_model_ = selection_br
            final_x = X_array[selection_train]
            final_y = Y_array[selection_train]
            self.refit_train_size_ = self.selection_train_size_

        self.cc_model_ = self._make_cc_model(order=desired_final_order)
        self.cc_model_.fit(final_x, final_y)
        if hasattr(self.cc_model_, "order_"):
            self.order_ = [int(label) for label in self.cc_model_.order_]
        else:
            self.order_ = list(desired_final_order)
        self.dependent_parent_map_ = self._dependent_parent_map(
            self.label_correlation_, self.order_, self.dependent_labels_
        )

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        """Return all probabilities under the frozen IL/DL configuration."""
        check_is_fitted(
            self,
            ("is_fitted_", "br_model_", "cc_model_", "independent_labels_"),
        )
        X_array = _as_dense_float(X)
        direct = self._direct_probabilities(self.br_model_, X_array)
        return self._configured_probabilities(
            X_array, direct, self.cc_model_, self.independent_labels_
        )

    def predict(self, X):
        """Return {0, abstain, 1} after one final BOP application."""
        probabilities = self.predict_proba(X)
        return self.predict_from_proba(probabilities, cost=self.cost)

    def predict_from_proba(self, probabilities, cost=None):
        """Apply BOP to one already-computed final probability matrix."""
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != self.n_labels_:
            raise ValueError("probabilities must have shape (n_samples, n_labels).")
        return self._apply_bop(
            np.clip(probabilities, 0.0, 1.0), cost=cost
        )

    def predict_full(self, X):
        """Return complete predictions for conventional benchmark metrics."""
        return self.predict_full_from_proba(self.predict_proba(X))

    def predict_full_from_proba(self, probabilities):
        """Threshold one already-computed final probability matrix at 0.5."""
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != self.n_labels_:
            raise ValueError("probabilities must have shape (n_samples, n_labels).")
        return (probabilities >= 0.5).astype(np.int32)

    def decision_function(self, X):
        probabilities = np.clip(self.predict_proba(X), 1e-7, 1.0 - 1e-7)
        return np.log(probabilities / (1.0 - probabilities))


GSIMLCPAClassifier = GSIMLCPartialAbstentionClassifier
