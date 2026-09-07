"""Bipartite GSI Classifier for Multi-Label Classification with Partial Abstention.

Theoretical Formulation:
-------------------------
In this architecture, labels are partitioned into two disjoint sets:
1. Independent Labels (IL):
   Predicted directly from input features X via Binary Relevance:
   P(Y_l = 1 | X) = P_BR(Y_l = 1 | X)

2. Dependent Labels (DL):
   Conditioned EXCLUSIVELY on input features X and the set of Independent Labels (IL):
   P(Y_d = 1 | X, Y_IL) = f_d(X, P_hat_IL)

Crucially, labels in DL do NOT depend on any other labels in DL.
There is NO chained dependency among DL labels (bipartite DAG structure).
"""

from copy import deepcopy
from time import perf_counter
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
except ImportError:
    MultilabelStratifiedShuffleSplit = None

from src.models.base_learners import (
    canonical_base_learner_name,
    create_binary_estimator,
    create_multilabel_estimator,
)


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
    """Return P(Y=1) from a binary classifier with tolerant shape adaptation."""
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
class _ConstantClassifier:
    """Fallback classifier when a label has only one unique class in training set."""
    def __init__(self, constant_value):
        self.constant_value = int(constant_value)

    def predict(self, X):
        n_samples = X.shape[0]
        return np.full(n_samples, self.constant_value, dtype=np.int32)

    def decision_function(self, X):
        n_samples = X.shape[0]
        return np.full(n_samples, 50.0 if self.constant_value == 1 else -50.0, dtype=np.float32)

    def predict_proba(self, X):
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, 2), dtype=np.float32)
        if self.constant_value == 1:
            probs[:, 1] = 1.0
        else:
            probs[:, 0] = 1.0
        return probs


def _safe_fit_binary(template_factory, X, y):
    unique_classes = np.unique(y)
    if len(unique_classes) <= 1:
        const_val = unique_classes[0] if len(unique_classes) == 1 else 0
        return _ConstantClassifier(const_val)
    clf = template_factory()
    clf.fit(X, y)
    return clf


class BipartiteGSIPartialAbstentionClassifier(BaseEstimator, ClassifierMixin):
    """Bipartite GSI Classifier where DL labels depend ONLY on IL labels.

    Parameters
    ----------
    cost : float, default=0.3
        Rejection cost c for Bayes-Optimal Prediction (BOP).
    validation_size : float, default=0.2
        Fraction of training fold used internally for IL/DL greedy selection.
    refit : bool, default=True
        Refit BR and DL models on the full training fold after partition is frozen.
    random_state : int, default=42
        Random seed for reproducibility.
    base_learner : {"mlp", "logistic"}, default="mlp"
        Base binary/multilabel estimator backend.
    abstain_value : int, default=-1
        Sentinel value representing abstention.
    selection_fast_backend : bool, default=True
        For large label spaces (e.g. >20 labels), use logistic regression during
        the inner greedy search loop to keep selection fast, then refit with
        the configured base_learner on the frozen partition.
    """

    def __init__(
        self,
        cost=0.3,
        validation_size=0.2,
        refit=True,
        random_state=42,
        base_learner="mlp",
        abstain_value=-1,
        selection_fast_backend=True,
    ):
        self.cost = cost
        self.validation_size = validation_size
        self.refit = refit
        self.random_state = random_state
        self.base_learner = base_learner
        self.abstain_value = abstain_value
        self.selection_fast_backend = selection_fast_backend

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

    def _make_br_model(self):
        return create_multilabel_estimator(
            self.base_learner, random_state=self.random_state
        )

    def _make_binary_classifier(self, seed=None, use_fast=False):
        backend = "logistic" if use_fast else self.base_learner
        return create_binary_estimator(
            backend, random_state=self.random_state if seed is None else seed
        )

    def _select_partition(self, X_sel, Y_sel, X_val, Y_val, val_direct):
        """Greedy forward selection of Independent Labels (IL).

        Starts with IL = {}, DL = all labels.
        Tests adding label k to IL. A label stays in IL if validation Macro-F1
        strictly improves.
        """
        independent = []
        dependent = list(range(self.n_labels_))

        # Initial baseline: IL = empty. Every label is in DL with no IL parents,
        # so each label is predicted by an unconditional binary classifier on X.
        use_fast_selection = self.selection_fast_backend
        current_probs = np.zeros_like(val_direct)
        for d in range(self.n_labels_):
            clf = _safe_fit_binary(
                lambda: self._make_binary_classifier(
                    seed=self.random_state + d,
                    use_fast=use_fast_selection,
                ),
                X_sel,
                Y_sel[:, d],
            )
            current_probs[:, d] = _positive_probability(clf, X_val)

        current_score = float(
            f1_score(Y_val, (current_probs >= 0.5).astype(np.int32), average="macro", zero_division=0)
        )

        history = [{
            "step": 0,
            "tested_label": None,
            "accepted": True,
            "score": current_score,
            "il_count": 0,
            "dl_count": self.n_labels_,
        }]

        for label_idx in range(self.n_labels_):
            cand_il = independent + [label_idx]
            cand_dl = [d for d in range(self.n_labels_) if d not in cand_il]

            cand_probs = np.zeros_like(val_direct)
            # IL labels use direct BR probabilities
            for l in cand_il:
                cand_probs[:, l] = val_direct[:, l]

            # DL labels are conditioned EXCLUSIVELY on cand_il
            if cand_il:
                X_sel_ext = np.hstack([X_sel, Y_sel[:, cand_il].astype(np.float32)])
                X_val_ext = np.hstack([X_val, val_direct[:, cand_il].astype(np.float32)])
            else:
                X_sel_ext = X_sel
                X_val_ext = X_val

            for d in cand_dl:
                clf = _safe_fit_binary(
                    lambda: self._make_binary_classifier(
                        seed=self.random_state + label_idx * 100 + d,
                        use_fast=use_fast_selection,
                    ),
                    X_sel_ext,
                    Y_sel[:, d],
                )
                cand_probs[:, d] = _positive_probability(clf, X_val_ext)

            cand_score = float(
                f1_score(Y_val, (cand_probs >= 0.5).astype(np.int32), average="macro", zero_division=0)
            )
            improvement = cand_score - current_score
            accepted = improvement > 1e-12

            if accepted:
                independent.append(label_idx)
                dependent.remove(label_idx)
                current_score = cand_score
                current_probs = cand_probs

            history.append({
                "step": len(history),
                "tested_label": int(label_idx),
                "accepted": bool(accepted),
                "score": float(cand_score),
                "improvement": float(improvement),
                "il_count": len(independent),
                "dl_count": len(dependent),
            })

        return sorted(independent), sorted(dependent), current_score, history

    def fit(self, X, Y):
        """Fit the Bipartite GSI model with leakage-safe IL/DL selection."""
        X_array = _as_dense_float(X)
        Y_array = np.asarray(Y, dtype=np.int32)

        if Y_array.ndim != 2 or Y_array.shape[0] != X_array.shape[0]:
            raise ValueError("Y must have shape (n_samples, n_labels).")
        self.n_labels_ = Y_array.shape[1]

        t_start = perf_counter()

        # Step 1: Internal split for IL/DL selection (Leakage-safe)
        train_idx, val_idx = self._split_train_validation(X_array, Y_array)
        X_sel, Y_sel = X_array[train_idx], Y_array[train_idx]
        X_val, Y_val = X_array[val_idx], Y_array[val_idx]

        # Fit selection BR to produce direct marginals
        selection_br = self._make_br_model()
        selection_br.fit(X_sel, Y_sel)
        val_direct = np.asarray(selection_br.predict_proba(X_val), dtype=np.float64)

        # Step 2: Select IL/DL partition
        (
            self.independent_labels_,
            self.dependent_labels_,
            self.validation_score_,
            self.selection_history_,
        ) = self._select_partition(X_sel, Y_sel, X_val, Y_val, val_direct)

        self.selection_time_ = perf_counter() - t_start

        # Step 3: Refit on full outer training data
        refit_x = X_array if self.refit else X_sel
        refit_y = Y_array if self.refit else Y_sel

        self.br_model_ = self._make_br_model()
        self.br_model_.fit(refit_x, refit_y)

        # Fit binary classifiers for DL labels (conditioned ONLY on IL)
        self.dl_models_ = {}
        if self.dependent_labels_:
            if self.independent_labels_:
                X_refit_ext = np.hstack([
                    refit_x,
                    refit_y[:, self.independent_labels_].astype(np.float32),
                ])
            else:
                X_refit_ext = refit_x

            for d in self.dependent_labels_:
                clf = _safe_fit_binary(
                    lambda: self._make_binary_classifier(seed=self.random_state + d),
                    X_refit_ext,
                    refit_y[:, d],
                )
                self.dl_models_[d] = clf

        self.total_fit_time_ = perf_counter() - t_start
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        """Predict probabilities under the frozen Bipartite IL/DL configuration."""
        check_is_fitted(self, ("is_fitted_", "br_model_", "independent_labels_"))
        X_array = _as_dense_float(X)
        n_samples = X_array.shape[0]

        direct_probs = np.asarray(self.br_model_.predict_proba(X_array), dtype=np.float64)
        probabilities = np.zeros((n_samples, self.n_labels_), dtype=np.float64)

        # IL labels take direct BR marginal probabilities
        for l in self.independent_labels_:
            probabilities[:, l] = direct_probs[:, l]

        # DL labels take conditional probability conditioned ONLY on IL
        if self.dependent_labels_:
            if self.independent_labels_:
                # Mean-field plug-in approximation: use soft probabilities of IL
                il_probs = direct_probs[:, self.independent_labels_].astype(np.float32)
                X_ext = np.hstack([X_array, il_probs])
            else:
                X_ext = X_array

            for d in self.dependent_labels_:
                clf = self.dl_models_[d]
                probabilities[:, d] = _positive_probability(clf, X_ext)

        return np.clip(probabilities, 0.0, 1.0)

    def predict_full_from_proba(self, probabilities):
        """Complete threshold prediction at 0.5 with no abstention."""
        return (probabilities >= 0.5).astype(np.int32)

    def predict_full(self, X):
        return self.predict_full_from_proba(self.predict_proba(X))

    def predict_from_proba(self, probabilities, cost=None):
        """Bayes-Optimal Prediction with rejection cost c (SEP linear policy).

        Decisions:
            p <= c      -> 0
            c < p < 1-c -> abstain (-1)
            p >= 1-c    -> 1
        """
        c = self.cost if cost is None else float(cost)
        if c >= 0.5:
            return self.predict_full_from_proba(probabilities)

        decisions = np.full(probabilities.shape, self.abstain_value, dtype=np.int32)
        decisions[probabilities <= c] = 0
        decisions[probabilities >= 1.0 - c] = 1
        return decisions

    def predict(self, X, cost=None):
        return self.predict_from_proba(self.predict_proba(X), cost=cost)
