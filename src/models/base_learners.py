"""Shared base-learner factory backed by the experiment manifest."""

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC


EXPERIMENT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "experiment.json"
)


class BackendUnavailableError(ImportError):
    """Raised when an explicitly configured learner backend is unavailable."""


@lru_cache(maxsize=1)
def load_experiment_config():
    with EXPERIMENT_CONFIG_PATH.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported experiment config schema version.")
    if not isinstance(config.get("base_learners"), dict):
        raise ValueError("Experiment config must define base_learners.")
    if not isinstance(config.get("models"), dict):
        raise ValueError("Experiment config must define models.")
    return config


_BASE_ALIASES = {
    "lr": "logistic",
    "logreg": "logistic",
    "logistic_regression": "logistic",
    "br_logistic": "logistic",
    "pytorch_mlp": "mlp",
    "gpu_mlp": "mlp",
    "br_mlp": "mlp",
    "nn": "mlp",
    "neural_network": "mlp",
    "mlpclassifier": "mlp",
    "svm": "svm_raw",
    "linearsvc": "svm_raw",
    "linear_svc": "svm_raw",
    "br_svm": "svm_raw",
    "sklearn_mlp": "mlp_sklearn",
    "cpu_mlp": "mlp_sklearn",
}


def canonical_base_learner_name(name, *, default="svm_raw"):
    if name is None:
        key = default
    else:
        key = str(name).strip().lower().replace("-", "_")
    key = _BASE_ALIASES.get(key, key)
    if key not in load_experiment_config()["base_learners"]:
        available = tuple(load_experiment_config()["base_learners"])
        raise ValueError(f"Unknown base learner: {name}. Available: {available}.")
    return key


def base_learner_manifest(name):
    canonical = canonical_base_learner_name(name)
    return {
        "name": canonical,
        **deepcopy(load_experiment_config()["base_learners"][canonical]),
    }


def _tuple_parameters(parameters):
    normalized = dict(parameters)
    if "hidden_layer_sizes" in normalized:
        normalized["hidden_layer_sizes"] = tuple(normalized["hidden_layer_sizes"])
    return normalized


def _pytorch_classes():
    try:
        from .pytorch_mlp import FastPyTorchBinaryMLP, MultiLabelMLPClassifier
    except (ImportError, OSError) as exc:
        raise BackendUnavailableError(
            "The configured 'mlp' learner requires the PyTorch backend. "
            "Install torch or select the explicit 'mlp_sklearn' legacy preset; "
            "the model ID will not silently change backend."
        ) from exc
    return FastPyTorchBinaryMLP, MultiLabelMLPClassifier


def create_binary_estimator(base_learner, random_state=42):
    """Create or clone one binary estimator from the shared configuration."""

    if hasattr(base_learner, "fit"):
        try:
            return clone(base_learner)
        except (TypeError, RuntimeError):
            return deepcopy(base_learner)
    name = canonical_base_learner_name(base_learner)
    config = load_experiment_config()["base_learners"][name]
    parameters = _tuple_parameters(config.get("binary_parameters", {}))
    parameters["random_state"] = random_state
    if name == "logistic":
        return LogisticRegression(**parameters)
    if name == "svm_raw":
        return LinearSVC(**parameters)
    if name == "svm_calibrated":
        from .probability_adapter import ProbabilityAdapter

        calibration = config["calibration"]
        return ProbabilityAdapter(
            estimator=LinearSVC(**parameters),
            method=calibration["method"],
            cv=int(calibration["cv"]),
            rare_label_strategy=calibration["rare_label_strategy"],
            random_state=random_state,
        )
    if name == "mlp_sklearn":
        return MLPClassifier(**parameters)
    if name == "mlp":
        binary_class, _ = _pytorch_classes()
        return binary_class(**parameters)
    raise AssertionError(f"Unhandled configured base learner: {name}")


def create_multilabel_estimator(base_learner, random_state=42):
    """Create the BR-style marginal estimator shared by BR/MLC/GSI."""

    if hasattr(base_learner, "fit") and hasattr(base_learner, "predict_proba"):
        try:
            return clone(base_learner)
        except (TypeError, RuntimeError):
            return deepcopy(base_learner)
    name = canonical_base_learner_name(base_learner, default="mlp")
    # Lazy import avoids a module cycle: binary_relevance itself delegates its
    # per-label estimator construction back to this module.
    from .binary_relevance import BinaryRelevanceClassifier, BinaryRelevanceMLP

    if name == "mlp":
        config = load_experiment_config()["base_learners"][name]
        parameters = _tuple_parameters(config.get("multilabel_parameters", {}))
        return BinaryRelevanceMLP(random_state=random_state, **parameters)
    return BinaryRelevanceClassifier(
        base_estimator=create_binary_estimator(name, random_state),
        random_state=random_state,
    )
