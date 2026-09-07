"""Matched model registry for reproducible BR/CC/MLC-PA/GSI baselines."""

from copy import deepcopy
from dataclasses import dataclass

from .base_learners import (
    base_learner_manifest,
    create_multilabel_estimator,
    load_experiment_config,
)
from .classifier_chain import ClassifierChainClassifier
from .gsi_mlc_pa import GSIMLCPartialAbstentionClassifier
from .mlc_pa import MLCPartialAbstentionClassifier


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    base_learner: str

    @property
    def selective(self):
        return self.family in ("MLC_PA", "GSI_MLC_PA")


def _configured_specs():
    return {
        model_id: ModelSpec(
            model_id=model_id,
            family=str(values["family"]),
            base_learner=str(values["base_learner"]),
        )
        for model_id, values in load_experiment_config()["models"].items()
    }


MATCHED_MODEL_IDS = tuple(
    load_experiment_config()["defaults"]["matched_model_ids"]
)
LOGISTIC_MLP_MODEL_IDS = tuple(
    load_experiment_config()["defaults"]["logistic_mlp_model_ids"]
)

_REGISTERED_ALIASES = {
    "BR_LR": "BR_Logistic",
    "BR_LOGREG": "BR_Logistic",
    "BR_NEURAL_NETWORK": "BR_MLP",
    "BR_NN": "BR_MLP",
    "CC_LR": "CC_Logistic",
    "CC_LOGREG": "CC_Logistic",
    "CC_NEURAL_NETWORK": "CC_MLP",
    "CC_NN": "CC_MLP",
    "MLCPA_LOGISTIC": "MLC_PA_Logistic",
    "MLCPA_MLP": "MLC_PA_MLP",
    "GSIMLCPA_LOGISTIC": "GSI_MLC_PA_Logistic",
    "GSIMLCPA_MLP": "GSI_MLC_PA_MLP",
    "BR_CALIBRATED_SVM": "BR_SVM",
    "CC_CALIBRATED_SVM": "CC_SVM",
    "MLCPA_SVM": "MLC_PA_SVM",
    "MLC_PA_CALIBRATED_SVM": "MLC_PA_SVM",
    "GSIMLCPA_SVM": "GSI_MLC_PA_SVM",
    "GSI_MLC_PA_CALIBRATED_SVM": "GSI_MLC_PA_SVM",
}

_LEGACY_FAMILIES = {
    "BR": "BR",
    "BR_SVC": "BR",
    "BR_LINEARSVC": "BR",
    "CC": "CC",
    "CC_SVC": "CC",
    "CC_LINEARSVC": "CC",
    "MLC_PA": "MLC_PA",
    "MLCPA": "MLC_PA",
    "MLC_PARTIAL_ABSTENTION": "MLC_PA",
    "GSI_MLC_PA": "GSI_MLC_PA",
    "GSIMLCPA": "GSI_MLC_PA",
    "GSI_MLC_PARTIAL_ABSTENTION": "GSI_MLC_PA",
}


def canonical_registered_model_id(model_id):
    key = str(model_id).strip().upper().replace("-", "_")
    specs = _configured_specs()
    by_upper = {name.upper(): name for name in specs}
    if key in _REGISTERED_ALIASES:
        return _REGISTERED_ALIASES[key]
    if key in by_upper:
        return by_upper[key]
    raise ValueError(
        f"Unknown registered model ID: {model_id}. Available: {MATCHED_MODEL_IDS}."
    )


def is_registered_model_id(model_id):
    try:
        canonical_registered_model_id(model_id)
    except ValueError:
        return False
    return True


def get_model_spec(model_id):
    return _configured_specs()[canonical_registered_model_id(model_id)]


def model_family(model_id):
    """Return a family for registered IDs and supported legacy aliases."""

    if is_registered_model_id(model_id):
        return get_model_spec(model_id).family
    key = str(model_id).strip().upper().replace("-", "_")
    if key in _LEGACY_FAMILIES:
        return _LEGACY_FAMILIES[key]
    raise ValueError(f"Unknown model ID: {model_id}.")


def model_base_learner(model_id, *, legacy_mlc_pa_base="mlp"):
    if is_registered_model_id(model_id):
        return get_model_spec(model_id).base_learner
    family = model_family(model_id)
    if family == "MLC_PA":
        return legacy_mlc_pa_base
    if family == "GSI_MLC_PA":
        return "mlp"
    if family in ("BR", "CC"):
        key = str(model_id).upper()
        return "svm_raw" if key in ("BR", "BR_SVC", "BR_LINEARSVC", "CC", "CC_SVC", "CC_LINEARSVC") else None
    return None


def _manifest_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _manifest_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_manifest_value(item) for item in value]
    if hasattr(value, "get_params"):
        return {
            "class": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "parameters": {
                str(key): _manifest_value(item)
                for key, item in value.get_params(deep=False).items()
            },
        }
    raise TypeError(
        f"Model parameter of type {type(value).__name__} cannot enter the manifest."
    )


def _attach_manifest(model, spec):
    model_parameters = deepcopy(model.get_params(deep=False))
    # Keep manifests/cache hashes bit-for-bit compatible for the default
    # behavior introduced before partition_random_state existed.  The field is
    # material and retained as soon as an ablation supplies a separate seed.
    if model_parameters.get("partition_random_state") is None:
        model_parameters.pop("partition_random_state", None)
    manifest = {
        "experiment_config_schema": int(
            load_experiment_config()["schema_version"]
        ),
        "model_id": spec.model_id,
        "family": spec.family,
        "selective": bool(spec.selective),
        "base_learner": base_learner_manifest(spec.base_learner),
        "model_parameters": _manifest_value(model_parameters),
    }
    model.model_id_ = spec.model_id
    model.family_ = spec.family
    model.base_learner_name_ = spec.base_learner
    model.experiment_manifest_ = manifest
    model.backend_ = manifest["base_learner"]["backend"]
    return model


def create_registered_model(
    model_id,
    *,
    random_state=42,
    abstention_cost=0.3,
    abstention_penalty="linear",
    gsi_validation_size=0.2,
    gsi_selection_objective="full_macro_f1",
    gsi_decision_policy="hamming",
    gsi_beta=1.0,
    gsi_penalty="linear",
    gsi_partition_mode="learned",
    gsi_partition_random_state=None,
    gsi_fixed_independent_labels=None,
    gsi_final_order="correlation",
    gsi_dependency_structure="bipartite",
):
    """Instantiate one of the eight preregistered Logistic/MLP baselines."""

    spec = get_model_spec(model_id)
    if spec.family == "BR":
        model = create_multilabel_estimator(
            spec.base_learner, random_state=random_state
        )
    elif spec.family == "CC":
        model = ClassifierChainClassifier(
            base_estimator=spec.base_learner,
            random_state=random_state,
        )
    elif spec.family == "MLC_PA":
        model = MLCPartialAbstentionClassifier(
            base_estimator=spec.base_learner,
            cost=abstention_cost,
            penalty=abstention_penalty,
            random_state=random_state,
        )
    else:
        model = GSIMLCPartialAbstentionClassifier(
            cost=abstention_cost,
            validation_size=gsi_validation_size,
            random_state=random_state,
            selection_objective=gsi_selection_objective,
            decision_policy=gsi_decision_policy,
            beta=gsi_beta,
            penalty=gsi_penalty,
            partition_mode=gsi_partition_mode,
            partition_random_state=gsi_partition_random_state,
            fixed_independent_labels=gsi_fixed_independent_labels,
            final_order=gsi_final_order,
            base_learner=spec.base_learner,
            dependency_structure=gsi_dependency_structure,
        )
    return _attach_manifest(model, spec)
