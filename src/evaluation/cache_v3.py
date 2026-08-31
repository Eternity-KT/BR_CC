"""Strict schema-v3 fold checkpoints isolated from legacy result caches."""

import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .metric_contract import METRIC_CONTRACT_VERSION, TARGET_RESULT_SCHEMA_VERSION


CACHE_SCHEMA_VERSION_V3 = TARGET_RESULT_SCHEMA_VERSION


class IncompatibleV3CacheError(ValueError):
    """Raised when a file at a v3 checkpoint path violates its contract."""


def _utc_timestamp():
    return datetime.now(timezone.utc).isoformat()


def json_safe(value):
    """Convert nested values to strict-JSON primitives, mapping nonfinite to null."""

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable.")


def canonical_config(config):
    """Return a deterministic strict-JSON copy of a run configuration."""

    normalized = json_safe(config)
    if not isinstance(normalized, dict):
        raise ValueError("config must be a mapping.")
    return normalized


def compute_config_hash(config):
    """Hash canonical settings so incompatible runs never share a checkpoint."""

    encoded = json.dumps(
        canonical_config(config),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _safe_component(value, field_name):
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    if not safe:
        raise ValueError(f"{field_name} must contain a safe filename character.")
    return safe


def fold_checkpoint_path(checkpoint_dir, model_name, dataset_name, config):
    """Return the isolated checkpoint path for one model/dataset/config tuple."""

    safe_model = _safe_component(model_name, "model_name")
    safe_dataset = _safe_component(dataset_name, "dataset_name")
    digest = compute_config_hash(config)
    return Path(checkpoint_dir) / safe_model / f"{safe_dataset}.{digest}.json"


def atomic_json_dump_v3(payload, path):
    """Atomically write strict JSON; NaN and infinities are serialized as null."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}_", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                json_safe(payload),
                stream,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return target


def _new_checkpoint(model_name, dataset_name, config):
    normalized_config = canonical_config(config)
    n_splits = normalized_config.get("n_splits")
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 1:
        raise ValueError("Checkpoint config must contain a positive integer n_splits.")
    timestamp = _utc_timestamp()
    return {
        "schema_version": CACHE_SCHEMA_VERSION_V3,
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "config_hash": compute_config_hash(normalized_config),
        "config": normalized_config,
        "model": str(model_name),
        "dataset": str(dataset_name),
        "status": "in_progress",
        "created_at": timestamp,
        "updated_at": timestamp,
        "folds": {},
    }


def _validate_checkpoint(payload, model_name, dataset_name, config):
    expected_hash = compute_config_hash(config)
    if not isinstance(payload, dict):
        raise IncompatibleV3CacheError("Checkpoint root must be a JSON object.")
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION_V3:
        raise IncompatibleV3CacheError(
            "Checkpoint is not schema v3; legacy caches are never migrated in place."
        )
    if payload.get("metric_contract_version") != METRIC_CONTRACT_VERSION:
        raise IncompatibleV3CacheError("Metric contract version differs from this run.")
    if payload.get("config_hash") != expected_hash:
        raise IncompatibleV3CacheError("Checkpoint config hash differs from this run.")
    if payload.get("config") != canonical_config(config):
        raise IncompatibleV3CacheError("Checkpoint settings differ despite hash match.")
    if payload.get("model") != str(model_name):
        raise IncompatibleV3CacheError("Checkpoint model name differs from this run.")
    if payload.get("dataset") != str(dataset_name):
        raise IncompatibleV3CacheError("Checkpoint dataset differs from this run.")
    if not isinstance(payload.get("folds"), dict):
        raise IncompatibleV3CacheError("Checkpoint folds must be an object.")
    if payload.get("status") not in ("in_progress", "complete"):
        raise IncompatibleV3CacheError("Checkpoint status is invalid.")
    if payload.get("status") == "complete":
        expected_count = int(canonical_config(config).get("n_splits", 0))
        expected_folds = tuple(range(1, expected_count + 1))
        if completed_fold_indices(payload) != expected_folds:
            raise IncompatibleV3CacheError(
                "Checkpoint is marked complete but does not contain every fold."
            )
    return payload


def load_or_create_fold_checkpoint(checkpoint_dir, model_name, dataset_name, config):
    """Load a compatible checkpoint or return a new in-memory checkpoint."""

    path = fold_checkpoint_path(checkpoint_dir, model_name, dataset_name, config)
    if not path.exists():
        return path, _new_checkpoint(model_name, dataset_name, config)
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return path, _validate_checkpoint(payload, model_name, dataset_name, config)


def completed_fold_indices(checkpoint):
    """Return completed positive fold indices in numeric order."""

    indices = []
    for key in checkpoint.get("folds", {}):
        try:
            index = int(key)
        except (TypeError, ValueError) as exc:
            raise IncompatibleV3CacheError(f"Invalid fold key: {key}") from exc
        if index < 1 or str(index) != str(key):
            raise IncompatibleV3CacheError(f"Invalid fold key: {key}")
        indices.append(index)
    if len(indices) != len(set(indices)):
        raise IncompatibleV3CacheError("Checkpoint has duplicate fold indices.")
    return tuple(sorted(indices))


def save_completed_fold(path, checkpoint, fold_index, metrics, metadata=None):
    """Append one completed fold and immediately persist the checkpoint."""

    if isinstance(fold_index, bool) or not isinstance(fold_index, (int, np.integer)):
        raise ValueError("fold_index must be a positive integer.")
    fold_index = int(fold_index)
    if fold_index < 1:
        raise ValueError("fold_index must be a positive integer.")
    key = str(fold_index)
    if key in checkpoint.get("folds", {}):
        raise ValueError(f"Fold {fold_index} is already checkpointed.")
    checkpoint.setdefault("folds", {})[key] = {
        "metrics": json_safe(metrics),
        "metadata": json_safe(metadata or {}),
    }
    checkpoint["status"] = "in_progress"
    checkpoint["updated_at"] = _utc_timestamp()
    atomic_json_dump_v3(checkpoint, path)
    return checkpoint


def mark_checkpoint_complete(path, checkpoint, expected_fold_count):
    """Mark a checkpoint complete only when every expected fold exists."""

    expected = tuple(range(1, int(expected_fold_count) + 1))
    actual = completed_fold_indices(checkpoint)
    if actual != expected:
        raise ValueError(f"Cannot complete checkpoint; expected folds {expected}, got {actual}.")
    checkpoint["status"] = "complete"
    checkpoint["updated_at"] = _utc_timestamp()
    atomic_json_dump_v3(checkpoint, path)
    return checkpoint
