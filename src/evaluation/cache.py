"""Per-model JSON result cache helpers for resumable experiments."""

import json
import os
import re
import shutil
import tempfile
from pathlib import Path


CACHE_SCHEMA_VERSION = 2
UNPERSISTED_METRICS = frozenset({"Macro Precision", "Macro Recall"})


def strip_unpersisted_metrics(payload):
    """Return a JSON-compatible copy without intentionally excluded metrics.

    Nested legacy summaries are migrated while cost-specific selective metrics
    remain unchanged.
    """
    if isinstance(payload, dict):
        return {
            key: strip_unpersisted_metrics(value)
            for key, value in payload.items()
            if key not in UNPERSISTED_METRICS
        }
    if isinstance(payload, list):
        return [strip_unpersisted_metrics(value) for value in payload]
    return payload


def model_cache_path(tables_dir, model_name):
    """Return a stable ``<MODEL>.json`` cache path."""
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(model_name)).strip("_")
    if not safe_name:
        raise ValueError("model_name must contain at least one safe filename character.")
    return Path(tables_dir) / f"{safe_name}.json"


def backup_model_cache(tables_dir, model_name, reason="incompatible"):
    """Copy an existing cache to a non-overwriting recovery file."""
    source = model_cache_path(tables_dir, model_name)
    if not source.exists():
        return None
    safe_reason = re.sub(r"[^A-Za-z0-9_-]+", "_", str(reason)).strip("_")
    safe_reason = safe_reason or "backup"
    candidate = source.with_name(f"{source.stem}.{safe_reason}.bak.json")
    counter = 1
    while candidate.exists():
        candidate = source.with_name(
            f"{source.stem}.{safe_reason}.{counter}.bak.json"
        )
        counter += 1
    shutil.copy2(source, candidate)
    return candidate


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _atomic_json_dump(payload, path):
    """Write JSON atomically so an interrupted run does not corrupt its cache."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}_", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                strip_unpersisted_metrics(payload),
                stream,
                indent=2,
                ensure_ascii=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_model_cache(tables_dir, model_name):
    """Load a per-model cache, accepting both schema-v1 and bare legacy maps."""
    path = model_cache_path(tables_dir, model_name)
    if not path.exists():
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "model": model_name,
            "settings": {},
            "datasets": {},
        }

    payload = _read_json(path)
    if "datasets" not in payload:
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "model": model_name,
            "settings": {"source": "bare_legacy_cache"},
            "datasets": payload,
        }
    payload.setdefault("schema_version", CACHE_SCHEMA_VERSION)
    payload.setdefault("model", model_name)
    payload.setdefault("settings", {})
    payload.setdefault("datasets", {})
    return strip_unpersisted_metrics(payload)


def save_model_cache(tables_dir, model_name, datasets, settings=None):
    """Persist all completed dataset results for one model."""
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model": model_name,
        "settings": dict(settings or {}),
        "datasets": datasets,
    }
    path = model_cache_path(tables_dir, model_name)
    _atomic_json_dump(payload, path)
    return path


def import_legacy_model_results(legacy_results, model_name, source_model=None):
    """Extract one model from the old dataset-first ``raw_results.json``."""
    source = source_model or model_name
    imported = {}
    for dataset_name, dataset_results in (legacy_results or {}).items():
        if source in dataset_results:
            imported[dataset_name] = dataset_results[source]
    return imported


def load_legacy_results(path):
    """Load the combined legacy cache, returning an empty map when absent."""
    path = Path(path)
    return _read_json(path) if path.exists() else {}


def save_legacy_results(path, all_results):
    """Update the combined compatibility cache without a non-atomic rewrite."""
    _atomic_json_dump(all_results, path)
    return Path(path)
