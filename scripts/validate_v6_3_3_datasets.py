"""Validate the frozen raw-data collection used by GSI-MLC-PA v6.3.3.

The validator is intentionally read-only. It verifies source-file hashes,
loader/manifest alignment, matrix contracts, label support, known duplicate
counts, and PseAAC sample identifiers without rewriting the raw datasets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.data.loader import (  # noqa: E402
    DATASET_CONFIG,
    V6_3_3_DATASETS,
    canonicalize_dataset_name,
    load_dataset,
)


DEFAULT_MANIFEST = WORKSPACE_ROOT / "configs" / "datasets_v6_3_3.json"
DEFAULT_SUMMARY = WORKSPACE_ROOT / "data" / "dataset_manifest.csv"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def configured_source_paths(dataset_name: str) -> List[str]:
    config = DATASET_CONFIG[dataset_name]
    fmt = config.get("format", "arff")
    if fmt == "mat":
        return [config["mat_path"]]
    if fmt == "pseaac":
        return [config["train_path"], config["test_path"]]

    paths = [config["arff_path"]]
    if config.get("xml_path"):
        paths.append(config["xml_path"])
    return paths


def count_duplicate_rows(X: np.ndarray, Y: np.ndarray) -> int:
    combined = np.ascontiguousarray(
        np.hstack([X.astype(np.float64), Y.astype(np.float64)])
    )
    _, counts = np.unique(combined, axis=0, return_counts=True)
    return int(np.sum(counts - 1))


def validate_pseaac_identifiers(paths: Sequence[Path]) -> Tuple[List[str], int]:
    errors: List[str] = []
    all_identifiers: List[str] = []
    for path in paths:
        identifiers: List[str] = []
        with path.open("r", encoding="utf-8", errors="ignore") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    errors.append(f"{path}: line {line_number} has fewer than 3 fields")
                    continue
                identifiers.append(parts[0])
        if len(identifiers) != len(set(identifiers)):
            errors.append(f"{path}: duplicate sample identifiers")
        all_identifiers.extend(identifiers)

    if len(all_identifiers) != len(set(all_identifiers)):
        errors.append("PseAAC train/test files contain overlapping sample identifiers")
    return errors, len(all_identifiers)


def _close(actual: float, expected: float) -> bool:
    return bool(np.isclose(actual, expected, rtol=0.0, atol=1e-12))


def validate_dataset(
    dataset_name: str,
    spec: Dict[str, Any],
    workspace_root: Path,
    check_checksums: bool = True,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = list(spec.get("known_warnings", []))
    file_specs = spec.get("files", [])
    manifest_paths = [item["path"].replace("\\", "/") for item in file_specs]
    loader_paths = [path.replace("\\", "/") for path in configured_source_paths(dataset_name)]
    if manifest_paths != loader_paths:
        errors.append(
            f"manifest paths {manifest_paths} differ from loader paths {loader_paths}"
        )

    resolved_paths: List[Path] = []
    for file_spec in file_specs:
        relative_path = Path(file_spec["path"])
        path = workspace_root / relative_path
        resolved_paths.append(path)
        if not path.is_file():
            errors.append(f"missing source file: {relative_path.as_posix()}")
            continue
        actual_bytes = path.stat().st_size
        if actual_bytes != int(file_spec["bytes"]):
            errors.append(
                f"size mismatch for {relative_path.as_posix()}: "
                f"{actual_bytes} != {file_spec['bytes']}"
            )
        if check_checksums:
            actual_hash = sha256_file(path)
            if actual_hash != file_spec["sha256"]:
                errors.append(
                    f"SHA256 mismatch for {relative_path.as_posix()}: {actual_hash}"
                )

    if errors:
        return {
            "dataset": dataset_name,
            "status": "FAIL",
            "errors": errors,
            "warnings": warnings,
        }

    if spec["format"] == "pseaac_train_test_text":
        id_errors, identifier_count = validate_pseaac_identifiers(resolved_paths)
        errors.extend(id_errors)
        if identifier_count != int(spec["expected"]["n_samples"]):
            errors.append(
                f"PseAAC identifier count {identifier_count} != "
                f"expected {spec['expected']['n_samples']}"
            )

    X, Y, feature_names, label_names = load_dataset(
        dataset_name, base_dir=str(workspace_root)
    )
    expected = spec["expected"]
    actual_shape = (int(X.shape[0]), int(X.shape[1]), int(Y.shape[1]))
    expected_shape = (
        int(expected["n_samples"]),
        int(expected["n_features"]),
        int(expected["n_labels"]),
    )
    if actual_shape != expected_shape:
        errors.append(f"shape {actual_shape} != expected {expected_shape}")
    if X.dtype != np.float32:
        errors.append(f"X dtype {X.dtype} != float32")
    if Y.dtype != np.int32:
        errors.append(f"Y dtype {Y.dtype} != int32")
    if not np.isfinite(X).all():
        errors.append("X contains NaN or infinite values")
    if not np.isin(Y, [0, 1]).all():
        errors.append("Y contains values outside {0, 1}")
    if len(feature_names) != X.shape[1] or len(set(feature_names)) != len(feature_names):
        errors.append("feature names are missing or non-unique")
    if len(label_names) != Y.shape[1] or len(set(label_names)) != len(label_names):
        errors.append("label names are missing or non-unique")

    positive_counts = Y.sum(axis=0)
    min_positive = int(positive_counts.min())
    max_positive = int(positive_counts.max())
    if min_positive <= 0:
        errors.append("at least one label has no positive sample")
    if max_positive >= Y.shape[0]:
        errors.append("at least one label has no negative sample")
    if min_positive != int(expected["min_label_positives"]):
        errors.append(
            f"minimum label support {min_positive} != "
            f"expected {expected['min_label_positives']}"
        )
    if max_positive != int(expected["max_label_positives"]):
        errors.append(
            f"maximum label support {max_positive} != "
            f"expected {expected['max_label_positives']}"
        )

    cardinality = float(Y.sum(axis=1).mean())
    density = float(Y.mean())
    if not _close(cardinality, float(expected["label_cardinality"])):
        errors.append(
            f"label cardinality {cardinality} != {expected['label_cardinality']}"
        )
    if not _close(density, float(expected["label_density"])):
        errors.append(f"label density {density} != {expected['label_density']}")

    duplicates = count_duplicate_rows(X, Y)
    if duplicates != int(expected["duplicate_rows"]):
        errors.append(
            f"duplicate row count {duplicates} != expected {expected['duplicate_rows']}"
        )

    manifest_ignored = spec.get("transforms", {}).get("ignored_feature_names", [])
    loader_ignored = DATASET_CONFIG[dataset_name].get("ignored_feature_names", [])
    if manifest_ignored != loader_ignored:
        errors.append(
            f"ignored features {manifest_ignored} differ from loader {loader_ignored}"
        )

    return {
        "dataset": dataset_name,
        "status": "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS"),
        "shape": actual_shape,
        "cardinality": cardinality,
        "density": density,
        "duplicates": duplicates,
        "errors": errors,
        "warnings": warnings,
    }


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported dataset manifest schema: {manifest.get('schema_version')}")
    order = tuple(manifest.get("dataset_order", []))
    if order != V6_3_3_DATASETS:
        raise ValueError(
            f"Manifest order {order} differs from loader registry {V6_3_3_DATASETS}"
        )
    if set(manifest.get("datasets", {})) != set(order):
        raise ValueError("Manifest datasets do not match dataset_order")
    return manifest


def validate_summary(path: Path, manifest: Dict[str, Any]) -> List[str]:
    """Check that the compact CSV matches the authoritative JSON manifest."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [row.get("canonical_name") for row in rows] != manifest["dataset_order"]:
        return ["CSV dataset names/order differ from the JSON manifest"]

    errors: List[str] = []
    integer_fields = (
        "n_samples", "n_features", "n_labels", "min_label_positives",
        "max_label_positives", "duplicate_rows",
    )
    for row in rows:
        name = row["canonical_name"]
        spec = manifest["datasets"][name]
        try:
            if row["format"] != spec["format"]:
                errors.append(f"{name}: CSV format differs from JSON")
            for field in integer_fields:
                if int(row[field]) != int(spec["expected"][field]):
                    errors.append(f"{name}: CSV {field} differs from JSON")
            for field in ("label_cardinality", "label_density"):
                if not _close(float(row[field]), float(spec["expected"][field])):
                    errors.append(f"{name}: CSV {field} differs from JSON")
            if int(row["source_file_count"]) != len(spec["files"]):
                errors.append(f"{name}: CSV source_file_count differs from JSON")
            if int(row["raw_bytes"]) != sum(item["bytes"] for item in spec["files"]):
                errors.append(f"{name}: CSV raw_bytes differs from JSON")
            digest = hashlib.sha256()
            for file_spec in spec["files"]:
                digest.update(
                    f"{file_spec['path']}\t{file_spec['sha256']}\n".encode("utf-8")
                )
            if row["source_set_sha256"] != digest.hexdigest():
                errors.append(f"{name}: CSV source_set_sha256 differs from JSON")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{name}: invalid CSV metadata: {exc}")
    return errors


def print_results(results: Iterable[Dict[str, Any]]) -> None:
    results = list(results)
    print(
        f"{'dataset':<18} {'status':<18} {'shape (N,d,K)':<20} "
        f"{'cardinality':>11} {'density':>10} {'dups':>5}"
    )
    print("-" * 88)
    for result in results:
        shape = result.get("shape")
        shape_text = "-" if shape is None else f"{shape[0]},{shape[1]},{shape[2]}"
        cardinality = result.get("cardinality")
        density = result.get("density")
        print(
            f"{result['dataset']:<18} {result['status']:<18} {shape_text:<20} "
            f"{cardinality if cardinality is not None else float('nan'):>11.6f} "
            f"{density if density is not None else float('nan'):>10.6f} "
            f"{result.get('duplicates', 0):>5}"
        )
        for warning in result.get("warnings", []):
            print(f"  WARNING: {warning}")
        for error in result.get("errors", []):
            print(f"  ERROR: {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=WORKSPACE_ROOT,
        help="Project root containing data/ and configs/.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST.relative_to(WORKSPACE_ROOT),
        help="Frozen JSON dataset manifest.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY.relative_to(WORKSPACE_ROOT),
        help="CSV summary checked against the JSON manifest.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Optional subset of canonical names or supported aliases.",
    )
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Skip SHA256 verification for a faster structural check.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace_root = args.workspace_root.resolve()
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = workspace_root / manifest_path
    manifest = load_manifest(manifest_path.resolve())
    summary_path = args.summary
    if not summary_path.is_absolute():
        summary_path = workspace_root / summary_path
    summary_errors = validate_summary(summary_path, manifest)
    if summary_errors:
        for error in summary_errors:
            print(f"ERROR: {error}")
        return 1

    selected = (
        [canonicalize_dataset_name(name) for name in args.datasets]
        if args.datasets
        else list(manifest["dataset_order"])
    )
    unknown = [name for name in selected if name not in manifest["datasets"]]
    if unknown:
        raise ValueError(f"Datasets are not part of v6.3.3: {unknown}")

    results = []
    for name in dict.fromkeys(selected):
        try:
            result = validate_dataset(
                name,
                manifest["datasets"][name],
                workspace_root,
                check_checksums=not args.skip_checksums,
            )
        except Exception as exc:
            result = {
                "dataset": name,
                "status": "FAIL",
                "errors": [f"{type(exc).__name__}: {exc}"],
                "warnings": [],
            }
        results.append(result)
    print_results(results)
    failures = sum(result["status"] == "FAIL" for result in results)
    print(f"\nValidated {len(results)} dataset(s); failures={failures}.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
