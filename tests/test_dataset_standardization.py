import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.validate_v6_3_3_datasets import (
    load_manifest,
    validate_dataset,
    validate_summary,
)
from src.data.loader import (
    DATASET_CONFIG,
    V6_3_3_DATASETS,
    canonicalize_dataset_name,
    load_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "datasets_v6_3_3.json"


def test_canonical_dataset_names_and_aliases():
    assert canonicalize_dataset_name(" Emotions ") == "emotions"
    assert canonicalize_dataset_name("CHD_49") == "chd49"
    assert canonicalize_dataset_name("HumanPseAAC") == "humanpseaac"
    assert canonicalize_dataset_name("gram_positive_pseaac") == "gpositivepseaac"
    with pytest.raises(ValueError):
        canonicalize_dataset_name("")


def test_frozen_manifest_matches_loader_registry():
    manifest = load_manifest(MANIFEST_PATH)
    assert tuple(manifest["dataset_order"]) == V6_3_3_DATASETS
    assert set(manifest["datasets"]) == set(V6_3_3_DATASETS)
    assert set(V6_3_3_DATASETS).issubset(DATASET_CONFIG)


def test_manifest_is_plain_json_without_absolute_paths():
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for dataset_spec in payload["datasets"].values():
        for file_spec in dataset_spec["files"]:
            assert not Path(file_spec["path"]).is_absolute()
            assert len(file_spec["sha256"]) == 64


def test_csv_summary_matches_json_manifest():
    assert not validate_summary(ROOT / "data" / "dataset_manifest.csv", load_manifest(MANIFEST_PATH))


def test_csv_summary_detects_dataset_statistics_assigned_to_wrong_dataset():
    manifest = copy.deepcopy(load_manifest(MANIFEST_PATH))
    manifest["datasets"]["emotions"]["expected"]["duplicate_rows"] = 455
    errors = validate_summary(ROOT / "data" / "dataset_manifest.csv", manifest)
    assert "emotions: CSV duplicate_rows differs from JSON" in errors


@pytest.fixture
def tiny_dataset(tmp_path, monkeypatch):
    """A standalone raw fixture; integrity tests do not require benchmark data."""
    source = tmp_path / "tiny.arff"
    source.write_text(
        "@relation tiny\n@attribute x numeric\n@attribute y {0,1}\n"
        "@data\n0,0\n1,1\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(DATASET_CONFIG, "tiny", {
        "arff_path": "tiny.arff", "xml_path": None,
        "label_location": "end", "num_labels": 1,
    })
    spec = {
        "format": "meka_arff",
        "files": [{
            "path": "tiny.arff", "bytes": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }],
        "expected": {
            "n_samples": 2, "n_features": 1, "n_labels": 1,
            "label_cardinality": 0.5, "label_density": 0.5,
            "min_label_positives": 1, "max_label_positives": 1,
            "duplicate_rows": 0,
        },
    }
    return source, spec


def test_validator_rejects_file_changed_without_size_change(tiny_dataset):
    source, spec = tiny_dataset
    source.write_text(source.read_text(encoding="utf-8").replace("1,1", "2,1"), encoding="utf-8")
    result = validate_dataset("tiny", spec, source.parent)
    assert result["status"] == "FAIL"
    assert any("SHA256 mismatch" in error for error in result["errors"])


def test_validator_rejects_missing_source(tiny_dataset):
    source, spec = tiny_dataset
    result = validate_dataset("tiny", spec, source.parent / "missing")
    assert result["status"] == "FAIL"
    assert any("missing source file" in error for error in result["errors"])


def test_validator_rejects_wrong_expected_shape(tiny_dataset):
    source, spec = tiny_dataset
    spec["expected"]["n_features"] = 2
    result = validate_dataset("tiny", spec, source.parent)
    assert result["status"] == "FAIL"
    assert any("shape" in error for error in result["errors"])


@pytest.mark.skipif(
    not (ROOT / "data" / "genbase" / "genbase.arff").is_file(),
    reason="raw benchmark data are not installed",
)
def test_genbase_identifier_is_excluded():
    X, _, feature_names, _ = load_dataset("genbase", base_dir=str(ROOT))
    assert X.shape[1] == 1185
    assert "protein" not in feature_names


@pytest.mark.parametrize("name", V6_3_3_DATASETS)
def test_collection_passes_validation(name):
    manifest = load_manifest(MANIFEST_PATH)
    spec = manifest["datasets"][name]
    if any(not (ROOT / item["path"]).is_file() for item in spec["files"]):
        pytest.skip(f"raw benchmark data for {name} are not installed")
    result = validate_dataset(name, spec, ROOT)
    assert result["status"] != "FAIL", result["errors"]
