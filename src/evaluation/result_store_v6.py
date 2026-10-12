"""Atomic, resumable result store for the v6.3 experiment contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd


COMMON_COLUMNS = [
    "version",
    "base_learner",
    "decision_policy",
    "model",
    "dataset",
    "mean_ir",
    "fold",
    "aggregation",
    "source_version",
    "provenance_status",
    "config_hash",
    "split_hash",
]

TABLE_SCHEMAS = {
    "complete_metric.csv": COMMON_COLUMNS
    + [
        "hamming_accuracy",
        "subset_accuracy",
        "micro_f1",
        "macro_f1",
        "instance_f1",
    ],
    "selective_metric.csv": COMMON_COLUMNS
    + [
        "cost",
        "selective_hamming_accuracy",
        "selective_subset_accuracy",
        "selective_micro_f1",
        "selective_macro_f1",
        "selective_instance_f1",
        "generalized_loss",
        "coverage",
        "abs",
        "aabs",
    ],
    "calibration_metric.csv": COMMON_COLUMNS
    + ["brier_score", "ece", "log_loss", "ece_bin_count"],
    "model_performance.csv": COMMON_COLUMNS
    + [
        "fit_time_sec_mean",
        "fit_time_sec_std",
        "inference_time_sec_mean",
        "inference_time_sec_std",
        "il_time_sec_mean",
        "il_time_sec_std",
        "dl_time_sec_mean",
        "dl_time_sec_std",
        "il_count_mean",
        "il_count_std",
        "dl_count_mean",
        "dl_count_std",
        "dl_parent_count_mean",
        "dl_parent_count_std",
        "dl_parent_map_by_fold",
        "chain_order_by_fold",
        "pcc_num_samples",
        "computed_fold_count",
        "inherited_fold_count",
        "missing_fold_count",
    ],
    "layer_performance.csv": COMMON_COLUMNS
    + [
        "layer_type",
        "layer_index",
        "label_indices",
        "label_count",
        "cumulative_label_count",
        "full_macro_f1",
        "selective_macro_f1",
        "coverage",
        "cumulative_full_macro_f1",
        "cumulative_selective_macro_f1",
        "cumulative_coverage",
        "status",
    ],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            _json_safe(payload),
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, na_rep="-")
    os.replace(temporary, path)


class V63ResultStore:
    """Create the locked result tree and persist model/dataset artifacts."""

    LEARNER_DIR = {
        "logistic": "logistic",
        "svm_calibrated": "svm",
        "mlp": "mlp",
    }

    def __init__(self, root: Path, learners: Iterable[str], models: Iterable[str]):
        self.root = Path(root)
        self.learners = list(learners)
        self.models = list(models)
        for learner in self.learners:
            learner_dir = self.learner_root(learner)
            for policy in ("hamming_bop", "f_measure_bop"):
                for child in ("checkpoints", "figures", "tables"):
                    (learner_dir / policy / child).mkdir(parents=True, exist_ok=True)
            for model in self.models:
                (learner_dir / "hamming_bop" / "checkpoints" / model / "predictions").mkdir(
                    parents=True, exist_ok=True
                )
        for policy in ("hamming_bop", "f_measure_bop"):
            for child in ("checkpoints", "figures", "tables"):
                (self.root / "all" / policy / child).mkdir(
                    parents=True, exist_ok=True
                )

    def learner_root(self, learner: str) -> Path:
        return self.root / self.LEARNER_DIR.get(learner, learner)

    def checkpoint_path(self, learner: str, model: str, dataset: str) -> Path:
        return (
            self.learner_root(learner)
            / "hamming_bop"
            / "checkpoints"
            / model
            / f"{dataset}.json"
        )

    def prediction_path(self, learner: str, model: str, dataset: str) -> Path:
        return (
            self.learner_root(learner)
            / "hamming_bop"
            / "checkpoints"
            / model
            / "predictions"
            / f"{dataset}.npz"
        )

    def load_checkpoint(self, learner: str, model: str, dataset: str) -> Optional[dict]:
        path = self.checkpoint_path(learner, model, dataset)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    def is_fold_complete(
        self,
        learner: str,
        model: str,
        dataset: str,
        fold: int,
        *,
        expected_config_hash: Optional[str] = None,
        expected_split_hash: Optional[str] = None,
    ) -> bool:
        checkpoint = self.load_checkpoint(learner, model, dataset)
        prediction_path = self.prediction_path(learner, model, dataset)
        if checkpoint is None or not prediction_path.exists():
            return False
        if expected_config_hash is not None and checkpoint.get("config_hash") != expected_config_hash:
            return False
        fold_record = next(
            (
                row
                for row in checkpoint.get("folds", [])
                if int(row.get("fold", -1)) == int(fold)
            ),
            None,
        )
        if fold_record is None or fold_record.get("status") != "complete":
            return False
        if expected_split_hash is not None and fold_record.get("split_hash") != expected_split_hash:
            return False
        expected = checkpoint.get("prediction", {}).get("sha256")
        return bool(expected and expected == sha256_file(prediction_path))

    @staticmethod
    def _merge_prediction_arrays(
        path: Path,
        fold_arrays: Mapping[str, np.ndarray],
        *,
        reset: bool = False,
    ):
        arrays = {key: np.asarray(value) for key, value in fold_arrays.items()}
        fold_value = int(np.asarray(arrays["fold_ids"]).ravel()[0])
        if path.exists() and not reset:
            with np.load(path, allow_pickle=False) as archive:
                old = {key: archive[key] for key in archive.files}
            keep = np.asarray(old["fold_ids"]) != fold_value
            merged = {}
            for key in sorted(set(old) | set(arrays)):
                if key not in old or key not in arrays:
                    raise ValueError(f"Prediction schema drift for key '{key}'.")
                merged[key] = np.concatenate([old[key][keep], arrays[key]], axis=0)
            order = np.lexsort((merged["test_indices"], merged["fold_ids"]))
            return {key: value[order] for key, value in merged.items()}
        order = np.lexsort((arrays["test_indices"], arrays["fold_ids"]))
        return {key: value[order] for key, value in arrays.items()}

    def save_computed_fold(
        self,
        *,
        learner: str,
        model: str,
        dataset: str,
        checkpoint_base: dict,
        fold_record: dict,
        fold_arrays: Mapping[str, np.ndarray],
    ):
        prediction_path = self.prediction_path(learner, model, dataset)
        existing_checkpoint = self.load_checkpoint(learner, model, dataset)
        reset = bool(
            existing_checkpoint is not None
            and existing_checkpoint.get("config_hash")
            != checkpoint_base.get("config_hash")
        )
        merged = self._merge_prediction_arrays(
            prediction_path, fold_arrays, reset=reset
        )
        _atomic_npz(prediction_path, merged)
        prediction_hash = sha256_file(prediction_path)

        checkpoint = (
            dict(checkpoint_base)
            if reset
            else (existing_checkpoint or dict(checkpoint_base))
        )
        records = [
            row
            for row in checkpoint.get("folds", [])
            if int(row.get("fold", -1)) != int(fold_record["fold"])
        ]
        records.append({**fold_record, "status": "complete"})
        records.sort(key=lambda row: int(row["fold"]))
        checkpoint.update(checkpoint_base)
        checkpoint["folds"] = records
        checkpoint["status"] = (
            "complete"
            if len(records) >= int(checkpoint.get("expected_fold_count", 5))
            else "partial"
        )
        checkpoint["prediction"] = {
            "path": str(prediction_path.relative_to(self.root)).replace("\\", "/"),
            "sha256": prediction_hash,
            "keys": sorted(merged),
            "row_count": int(merged["y_true"].shape[0]),
        }
        _atomic_json(self.checkpoint_path(learner, model, dataset), checkpoint)

    def save_inherited_checkpoint(
        self, *, learner: str, model: str, dataset: str, payload: dict
    ):
        _atomic_json(self.checkpoint_path(learner, model, dataset), payload)

    @staticmethod
    def _common(checkpoint: dict, fold_record: dict, aggregation: str = "fold"):
        return {
            "version": "v6_3",
            "base_learner": checkpoint["base_learner"],
            "decision_policy": checkpoint.get("decision_policy", "hamming_bop"),
            "model": checkpoint["model"],
            "dataset": checkpoint["dataset"],
            "mean_ir": checkpoint.get("mean_ir"),
            "fold": fold_record.get("fold", "-"),
            "aggregation": aggregation,
            "source_version": checkpoint.get("source_version", "v6_3"),
            "provenance_status": fold_record.get(
                "provenance_status", checkpoint.get("provenance_status", "computed")
            ),
            "config_hash": checkpoint.get("config_hash", "-"),
            "split_hash": fold_record.get("split_hash", "-"),
        }

    @staticmethod
    def _append_mean_std(rows: List[dict], metric_columns: List[str], group_columns: List[str]):
        fold_rows = [row for row in rows if row.get("aggregation") == "fold"]
        if not fold_rows:
            return rows
        frame = pd.DataFrame(fold_rows)
        additions = []
        for _, group in frame.groupby(group_columns, dropna=False, sort=False):
            prototype = group.iloc[0].to_dict()
            for aggregation in ("mean", "std"):
                row = dict(prototype)
                row["fold"] = "-"
                row["aggregation"] = aggregation
                for column in metric_columns:
                    numeric = pd.to_numeric(group[column], errors="coerce")
                    row[column] = (
                        numeric.mean()
                        if aggregation == "mean"
                        else numeric.std(ddof=0)
                    )
                additions.append(row)
        return rows + additions

    def _rows_from_checkpoints(self, learner: str):
        complete_rows: List[dict] = []
        selective_rows: List[dict] = []
        calibration_rows: List[dict] = []
        layer_rows: List[dict] = []
        performance_rows: List[dict] = []
        checkpoint_root = self.learner_root(learner) / "hamming_bop" / "checkpoints"
        for path in sorted(checkpoint_root.glob("*/*.json")):
            with path.open("r", encoding="utf-8") as stream:
                checkpoint = json.load(stream)
            folds = checkpoint.get("folds", [])
            for fold in folds:
                common = self._common(checkpoint, fold)
                complete_rows.append({**common, **fold.get("complete_metrics", {})})
                for selective in fold.get("selective_metrics", []):
                    selective_rows.append({**common, **selective})
                calibration_rows.append(
                    {**common, **fold.get("calibration_metrics", {})}
                )
                for layer in fold.get("layer_metrics", []):
                    layer_rows.append({**common, **layer})

            if folds:
                times = {
                    key: pd.to_numeric(
                        pd.Series([fold.get("timing", {}).get(key) for fold in folds]),
                        errors="coerce",
                    )
                    for key in ("fit_time_sec", "inference_time_sec", "il_time_sec", "dl_time_sec")
                }
                counts = {
                    key: pd.to_numeric(
                        pd.Series([fold.get(key) for fold in folds]), errors="coerce"
                    )
                    for key in ("il_count", "dl_count", "dl_parent_count")
                }
                status_values = [
                    fold.get("provenance_status", checkpoint.get("provenance_status"))
                    for fold in folds
                ]
                common = self._common(checkpoint, {"fold": "-"}, "mean")
                common["provenance_status"] = checkpoint.get(
                    "provenance_status", "computed"
                )
                performance_rows.append(
                    {
                        **common,
                        **{
                            f"{key}_mean": values.mean()
                            for key, values in times.items()
                        },
                        **{
                            f"{key}_std": values.std(ddof=0)
                            for key, values in times.items()
                        },
                        **{
                            f"{key}_mean": values.mean()
                            for key, values in counts.items()
                        },
                        **{
                            f"{key}_std": values.std(ddof=0)
                            for key, values in counts.items()
                        },
                        "dl_parent_map_by_fold": json.dumps(
                            {
                                str(fold.get("fold")): fold.get("dl_parent_map")
                                for fold in folds
                            },
                            separators=(",", ":"),
                        ),
                        "chain_order_by_fold": json.dumps(
                            {
                                str(fold.get("fold")): fold.get("chain_order")
                                for fold in folds
                            },
                            separators=(",", ":"),
                        ),
                        "pcc_num_samples": checkpoint.get("pcc_num_samples"),
                        "computed_fold_count": sum(
                            status == "computed" for status in status_values
                        ),
                        "inherited_fold_count": sum(
                            str(status).startswith("inherited") for status in status_values
                        ),
                        "missing_fold_count": max(
                            0,
                            int(checkpoint.get("expected_fold_count", 5)) - len(folds),
                        ),
                    }
                )
        return complete_rows, selective_rows, calibration_rows, performance_rows, layer_rows

    def rebuild_tables(self, learner: str):
        table_root = self.learner_root(learner) / "hamming_bop" / "tables"
        complete, selective, calibration, performance, layers = self._rows_from_checkpoints(
            learner
        )
        complete = self._append_mean_std(
            complete,
            TABLE_SCHEMAS["complete_metric.csv"][len(COMMON_COLUMNS) :],
            ["base_learner", "model", "dataset"],
        )
        selective = self._append_mean_std(
            selective,
            TABLE_SCHEMAS["selective_metric.csv"][len(COMMON_COLUMNS) + 1 :],
            ["base_learner", "model", "dataset", "cost"],
        )
        calibration = self._append_mean_std(
            calibration,
            ["brier_score", "ece", "log_loss"],
            ["base_learner", "model", "dataset"],
        )
        layers = self._append_mean_std(
            layers,
            [
                "label_count",
                "cumulative_label_count",
                "full_macro_f1",
                "selective_macro_f1",
                "coverage",
                "cumulative_full_macro_f1",
                "cumulative_selective_macro_f1",
                "cumulative_coverage",
            ],
            ["base_learner", "model", "dataset", "layer_type", "layer_index"],
        )
        rows_by_name = {
            "complete_metric.csv": complete,
            "selective_metric.csv": selective,
            "calibration_metric.csv": calibration,
            "model_performance.csv": performance,
            "layer_performance.csv": layers,
        }
        for filename, rows in rows_by_name.items():
            schema = TABLE_SCHEMAS[filename]
            frame = pd.DataFrame(rows)
            for column in schema:
                if column not in frame:
                    frame[column] = np.nan
            frame = frame[schema]
            _atomic_csv(table_root / filename, frame)

    def rebuild_all_tables(self):
        destination = self.root / "all" / "hamming_bop" / "tables"
        for filename, schema in TABLE_SCHEMAS.items():
            frames = []
            for learner in self.learners:
                source = self.learner_root(learner) / "hamming_bop" / "tables" / filename
                if source.exists():
                    frames.append(pd.read_csv(source, keep_default_na=False))
            merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=schema)
            for column in schema:
                if column not in merged:
                    merged[column] = "-"
            _atomic_csv(destination / filename, merged[schema])

        manifests = []
        for learner in self.learners:
            for path in sorted(
                (self.learner_root(learner) / "hamming_bop" / "checkpoints").glob("*/*.json")
            ):
                manifests.append(
                    {
                        "base_learner": learner,
                        "path": str(path.relative_to(self.root)).replace("\\", "/"),
                        "sha256": sha256_file(path),
                    }
                )
        _atomic_json(
            self.root / "all" / "hamming_bop" / "checkpoints" / "manifest.json",
            {"version": "v6_3", "checkpoints": manifests},
        )


__all__ = [
    "COMMON_COLUMNS",
    "TABLE_SCHEMAS",
    "V63ResultStore",
    "sha256_file",
    "stable_hash",
]
