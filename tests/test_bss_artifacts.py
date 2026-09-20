import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from main import _create_model, _evaluate_model
from scripts.audit_v3_results import audit_bss_results
from src.evaluation.cv import get_multilabel_cv
from src.evaluation.metric_contract import (
    BSS_SPCC_FULL_METRIC_NAMES,
    BSS_SPCC_SELECTIVE_METRIC_NAMES,
)
from src.evaluation.pipeline_v3 import run_experiment_v3
from src.visualization.plots import (
    BSS_DATASET_FIGURES,
    BSS_REJECTION_FIGURES,
)


class BSSArtifactTests(unittest.TestCase):
    @staticmethod
    def _loader(name):
        if name != "tiny_bss":
            raise ValueError(name)
        rng = np.random.RandomState(31)
        X = rng.normal(size=(24, 4))
        y0 = (X[:, 0] > 0).astype(np.int32)
        y1 = ((y0 + (X[:, 1] > 0)) >= 1).astype(np.int32)
        y2 = ((y0 + y1 + (X[:, 2] > 0)) >= 2).astype(np.int32)
        Y = np.column_stack((y0, y1, y2))
        return X, Y, None, ["y0", "y1", "y2"]

    def test_synthetic_pipeline_exports_exact_profile_and_fifteen_figures(self):
        with tempfile.TemporaryDirectory() as temporary:
            document = run_experiment_v3(
                datasets=["tiny_bss"],
                models=["BSS_UG_SPCC_PA_Logistic"],
                n_splits=2,
                random_state=37,
                output_dir=Path(temporary) / "bss_v3",
                abstention_costs=(0.2, 0.3),
                report_cost=0.3,
                abstention_penalty="linear",
                mlc_pa_base="logistic",
                gsi_validation_size=0.2,
                dataset_loader=self._loader,
                cv_factory=get_multilabel_cv,
                model_factory=_create_model,
                evaluator=_evaluate_model,
                bss_parameters={
                    "inner_oof_splits": 2,
                    "alpha_grid": (0.2,),
                    "q_max": 1,
                    "calibration_splits": 2,
                },
                metric_profile="bss_ug_spcc_pa_v1",
                refresh_artifacts_per_dataset=True,
            )
            self.assertEqual(document["Status"], "complete")
            figure_dir = (
                Path(temporary)
                / "bss_v3"
                / "figures"
                / document["Config Hash"]
            )
            expected_figures = set(BSS_DATASET_FIGURES.values()) | set(
                BSS_REJECTION_FIGURES.values()
            )
            self.assertEqual(
                {path.name for path in figure_dir.glob("*.png")}, expected_figures
            )

            table_dir = (
                Path(temporary)
                / "bss_v3"
                / "tables"
                / document["Config Hash"]
            )
            with (table_dir / "results.json").open(encoding="utf-8") as stream:
                exported = json.load(stream)
            summary = exported["Results"]["tiny_bss"][
                "BSS_UG_SPCC_PA_Logistic"
            ]
            self.assertEqual(
                set(summary["Full"]["Mean"]), set(BSS_SPCC_FULL_METRIC_NAMES)
            )
            self.assertEqual(
                set(summary["Costs"]["0.30"]["Selective"]["Mean"]),
                set(BSS_SPCC_SELECTIVE_METRIC_NAMES),
            )
            for required in (
                "fold_results.csv",
                "summary_results.csv",
                "structure_audit.json",
                "artifact_manifest.json",
            ):
                self.assertTrue((table_dir / required).is_file())
            with (table_dir / "fold_results.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(rows)
            self.assertNotIn("Hamming Loss", rows[0])
            self.assertNotIn("AURC", rows[0])
            audit = audit_bss_results(
                Path(temporary) / "bss_v3",
                config_hash=document["Config Hash"],
                expected_datasets=["tiny_bss"],
                expected_models=["BSS_UG_SPCC_PA_Logistic"],
                expected_folds=2,
                expected_costs=[0.2, 0.3],
            )
            self.assertEqual(audit["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
