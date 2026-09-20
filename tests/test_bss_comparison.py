import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.bss_comparison import (
    comparison_status,
    load_bss_comparison_bundle,
)
from src.visualization.plots import (
    BSS_DATASET_FIGURES,
    BSS_REJECTION_FIGURES,
    generate_bss_spcc_plots,
)


def _full_legacy_row(offset=0.0):
    return {
        "Macro-F1": 0.60 + offset,
        "Micro-F1": 0.65 + offset,
        "Hamming Loss": 0.20 - offset,
        "Subset Accuracy": 0.25 + offset,
        "Example-F1": 0.55 + offset,
    }


class BSSComparisonTests(unittest.TestCase):
    def _fixtures(self, root):
        root = Path(root)
        legacy_path = root / "raw_results.json"
        mlc_path = root / "MLC_PA.json"
        cc_root = root / "cc"
        mlc_v3_root = root / "mlc_v3"
        legacy_path.write_text(
            json.dumps({
                "tiny": {
                    "BR_Logistic": {
                        "raw_folds": [
                            _full_legacy_row(),
                            _full_legacy_row(0.02),
                        ]
                    }
                }
            }),
            encoding="utf-8",
        )
        selective_rows = [
            {
                "Coverage": 0.7,
                "AABS": 0.3,
                "ABS": 0.8,
                "Generalized Loss": 0.18,
                "Selective Macro-F1": 0.66,
                "Selective Micro-F1": 0.70,
                "Selective Hamming Loss": 0.10,
            },
            {
                "Coverage": 0.8,
                "AABS": 0.2,
                "ABS": 0.7,
                "Generalized Loss": 0.16,
                "Selective Macro-F1": 0.68,
                "Selective Micro-F1": 0.72,
                "Selective Hamming Loss": 0.08,
            },
        ]
        mlc_path.write_text(
            json.dumps({
                "schema_version": 2,
                "model": "MLC_PA",
                "settings": {
                    "n_splits": 2,
                    "random_state": 42,
                    "abstention_costs": [0.2, 0.3],
                    "report_cost": 0.3,
                    "abstention_penalty": "linear",
                    "base_estimator": "logistic",
                },
                "datasets": {
                    "tiny": {
                        "full": {
                            "raw_folds": [
                                _full_legacy_row(0.01),
                                _full_legacy_row(0.03),
                            ]
                        },
                        "costs": {
                            "0.20": {"raw_folds": selective_rows},
                            "0.30": {"raw_folds": selective_rows},
                        },
                    }
                },
            }),
            encoding="utf-8",
        )
        table_dir = cc_root / "tables" / "fixture"
        table_dir.mkdir(parents=True)
        table_dir.joinpath("results_v3.json").write_text(
            json.dumps({
                "Schema Version": 3,
                "Config Hash": "fixture",
                "Settings": {
                    "models": ["CC_Logistic"],
                    "n_splits": 2,
                    "random_state": 42,
                },
                "Results": {
                    "tiny": {
                        "CC_Logistic": {
                            "Status": "complete",
                            "Expected Fold Count": 2,
                            "Completed Folds": [1, 2],
                            "Full": {
                                "Raw Folds": [
                                    {
                                        "Fold": 1,
                                        "Hamming Accuracy": 0.81,
                                        "Subset Accuracy": 0.26,
                                        "Macro-F1": 0.61,
                                        "Micro-F1": 0.66,
                                        "Instance-F1": 0.56,
                                    },
                                    {
                                        "Fold": 2,
                                        "Hamming Accuracy": 0.83,
                                        "Subset Accuracy": 0.28,
                                        "Macro-F1": 0.63,
                                        "Micro-F1": 0.68,
                                        "Instance-F1": 0.58,
                                    },
                                ]
                            },
                        }
                    }
                },
            }),
            encoding="utf-8",
        )
        return {
            "legacy_results_path": str(legacy_path),
            "mlc_pa_cache_path": str(mlc_path),
            "cc_logistic_output_dir": str(cc_root),
            "mlc_pa_logistic_output_dir": str(mlc_v3_root),
        }

    def test_strict_adapters_preserve_missing_metric_and_exact_complements(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self._fixtures(temporary)
            expected = {
                "n_splits": 2,
                "random_state": 42,
                "abstention_costs": [0.2, 0.3],
                "report_cost": 0.3,
                "abstention_penalty": "linear",
            }
            bundle = load_bss_comparison_bundle(config, ["tiny"], expected)
            self.assertEqual(
                bundle["Missing"],
                {
                    "BR_Logistic": [],
                    "CC_Logistic": [],
                    "MLC_PA_Logistic": [],
                },
            )
            br = bundle["Models"]["BR_Logistic"]["Datasets"]["tiny"]
            self.assertAlmostEqual(br["Full"]["Mean"]["Hamming Accuracy"], 0.81)
            self.assertAlmostEqual(br["Full"]["Mean"]["Instance-F1"], 0.56)
            mlc = bundle["Models"]["MLC_PA_Logistic"]["Datasets"]["tiny"]
            selective = mlc["Costs"]["0.30"]["Selective"]
            self.assertAlmostEqual(
                selective["Mean"]["Selective Hamming Accuracy"], 0.91
            )
            self.assertNotIn("Selective Instance-F1", selective["Mean"])
            self.assertNotIn("Selective Instance-F1", selective["Raw Folds"][0])

            status = comparison_status(config, ["tiny"])
            self.assertEqual(status["CC_Logistic"]["missing_datasets"], [])

    def test_mlc_settings_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self._fixtures(temporary)
            expected = {
                "n_splits": 2,
                "random_state": 42,
                "abstention_costs": [0.2, 0.3],
                "report_cost": 0.3,
                "abstention_penalty": "nonlinear",
            }
            with self.assertRaisesRegex(ValueError, "not scientifically compatible"):
                load_bss_comparison_bundle(config, ["tiny"], expected)

    def test_comparison_figures_render_without_fabricating_missing_instance_f1(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self._fixtures(temporary)
            expected = {
                "n_splits": 2,
                "random_state": 42,
                "abstention_costs": [0.2, 0.3],
                "report_cost": 0.3,
                "abstention_penalty": "linear",
            }
            comparison = load_bss_comparison_bundle(config, ["tiny"], expected)
            full_rows = [
                {
                    "Fold": 1,
                    "Hamming Accuracy": 0.84,
                    "Subset Accuracy": 0.30,
                    "Macro-F1": 0.64,
                    "Micro-F1": 0.69,
                    "Instance-F1": 0.59,
                },
                {
                    "Fold": 2,
                    "Hamming Accuracy": 0.86,
                    "Subset Accuracy": 0.32,
                    "Macro-F1": 0.66,
                    "Micro-F1": 0.71,
                    "Instance-F1": 0.61,
                },
            ]
            selective_rows = [
                {
                    "Fold": 1,
                    "Coverage": 0.75,
                    "AABS": 0.25,
                    "ABS": 0.70,
                    "Generalized Loss": 0.14,
                    "Selective Macro-F1": 0.70,
                    "Selective Micro-F1": 0.74,
                    "Selective Instance-F1": 0.72,
                    "Selective Hamming Accuracy": 0.92,
                },
                {
                    "Fold": 2,
                    "Coverage": 0.77,
                    "AABS": 0.23,
                    "ABS": 0.68,
                    "Generalized Loss": 0.13,
                    "Selective Macro-F1": 0.72,
                    "Selective Micro-F1": 0.76,
                    "Selective Instance-F1": 0.74,
                    "Selective Hamming Accuracy": 0.93,
                },
            ]

            def summary(rows):
                keys = [key for key in rows[0] if key != "Fold"]
                return {
                    "Mean": {
                        key: sum(row[key] for row in rows) / len(rows)
                        for key in keys
                    },
                    "Std": {key: 0.01 for key in keys},
                    "Raw Folds": rows,
                }

            document = {
                "Settings": {
                    "models": ["BSS_UG_SPCC_PA_Logistic"],
                    "report_cost": 0.3,
                    "abstention_costs": [0.2, 0.3],
                },
                "Comparison": comparison,
                "Results": {
                    "tiny": {
                        "BSS_UG_SPCC_PA_Logistic": {
                            "Status": "complete",
                            "Expected Fold Count": 2,
                            "Completed Folds": [1, 2],
                            "Full": summary(full_rows),
                            "Costs": {
                                "0.20": {"Selective": summary(selective_rows)},
                                "0.30": {"Selective": summary(selective_rows)},
                            },
                        }
                    }
                },
            }
            output = Path(temporary) / "figures"
            generated = generate_bss_spcc_plots(document, output)
            expected_names = set(BSS_DATASET_FIGURES.values()) | set(
                BSS_REJECTION_FIGURES.values()
            )
            self.assertEqual(set(generated), expected_names)
            self.assertEqual({path.name for path in output.glob("*.png")}, expected_names)


if __name__ == "__main__":
    unittest.main()
