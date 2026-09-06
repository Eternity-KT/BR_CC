"""Tests for the frozen Q12 ablation orchestration contract."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.run_frozen_ablation import (
    iter_ablation_jobs,
    run_jobs,
    verify_ablation_config,
)


class FrozenAblationRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload, cls.digest = verify_ablation_config(
            "configs/ablation_run.json"
        )

    def test_grid_is_complete_unique_and_keeps_outer_seed_fixed(self):
        jobs = list(iter_ablation_jobs(self.payload))
        self.assertEqual(len(jobs), 39)
        self.assertEqual(len({job_id for job_id, _ in jobs}), 39)
        self.assertTrue(all(settings["random_state"] == 42 for _, settings in jobs))
        random_jobs = [row for row in jobs if row[0].startswith("random_matched/")]
        self.assertEqual(len(random_jobs), 30)
        self.assertEqual(
            len({settings["gsi_partition_random_state"] for _, settings in random_jobs}),
            30,
        )
        self.assertTrue(
            all(settings["gsi_partition_mode"] == "random_matched" for _, settings in random_jobs)
        )

    def test_checksum_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "ablation.json"
            checksum = Path(directory) / "ablation.sha256"
            config.write_text(json.dumps(self.payload), encoding="utf-8")
            checksum.write_text(f"{self.digest}  ablation.json\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_ablation_config(config, checksum)

    def test_global_fold_quota_stops_before_next_job(self):
        payload = {
            "common_settings": {"result_schema": 3},
            "output_root": "unused",
            "objective_ablation": {"models": ["GSI"], "objectives": ["a", "b"]},
            "partition_ablation": {"models": [], "modes": []},
            "random_matched_ablation": {"models": [], "partition_seeds": []},
        }
        counts = iter([0, 2])
        with mock.patch(
            "scripts.run_frozen_ablation.checkpointed_fold_count",
            side_effect=lambda output: next(counts),
        ), mock.patch("scripts.run_frozen_ablation.run_experiment") as run:
            run_jobs(payload, max_new_folds=2)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.kwargs["max_new_folds"], 2)


if __name__ == "__main__":
    unittest.main()
