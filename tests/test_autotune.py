import json
import tempfile
import unittest
from pathlib import Path

from scripts.autotune_vit_workstation import (
    acceptance_criteria,
    best_trial,
    pipeline_decisions,
)


class AutoTuneTests(unittest.TestCase):
    def test_best_trial_ignores_failures_and_uses_throughput(self):
        trials = [
            {"status": "oom", "batch_size": 4096},
            {"status": "ok", "batch_size": 256, "samples_per_second": 100.0},
            {"status": "ok", "batch_size": 512, "samples_per_second": 140.0},
        ]
        self.assertEqual(best_trial(trials)["batch_size"], 512)

    def test_best_trial_rejects_all_failed_search(self):
        with self.assertRaises(RuntimeError):
            best_trial([{"status": "error"}, {"status": "oom"}])

    def test_acceptance_criteria_distinguishes_existing_swap_from_growth(self):
        criteria = acceptance_criteria({
            "resource_monitor": {
                "gpu_utilization_percent_mean": 90.0,
                "minimum_available_ram_bytes": 40 * 1024 ** 3,
            },
            "swap_used_before_bytes": 1024,
            "swap_used_after_bytes": 1024,
            "data_wait_fraction": 0.01,
        })
        self.assertTrue(criteria["gpu_utilization_pass"])
        self.assertTrue(criteria["ram_headroom_pass"])
        self.assertFalse(criteria["swap_zero_absolute"])
        self.assertTrue(criteria["swap_growth_zero"])

    def test_pipeline_decisions_keep_cpu_dense_when_wait_is_low(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "manifest.json"
            manifest.write_text(json.dumps({
                "config": {"rounds_per_shard": 4096},
            }))
            decisions = pipeline_decisions({"data_wait_fraction": 0.002}, manifest)

        self.assertEqual(
            decisions["feature_expansion"]["selected"], "cpu_dense_encode"
        )
        self.assertFalse(
            decisions["feature_expansion"]["gpu_feature_expander_adopted"]
        )
        self.assertEqual(decisions["storage"]["rounds_per_shard"], 4096)


if __name__ == "__main__":
    unittest.main()
