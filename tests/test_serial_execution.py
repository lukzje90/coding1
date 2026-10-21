"""Removed worker settings must not change solver results or saved run recipes."""
from __future__ import annotations

import copy
import unittest

import numpy as np

from tdm_jmp import build_jsl, _validate_jsl
from tdm_minimal_io import MODEL_EDM_LANGMUIR, normalize_config
from tdm_minimal_model import run_batch, simulate
from verify_model import make_config


class SerialExecutionTests(unittest.TestCase):
    def test_jmp_has_no_worker_controls(self):
        jsl = build_jsl()
        _validate_jsl(jsl)
        for forbidden in ("predictionCPUThreads", "lsqParallelWorkers",
                          "tdm_ui_prediction_cpu_threads", "tdm_ui_lsq_parallel_workers"):
            self.assertNotIn(forbidden, jsl)

    def test_old_cpu_settings_are_ignored_and_batch_matches_individual_runs(self):
        config = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        config["numerics"].update(time_steps=50, axial_positions=6, cpu_threads=12)
        config["least_squares"]["parallel_workers"] = 4
        clean = normalize_config(config)
        self.assertNotIn("cpu_threads", clean["numerics"])
        self.assertNotIn("parallel_workers", clean["least_squares"])
        runs = [copy.deepcopy(clean), copy.deepcopy(clean)]
        runs[1]["process"]["load_CV"] = 0.3
        runs[1]["feed"]["load_density_mg_mL_resin"] = (
            0.3 * runs[1]["feed"]["total_concentration_mg_mL"])
        results = run_batch(clean, runs)
        self.assertEqual([row["result"]["batch"]["run_number"] for row in results], [1, 2])
        for row in results:
            expected = simulate(row["config"])
            np.testing.assert_array_equal(row["result"]["trace"]["uv_mAU"],
                                          expected["trace"]["uv_mAU"])
            self.assertEqual(row["result"]["mass_balance"], expected["mass_balance"])


if __name__ == "__main__":
    unittest.main()
