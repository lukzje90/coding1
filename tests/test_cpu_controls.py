"""Separate CPU budgets preserve the unchanged chromatography solutions."""
from __future__ import annotations

import copy
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import tdm_bridge
import tdm_least_squares as lsq
import tdm_minimal_model as model
from tdm_jmp import build_jsl, _validate_jsl
from tdm_minimal_io import (
    MODEL_EDM_CPA, MODEL_EDM_LANGMUIR, MODEL_TDM_CPA,
    MODEL_TDM_LANGMUIR, MODEL_TDM_WANG, MAX_CPU_THREADS,
    apply_batch_shared_parameters, sanitize_config_for_selected_model,
    validate_config,
)
from verify_model import make_config


class CpuControlTests(unittest.TestCase):
    def test_independent_jmp_controls_validate_twelve_and_reject_bad_values(self):
        jsl = build_jsl()
        _validate_jsl(jsl)
        self.assertLess(jsl.index('axialPositionsBox = Number Edit Box'),
                        jsl.index('predictionCPUThreads = Number Edit Box'))
        self.assertIn('Python Name("tdm_ui_prediction_cpu_threads")', jsl)
        self.assertIn('Python Name("tdm_ui_lsq_parallel_workers")', jsl)
        self.assertNotIn('Column(dtRuns, "CPU_Threads")', jsl)
        with patch.dict(tdm_bridge.__dict__, {
                "tdm_ui_prediction_cpu_threads": 12,
                "tdm_ui_lsq_parallel_workers": 4}):
            cfg = tdm_bridge.build_shared_config_from_jmp()
        self.assertEqual(cfg["numerics"]["cpu_threads"], 12)
        self.assertEqual(cfg["least_squares"]["parallel_workers"], 4)
        for bad in (0, 1.5, MAX_CPU_THREADS + 1):
            for key in ("tdm_ui_prediction_cpu_threads", "tdm_ui_lsq_parallel_workers"):
                with patch.dict(tdm_bridge.__dict__, {key: bad}):
                    with self.assertRaisesRegex(ValueError, "CPU"):
                        tdm_bridge.build_shared_config_from_jmp()
        valid = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        valid["numerics"].update(time_steps=50, axial_positions=6, cpu_threads=12)
        valid["least_squares"]["parallel_workers"] = 12
        self.assertEqual(validate_config(valid, strict=True), [])
        with self.assertRaisesRegex(ValueError, "whole number"):
            model.run_batch(valid, [valid], parallel_workers=1.5)
        with self.assertRaisesRegex(ValueError, "whole number"):
            model.simulate(valid, native_threads=1.5)

    def test_shared_normal_limit_survives_run_recipe_and_is_separate_from_lsq(self):
        shared = sanitize_config_for_selected_model(
            make_config(MODEL_EDM_LANGMUIR, 0, multistep=False))
        shared["numerics"]["cpu_threads"] = 12
        shared["least_squares"]["parallel_workers"] = 3
        run = copy.deepcopy(shared)
        run["numerics"].update(time_steps=60, axial_positions=8, cpu_threads=1)
        effective, _ = apply_batch_shared_parameters(shared, run)
        self.assertEqual(effective["numerics"],
                         {"time_steps": 60, "axial_positions": 8, "cpu_threads": 12})
        self.assertEqual(effective["least_squares"]["parallel_workers"], 3)

    def test_native_limit_and_lsq_override_leave_solver_results_equal(self):
        cfg = sanitize_config_for_selected_model(
            make_config(MODEL_TDM_CPA, 1, multistep=False))
        cfg["numerics"].update(time_steps=50, axial_positions=6, cpu_threads=12)
        baseline = model.simulate(cfg, native_threads=1)
        requested = []
        real_limit = model.threadpool_limits

        def observed_limit(*, limits):
            requested.append(limits)
            return real_limit(limits=limits)

        with patch.object(model, "threadpool_limits", side_effect=observed_limit):
            selected = model.simulate(cfg)
        self.assertEqual(requested, [min(12, os.cpu_count() or 1)])
        np.testing.assert_allclose(baseline["trace"]["uv_mAU"], selected["trace"]["uv_mAU"],
                                   rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(baseline["trace"]["conductivity_mS_cm"],
                                   selected["trace"]["conductivity_mS_cm"], rtol=1e-10, atol=1e-10)

        # The LSQ objective uses one native thread per process regardless of
        # the normal predictor's saved limit.
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "measured.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["CV", "UV 280 [mAU]"])
                writer.writerows(zip(baseline["trace"]["CV"], baseline["trace"]["uv_mAU"]))
            ref = lsq.load_chromatogram_reference(source, cfg, require_uv=True)
            calls = []
            real_simulate = lsq.simulate

            def observed_simulate(config, **kwargs):
                calls.append(kwargs)
                return real_simulate(config, **kwargs)

            with patch.object(lsq, "simulate", side_effect=observed_simulate):
                residual, _ = lsq._objective_parts(cfg, ref, lsq.MODE_CHROMATOGRAM, [])
            self.assertEqual(calls, [{"native_threads": 1}])
            np.testing.assert_allclose(residual, 0.0, atol=1e-10)

    def test_normal_cpu_limit_preserves_all_five_model_combinations(self):
        for combination in (MODEL_EDM_CPA, MODEL_EDM_LANGMUIR, MODEL_TDM_CPA,
                            MODEL_TDM_LANGMUIR, MODEL_TDM_WANG):
            with self.subTest(model=combination):
                cfg = make_config(combination, 0, multistep=False)
                cfg["numerics"].update(time_steps=50, axial_positions=6, cpu_threads=12)
                if combination == MODEL_TDM_WANG:
                    cfg["chromatography_mode"] = "HIC"
                serial = model.simulate(cfg, native_threads=1)
                selected = model.simulate(cfg)
                for channel in ("CV", "uv_mAU", "conductivity_mS_cm", "component_g_L"):
                    np.testing.assert_allclose(serial["trace"][channel],
                                               selected["trace"][channel], rtol=1e-10, atol=1e-10)
                self.assertEqual(serial["mass_balance"], selected["mass_balance"])

    def test_parallel_batch_matches_serial_traces_and_saved_run_receipts(self):
        for combination in (MODEL_EDM_LANGMUIR, MODEL_TDM_CPA):
            with self.subTest(model=combination), tempfile.TemporaryDirectory() as directory:
                cfg = sanitize_config_for_selected_model(
                    make_config(combination, 1, multistep=False))
                cfg["numerics"].update(time_steps=50, axial_positions=6, cpu_threads=12)
                runs = [copy.deepcopy(cfg), copy.deepcopy(cfg)]
                runs[1]["process"]["load_CV"] = 0.3
                runs[1]["feed"]["load_density_mg_mL_resin"] = 0.6
                serial = model.run_batch(cfg, runs, parallel_workers=1)
                parallel = model.run_batch(cfg, runs, parallel_workers=2)
                self.assertEqual([r["result"]["batch"]["run_number"] for r in parallel], [1, 2])
                for left, right in zip(serial, parallel):
                    for channel in ("CV", "uv_mAU", "conductivity_mS_cm", "component_g_L"):
                        np.testing.assert_allclose(left["result"]["trace"][channel],
                                                   right["result"]["trace"][channel], rtol=0, atol=0)
                    self.assertEqual(left["result"]["mass_balance"], right["result"]["mass_balance"])
                    self.assertEqual(right["result"]["batch"]["cpu_workers_used"], 2)
                root = Path(directory) / "batch_results"
                saved = model.run_batch_and_write(cfg, runs, root, parallel_workers=2)
                self.assertEqual(saved["cpu_workers_used"], 2)
                self.assertEqual([r["run_number"] for r in saved["runs"]], [1, 2])
                for run_number in (1, 2):
                    receipt = json.loads((root / f"run_{run_number:03d}"
                                          / "latest_results_manifest.json").read_text())
                    self.assertEqual(receipt["run_number"], run_number)


if __name__ == "__main__":
    unittest.main()
