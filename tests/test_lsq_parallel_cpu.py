"""Parallel run evaluation must preserve the serial multi-channel fit."""
from __future__ import annotations

import copy
import csv
import json
from multiprocessing import TimeoutError as ProcessTimeoutError
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import tdm_bridge
import tdm_least_squares as lsq
from tdm_jmp import build_jsl, _validate_jsl
from tdm_minimal_io import MODEL_EDM_LANGMUIR, MODEL_TDM_CPA, validate_config
from tdm_minimal_model import simulate
from verify_model import make_config


OUTPUTS = ("FIT_RESULTS_CSV", "FIT_TRACE_CSV", "FIT_COMPOSITION_CSV",
           "FIT_CONDUCTIVITY_CSV", "FIT_REPORT_HTML", "FIT_CONFIG_JSON",
           "FIT_APPLY_JSL", "FIT_SETTINGS_JSON", "FIT_HISTORY_CSV",
           "FIT_IMPROVED_CSV", "FIT_IMPROVED_TXT", "CONFIG_PATH")


class ParallelCpuFitTests(unittest.TestCase):
    def test_ui_and_validation(self):
        jsl = build_jsl()
        _validate_jsl(jsl)
        self.assertIn('lsqParallelWorkers = Number Edit Box', jsl)
        self.assertIn('Python Name("tdm_ui_lsq_parallel_workers")', jsl)
        cfg = make_config(MODEL_EDM_LANGMUIR, 1, multistep=False)
        cfg["least_squares"]["parallel_workers"] = 1.5
        self.assertTrue(any("CPU workers" in error for error in validate_config(cfg, strict=True)))
        with patch.dict(tdm_bridge.__dict__, {"tdm_ui_lsq_parallel_workers": 1.5}):
            with self.assertRaisesRegex(ValueError, "CPU workers"):
                tdm_bridge.build_shared_config_from_jmp()

    def test_two_run_mass_conductivity_fit_matches_serial(self):
        truth = make_config(MODEL_EDM_LANGMUIR, 1, multistep=False)
        truth["numerics"].update(time_steps=50, axial_positions=6)
        truth["least_squares"]["objective"] = "RAW_SSE"
        self.assertFalse(validate_config(truth, strict=True))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = []
            for number, load_cv in ((2, 0.20), (4, 0.35)):
                actual = copy.deepcopy(truth)
                actual["process"]["load_CV"] = load_cv
                actual["feed"]["load_density_mg_mL_resin"] = (
                    load_cv * actual["feed"]["total_concentration_mg_mL"])
                trace = simulate(actual)["trace"]
                source = root / f"measured_{number}.csv"
                with source.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["CV", "UV 280 [mAU]", "Conductivity [mS/cm]"])
                    writer.writerows(zip(trace["CV"], trace["uv_mAU"], trace["conductivity_mS_cm"]))
                initial = copy.deepcopy(actual)
                initial["components"][0]["qmax_g_L"] *= 1.15
                initial["least_squares"]["parallel_workers"] = 2
                profiles.append({
                    "config": initial, "chromatogram_csv": source,
                    "fit_conductivity": True,
                    "composition_references": [{"fraction_index": 1, "start": 0.05,
                                                "end": 0.1, "x_unit": "CV",
                                                "mass_percent": {"Target protein": 50.0}}],
                    "run_context": {"run_number": number, "run_name": f"LSQ Run {number}"},
                })

            outputs = []
            for workers in (1, 2):
                result_dir = root / f"result_{workers}"
                result_dir.mkdir()
                with patch.multiple(lsq, **{name: result_dir / getattr(lsq, name).name
                                            for name in OUTPUTS}):
                    result = lsq.run_global_least_squares_refinement(
                        profiles, mode=lsq.MODE_CHROMATOGRAM_AND_COMPOSITION,
                        unlocked_paths=["components[0].qmax_g_L"], max_nfev=3,
                        parallel_workers=workers)
                settings = json.loads((result_dir / lsq.FIT_SETTINGS_JSON.name).read_text())
                with (result_dir / lsq.FIT_COMPOSITION_CSV.name).open(newline="") as stream:
                    mass_rows = list(csv.DictReader(stream))
                with (result_dir / lsq.FIT_CONDUCTIVITY_CSV.name).open(newline="") as stream:
                    cond_rows = list(csv.DictReader(stream))
                outputs.append((result, settings, mass_rows, cond_rows))

            serial, parallel = outputs
            for key in ("initial_objective", "final_objective", "improvement_percent"):
                np.testing.assert_allclose(serial[0][key], parallel[0][key], rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(
                serial[0]["fitted_config"]["components"][0]["qmax_g_L"],
                parallel[0]["fitted_config"]["components"][0]["qmax_g_L"], rtol=1e-11)
            self.assertEqual([m["run_number"] for m in parallel[0]["profile_metrics"]], [2, 4])
            self.assertEqual({r["Run_Number"] for r in parallel[2]}, {"2", "4"})
            self.assertEqual({r["Run_Number"] for r in parallel[3]}, {"2", "4"})
            self.assertEqual(serial[1]["parallel_workers_used"], 1)
            self.assertEqual(parallel[1]["parallel_workers_used"], 2)
            self.assertEqual(parallel[0]["parallel_workers_used"], 2)

    def test_one_run_parallel_parameter_trials_match_serial(self):
        truth = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        truth["numerics"].update(time_steps=50, axial_positions=6)
        trace = simulate(truth)["trace"]
        initial = copy.deepcopy(truth)
        initial["components"][0]["qmax_g_L"] *= 1.2
        initial["components"][0]["b_L_g"] *= 0.8
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "measured.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["CV", "UV 280 [mAU]"])
                writer.writerows(zip(trace["CV"], trace["uv_mAU"]))
            results = []
            for workers in (1, 2):
                result_dir = root / f"result_{workers}"
                result_dir.mkdir()
                with patch.multiple(lsq, **{name: result_dir / getattr(lsq, name).name
                                            for name in OUTPUTS}):
                    results.append(lsq.run_global_least_squares_refinement(
                        [{"config": initial, "chromatogram_csv": source}],
                        unlocked_paths=["components[0].qmax_g_L", "components[0].b_L_g"],
                        max_nfev=3, parallel_workers=workers))
                if workers == 2:
                    with (result_dir / lsq.FIT_HISTORY_CSV.name).open(newline="") as stream:
                        history = list(csv.DictReader(stream))
                    trials = [row for row in history if row["phase"] == "jacobian"]
                    self.assertTrue(trials)
                    self.assertTrue(any("parameter=components[0].qmax_g_L" in row["status"]
                                        for row in trials))
                    self.assertEqual(len(history), results[-1]["residual_evaluations"] + 1)
            for key in ("initial_objective", "final_objective", "improvement_percent"):
                np.testing.assert_allclose(results[0][key], results[1][key], rtol=1e-11, atol=1e-11)
            for parameter in ("qmax_g_L", "b_L_g"):
                np.testing.assert_allclose(
                    results[0]["fitted_config"]["components"][0][parameter],
                    results[1]["fitted_config"]["components"][0][parameter], rtol=1e-11)
            self.assertEqual(results[0]["parallel_workers_used"], 1)
            self.assertEqual(results[1]["parallel_workers_used"], 2)

    def test_tdm_cpa_parallel_trials_match_serial_with_mass_and_conductivity(self):
        truth = make_config(MODEL_TDM_CPA, 1, multistep=False)
        truth["numerics"].update(time_steps=50, axial_positions=6)
        trace = simulate(truth)["trace"]
        initial = copy.deepcopy(truth)
        initial["components"][0]["As_m_inv"] *= 1.05
        initial["components"][0]["Z_ref"] *= 1.03
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "tdm_cpa_measured.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["CV", "UV 280 [mAU]", "Conductivity [mS/cm]"])
                writer.writerows(zip(trace["CV"], trace["uv_mAU"], trace["conductivity_mS_cm"]))
            profile = {"config": initial, "chromatogram_csv": source,
                       "fit_conductivity": True,
                       "composition_references": [{"fraction_index": 1, "start": 0.05,
                                                   "end": 0.1, "x_unit": "CV",
                                                   "mass_percent": {"Target protein": 50.0}}]}
            results = []
            for workers in (1, 2):
                result_dir = root / f"result_{workers}"
                result_dir.mkdir()
                with patch.multiple(lsq, **{name: result_dir / getattr(lsq, name).name
                                            for name in OUTPUTS}):
                    results.append(lsq.run_global_least_squares_refinement(
                        [profile], mode=lsq.MODE_CHROMATOGRAM_AND_COMPOSITION,
                        unlocked_paths=["components[0].As_m_inv", "components[0].Z_ref"],
                        max_nfev=2, parallel_workers=workers))
            for key in ("initial_objective", "final_objective"):
                np.testing.assert_allclose(results[0][key], results[1][key], rtol=1e-11, atol=1e-11)
            for parameter in ("As_m_inv", "Z_ref"):
                np.testing.assert_allclose(
                    results[0]["fitted_config"]["components"][0][parameter],
                    results[1]["fitted_config"]["components"][0][parameter], rtol=1e-11)
            self.assertEqual(results[1]["parallel_workers_used"], 2)

    def test_timed_out_trial_pool_retries_same_vectors_with_fewer_workers(self):
        truth = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        truth["numerics"].update(time_steps=50, axial_positions=6)
        trace = simulate(truth)["trace"]
        initial = copy.deepcopy(truth)
        initial["components"][0]["qmax_g_L"] *= 1.2
        initial["components"][0]["b_L_g"] *= 0.8
        paths = ["components[0].qmax_g_L", "components[0].b_L_g",
                 "components[0].D_app_mm2_s"]

        class TimeoutOnceParallel:
            calls = 0

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def __call__(self, jobs):
                if self.kwargs.get("return_as") == "generator_unordered":
                    TimeoutOnceParallel.calls += 1

                    def results():
                        for index, (function, args, kwargs) in enumerate(jobs):
                            if TimeoutOnceParallel.calls == 1 and index == 1:
                                raise ProcessTimeoutError()
                            yield function(*args, **kwargs)
                    return results()
                return [function(*args, **kwargs) for function, args, kwargs in jobs]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "measured.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["CV", "UV 280 [mAU]"])
                writer.writerows(zip(trace["CV"], trace["uv_mAU"]))
            results = []
            for label, workers in (("serial", 1), ("retry", 3)):
                result_dir = root / label
                result_dir.mkdir()
                patches = {name: result_dir / getattr(lsq, name).name for name in OUTPUTS}
                with patch.multiple(lsq, **patches):
                    if workers == 3:
                        with patch("joblib.Parallel", TimeoutOnceParallel):
                            results.append(lsq.run_global_least_squares_refinement(
                                [{"config": initial, "chromatogram_csv": source}],
                                unlocked_paths=paths, max_nfev=2,
                                parallel_workers=workers))
                    else:
                        results.append(lsq.run_global_least_squares_refinement(
                            [{"config": initial, "chromatogram_csv": source}],
                            unlocked_paths=paths, max_nfev=2,
                            parallel_workers=workers))
            self.assertGreater(TimeoutOnceParallel.calls, 1)
            self.assertEqual(results[1]["parallel_workers_used"], 2)
            np.testing.assert_allclose(results[0]["final_objective"],
                                       results[1]["final_objective"], rtol=1e-11)
            for path in paths:
                field = path.rsplit(".", 1)[-1]
                np.testing.assert_allclose(
                    results[0]["fitted_config"]["components"][0][field],
                    results[1]["fitted_config"]["components"][0][field], rtol=1e-11)
            self.assertTrue(results[1]["parallel_worker_fallback_used"])
            settings = json.loads((root / "retry" / lsq.FIT_SETTINGS_JSON.name).read_text())
            self.assertTrue(settings["parallel_worker_fallback_used"])


if __name__ == "__main__":
    unittest.main()
