"""Optional per-parameter percentage limits reach the bounded LSQ optimizer."""
import copy
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import tdm_bridge
import tdm_least_squares as lsq
from tdm_jmp import _validate_jsl, build_jsl
from tdm_minimal_io import (
    MODEL_EDM_LANGMUIR, apply_batch_shared_parameters,
    batch_row_to_run_config, default_batch_row, load_config, save_config,
)
from tdm_minimal_model import simulate
from verify_model import make_config


class FitRangeTests(unittest.TestCase):
    def test_jmp_controls_and_bridge_parse_limited_ranges(self):
        config = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        path = "components[0].D_app_mm2_s"
        config["least_squares"]["fit_range_percent"] = {path: 20.0}
        jsl = build_jsl(config)
        _validate_jsl(jsl)
        self.assertIn('c1DappFitRangeMode = Combo Box({"Default", "Limited"}, << Set(2)', jsl)
        self.assertIn('c1DappFitRangePercent = Number Edit Box(20', jsl)
        self.assertIn('tdm_ui_lsq_parameter_fit_range_chunk_count', jsl)
        self.assertIn('tdm_ui_lsq_parameter_fit_ranges_1', jsl)
        self.assertLess(max(len(line) for line in jsl.splitlines()
                            if 'tdm_ui_lsq_parameter_fit_ranges_' in line), 8000)
        with patch.dict(tdm_bridge.__dict__, {
            "tdm_ui_lsq_parameter_fit_range_chunk_count": 2,
            "tdm_ui_lsq_parameter_fit_ranges_1": f"{path}|Limited|",
            "tdm_ui_lsq_parameter_fit_ranges_2": "20;system.uv_dead_volume_mL|Default|.;",
        }):
            self.assertEqual(tdm_bridge._lsq_fit_range_percent_from_jmp(), {path: 20.0})
            self.assertEqual(tdm_bridge.build_shared_config_from_jmp()["least_squares"]["fit_range_percent"],
                             {path: 20.0})
        with patch.dict(tdm_bridge.__dict__, {
            "tdm_ui_lsq_parameter_fit_range_chunk_count": 2,
            "tdm_ui_lsq_parameter_fit_ranges_1": f"{path}|Limited|20;",
        }):
            with self.assertRaisesRegex(ValueError, "Incomplete per-parameter"):
                tdm_bridge._lsq_fit_range_percent_from_jmp()
        with patch.dict(tdm_bridge.__dict__, {
            "tdm_ui_lsq_parameter_fit_ranges":
                f"{path}|Limited|20;system.uv_dead_volume_mL|Default|.;",
        }):
            self.assertEqual(tdm_bridge._lsq_fit_range_percent_from_jmp(), {path: 20.0})
            self.assertEqual(tdm_bridge.build_shared_config_from_jmp()["least_squares"]["fit_range_percent"],
                             {path: 20.0})
        with patch.dict(tdm_bridge.__dict__, {
            "tdm_ui_lsq_parameter_fit_ranges": f"{path}|Limited|0;",
        }):
            with self.assertRaisesRegex(ValueError, "greater than 0"):
                tdm_bridge._lsq_fit_range_percent_from_jmp()

    def test_percentage_limits_intersect_default_bounds(self):
        config = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        limits = {"components[0].D_app_mm2_s": 20.0,
                  "conversion.uv_to_protein_mAU_L_g": 10.0,
                  "system.uv_dead_volume_mL": 20.0}
        config["least_squares"]["fit_range_percent"] = limits
        specs = {spec.path: spec for spec in lsq.fit_parameter_specs(
            config, unlocked_paths=limits)}
        np.testing.assert_allclose(lsq._physical_fit_bounds(specs["components[0].D_app_mm2_s"]),
                                   [0.067 * 0.8, 0.067 * 1.2])
        np.testing.assert_allclose(lsq._physical_fit_bounds(specs["conversion.uv_to_protein_mAU_L_g"]),
                                   [180.0, 220.0])
        np.testing.assert_allclose(lsq._physical_fit_bounds(specs["system.uv_dead_volume_mL"]),
                                   [0.0, 2.0])
        self.assertEqual(lsq.fit_parameter_specs(config, unlocked_paths=[]), [])
        config["least_squares"]["fit_range_percent"]["components[0].D_app_mm2_s"] = -10
        with self.assertRaisesRegex(ValueError, "greater than 0"):
            lsq.fit_parameter_specs(config, unlocked_paths=["components[0].D_app_mm2_s"])

    def test_saved_range_survives_shared_model_and_independent_run_merges(self):
        config = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        path = "conversion.uv_to_protein_mAU_L_g"
        config["least_squares"]["fit_range_percent"] = {path: 12.5}
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / "model.json"
            save_config(config, saved)
            restored = load_config(saved)
        self.assertEqual(restored["least_squares"]["fit_range_percent"], {path: 12.5})
        independent = batch_row_to_run_config(default_batch_row(2))
        effective, _ = apply_batch_shared_parameters(restored, independent)
        self.assertEqual(effective["least_squares"]["fit_range_percent"], {path: 12.5})
        self.assertEqual(len(lsq.fit_parameter_specs(effective, unlocked_paths=[path])), 1)

    def test_two_run_fit_respects_saved_percent_limit_without_changing_simulation(self):
        truth = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        truth["numerics"].update(time_steps=50, axial_positions=6)
        truth["least_squares"]["objective"] = "NORMALIZED_MSE"
        path = "conversion.uv_to_protein_mAU_L_g"
        with patch.dict(tdm_bridge.__dict__, {
            "tdm_ui_lsq_parameter_fit_range_chunk_count": 1,
            "tdm_ui_lsq_parameter_fit_ranges_1": f"{path}|Limited|10;",
        }):
            bounds_from_ui = tdm_bridge.build_shared_config_from_jmp()["least_squares"]["fit_range_percent"]
        start_factor = truth["conversion"]["uv_to_protein_mAU_L_g"] / 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = []
            for number in (1, 2):
                target = copy.deepcopy(truth)
                if number == 2:
                    target["process"]["elution_steps"][0]["CV"] *= 1.5
                measured = simulate(target)["trace"]
                source = root / f"reference_{number}.csv"
                with source.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["CV", "UV 280 [mAU]"])
                    writer.writerows(zip(measured["CV"], measured["uv_mAU"]))
                starting = copy.deepcopy(target)
                starting["conversion"]["uv_to_protein_mAU_L_g"] = start_factor
                starting["least_squares"]["fit_range_percent"] = bounds_from_ui
                unrestricted = copy.deepcopy(starting)
                unrestricted["least_squares"]["fit_range_percent"] = {}
                np.testing.assert_allclose(simulate(starting)["trace"]["uv_mAU"],
                                           simulate(unrestricted)["trace"]["uv_mAU"])
                profiles.append({"config": starting, "chromatogram_csv": source,
                                 "run_context": {"run_number": number}})
            outputs = ("FIT_RESULTS_CSV", "FIT_TRACE_CSV", "FIT_COMPOSITION_CSV",
                       "FIT_CONDUCTIVITY_CSV", "FIT_REPORT_HTML", "FIT_CONFIG_JSON",
                       "FIT_APPLY_JSL", "FIT_SETTINGS_JSON", "FIT_HISTORY_CSV",
                       "FIT_IMPROVED_CSV", "FIT_IMPROVED_TXT", "CONFIG_PATH")
            with patch.multiple(lsq, **{name: root / getattr(lsq, name).name
                                        for name in outputs}):
                fit = lsq.run_global_least_squares_refinement(
                    profiles, unlocked_paths=[path], max_nfev=8,
                    max_model_evaluations=25)
                with lsq.FIT_RESULTS_CSV.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                report = lsq.FIT_REPORT_HTML.read_text(encoding="utf-8")
            fitted = fit["fitted_config"]["conversion"]["uv_to_protein_mAU_L_g"]
            self.assertTrue(fit["accepted"])
            self.assertLess(fit["final_objective"], fit["initial_objective"])
            self.assertGreater(fitted, start_factor)
            self.assertLessEqual(fitted, start_factor * 1.1 + 1e-8)
            self.assertAlmostEqual(fitted, start_factor * 1.1, delta=1e-5)
            row = next(row for row in rows if row["Parameter_Path"] == path)
            self.assertEqual(float(row["Fit_Range_Percent"]), 10.0)
            self.assertAlmostEqual(float(row["Allowed_Max"]), start_factor * 1.1)
            self.assertEqual(row["At_Upper_Bound"], "True")
            self.assertNotIn("Residual_DOF", row)
            self.assertIn("Fit limit ±%", report)
            self.assertNotIn("Residual DOF", report)
            different = copy.deepcopy(profiles)
            different[1]["config"]["least_squares"]["fit_range_percent"] = {path: 5.0}
            with self.assertRaisesRegex(ValueError, "different per-parameter fit ranges"):
                lsq.run_global_least_squares_refinement(
                    different, unlocked_paths=[path], max_nfev=2,
                    max_model_evaluations=5)


if __name__ == "__main__":
    unittest.main()
