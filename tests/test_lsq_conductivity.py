"""Conductivity fitting uses the selected run's measured detector channel."""
from __future__ import annotations

import copy
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

import tdm_least_squares as ls
import tdm_bridge
from tdm_jmp import build_jsl, _validate_jsl
from tdm_minimal_model import simulate
from tests.test_lsq_r23 import config_for


class ConductivityFitTests(unittest.TestCase):
    def test_jmp_saves_and_restores_each_run_conductivity_choice(self):
        jsl = build_jsl()
        _validate_jsl(jsl)
        for slot in (1, 4, 10):
            self.assertIn(f"lsqConductivityFit{slot} << Set(condFitIndex)", jsl)
            self.assertIn(f"lsqConductivityColumn{slot} << Set Text(condColumn)", jsl)
        self.assertIn('Column(dtLSQRuns, "LSQ_Conductivity_Fit")[r] = LSQConductivityFitSelected(slot)', jsl)
        self.assertIn('Column(dtLSQRuns, "LSQ_Conductivity_Column")[r] = Trim(Char(LSQConductivityColumnText(slot)))', jsl)

    def test_saved_run_choices_stay_independent(self):
        on = tdm_bridge._lsq_reference_column_options({
            "LSQ_X_Unit": "ML", "LSQ_Conductivity_Fit": "On",
            "LSQ_Conductivity_Column": "Cond run 2"})
        off = tdm_bridge._lsq_reference_column_options({
            "LSQ_X_Unit": "CV", "LSQ_Conductivity_Fit": "Off",
            "LSQ_Conductivity_Column": "Cond run 4"})
        self.assertEqual((on["fit_conductivity"], on["conductivity_column"], on["x_unit"]),
                         (True, "Cond run 2", "ML"))
        self.assertEqual((off["fit_conductivity"], off["conductivity_column"], off["x_unit"]),
                         (False, "Cond run 4", "CV"))
        cfg = config_for()
        cfg["impurity_count"] = 1
        cfg["components"][1]["name"] = "Host-cell protein"
        self.assertEqual(ls._report_parameter_label(cfg, "components[1].qmax_g_L", "Saturation capacity"),
                         "Impurity 1 (Host-cell protein) — Saturation capacity")
        cfg["impurity_count"] = 2
        cfg["components"][1]["mass_percent"] = 0
        cfg["components"][2]["mass_percent"] = 5
        default_paths = ls._default_unlocked_parameter_paths(cfg)
        self.assertIn("components[2].qmax_g_L", default_paths)
        self.assertNotIn("components[1].qmax_g_L", default_paths)

    def test_opt_in_conductivity_recovers_detector_delay_and_labels_species(self):
        truth = config_for()
        truth["numerics"] = {"time_steps": 50, "axial_positions": 6}
        truth["least_squares"]["objective"] = "RAW_SSE"
        trace = simulate(truth)["trace"]
        initial = copy.deepcopy(truth)
        initial["system"]["conductivity_dead_volume_mL"] = 1.5

        with TemporaryDirectory() as directory:
            tmp = Path(directory)
            source = tmp / "run_4.csv"
            with source.open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["CV", "UV 280 [mAU]", "Conductivity [mS/cm]"])
                writer.writerows(zip(trace["CV"], trace["uv_mAU"], trace["conductivity_mS_cm"]))

            uv_only = ls.load_chromatogram_reference(source, initial, require_uv=True)
            self.assertIsNone(uv_only.conductivity_signal)
            measured = ls.load_chromatogram_reference(
                source, initial, require_uv=True, fit_conductivity=True)
            self.assertEqual(measured.conductivity_column, "Conductivity [mS/cm]")
            self.assertEqual(len(measured.conductivity_cv), len(trace["CV"]))
            uv_only_r, _ = ls._objective_parts(initial, uv_only, ls.MODE_CHROMATOGRAM, [])
            combined_r, combined_info = ls._objective_parts(
                initial, measured, ls.MODE_CHROMATOGRAM, [], True)
            self.assertEqual(len(combined_r), len(uv_only_r) + len(measured.conductivity_cv))
            self.assertGreater(combined_info["conductivity"]["rmse_mS_cm"], 0.1)
            self.assertNotIn("system.conductivity_dead_volume_mL", {
                p.path for p in ls.fit_parameter_specs(initial,
                    unlocked_paths=["system.conductivity_dead_volume_mL"])})

            outputs = ["FIT_RESULTS_CSV", "FIT_TRACE_CSV", "FIT_COMPOSITION_CSV",
                       "FIT_REPORT_HTML", "FIT_CONFIG_JSON", "FIT_APPLY_JSL",
                       "FIT_SETTINGS_JSON", "FIT_HISTORY_CSV", "FIT_IMPROVED_CSV",
                       "FIT_IMPROVED_TXT", "CONFIG_PATH"]
            with patch.multiple(ls, **{name: tmp / getattr(ls, name).name for name in outputs}):
                result = ls.run_global_least_squares_refinement(
                    [{"config": initial, "chromatogram_csv": source,
                      "fit_conductivity": True,
                      "run_context": {"run_number": 4, "run_name": "LSQ Run 4"}}],
                    unlocked_paths=["system.conductivity_dead_volume_mL"], max_nfev=8)
            self.assertTrue(result["accepted"])
            self.assertLess(result["final_objective"], result["initial_objective"] * 0.1)
            self.assertAlmostEqual(result["fitted_config"]["system"]["conductivity_dead_volume_mL"],
                                   truth["system"]["conductivity_dead_volume_mL"], delta=0.12)
            self.assertEqual(result["parameters"], ["system.conductivity_dead_volume_mL"])
            with (tmp / "least_squares_fit_results.csv").open(newline="") as fh:
                rows = list(csv.DictReader(fh))
            target = next(row for row in rows if row["Parameter_Path"] == "components[0].qmax_g_L")
            self.assertIn("Target protein", target["Parameter"])
            self.assertEqual(target["Fit_State"], "Locked")
            with (tmp / "least_squares_fit_conductivity.csv").open(newline="") as fh:
                cond_rows = list(csv.DictReader(fh))
            self.assertEqual(len(cond_rows), len(trace["CV"]))
            self.assertEqual({r["Run_Number"] for r in cond_rows}, {"4"})
            self.assertIn("Measured vs predicted conductivity", (tmp / "least_squares_fit_report.html").read_text())

    def test_selected_conductivity_column_and_units_are_checked(self):
        cfg = config_for()
        with TemporaryDirectory() as directory:
            source = Path(directory) / "reference.csv"
            source.write_text("CV,UV [mAU],Conductivity [uS/cm],Other\n0,1,1000,2\n1,2,2000,3\n2,3,3000,4\n", encoding="utf-8")
            ref = ls.load_chromatogram_reference(source, cfg, require_uv=True,
                fit_conductivity=True, conductivity_column="Conductivity [uS/cm]")
            np.testing.assert_allclose(ref.conductivity_signal, [1, 2, 3])
            with self.assertRaisesRegex(ValueError, "does not exist"):
                ls.load_chromatogram_reference(source, cfg, require_uv=True,
                    fit_conductivity=True, conductivity_column="Missing")
            with self.assertRaisesRegex(ValueError, "must differ"):
                ls.load_chromatogram_reference(source, cfg, require_uv=True,
                    fit_conductivity=True, conductivity_column="UV [mAU]")

    def test_sparse_conductivity_indices_keep_full_uv_axis(self):
        cfg = config_for()
        with TemporaryDirectory() as directory:
            source = Path(directory) / "indexed.csv"
            source.write_text(
                "Index,UV [mAU],Conductivity [mS/cm]\n"
                "0,1,\n1,2,10\n2,3,20\n3,4,30\n4,5,\n", encoding="utf-8")
            ref = ls.load_chromatogram_reference(source, cfg, require_uv=True,
                x_column="Index", x_unit="INDEX", fit_conductivity=True)
            np.testing.assert_allclose(ref.conductivity_cv,
                ref.cv[-1] * np.array([0.25, 0.5, 0.75]))


if __name__ == "__main__":
    unittest.main()
