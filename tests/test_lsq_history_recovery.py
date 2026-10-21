"""Progress-file locks must not alter or stop least-squares fitting."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import tdm_least_squares as lsq
from tdm_minimal_io import MODEL_EDM_LANGMUIR
from tdm_minimal_model import simulate
from verify_model import make_config


OUTPUTS = ("FIT_RESULTS_CSV", "FIT_TRACE_CSV", "FIT_COMPOSITION_CSV",
           "FIT_CONDUCTIVITY_CSV", "FIT_REPORT_HTML", "FIT_CONFIG_JSON",
           "FIT_APPLY_JSL", "FIT_SETTINGS_JSON", "FIT_HISTORY_CSV",
           "FIT_IMPROVED_CSV", "FIT_IMPROVED_TXT", "CONFIG_PATH")


class HistoryRecoveryTests(unittest.TestCase):
    def test_fitted_mechanistic_shape_and_uv_response_reproduce_exported_trace(self):
        truth = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        truth["numerics"].update(time_steps=50, axial_positions=6)
        truth_trace = simulate(truth)["trace"]
        initial = copy.deepcopy(truth)
        initial["components"][0]["qmax_g_L"] *= 3
        initial["conversion"]["uv_to_protein_mAU_L_g"] *= 0.5
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "measured.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["CV", "UV 280 [mAU]"])
                writer.writerows(zip(truth_trace["CV"], truth_trace["uv_mAU"]))
            result_dir = root / "result"
            result_dir.mkdir()
            patched_outputs = {name: result_dir / getattr(lsq, name).name for name in OUTPUTS}
            with patch.multiple(lsq, **patched_outputs):
                result = lsq.run_global_least_squares_refinement(
                    [{"config": initial, "chromatogram_csv": source}],
                    unlocked_paths=["components[0].qmax_g_L",
                                    "conversion.uv_to_protein_mAU_L_g"],
                    max_nfev=15, max_model_evaluations=40)
            fitted = result["fitted_config"]
            reproduced = simulate(fitted)["trace"]
            before = simulate(initial)["trace"]
            self.assertTrue(result["accepted"])
            self.assertLess(result["final_objective"], result["initial_objective"] * 1e-5)
            self.assertAlmostEqual(fitted["components"][0]["qmax_g_L"],
                                   truth["components"][0]["qmax_g_L"], delta=1)
            self.assertAlmostEqual(fitted["conversion"]["uv_to_protein_mAU_L_g"],
                                   truth["conversion"]["uv_to_protein_mAU_L_g"], delta=1)
            self.assertGreater(np.max(np.abs(before["uv_mAU"] - reproduced["uv_mAU"])), 0.001)
            with patched_outputs["FIT_TRACE_CSV"].open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            np.testing.assert_allclose(
                [float(row["Predicted_Detector_Signal"]) for row in rows],
                np.interp([float(row["CV"]) for row in rows],
                          reproduced["CV"], reproduced["uv_mAU"]), rtol=1e-8, atol=1e-8)

    def test_locked_progress_csv_keeps_mass_percent_fit_and_complete_history(self):
        truth = make_config(MODEL_EDM_LANGMUIR, 1, multistep=False)
        truth["numerics"].update(time_steps=50, axial_positions=6)
        trace = simulate(truth)["trace"]
        initial = copy.deepcopy(truth)
        initial["components"][0]["qmax_g_L"] *= 1.2
        names = [component["name"] for component in truth["components"][:2]]
        fractions = [{"fraction_index": 1, "start": 0.20, "end": 0.24,
                      "x_unit": "CV", "mass_percent": {names[0]: 70, names[1]: 30}}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "measured.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["CV", "UV 280 [mAU]"])
                writer.writerows(zip(trace["CV"], trace["uv_mAU"]))
            profiles = [{"config": initial, "chromatogram_csv": source,
                         "composition_references": fractions}]
            outcomes = []
            original_open = Path.open
            for scenario in ("normal", "append_locked", "creation_locked", "folder_locked"):
                result_dir = root / scenario
                result_dir.mkdir()
                patched_outputs = {name: result_dir / getattr(lsq, name).name for name in OUTPUTS}
                preferred = patched_outputs["FIT_HISTORY_CSV"]

                def controlled_open(path, mode="r", *args, **kwargs):
                    if (path == preferred and (
                            scenario == "creation_locked" or
                            scenario == "append_locked" and mode == "a") or
                            scenario == "folder_locked" and path.parent == result_dir
                            and path.name.startswith("least_squares_objective_history")):
                        raise PermissionError(13, "simulated Excel/OneDrive lock", str(path))
                    return original_open(path, mode, *args, **kwargs)

                with patch.multiple(lsq, **patched_outputs), patch.object(Path, "open", controlled_open):
                    result = lsq.run_global_least_squares_refinement(
                        profiles, mode=lsq.MODE_CHROMATOGRAM_AND_COMPOSITION,
                        unlocked_paths=["components[0].qmax_g_L"], max_nfev=2)
                history_path = Path(result["latest_fit_history_csv"])
                self.assertTrue(history_path.is_file())
                with history_path.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), result["residual_evaluations"] + 1)
                self.assertFalse(any("PermissionError" in row["status"] for row in rows))
                settings = json.loads(patched_outputs["FIT_SETTINGS_JSON"].read_text())
                self.assertEqual(settings["objective_history_csv"], str(history_path))
                report = patched_outputs["FIT_REPORT_HTML"].read_text()
                self.assertIn(history_path.name, report)
                self.assertIn("Fraction mass% fit by protein species", report)
                if scenario == "normal":
                    self.assertEqual(history_path, preferred)
                else:
                    self.assertNotEqual(history_path, preferred)
                    self.assertIn("_recovered_", history_path.name)
                    self.assertIn("PermissionError", settings["objective_history_write_error"])
                if scenario == "folder_locked":
                    self.assertEqual(history_path.parent, Path(tempfile.gettempdir()))
                    history_path.unlink()
                outcomes.append(result)
            for result in outcomes[1:]:
                np.testing.assert_allclose(result["final_objective"],
                                           outcomes[0]["final_objective"], rtol=1e-12)
                np.testing.assert_allclose(
                    result["fitted_config"]["components"][0]["qmax_g_L"],
                    outcomes[0]["fitted_config"]["components"][0]["qmax_g_L"], rtol=1e-12)

    def test_unwritable_history_paths_still_produce_fit_report(self):
        truth = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        truth["numerics"].update(time_steps=50, axial_positions=6)
        trace = simulate(truth)["trace"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "measured.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["CV", "UV 280 [mAU]"])
                writer.writerows(zip(trace["CV"], trace["uv_mAU"]))
            output_dir = root / "result"
            output_dir.mkdir()
            patched_outputs = {name: output_dir / getattr(lsq, name).name for name in OUTPUTS}
            original_open = Path.open

            def deny_history(path, mode="r", *args, **kwargs):
                if path.name.startswith("least_squares_objective_history") and mode in {"w", "a"}:
                    raise PermissionError(13, "simulated locked history path", str(path))
                return original_open(path, mode, *args, **kwargs)

            with patch.multiple(lsq, **patched_outputs), patch.object(Path, "open", deny_history):
                result = lsq.run_global_least_squares_refinement(
                    [{"config": truth, "chromatogram_csv": source}],
                    unlocked_paths=["components[0].qmax_g_L"], max_nfev=2)
            self.assertIsNone(result["latest_fit_history_csv"])
            settings = json.loads(patched_outputs["FIT_SETTINGS_JSON"].read_text())
            self.assertIsNone(settings["objective_history_csv"])
            self.assertIn("PermissionError", settings["objective_history_write_error"])
            report = patched_outputs["FIT_REPORT_HTML"].read_text()
            self.assertIn("History CSV could not be written", report)
            self.assertIn("Best least-squares objective", report)


if __name__ == "__main__":
    unittest.main()
