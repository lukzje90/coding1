"""Selected LSQ run numbers own their recipes, UV files, and fraction mass%."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import tdm_bridge
import tdm_least_squares as lsq
from tdm_jmp import build_jsl, _validate_jsl
from tdm_minimal_io import (
    MODEL_EDM_LANGMUIR, default_batch_row, ensure_lsq_mass_csv,
    load_lsq_mass_rows, LSQ_MASS_FIELDS, LSQ_MODEL_SOURCE_BATCH,
    LSQ_MODEL_SOURCE_INDEPENDENT,
)
from tdm_minimal_model import simulate
from verify_model import make_config


class TenRunMassTests(unittest.TestCase):
    def test_ten_slots_and_mass_rows_remain_keyed_to_run_number(self):
        jsl = build_jsl()
        _validate_jsl(jsl)
        self.assertEqual(jsl.count('Row = Outline Box("LSQ run '), 10)
        self.assertEqual(jsl.count('Text Box("Use this run\'s fraction mass%"'), 10)
        self.assertIn('lsqMassEnabled10 = Combo Box', jsl)
        self.assertIn('SaveLSQMassReference(lsqMassEditorRun)', jsl)
        self.assertIn('lsqLoadedRuns = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}', jsl)
        self.assertLess(jsl.index('isLoading = 1; isLSQSetup = 0'),
                        jsl.index('modelWindow = New Window'))
        self.assertIn('lsqRunCount = Combo Box({"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}', jsl)
        self.assertIn('If(!isLoading, UpdateUI(); UpdateLSQRunSummary())', jsl)
        self.assertIn('massRow = (r - 1) * 30 + 1', jsl)
        self.assertIn('Column(dtLSQMass, "Enabled")[massRow] = LSQMassEnabledSelected(slot)', jsl)
        for field in LSQ_MASS_FIELDS:
            kind = "Character" if field in {"Enabled", "Unit"} else "Numeric"
            self.assertIn(f'Column(dtLSQMass, "{field}") << Data Type({kind});', jsl)
        self.assertIn('lsqR30C6Mass << Set(Num(Column(dtLSQMass, "Impurity5_mass_percent")[massRow + 29]))', jsl)
        self.assertEqual(jsl.count('Column(dtLSQMass, "Start")[massRow +'), 60)
        ui = {"tdm_ui_lsq_run_count": 10}
        ui.update({f"tdm_ui_lsq_selected_run_{slot}": 11 - slot for slot in range(1, 11)})
        with patch.dict(tdm_bridge.__dict__, ui):
            self.assertEqual(tdm_bridge._selected_lsq_run_numbers(), list(range(10, 0, -1)))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mass.csv"
            rows = load_lsq_mass_rows(path)
            self.assertEqual(len(rows), 300)
            rows[30].update(Enabled="On", Fraction_Count="2", Unit="CV", Start="0.1",
                            End="0.2", Target_mass_percent="77")
            rows[90].update(Enabled="On", Fraction_Count="1", Unit="mL", Start="0.8",
                            End="1.0", Target_mass_percent="35")
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(reversed(rows))
            ensure_lsq_mass_csv(path)  # Restores positional JMP indexing without moving values.
            restored = load_lsq_mass_rows(path)
            self.assertEqual(restored[30]["Target_mass_percent"], "77")
            self.assertEqual(restored[90]["Target_mass_percent"], "35")
            self.assertEqual(restored[299]["Run_Number"], "10")

    def test_full_ten_profile_fit_keeps_each_mass_observation_on_its_run(self):
        shared = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        shared["numerics"].update(time_steps=50, axial_positions=6)
        trace = simulate(shared)["trace"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = []
            for number in range(1, 11):
                source = root / f"measured_run_{number}.csv"
                with source.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["CV", "UV 280 [mAU]"])
                    writer.writerows(zip(trace["CV"], trace["uv_mAU"]))
                profiles.append({
                    "config": shared,
                    "chromatogram_csv": source,
                    "run_context": {"run_number": number, "run_name": f"LSQ Run {number}"},
                    "composition_references": [{"fraction_index": 1, "start": 0.1, "end": 0.2,
                                                "x_unit": "CV", "mass_percent": {
                                                    "Target protein": float(number)}}],
                })
            outputs = ("FIT_RESULTS_CSV", "FIT_TRACE_CSV", "FIT_COMPOSITION_CSV",
                       "FIT_REPORT_HTML", "FIT_CONFIG_JSON", "FIT_APPLY_JSL",
                       "FIT_SETTINGS_JSON", "FIT_HISTORY_CSV", "FIT_IMPROVED_CSV",
                       "FIT_IMPROVED_TXT", "CONFIG_PATH")
            with patch.multiple(lsq, **{name: root / getattr(lsq, name).name for name in outputs}):
                result = lsq.run_global_least_squares_refinement(
                    profiles, mode="CHROMATOGRAM_AND_COMPOSITION", unlocked_paths=[])
                with lsq.FIT_COMPOSITION_CSV.open(newline="", encoding="utf-8") as stream:
                    mass_fit = list(csv.DictReader(stream))
                with lsq.FIT_TRACE_CSV.open(newline="", encoding="utf-8") as stream:
                    uv_fit = list(csv.DictReader(stream))
            self.assertEqual(len(result["profile_metrics"]), 10)
            self.assertEqual({(row["Run_Number"], row["Measured_mass_percent"])
                              for row in mass_fit}, {(str(i), str(float(i))) for i in range(1, 11)})
            self.assertEqual({row["Run_Number"] for row in uv_fit}, {str(i) for i in range(1, 11)})
            self.assertEqual({(row["run_number"], row["initial_rmse_percentage_points"],
                               row["final_rmse_percentage_points"])
                              for row in result["composition_profile_metrics"]},
                             {(i, 100.0 - i, 100.0 - i) for i in range(1, 11)})

    def test_reversed_run_4_and_2_keep_their_own_recipe_uv_and_mass_in_preview_and_fit(self):
        shared = make_config(MODEL_EDM_LANGMUIR, 1, multistep=False)
        shared["numerics"].update(time_steps=50, axial_positions=6)
        shared["column"]["volume_mL"] = 2.0
        batch = [default_batch_row(i) for i in range(1, 31)]
        batch[6].update(Run_Name="Batch recipe seven", Load_CV=0.22, PLW_Count=0,
                        Elution_Count=1, Elution1_CV=0.28, Time_Steps=50, Axial_Positions=6)
        lsq_rows = [default_batch_row(i) for i in range(1, 31)]
        lsq_rows[1].update(Run_Name="LSQ profile two", LSQ_Model_Source=LSQ_MODEL_SOURCE_BATCH,
                           LSQ_Model_Run_Number=7)
        lsq_rows[3].update(Run_Name="LSQ profile four", LSQ_Model_Source=LSQ_MODEL_SOURCE_INDEPENDENT,
                           Load_CV=0.38, PLW_Count=0, Elution_Count=1, Elution1_CV=0.36,
                           Time_Steps=50, Axial_Positions=6)
        cfg2, _ = tdm_bridge._lsq_model_run_config(shared, 2, lsq_rows[1], batch)
        cfg4, _ = tdm_bridge._lsq_model_run_config(shared, 4, lsq_rows[3], batch)
        self.assertAlmostEqual(cfg2["process"]["load_CV"], 0.22)
        self.assertAlmostEqual(cfg4["process"]["load_CV"], 0.38)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for number, config in ((2, cfg2), (4, cfg4)):
                trace = simulate(config)["trace"]
                source = root / f"uv_run_{number}.csv"
                with source.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["CV", "UV 280 [mAU]"])
                    writer.writerows(zip(trace["CV"], trace["uv_mAU"]))
                lsq_rows[number - 1]["LSQ_Chromatogram_CSV"] = str(source)
            mass_rows = load_lsq_mass_rows(root / "mass.csv")
            mass_rows[30].update(Enabled="On", Fraction_Count="2", Unit="CV",
                                 Start="0.10", End="0.14", Target_mass_percent="77")
            mass_rows[31].update(Start="0.14", End="0.18", Impurity1_mass_percent="23")
            mass_rows[90].update(Enabled="On", Fraction_Count="2", Unit="mL",
                                 Start="0.80", End="0.90", Target_mass_percent="35")
            mass_rows[91].update(Start="0.90", End="1.00", Impurity1_mass_percent="65")
            ui = {"tdm_ui_action": "PREVIEW_LSQ_ALIGNMENT", "tdm_ui_lsq_run_count": 2,
                  "tdm_ui_lsq_selected_run_1": 4, "tdm_ui_lsq_selected_run_2": 2,
                  "tdm_ui_lsq_target_run": 10,
                  "tdm_ui_lsq_mode": "Least-squares: chromatogram + species mass% by fraction"}
            captured = []
            actual_preview = lsq.preview_lsq_alignment
            report = root / "preview.html"

            def preview(profiles):
                captured.extend(profiles)
                return actual_preview(profiles, output_html=report)

            with patch.dict(tdm_bridge.__dict__, ui), \
                 patch.object(tdm_bridge, "build_shared_config_from_jmp", return_value=shared), \
                 patch.object(tdm_bridge, "load_lsq_rows", return_value=lsq_rows), \
                 patch.object(tdm_bridge, "load_batch_rows", return_value=batch), \
                 patch.object(tdm_bridge, "load_lsq_mass_rows", return_value=mass_rows), \
                 patch.object(tdm_bridge, "BATCH_COUNT_FILE", root / "count.txt"), \
                 patch.object(tdm_bridge, "save_config"), \
                 patch.object(tdm_bridge, "_write_status"), \
                 patch.object(lsq, "preview_lsq_alignment", side_effect=preview):
                self.assertEqual(tdm_bridge.main(), 0)
            self.assertEqual([p["run_context"]["run_number"] for p in captured], [4, 2])
            self.assertEqual([p["run_context"]["model_run_number"] for p in captured], [4, 7])
            self.assertEqual([p["composition_references"][0]["mass_percent"]["Target protein"]
                              for p in captured], ["35", "77"])
            self.assertEqual([p["composition_references"][0]["x_unit"] for p in captured], ["ML", "CV"])
            self.assertEqual([Path(p["chromatogram_csv"]).name for p in captured],
                             ["uv_run_4.csv", "uv_run_2.csv"])
            markup = report.read_text()
            self.assertIn("LSQ Run 4", markup)
            self.assertIn("LSQ Run 2", markup)
            self.assertEqual(markup.count("Fraction mass% at midpoints"), 2)

            outputs = ("FIT_RESULTS_CSV", "FIT_TRACE_CSV", "FIT_COMPOSITION_CSV",
                       "FIT_REPORT_HTML", "FIT_CONFIG_JSON", "FIT_APPLY_JSL",
                       "FIT_SETTINGS_JSON", "FIT_HISTORY_CSV", "FIT_IMPROVED_CSV",
                       "FIT_IMPROVED_TXT", "CONFIG_PATH")
            fit_inputs = []
            fit_results = []
            actual_fit = lsq.run_global_least_squares_refinement

            def fit(profiles, **kwargs):
                fit_inputs.extend(profiles)
                result = actual_fit(profiles, **kwargs)
                fit_results.append(result)
                return result

            ui["tdm_ui_action"] = "FIT_LSQ"
            ui["tdm_ui_lsq_parameter_lock_states"] = "conversion.uv_to_protein_mAU_L_g=Locked"
            with patch.dict(tdm_bridge.__dict__, ui), \
                 patch.object(tdm_bridge, "build_shared_config_from_jmp", return_value=shared), \
                 patch.object(tdm_bridge, "load_lsq_rows", return_value=lsq_rows), \
                 patch.object(tdm_bridge, "load_batch_rows", return_value=batch), \
                 patch.object(tdm_bridge, "load_lsq_mass_rows", return_value=mass_rows), \
                 patch.object(tdm_bridge, "BATCH_COUNT_FILE", root / "count.txt"), \
                 patch.object(tdm_bridge, "save_config"), \
                 patch.object(tdm_bridge, "_write_status") as status, \
                 patch.object(lsq, "run_global_least_squares_refinement", side_effect=fit), \
                 patch.multiple(lsq, **{name: root / getattr(lsq, name).name for name in outputs}):
                self.assertEqual(tdm_bridge.main(), 0)
                self.assertIn(f"Elapsed fit time: {fit_results[0]['fit_elapsed_hms']}",
                              status.call_args[0][0])
                with lsq.FIT_COMPOSITION_CSV.open(newline="", encoding="utf-8") as stream:
                    mass_fit = list(csv.DictReader(stream))
                with lsq.FIT_TRACE_CSV.open(newline="", encoding="utf-8") as stream:
                    uv_fit = list(csv.DictReader(stream))
                settings = json.loads(lsq.FIT_SETTINGS_JSON.read_text())
            result = fit_results[0]
            self.assertEqual([p["run_context"]["run_number"] for p in fit_inputs], [4, 2])
            self.assertEqual(settings["profile_count"], 2)
            self.assertEqual(len(result["profile_metrics"]), 2)
            self.assertEqual({(row["Run_Number"], row["Measured_mass_percent"])
                              for row in mass_fit}, {("2", "77.0"), ("2", "23.0"),
                                                      ("4", "35.0"), ("4", "65.0")})
            self.assertEqual({row["Run_Number"] for row in uv_fit}, {"2", "4"})
            self.assertEqual([item["run_context"]["run_number"] for item in settings["profiles"]], [4, 2])


if __name__ == "__main__":
    unittest.main()
