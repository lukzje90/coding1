"""The pre-fit overlay must use the same UV and CV alignment as refinement."""
import copy
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from openpyxl import Workbook

from tdm_jmp import _validate_jsl, build_jsl
import tdm_bridge
import tdm_minimal_io as io
import tdm_least_squares as lsq
from tdm_least_squares import preview_lsq_alignment
from tdm_minimal_io import (
    LSQ_MODEL_SOURCE_BATCH, LSQ_MODEL_SOURCE_INDEPENDENT,
    MODEL_EDM_LANGMUIR, apply_batch_shared_parameters, batch_row_to_run_config,
    default_batch_row, load_config,
)
from tdm_minimal_model import simulate
from verify_model import make_config


def _write_uv(path: Path, x, uv, x_header: str):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([x_header, "UV 280 [mAU]"])
        writer.writerows(zip(x, uv))


class AlignmentPreviewTests(unittest.TestCase):
    def test_dense_reference_plot_is_robust_but_fit_uses_every_raw_sample(self):
        raw_cv = np.linspace(0.0, 1.0, 5001)
        raw_uv = np.exp(-0.5 * ((raw_cv - 0.6) / 0.03) ** 2)
        repeated_cv = np.full(301, 0.65)
        repeated_uv = np.full(301, float(np.interp(0.65, raw_cv, raw_uv)))
        repeated_uv[:10] = 100.0  # export artifacts must not paint a solid black bar
        cv = np.r_[raw_cv, repeated_cv]
        uv = np.r_[raw_uv, repeated_uv]
        display_cv, display_uv = lsq._reference_display_series(cv, uv, 1.0)
        self.assertLessEqual(len(display_cv), 744)
        self.assertTrue(np.all(np.diff(display_cv) > 0))
        self.assertAlmostEqual(float(np.max(display_uv)), 1.0, delta=0.01)
        self.assertAlmostEqual(float(display_uv[np.argmin(abs(display_cv - 0.65))]),
                               float(np.interp(0.65, raw_cv, raw_uv)), delta=0.01)

        reference = lsq.ChromatogramReference(cv, uv, "CV", "UV 280 [mAU]", "CV", "test.csv")
        trace = {"CV": raw_cv, "total_g_L": np.zeros_like(raw_cv), "uv_mAU": raw_uv}
        residual, fit_info = lsq._chromatogram_residuals(
            make_config(MODEL_EDM_LANGMUIR, 0, multistep=False), reference, trace)
        self.assertEqual(len(residual), len(cv))
        self.assertEqual(fit_info["overlap_points"], len(cv))
        self.assertGreater(float(np.max(abs(residual))), 90.0)

    def test_new_lsq_table_tracks_normal_runs_but_legacy_table_keeps_independent_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            new_table = root / "new_lsq.csv"
            with patch.object(io, "LSQ_RUNS_CSV", new_table), \
                 patch.object(io, "load_batch_rows", return_value=[default_batch_row(i) for i in range(1, 31)]):
                io.ensure_lsq_runs_csv()
            with new_table.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["LSQ_Model_Source"], LSQ_MODEL_SOURCE_BATCH)
            self.assertEqual(rows[4]["LSQ_Model_Run_Number"], "5")
            self.assertEqual(rows[9]["LSQ_Model_Run_Number"], "10")

            legacy_table = root / "legacy_lsq.csv"
            legacy_fields = [name for name in io.batch_run_fields()
                             if name not in {"LSQ_Model_Source", "LSQ_Model_Run_Number"}]
            with legacy_table.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=legacy_fields)
                writer.writeheader()
                writer.writerow({key: default_batch_row(1)[key] for key in legacy_fields})
            io.ensure_batch_runs_csv(legacy_table)
            with legacy_table.open(newline="", encoding="utf-8") as stream:
                migrated = next(csv.DictReader(stream))
            self.assertEqual(migrated["LSQ_Model_Source"], LSQ_MODEL_SOURCE_INDEPENDENT)

    def test_saved_batch_run_and_latest_uv_are_used_by_preview_and_refinement(self):
        shared = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        shared["numerics"].update(time_steps=50, axial_positions=6)
        batch_rows = [default_batch_row(1), default_batch_row(2)]
        batch_rows[0].update(Run_Name="Old batch run", Load_CV=1.5, PLW_Count=0,
                             Elution_Count=1, Elution1_CV=0.2)
        batch_rows[1].update(Run_Name="Chosen batch run", Load_CV=0.4, PLW_Count=0,
                             Elution_Count=1, Elution1_CV=0.3,
                             Time_Steps=50, Axial_Positions=6)
        lsq_row = default_batch_row(1)
        lsq_row.update(Run_Name="Measured profile", Load_CV=2.0, PLW_Count=0,
                       Elution_Count=1, Elution1_CV=0.2,
                       LSQ_Model_Source=LSQ_MODEL_SOURCE_BATCH, LSQ_Model_Run_Number=2)

        def selected_trace():
            cfg, model_run = tdm_bridge._lsq_model_run_config(shared, 1, lsq_row, batch_rows)
            self.assertEqual(model_run["model_run_number"], 2)
            return simulate(cfg)["trace"]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, report = root / "measured.csv", root / "alignment.html"
            lsq_row["LSQ_Chromatogram_CSV"] = str(reference)
            trace = selected_trace()
            _write_uv(reference, trace["CV"], trace["uv_mAU"], "CV")
            previews = []
            actual_preview = lsq.preview_lsq_alignment

            def preview_with_receipt(profiles):
                result = actual_preview(profiles, output_html=report)
                previews.append(result)
                return result

            output_paths = {
                name: root / getattr(lsq, name).name for name in (
                    "FIT_RESULTS_CSV", "FIT_TRACE_CSV", "FIT_COMPOSITION_CSV",
                    "FIT_REPORT_HTML", "FIT_CONFIG_JSON", "FIT_APPLY_JSL",
                    "FIT_SETTINGS_JSON", "FIT_HISTORY_CSV", "FIT_IMPROVED_CSV",
                    "FIT_IMPROVED_TXT", "CONFIG_PATH")
            }
            ui = {"tdm_ui_action": "PREVIEW_LSQ_ALIGNMENT", "tdm_ui_lsq_target_run": 1,
                  "tdm_ui_lsq_run_count": 1, "tdm_ui_lsq_selected_run_1": 1,
                  "tdm_ui_lsq_mode": "Normal least-squares: chromatogram only",
                  "tdm_ui_lsq_parameter_lock_states": "conversion.uv_to_protein_mAU_L_g=Locked"}
            with patch.dict(tdm_bridge.__dict__, ui), \
                 patch.object(tdm_bridge, "build_shared_config_from_jmp", return_value=shared), \
                 patch.object(tdm_bridge, "_effective_run", return_value=shared), \
                 patch.object(tdm_bridge, "load_lsq_rows", return_value=[lsq_row]), \
                 patch.object(tdm_bridge, "load_batch_rows", return_value=batch_rows), \
                 patch.object(tdm_bridge, "BATCH_COUNT_FILE", root / "batch_count.txt"), \
                 patch.object(tdm_bridge, "save_config"), \
                 patch.object(tdm_bridge, "_write_status"), \
                 patch.object(lsq, "preview_lsq_alignment", side_effect=preview_with_receipt), \
                 patch.multiple(lsq, **output_paths):
                self.assertEqual(tdm_bridge.main(), 0)
                first = previews[-1]["profiles"][0]
                self.assertEqual(first["model_run_label"], "Normal batch Run 2")
                self.assertAlmostEqual(first["programmed_end_CV"], 0.7)
                self.assertAlmostEqual(first["uv_rmse_mAU"], 0.0, places=6)

                # Change the saved batch recipe and replace the file at its
                # existing path. Both inputs must be reread on the next action.
                batch_rows[1]["Elution1_CV"] = 0.45
                trace = selected_trace()
                _write_uv(reference, trace["CV"], trace["uv_mAU"], "CV")
                self.assertEqual(tdm_bridge.main(), 0)
                second = previews[-1]["profiles"][0]
                self.assertAlmostEqual(second["programmed_end_CV"], 0.85)
                self.assertAlmostEqual(second["uv_rmse_mAU"], 0.0, places=6)

                # Refinement reads the same selected recipe and latest UV
                # values, rather than a previous preview's cached reference.
                _write_uv(reference, trace["CV"], np.asarray(trace["uv_mAU"]) + 3.0, "CV")
                tdm_bridge.tdm_ui_action = "FIT_LSQ"
                self.assertEqual(tdm_bridge.main(), 0)
                with output_paths["FIT_TRACE_CSV"].open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(rows[0]["Model_Run"], "Normal batch Run 2")
                self.assertAlmostEqual(float(rows[0]["Measured_Signal"]),
                                       float(trace["uv_mAU"][0]) + 3.0)
                self.assertAlmostEqual(float(rows[0]["Predicted_Detector_Signal"]),
                                       float(trace["uv_mAU"][0]))
                self.assertIn("Normal batch Run 2", output_paths["FIT_REPORT_HTML"].read_text())

    def test_independent_lsq_recipe_remains_selectable(self):
        shared = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        lsq_row = default_batch_row(1)
        lsq_row.update(Load_CV=0.25, PLW_Count=0, Elution_Count=1,
                       Elution1_CV=0.2, LSQ_Model_Source=LSQ_MODEL_SOURCE_INDEPENDENT)
        batch = default_batch_row(1)
        batch.update(Load_CV=1.5, PLW_Count=0, Elution_Count=1, Elution1_CV=1.0)
        actual, receipt = tdm_bridge._lsq_model_run_config(shared, 1, lsq_row, [batch])
        expected, _ = apply_batch_shared_parameters(shared, batch_row_to_run_config(lsq_row))
        self.assertEqual(actual["process"], expected["process"])
        self.assertEqual(receipt["model_run_source"], LSQ_MODEL_SOURCE_INDEPENDENT)

    def test_two_independent_recipes_overlay_current_model_without_fitting(self):
        first = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        first["numerics"].update(time_steps=60, axial_positions=6)
        first["process"].update(load_CV=0.4, plw_count=0)
        second = copy.deepcopy(first)
        second["column"]["volume_mL"] = 2.0
        second["process"].update(load_amount_basis="VOLUME_ML", load_volume_mL=0.6,
                                 load_CV=99.0)
        second["process"]["elution_steps"][0]["CV"] = 0.25
        first_trace = simulate(first)["trace"]
        second_trace = simulate(second)["trace"]
        elution_start = 0.4
        first_mask = np.asarray(first_trace["CV"]) >= elution_start - 1e-10
        original_first = copy.deepcopy(first)
        original_second = copy.deepcopy(second)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_csv, second_csv = root / "elution_relative.csv", root / "whole_run_ml.csv"
            report = root / "alignment.html"
            _write_uv(first_csv,
                      np.asarray(first_trace["CV"])[first_mask] - elution_start,
                      np.asarray(first_trace["uv_mAU"])[first_mask], "CV")
            _write_uv(second_csv,
                      np.asarray(second_trace["CV"]) * second["column"]["volume_mL"],
                      np.asarray(second_trace["uv_mAU"]) + 5.0, "mL")
            preview = preview_lsq_alignment([
                {"config": first, "chromatogram_csv": first_csv,
                 "x_unit": "CV", "x_origin": "ELUTION_START",
                 "run_context": {"run_number": 1, "run_name": "Elution only"}},
                {"config": second, "chromatogram_csv": second_csv,
                 "x_unit": "ML", "x_origin": "RUN_START",
                 "run_context": {"run_number": 2, "run_name": "mL full run"}},
            ], output_html=report)
            self.assertEqual(preview["report_html"], str(report))
            self.assertEqual(len(preview["profiles"]), 2)
            self.assertGreaterEqual(preview["profiles"][0]["measured_start_CV"], elution_start)
            self.assertLess(preview["profiles"][0]["measured_start_CV"], elution_start + 0.02)
            self.assertAlmostEqual(preview["profiles"][0]["uv_rmse_mAU"], 0.0, places=6)
            self.assertAlmostEqual(preview["profiles"][1]["programmed_end_CV"], 0.55)
            self.assertAlmostEqual(preview["profiles"][1]["uv_rmse_mAU"], 5.0, places=6)
            markup = report.read_text(encoding="utf-8")
            self.assertEqual(markup.count("<svg"), 2)
            self.assertIn("Measured UV", markup)
            self.assertIn("Predicted UV", markup)
            self.assertIn("Elution only", markup)
            self.assertIn("mL full run", markup)
            self.assertIn('id="lsq-preview-run-picker"', markup)
            self.assertIn('data-run="1"', markup)
            self.assertIn('data-run="2" hidden', markup)
            self.assertIn(str(first_csv), markup)
            self.assertIn(str(second_csv), markup)
            self.assertIn("<b>Process setup:</b> column volume", markup)
            self.assertFalse((root / "least_squares_fit_report.html").exists())
        self.assertEqual(first, original_first)
        self.assertEqual(second, original_second)

    def test_out_of_recipe_reference_is_rejected_before_refinement(self):
        config = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        config["numerics"].update(time_steps=50, axial_positions=6)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "too_long.csv"
            _write_uv(reference, [0.0, 0.2, 5.0], [0.0, 1.0, 0.0], "CV")
            with self.assertRaisesRegex(ValueError, "outside the simulated recipe"):
                preview_lsq_alignment([{"config": config, "chromatogram_csv": reference}],
                                      output_html=root / "alignment.html")
            self.assertFalse((root / "alignment.html").exists())

    def test_preview_opens_on_run_with_mass_percent_and_labels_other_run(self):
        config = make_config(MODEL_EDM_LANGMUIR, 1, multistep=False)
        config["numerics"].update(time_steps=50, axial_positions=6)
        trace = simulate(config)["trace"]
        names = [component["name"] for component in config["components"][:2]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uv = root / "measured.csv"
            report = root / "alignment.html"
            _write_uv(uv, trace["CV"], trace["uv_mAU"], "CV")
            preview = preview_lsq_alignment([
                {"config": config, "chromatogram_csv": uv,
                 "run_context": {"run_number": 1, "run_name": "UV only"}},
                {"config": config, "chromatogram_csv": uv,
                 "run_context": {"run_number": 2, "run_name": "Fraction run"},
                 "composition_references": [
                     {"fraction_index": 1, "start": 0.20, "end": 0.24,
                      "x_unit": "CV", "mass_percent": {names[0]: 70, names[1]: 30}},
                     {"fraction_index": 2, "start": 0.24, "end": 0.28,
                      "x_unit": "CV", "mass_percent": {names[0]: 65, names[1]: 35}},
                 ]},
            ], output_html=report)
            self.assertEqual([item["mass_percent_observations"] for item in preview["profiles"]], [0, 4])
            markup = report.read_text(encoding="utf-8")
            self.assertIn('<option value="2" selected>LSQ Run 2: Fraction run — 4 mass% points</option>', markup)
            self.assertIn('data-run="1" hidden', markup)
            self.assertIn('data-run="2"><h2>', markup)
            self.assertIn('<h3 id="mass-preview-run-2">Fraction mass% at midpoints</h3>', markup)
            self.assertIn('href="#mass-preview-run-2"', markup)
            self.assertIn('no observations enabled for this run', markup)
            self.assertIn('stroke-width="2.3" stroke-dasharray="2 4"', markup)

    def test_jmp_has_a_separate_preview_action_and_report(self):
        jsl = build_jsl(load_config())
        _validate_jsl(jsl)
        self.assertIn('Button Box("Preview UV, conductivity and mass%", lsqPreviewRun = 0; SendAndRun("PREVIEW_LSQ_ALIGNMENT"))', jsl)
        self.assertEqual(jsl.count('Button Box("Preview this run", PreviewLSQSlot('), 10)
        self.assertIn('Python Name("tdm_ui_lsq_preview_run")', jsl)
        self.assertIn('Button Box("Run least-squares refinement", SendAndRun("FIT_LSQ"))', jsl)
        self.assertIn('If(actionText == "PREVIEW_LSQ_ALIGNMENT" & Contains(statusValue, "LSQ_ALIGNMENT_READY:")', jsl)
        self.assertIn('Button Box("Open alignment preview", OpenAlignmentPreview())', jsl)
        open_preview = jsl.split('OpenAlignmentPreview = Function', 1)[1].split('OpenFitReport = Function', 1)[0]
        self.assertIn('lsqPreviewRun = 0;', open_preview)
        self.assertIn('SendAndRun("PREVIEW_LSQ_ALIGNMENT")', open_preview)
        self.assertNotIn('Web(lsqPreviewReport)', open_preview)
        self.assertIn('If(actionText == "FIT_LSQ" | actionText == "PREVIEW_LSQ_ALIGNMENT" | actionText == "ATTACH_LSQ_CSV", If(!SaveAllLSQReferences(), Return()));', jsl)
        self.assertEqual(jsl.count('Text Box("Model run"'), 10)
        self.assertIn('Column(dtLSQRuns, "LSQ_Model_Source")[r] = If(modelChoice', jsl)
        self.assertIn('Column(dtLSQRuns, "LSQ_Model_Run_Number")[r] = If(modelChoice', jsl)
        self.assertIn('modelChoice = LSQModelRunSelected(slot);', jsl)

    def test_section_preview_uses_its_own_recipe_and_excel_file(self):
        shared = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        shared["numerics"].update(time_steps=50, axial_positions=6)
        lsq_rows = [default_batch_row(i) for i in range(1, 11)]
        batch_rows = [default_batch_row(i) for i in range(1, 31)]
        for number, load, elution in ((2, 0.4, 0.2), (4, 0.5, 0.4)):
            lsq_rows[number - 1].update(
                Run_Name=f"Reference {number}", Load_CV=load, PLW_Count=0,
                Elution_Count=1, Elution1_CV=elution, Time_Steps=50,
                Axial_Positions=6, LSQ_Model_Source=LSQ_MODEL_SOURCE_INDEPENDENT,
                LSQ_Sheet=f"Measured {number}", LSQ_X_Column="CV",
                LSQ_Signal_Column="UV 280 [mAU]",
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "alignment.html"

            def write_excel(number):
                path = root / f"run_{number}.xlsx"
                workbook = Workbook()
                decoy = workbook.active
                decoy.title = "Decoy"
                decoy.append(["CV", "UV 280 [mAU]"])
                for cv in (0, 0.1, 0.2, 0.3, 0.4):
                    decoy.append([cv, 100])
                measured = workbook.create_sheet(f"Measured {number}")
                measured.append(["CV", "UV 280 [mAU]"])
                for cv in (0, 0.1, 0.2, 0.3, 0.4):
                    measured.append([cv, number])
                workbook.save(path)
                return path

            path_2 = write_excel(2)
            path_4 = write_excel(4)
            lsq_rows[1]["LSQ_Chromatogram_CSV"] = str(path_2)
            actual_preview = lsq.preview_lsq_alignment
            previews = []

            def capture_preview(profiles):
                result = actual_preview(profiles, output_html=report)
                previews.append(result)
                return result

            ui = {
                "tdm_ui_action": "PREVIEW_LSQ_ALIGNMENT",
                "tdm_ui_lsq_run_count": 2,
                "tdm_ui_lsq_selected_run_1": 2,
                "tdm_ui_lsq_selected_run_2": 4,
                "tdm_ui_lsq_preview_run": 2,
                "tdm_ui_lsq_mode": "Normal least-squares: chromatogram only",
            }
            with patch.dict(tdm_bridge.__dict__, ui), \
                 patch.object(tdm_bridge, "build_shared_config_from_jmp", return_value=shared), \
                 patch.object(tdm_bridge, "_effective_run", return_value=shared), \
                 patch.object(tdm_bridge, "load_lsq_rows", return_value=lsq_rows), \
                 patch.object(tdm_bridge, "load_batch_rows", return_value=batch_rows), \
                 patch.object(tdm_bridge, "BATCH_COUNT_FILE", root / "batch_count.txt"), \
                 patch.object(tdm_bridge, "save_config"), \
                 patch.object(tdm_bridge, "_write_status"), \
                 patch.object(lsq, "preview_lsq_alignment", side_effect=capture_preview):
                # Another selected run need not have a file to preview Run 2.
                self.assertEqual(tdm_bridge.main(), 0)
                first = previews[-1]["profiles"]
                self.assertEqual(len(first), 1)
                self.assertEqual(first[0]["run_number"], 2)
                self.assertEqual(first[0]["source_path"], str(path_2))
                self.assertEqual(first[0]["sheet_name"], "Measured 2")
                self.assertAlmostEqual(first[0]["programmed_end_CV"], 0.6)
                self.assertEqual(report.read_text().count('class="lsq-preview-run"'), 1)

                lsq_rows[3]["LSQ_Chromatogram_CSV"] = str(path_4)
                tdm_bridge.tdm_ui_lsq_preview_run = 4
                self.assertEqual(tdm_bridge.main(), 0)
                second = previews[-1]["profiles"]
                self.assertEqual(len(second), 1)
                self.assertEqual(second[0]["run_number"], 4)
                self.assertEqual(second[0]["source_path"], str(path_4))
                self.assertEqual(second[0]["sheet_name"], "Measured 4")
                self.assertAlmostEqual(second[0]["programmed_end_CV"], 0.9)
                self.assertIn("Reference 4", report.read_text())
                self.assertNotIn(str(path_2), report.read_text())

    def test_bridge_preview_action_does_not_start_optimizer(self):
        shared = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        shared["numerics"].update(time_steps=50, axial_positions=6)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "measured.csv"
            _write_uv(source, [0.0, 0.1, 0.2], [0.0, 0.0, 0.0], "CV")
            row = default_batch_row(1)
            row.update(LSQ_Chromatogram_CSV=str(source), Load_CV=0.4,
                       PLW_Count=0, Elution_Count=1, Elution1_CV=0.18,
                       Time_Steps=50, Axial_Positions=6)
            report = root / "alignment.html"
            actual_preview = lsq.preview_lsq_alignment

            def write_preview(profiles):
                return actual_preview(profiles, output_html=report)

            with patch.dict(tdm_bridge.__dict__, {
                "tdm_ui_action": "PREVIEW_LSQ_ALIGNMENT", "tdm_ui_lsq_target_run": 1,
                "tdm_ui_lsq_run_count": 1, "tdm_ui_lsq_selected_run_1": 1,
                "tdm_ui_lsq_mode": "Normal least-squares: chromatogram only",
            }), patch.object(tdm_bridge, "build_shared_config_from_jmp", return_value=shared), \
                 patch.object(tdm_bridge, "_effective_run", return_value=shared), \
                 patch.object(tdm_bridge, "_selected_lsq_profile_rows", return_value=[(1, row, str(source))]), \
                 patch.object(tdm_bridge, "load_lsq_rows", return_value=[row]), \
                 patch.object(tdm_bridge, "BATCH_COUNT_FILE", root / "batch_count.txt"), \
                 patch.object(tdm_bridge, "save_config") as save_model, \
                 patch.object(tdm_bridge, "_write_status") as write_status, \
                 patch.object(lsq, "preview_lsq_alignment", side_effect=write_preview), \
                 patch.object(lsq, "run_global_least_squares_refinement", side_effect=AssertionError("optimizer ran")):
                self.assertEqual(tdm_bridge.main(), 0)
                save_model.assert_called_once()
                self.assertIn("LSQ_ALIGNMENT_READY:", write_status.call_args[0][0])
            self.assertTrue(report.is_file())


if __name__ == "__main__":
    unittest.main()
