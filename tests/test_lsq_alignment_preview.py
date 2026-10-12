"""The pre-fit overlay must use the same UV and CV alignment as refinement."""
import copy
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tdm_jmp import _validate_jsl, build_jsl
import tdm_bridge
import tdm_least_squares as lsq
from tdm_least_squares import preview_lsq_alignment
from tdm_minimal_io import MODEL_EDM_LANGMUIR, default_batch_row, load_config
from tdm_minimal_model import simulate
from verify_model import make_config


def _write_uv(path: Path, x, uv, x_header: str):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([x_header, "UV 280 [mAU]"])
        writer.writerows(zip(x, uv))


class AlignmentPreviewTests(unittest.TestCase):
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

    def test_jmp_has_a_separate_preview_action_and_report(self):
        jsl = build_jsl(load_config())
        _validate_jsl(jsl)
        self.assertIn('Button Box("Preview measured vs model UV", SendAndRun("PREVIEW_LSQ_ALIGNMENT"))', jsl)
        self.assertIn('Button Box("Run least-squares refinement", SendAndRun("FIT_LSQ"))', jsl)
        self.assertIn('If(actionText == "PREVIEW_LSQ_ALIGNMENT" & Contains(statusValue, "LSQ_ALIGNMENT_READY:")', jsl)
        self.assertIn('Button Box("Open alignment preview", OpenAlignmentPreview())', jsl)

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
                "tdm_ui_lsq_mode": "Least-squares: chromatogram + species mass% at CV",
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
