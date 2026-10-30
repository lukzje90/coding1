"""Saved prediction and LSQ folders preserve each run's evidence."""
import csv
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import tdm_bridge
from tdm_minimal_io import MODEL_EDM_LANGMUIR
from tdm_minimal_model import (
    _saved_run_folder_name, _uk_archive_stamp, check_result_freshness,
    run_batch_and_write, simulate, write_outputs,
)
from verify_model import make_config


class ResultArchiveTests(unittest.TestCase):
    def test_prediction_archive_keeps_run_receipt_and_does_not_replace_older_copy(self):
        cfg = make_config(MODEL_EDM_LANGMUIR, 1, multistep=False)
        cfg["numerics"].update(time_steps=50, axial_positions=6)
        context = {"run_number": 4, "run_name": 'Product/Impurity: trial?'}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = write_outputs(cfg, simulate(cfg), root, context)
            first = tdm_bridge._archive_prediction(paths, context)
            second = tdm_bridge._archive_prediction(paths, context)
            self.assertNotEqual(first, second)
            for folder in (first, second):
                self.assertEqual(folder.parent, root / "Predictions")
                self.assertIn("Run_04_Product_Impurity_", folder.name)
                self.assertTrue(folder.name.endswith("_GMT") or "_GMT_" in folder.name or "_BST_" in folder.name)
                valid, message = check_result_freshness(cfg, context, folder)
                self.assertTrue(valid, message)
                for key in ("html", "csv", "svg", "manifest", "impurity_svg", "fraction_csv"):
                    if key in paths:
                        self.assertEqual((folder / Path(paths[key]).name).read_bytes(),
                                         Path(paths[key]).read_bytes())
            self.assertTrue(Path(paths["manifest"]).is_file())

    def test_batch_gallery_points_to_dated_named_run_folders(self):
        cfg = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        cfg["numerics"].update(time_steps=50, axial_positions=5)
        first = dict(cfg, _run_name="Early/product")
        second = dict(cfg, _run_name="Late: product")
        with tempfile.TemporaryDirectory() as temp:
            info = run_batch_and_write(cfg, [first, second], Path(temp) / "Predictions" / "Batch_test")
            gallery = Path(info["gallery_html"]).read_text(encoding="utf-8")
            folders = [Path(row["html"]).parent for row in info["runs"]]
            self.assertEqual(len(set(folders)), 2)
            self.assertIn("Run_01_Early_product_", folders[0].name)
            self.assertIn("Run_02_Late_product_", folders[1].name)
            for folder, row in zip(folders, info["runs"]):
                self.assertIn(folder.name + "/latest_results.svg", gallery)
                context = {"run_number": row["run_number"], "run_name": row["run_name"]}
                valid, message = check_result_freshness(cfg, context, folder)
                self.assertTrue(valid, message)
            with patch.object(tdm_bridge, "BATCH_GALLERY_HTML", Path(temp) / "latest_gallery.html"), \
                 patch.object(tdm_bridge, "BATCH_SUMMARY_CSV", Path(temp) / "latest_summary.csv"):
                tdm_bridge._save_batch_summary_to_root(info)
                latest_gallery = tdm_bridge.BATCH_GALLERY_HTML.read_text(encoding="utf-8")
                self.assertIn('<base href="Predictions/Batch_test/">', latest_gallery)
                self.assertTrue(tdm_bridge.BATCH_SUMMARY_CSV.is_file())

    def test_global_lsq_archive_separates_selected_run_data_and_references(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source2 = root / "reference2.csv"
            source4 = root / "reference4.xlsx"
            source2.write_text("CV,UV\n0,1\n", encoding="utf-8")
            source4.write_bytes(b"example workbook bytes")
            report = root / "least_squares_fit_report.html"
            report.write_text("<h1>Global fit</h1>", encoding="utf-8")
            apply_script = root / "least_squares_apply_to_ui.jsl"
            apply_script.write_text("// applied fit", encoding="utf-8")
            trace = root / "least_squares_fit_trace.csv"
            composition = root / "least_squares_fit_composition.csv"
            conductivity = root / "least_squares_fit_conductivity.csv"
            for path in (trace, composition, conductivity):
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["Run_Number", "Value"])
                    writer.writeheader()
                    writer.writerows([{"Run_Number": 2, "Value": "two"},
                                     {"Run_Number": 4, "Value": "four"}])
            initial2 = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
            initial2["process"]["load_CV"] = 2.0
            initial4 = copy.deepcopy(initial2)
            initial4["process"]["load_CV"] = 4.0
            fitted = copy.deepcopy(initial2)
            fitted["components"][0]["qmax_g_L"] *= 1.5
            fit = {
                "fit_report_html": str(report), "fit_trace_csv": str(trace),
                "fit_composition_csv": str(composition),
                "fit_conductivity_csv": str(conductivity),
                "accepted": True, "apply_jsl": str(apply_script),
                "fitted_config": fitted,
                "profile_metrics": [{"run_number": 2, "raw_rmse": 1.2},
                                    {"run_number": 4, "raw_rmse": 0.4}],
            }
            profiles = [
                {"run_context": {"run_number": 2, "run_name": "Acidic/variant",
                                 "model_run_name": "Process 2", "model_run_label": "Run 2"},
                 "chromatogram_csv": str(source2), "config": initial2},
                {"run_context": {"run_number": 4, "run_name": "Main: product",
                                 "model_run_name": "Process 4", "model_run_label": "Run 4"},
                 "chromatogram_csv": str(source4), "config": initial4},
            ]
            saved = tdm_bridge._archive_lsq_fit(fit, profiles)
            index = json.loads((Path(saved["global_folder"]) / "archive_index.json").read_text())
            self.assertEqual((Path(saved["global_folder"]) / apply_script.name).read_bytes(),
                             apply_script.read_bytes())
            self.assertEqual([row["run_number"] for row in index["runs"]], [2, 4])
            self.assertEqual(len(saved["runs"]), 2)
            for number, source, row in zip((2, 4), (source2, source4), saved["runs"]):
                folder = Path(row["folder"])
                self.assertIn(f"Run_{number:02d}_", folder.name)
                receipt = json.loads((folder / "run_manifest.json").read_text())
                self.assertEqual(receipt["run_number"], number)
                self.assertEqual(receipt["recipe"]["process"]["load_CV"], number)
                fitted_recipe = json.loads((folder / receipt["fitted_recipe_file"]).read_text())
                self.assertEqual(fitted_recipe["process"]["load_CV"], number)
                self.assertEqual(fitted_recipe["components"][0]["qmax_g_L"],
                                 fitted["components"][0]["qmax_g_L"])
                self.assertEqual((folder / receipt["reference_copy"]).read_bytes(), source.read_bytes())
                self.assertTrue((folder / receipt["global_report"]).is_file())
                for filename in (trace.name, composition.name, conductivity.name):
                    with (folder / filename).open(newline="", encoding="utf-8") as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual([int(item["Run_Number"]) for item in rows], [number])
                self.assertIn("Global fit", (Path(saved["global_folder"]) / report.name).read_text())
            second = tdm_bridge._archive_lsq_fit(fit, profiles)
            self.assertNotEqual(saved["global_folder"], second["global_folder"])
            self.assertTrue(Path(saved["global_folder"]).is_dir())
            history = root / "least_squares_objective_history.csv"
            history.write_text("evaluation,objective\n0,1\n", encoding="utf-8")
            fit["fit_history_csv"] = str(history)
            original_copy = shutil.copy2

            def locked_history(source, destination):
                if Path(source) == history:
                    raise PermissionError("history is open in JMP")
                return original_copy(source, destination)

            with patch.object(tdm_bridge.shutil, "copy2", side_effect=locked_history):
                recovered = tdm_bridge._archive_lsq_fit(fit, profiles)
            self.assertTrue(Path(recovered["global_folder"]).is_dir())
            self.assertIn("could not be copied", recovered["warnings"][0])

    def test_archive_names_use_uk_civil_time_and_safe_windows_characters(self):
        winter = datetime(2026, 1, 15, 12, tzinfo=timezone.utc)
        summer = datetime(2026, 7, 15, 12, tzinfo=timezone.utc)
        self.assertEqual(_uk_archive_stamp(winter), "2026-01-15_12-00-00_GMT")
        self.assertEqual(_uk_archive_stamp(summer), "2026-07-15_13-00-00_BST")
        name = _saved_run_folder_name(3, '../CON\\sample:*?<>"', winter)
        self.assertTrue(name.startswith("Run_03_CON_sample_2026-01-15_12-00-00_GMT_"))
        self.assertFalse(any(char in name for char in '/\\:*?<>"'))
