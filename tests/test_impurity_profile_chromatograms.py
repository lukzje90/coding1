"""The separate impurity charts use the actual per-run detector traces."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

import tdm_least_squares as lsq
from tdm_minimal_io import MODEL_EDM_LANGMUIR
from tdm_minimal_model import (
    _predicted_fraction_analysis, check_result_freshness, simulate, write_outputs,
)
from verify_model import make_config


class ImpurityChromatogramTests(unittest.TestCase):
    def make_run(self, impurity_count=2, elution_cv=1.5):
        cfg = make_config(MODEL_EDM_LANGMUIR, impurity_count, multistep=False)
        cfg["numerics"].update(time_steps=50, axial_positions=6)
        cfg["process"]["elution_steps"][0]["CV"] = elution_cv
        return cfg

    def test_forward_output_uses_detector_impurity_concentrations_and_checks_saved_chart(self):
        cfg = self.make_run()
        result = simulate(cfg)
        trace = result["trace"]
        names = result["simulation"]["component_names"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = write_outputs(cfg, result, root, {"run_number": 2, "run_name": "Impurity trial"})
            markup = Path(paths["html"]).read_text(encoding="utf-8")
            impurity_svg = Path(paths["impurity_svg"]).read_text(encoding="utf-8")
            chart = ET.fromstring(impurity_svg)
            text = [node.text or "" for node in chart.findall("{http://www.w3.org/2000/svg}text")]
            self.assertIn("Predicted main protein and impurity chromatogram", markup)
            self.assertEqual(markup.count('aria-label="Predicted main protein and impurity concentrations by CV"'), 1)
            self.assertEqual([label for label in text if label in names], names)
            self.assertIn(f'run ends at {result["simulation"]["program"].total_CV:g} CV', impurity_svg)
            self.assertIn("Predicted impurities at detector (g/L)", impurity_svg)
            self.assertIn("Main protein at detector (g/L)", impurity_svg)
            self.assertIn("Predicted fraction analysis by CV", markup)
            with Path(paths["fraction_csv"]).open(newline="", encoding="utf-8") as stream:
                fraction_rows = list(csv.DictReader(stream))
            self.assertEqual(len(fraction_rows), int(np.ceil(result["simulation"]["program"].total_CV)))
            self.assertEqual(float(fraction_rows[-1]["End_CV"]),
                             result["simulation"]["program"].total_CV)
            for row in fraction_rows:
                start, end = float(row["Start_CV"]), float(row["End_CV"])
                if row["Protein_Detected"] == "1":
                    actual = np.array([float(row[f"{name}_mass_percent"]) for name in names])
                    expected, quantifiable = lsq._predicted_fraction_mass_percent(
                        cfg, trace, {"start": start, "end": end,
                                     "x_unit": "CV", "fraction_index": 1})
                    self.assertTrue(quantifiable)
                    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                    self.assertAlmostEqual(float(actual.sum()), 100.0, places=9)
                else:
                    self.assertTrue(all(row[f"{name}_mass_percent"] == "" for name in names))
            plotted = chart.findall("{http://www.w3.org/2000/svg}polyline")
            self.assertEqual(len(plotted), 3)
            def peak_height(line):
                return min(float(point.split(",")[1]) for point in line.attrib["points"].split())
            # The separate axes give the main protein and the largest impurity
            # useful vertical range, even when their g/L values differ widely.
            self.assertAlmostEqual(peak_height(plotted[0]),
                                   min(peak_height(line) for line in plotted[1:]), delta=0.02)
            with Path(paths["csv"]).open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            for i, name in enumerate(names):
                np.testing.assert_allclose([float(row[f"{name}_g_L"]) for row in rows],
                                           trace["component_g_L"][:, i], rtol=1e-12, atol=1e-12)
            receipt = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
            self.assertIn("impurity_svg", receipt["output_sha256"])
            self.assertIn("fraction_csv", receipt["output_sha256"])
            context = {"run_number": 2, "run_name": "Impurity trial"}
            self.assertTrue(check_result_freshness(cfg, context, root)[0])
            Path(paths["impurity_svg"]).write_text(impurity_svg + " ", encoding="utf-8")
            self.assertFalse(check_result_freshness(cfg, context, root)[0])

    def test_target_only_run_has_no_empty_impurity_artifact(self):
        cfg = self.make_run(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = write_outputs(cfg, simulate(cfg), root)
            self.assertNotIn("impurity_svg", paths)
            self.assertNotIn("fraction_csv", paths)
            self.assertTrue(all(Path(path).is_file() for path in paths.values()))
            self.assertFalse((root / "latest_impurity_profile.svg").exists())
            self.assertNotIn("Predicted impurity chromatogram", Path(paths["html"]).read_text())
            self.assertNotIn("Predicted fraction analysis by CV", Path(paths["html"]).read_text())
            self.assertTrue(check_result_freshness(cfg, None, root)[0])

    def test_lsq_preview_and_fit_use_each_selected_run(self):
        first = self.make_run(1, 1.0)
        second = self.make_run(1, 2.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = []
            end_cvs = []
            for number, cfg in ((2, first), (4, second)):
                result = simulate(cfg)
                trace = result["trace"]
                end_cvs.append(result["simulation"]["program"].total_CV)
                path = root / f"reference_{number}.csv"
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["CV", "UV 280 [mAU]"])
                    writer.writerows(zip(trace["CV"], trace["uv_mAU"]))
                profiles.append({"config": cfg, "chromatogram_csv": path,
                                 "run_context": {"run_number": number,
                                                 "run_name": f"LSQ Run {number}",
                                                 "model_run_name": f"Process {number}"}})
            preview_path = root / "preview.html"
            lsq.preview_lsq_alignment(profiles, output_html=preview_path)
            preview = preview_path.read_text(encoding="utf-8")
            self.assertEqual(preview.count("Predicted main protein and impurity chromatogram before refinement"), 2)
            self.assertIn("Run 2 LSQ — Process 2", preview)
            self.assertIn("Run 4 LSQ — Process 4", preview)
            self.assertEqual(preview.count('aria-label="Predicted main protein and impurity concentrations by CV"'), 2)
            self.assertEqual(preview.count("Predicted fraction analysis by CV"), 2)

            outputs = ("FIT_RESULTS_CSV", "FIT_TRACE_CSV", "FIT_COMPOSITION_CSV",
                       "FIT_CONDUCTIVITY_CSV", "FIT_REPORT_HTML", "FIT_CONFIG_JSON",
                       "FIT_APPLY_JSL", "FIT_SETTINGS_JSON", "FIT_HISTORY_CSV",
                       "FIT_IMPROVED_CSV", "FIT_IMPROVED_TXT", "CONFIG_PATH")
            with patch.multiple(lsq, **{name: root / getattr(lsq, name).name for name in outputs}):
                lsq.run_global_least_squares_refinement(profiles, unlocked_paths=[])
                report = lsq.FIT_REPORT_HTML.read_text(encoding="utf-8")
            self.assertEqual(report.count("Predicted main protein and impurity chromatogram after refinement"), 2)
            self.assertEqual(report.count('aria-label="Predicted main protein and impurity concentrations by CV"'), 2)
            self.assertEqual(report.count("Predicted fraction analysis by CV"), 2)
            for end_cv in end_cvs:
                self.assertIn(f"run ends at {end_cv:g} CV", report)

    def test_partial_cv_and_no_protein_are_not_reported_as_zero_percent(self):
        class Program:
            total_CV = 2.5

        result = {
            "simulation": {"component_names": ["Product", "Impurity A"], "program": Program()},
            "trace": {"CV": np.array([0.0, 0.75, 1.0, 1.5, 2.5]),
                      "component_g_L": np.array([[0.0, 0.0], [0.0, 0.0],
                                                  [0.0, 0.0], [2.0, 1.0], [2.0, 1.0]])},
        }
        names, rows = _predicted_fraction_analysis(result)
        self.assertEqual(names, ["Product", "Impurity A"])
        self.assertEqual([(row["start_cv"], row["end_cv"]) for row in rows],
                         [(0.0, 1.0), (1.0, 2.0), (2.0, 2.5)])
        self.assertFalse(rows[0]["quantifiable"])
        self.assertEqual(rows[0]["percentages"], [None, None])
        np.testing.assert_allclose(rows[-1]["percentages"], [200 / 3, 100 / 3])
        self.assertAlmostEqual(sum(rows[-1]["percentages"]), 100.0)


if __name__ == "__main__":
    unittest.main()
