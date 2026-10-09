"""Keep reported chromatograms aligned with the complete programmed recipe."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

import tdm_least_squares as lsq
from tdm_minimal_io import (
    MODEL_EDM_LANGMUIR, apply_batch_shared_parameters, batch_row_to_run_config,
    load_batch_rows, load_config,
)
from tdm_minimal_model import _build_program, _stage_receipt, _svg_plot, simulate, write_outputs
from verify_model import make_config


SVG_NS = {"svg": "http://www.w3.org/2000/svg"}


def _chromatogram_svg(document):
    for markup in re.findall(r"<svg\b.*?</svg>", document, flags=re.DOTALL):
        svg = ET.fromstring(markup)
        if any("Column volumes" in (text.text or "")
               for text in svg.findall("svg:text", SVG_NS)):
            return svg
    raise AssertionError("No chromatogram SVG was written")


def _points(polyline):
    return np.asarray([[float(value) for value in point.split(",")]
                       for point in polyline.attrib["points"].split()])


class ChromatogramExtentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        cfg["chromatography_mode"] = "HIC"
        cfg["numerics"].update(time_steps=148, axial_positions=6)
        cfg["components"][0]["salt_sensitivity_per_M"] = 4.0
        cfg["least_squares"].update(objective="RAW_SSE", baseline_mode="NONE",
                                    detector_baseline=23.0)
        cfg["process"].update(load_amount_basis="LOAD_VOLUME", load_CV=1.0,
                              load_start_percent_B=100.0, load_end_percent_B=100.0,
                              plw_count=2, elution_count=1)
        for step, flow in zip(cfg["process"]["plw_steps"][:2], (0.5, 2.0)):
            step.update(mode="SET_POINT", CV=2.0, flow_mL_min=flow,
                        start_percent_B=100.0, end_percent_B=100.0)
        cfg["process"]["elution_steps"][0].update(
            mode="LINEAR", CV=10.0, flow_mL_min=0.8,
            start_percent_B=100.0, end_percent_B=0.0)
        cls.config = cfg
        cls.result = simulate(cfg)

    def assert_full_recipe_chart(self, svg):
        stages = [rect for rect in svg.findall("svg:rect", SVG_NS)
                  if rect.attrib.get("stroke") == "#777"]
        self.assertEqual(len(stages), 4)
        start = float(stages[0].attrib["x"])
        end = float(stages[-1].attrib["x"]) + float(stages[-1].attrib["width"])
        span = end - start
        for stage, start_cv, duration_cv in zip(stages, (0, 1, 3, 5), (1, 2, 2, 10)):
            self.assertAlmostEqual(float(stage.attrib["x"]), start + start_cv / 15 * span, delta=0.02)
            self.assertAlmostEqual(float(stage.attrib["width"]), duration_cv / 15 * span, delta=0.02)
        texts = svg.findall("svg:text", SVG_NS)
        self.assertTrue(any("run ends at 15 CV" in (text.text or "") for text in texts))
        # Different stage flows give 1 + 4 + 1 + 12.5 minutes, not 15 minutes.
        end_time_ticks = [text for text in texts if text.attrib.get("y") == "131"
                          and abs(float(text.attrib["x"]) - end) < 0.02]
        self.assertEqual(len(end_time_ticks), 1)
        self.assertAlmostEqual(float(end_time_ticks[0].text), 18.5)
        for name in ("percent-b", "conductivity", "flow-rate"):
            group = svg.find(f"svg:g[@data-overlay='{name}']", SVG_NS)
            self.assertIsNotNone(group)
            curves = group.findall("svg:polyline", SVG_NS)
            self.assertTrue(curves, f"Missing {name} trace")
            for curve in curves:
                self.assertAlmostEqual(_points(curve)[-1, 0], end, delta=0.02)
        return start, end

    def test_normal_chromatogram_covers_ten_cv_elution_after_load_and_washes(self):
        trace = self.result["trace"]
        program = self.result["simulation"]["program"]
        self.assertEqual(program.stage_start_CV, [0.0, 1.0, 3.0, 5.0])
        self.assertEqual(program.stage_end_CV, [1.0, 3.0, 5.0, 15.0])
        self.assertAlmostEqual(float(trace["CV"][-1]), 15.0)
        self.assertAlmostEqual(float(trace["time_min"][-1]), 18.5)
        self.assertEqual(len(program.stages), 4)
        with tempfile.TemporaryDirectory() as directory:
            outputs = write_outputs(self.config, self.result, Path(directory))
            report = Path(outputs["html"]).read_text(encoding="utf-8")
            svg = _chromatogram_svg(report)
            _start, end = self.assert_full_recipe_chart(svg)
            for curve in svg.findall("svg:polyline", SVG_NS):
                self.assertAlmostEqual(_points(curve)[-1, 0], end, delta=0.02)
            manifest = json.loads(Path(outputs["manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["total_CV"], 15.0)
            self.assertEqual(manifest["stages"][-1]["duration_CV"], 10.0)
            self.assertEqual(manifest["stages"][-1]["end_CV"], 15.0)
            with Path(outputs["csv"]).open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertAlmostEqual(float(rows[-1]["CV"]), 15.0)
            self.assertAlmostEqual(float(rows[-1]["time_min"]), 18.5)

    def test_saved_three_cv_load_and_ten_cv_elution_end_at_thirteen_cv(self):
        row = dict(load_batch_rows()[0])
        row.update(Load_Amount_Basis="LOAD_VOLUME", Load_CV=3.0,
                   PLW_Count=0, Elution_Count=1, Elution1_CV=10.0)
        run_config = batch_row_to_run_config(row)
        config, _ = apply_batch_shared_parameters(load_config(), run_config)
        config["numerics"].update(time_steps=148, axial_positions=6)
        result = simulate(config)
        program = result["simulation"]["program"]
        self.assertEqual(program.stage_start_CV, [0.0, 3.0])
        self.assertEqual(program.stage_end_CV, [3.0, 13.0])
        with tempfile.TemporaryDirectory() as directory:
            outputs = write_outputs(config, result, Path(directory))
            svg = _chromatogram_svg(Path(outputs["html"]).read_text(encoding="utf-8"))
            self.assertTrue(any("run ends at 13 CV" in (text.text or "")
                                for text in svg.findall("svg:text", SVG_NS)))
            with Path(outputs["csv"]).open(newline="", encoding="utf-8") as fh:
                last = list(csv.DictReader(fh))[-1]
            self.assertAlmostEqual(float(last["CV"]), 13.0)

    def test_global_fit_plots_full_prediction_without_extending_measured_data(self):
        trace = self.result["trace"]
        sample_indices = np.flatnonzero(np.asarray(trace["CV"]) <= 10.0 + 1e-9)[::2]
        measured_cv = np.asarray(trace["CV"])[sample_indices]
        baseline = self.config["least_squares"]["detector_baseline"]
        measured_uv = np.asarray(trace["uv_mAU"])[sample_indices] + baseline
        self.assertAlmostEqual(float(measured_cv[-1]), 10.0)
        names = ["FIT_RESULTS_CSV", "FIT_TRACE_CSV", "FIT_COMPOSITION_CSV",
                 "FIT_REPORT_HTML", "FIT_CONFIG_JSON", "FIT_APPLY_JSL",
                 "FIT_SETTINGS_JSON", "FIT_HISTORY_CSV", "FIT_IMPROVED_CSV",
                 "FIT_IMPROVED_TXT", "CONFIG_PATH"]
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            source = tmp / "short_uv_reference.csv"
            with source.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["CV", "UV 280 [mAU]"])
                writer.writerows(zip(measured_cv, measured_uv))
            with patch.multiple(lsq, **{name: tmp / getattr(lsq, name).name for name in names}), \
                    patch.object(lsq, "simulate", return_value=self.result):
                fit = lsq.run_global_least_squares_refinement(
                    [{"config": self.config, "chromatogram_csv": source}], unlocked_paths=[])
            self.assertTrue(fit["success"])
            self.assertEqual(fit["parameters"], [])
            self.assertEqual(fit["nfev"], 0)
            self.assertLess(fit["final_objective"], 1e-16)
            self.assertEqual(fit["profile_metrics"][0]["overlap_points"], len(measured_cv))
            report = Path(fit["fit_report_html"]).read_text(encoding="utf-8")
            svg = _chromatogram_svg(report)
            start, end = self.assert_full_recipe_chart(svg)
            measured_curve, predicted_curve = svg.findall("svg:polyline", SVG_NS)
            measured_points = _points(measured_curve)
            predicted_points = _points(predicted_curve)
            self.assertEqual(len(measured_points), len(measured_cv))
            self.assertEqual(len(predicted_points), len(trace["CV"]))
            self.assertAlmostEqual(measured_points[-1, 0], start + 10.0 / 15.0 * (end - start), delta=0.02)
            self.assertAlmostEqual(predicted_points[-1, 0], end, delta=0.02)
            # Every observed sample remains on the predicted UV curve with its
            # fixed, nonzero detector baseline, including the initial zero UV.
            np.testing.assert_allclose(measured_points, predicted_points[sample_indices], atol=0.01)
            uv_axis_bottom = max(float(line.attrib["y2"]) for line in svg.findall("svg:line", SVG_NS)
                                 if line.attrib.get("x1") == line.attrib.get("x2"))
            self.assertLess(predicted_points[0, 1], uv_axis_bottom)
            with Path(fit["fit_trace_csv"]).open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), len(measured_cv))
            np.testing.assert_allclose([float(row["CV"]) for row in rows], measured_cv)
            np.testing.assert_allclose([float(row["Predicted_Detector_Signal"]) for row in rows], measured_uv)
            np.testing.assert_allclose([float(row["Residual"]) for row in rows], 0.0, atol=1e-12)

    def test_changing_recipe_and_adding_elution_extends_normal_report(self):
        cfg = copy.deepcopy(self.config)
        cfg["process"]["elution_steps"][0]["CV"] = 12.0
        cfg["process"]["elution_count"] = 2
        cfg["process"]["elution_steps"][1].update(
            mode="SET_POINT", CV=4.0, flow_mL_min=1.6,
            start_percent_B=0.0, end_percent_B=0.0)
        result = simulate(cfg)
        self.assertAlmostEqual(float(result["trace"]["CV"][-1]), 21.0)
        self.assertAlmostEqual(float(result["trace"]["time_min"][-1]), 23.5)
        with tempfile.TemporaryDirectory() as directory:
            outputs = write_outputs(cfg, result, Path(directory))
            svg = _chromatogram_svg(Path(outputs["html"]).read_text(encoding="utf-8"))
            self.assertTrue(any("run ends at 21 CV" in (text.text or "")
                                for text in svg.findall("svg:text", SVG_NS)))
            manifest = json.loads(Path(outputs["manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["total_CV"], 21.0)
            self.assertEqual([stage["end_CV"] for stage in manifest["stages"]],
                             [1.0, 3.0, 5.0, 17.0, 21.0])
            stages = [rect for rect in svg.findall("svg:rect", SVG_NS)
                      if rect.attrib.get("stroke") == "#777"]
            self.assertEqual(len(stages), 5)
            end = float(stages[-1].attrib["x"]) + float(stages[-1].attrib["width"])
            for curve in svg.findall("svg:polyline", SVG_NS):
                self.assertAlmostEqual(_points(curve)[-1, 0], end, delta=0.02)

    def test_axis_and_commands_follow_full_recipe_despite_short_display_limit(self):
        cfg = copy.deepcopy(self.config)
        cfg["process"]["plw_steps"][0]["end_percent_B"] = 20.0
        cfg["process"]["plw_steps"][1].update(mode="STEP", end_percent_B=80.0)
        cfg["process"]["elution_steps"][0]["start_percent_B"] = 80.0
        program = _build_program(cfg, np.array([2.0]))
        stages = _stage_receipt(cfg, program)
        # A sparse signal contains no samples at the wash or elution boundaries.
        # A stale display limit must not shorten the enabled 15 CV recipe.
        x = np.array([0.0, 15.0])
        svg = ET.fromstring(_svg_plot(
            x, [("UV", np.array([0.0, 1.0]))], stages=stages, total_cv=10.0,
            time_min=np.array([0.0, 18.5]), percent_B=np.array([100.0, 0.0]),
            flow_mL_min=np.array([1.0, 0.8])))
        self.assertTrue(any("run ends at 15 CV" in (text.text or "")
                            for text in svg.findall("svg:text", SVG_NS)))
        horizontal_axis = next(line for line in svg.findall("svg:line", SVG_NS)
                               if line.attrib.get("y1") == "520"
                               and line.attrib.get("y2") == "520"
                               and line.attrib.get("stroke") == "#222")
        start = float(horizontal_axis.attrib["x1"])
        end = float(horizontal_axis.attrib["x2"])
        rectangles = [rect for rect in svg.findall("svg:rect", SVG_NS)
                      if rect.attrib.get("stroke") == "#777"]
        self.assertAlmostEqual(float(rectangles[-1].attrib["x"])
                               + float(rectangles[-1].attrib["width"]), end, delta=0.02)
        expected_x = start + np.array([0, 1, 1, 3, 3, 5, 5, 15]) / 15.0 * (end - start)
        percent_b = _points(svg.find("svg:g[@data-overlay='percent-b']/svg:polyline", SVG_NS))
        np.testing.assert_allclose(percent_b[:, 0], expected_x, atol=0.01)
        # SET_POINT holds its start value, STEP uses its end value immediately,
        # and the final LINEAR gradient spans all ten programmed CV.
        np.testing.assert_allclose(percent_b[:4, 1], percent_b[0, 1])
        np.testing.assert_allclose(percent_b[4:7, 1], percent_b[4, 1])
        self.assertGreater(percent_b[4, 1], percent_b[3, 1])
        self.assertGreater(percent_b[7, 1], percent_b[6, 1])
        flow = _points(svg.find("svg:g[@data-overlay='flow-rate']/svg:polyline", SVG_NS))
        np.testing.assert_allclose(flow[:, 0], expected_x, atol=0.01)
        np.testing.assert_allclose(flow[::2, 1], flow[1::2, 1])
        self.assertEqual(len(set(flow[::2, 1])), 4)

    def test_direct_endpoint_stage_has_no_programmed_percent_b(self):
        program = _build_program(self.config, np.array([2.0]))
        stages = _stage_receipt(self.config, program)
        stages[-1]["start_percent_B"] = None
        stages[-1]["end_percent_B"] = None
        svg = ET.fromstring(_svg_plot(
            np.array([0.0, 15.0]), [("UV", np.array([0.0, 1.0]))],
            stages=stages, percent_B=np.array([100.0, np.nan]),
            flow_mL_min=np.array([1.0, 0.8]), time_min=np.array([0.0, 18.5])))
        percent_b = _points(svg.find("svg:g[@data-overlay='percent-b']/svg:polyline", SVG_NS))
        flow = _points(svg.find("svg:g[@data-overlay='flow-rate']/svg:polyline", SVG_NS))
        self.assertLess(percent_b[-1, 0], flow[-1, 0])


if __name__ == "__main__":
    unittest.main()
