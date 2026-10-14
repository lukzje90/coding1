"""Fraction mass% observations use real midpoints and preserve missing-data gaps."""
import copy
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import tdm_bridge
import tdm_least_squares as lsq
from tdm_jmp import build_jsl, _validate_jsl
from tdm_minimal_io import MODEL_EDM_LANGMUIR, default_batch_row
from tdm_minimal_model import simulate
from verify_model import make_config


class FractionMassTests(unittest.TestCase):
    def test_bridge_alignment_preview_contains_dotted_impurity_profile(self):
        cfg = make_config(MODEL_EDM_LANGMUIR, 1, multistep=False)
        cfg["numerics"].update(time_steps=50, axial_positions=6)
        trace = simulate(cfg)["trace"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uv = root / "uv.csv"
            with uv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["CV", "UV 280 [mAU]"])
                writer.writerows(zip(trace["CV"], trace["uv_mAU"]))
            row = default_batch_row(1)
            row["LSQ_Chromatogram_CSV"] = str(uv)
            report = root / "preview.html"
            actual_preview = lsq.preview_lsq_alignment
            ui = {"tdm_ui_action": "PREVIEW_LSQ_ALIGNMENT", "tdm_ui_lsq_target_run": 1,
                  "tdm_ui_lsq_run_count": 1, "tdm_ui_lsq_selected_run_1": 1,
                  "tdm_ui_lsq_mode": "Least-squares: chromatogram + species mass% by fraction",
                  "tdm_ui_lsq_reference_count": 4, "tdm_ui_lsq_fraction_unit": "CV"}
            for i, (lo, hi) in enumerate(((0.20, 0.22), (0.22, 0.24),
                                          (0.24, 0.26), (0.26, 0.28)), 1):
                ui[f"tdm_ui_lsq_r{i}_start"] = lo
                ui[f"tdm_ui_lsq_r{i}_end"] = hi
                if i != 3:
                    ui[f"tdm_ui_lsq_r{i}_c1_mass_percent"] = 60.0
                    ui[f"tdm_ui_lsq_r{i}_c2_mass_percent"] = 40.0
            receipt = {"model_run_source": "INDEPENDENT_LSQ", "model_run_number": 1,
                       "model_run_name": "Run 1", "model_run_label": "Independent LSQ recipe Run 1"}
            with patch.dict(tdm_bridge.__dict__, ui), \
                 patch.object(tdm_bridge, "build_shared_config_from_jmp", return_value=cfg), \
                 patch.object(tdm_bridge, "_effective_run", return_value=cfg), \
                 patch.object(tdm_bridge, "_selected_lsq_profile_rows", return_value=[(1, row, str(uv))]), \
                 patch.object(tdm_bridge, "load_lsq_rows", return_value=[row]), \
                 patch.object(tdm_bridge, "load_batch_rows", return_value=[row]), \
                 patch.object(tdm_bridge, "_lsq_model_run_config", return_value=(cfg, receipt)), \
                 patch.object(tdm_bridge, "BATCH_COUNT_FILE", root / "count.txt"), \
                 patch.object(tdm_bridge, "save_config"), \
                 patch.object(tdm_bridge, "_write_status") as status, \
                 patch.object(lsq, "preview_lsq_alignment",
                              side_effect=lambda profiles: actual_preview(profiles, output_html=report)):
                self.assertEqual(tdm_bridge.main(), 0)
            markup = report.read_text()
            self.assertIn("Fraction mass% at midpoints", markup)
            self.assertIn('stroke-width="2.3" stroke-dasharray="2 4"', markup)
            self.assertIn('stroke-dasharray="1 6"', markup)
            self.assertIn("6 fraction mass% observations", status.call_args[0][0])

    def test_midpoints_units_missing_species_and_measured_zero(self):
        cfg = make_config(MODEL_EDM_LANGMUIR, 1, multistep=False)
        cfg["column"]["volume_mL"] = 2.0
        names = [item["name"] for item in cfg["components"][:2]]
        refs = lsq.validate_composition_references(cfg, [
            {"fraction_index": 1, "start": 0.4, "end": 0.8, "x_unit": "ML",
             "mass_percent": {names[0]: 55.0, names[1]: None}},
            {"fraction_index": 2, "start": 0.8, "end": 1.2, "x_unit": "ML",
             "mass_percent": {names[0]: None, names[1]: 0.0}},
            {"fraction_index": 3, "start": 1.2, "end": 1.6, "x_unit": "ML",
             "mass_percent": {}},
        ])
        self.assertAlmostEqual(refs[0]["CV"], 0.3)
        self.assertAlmostEqual(refs[1]["CV"], 0.5)
        self.assertEqual(refs[0]["mass_percent"], {names[0]: 55.0})
        self.assertEqual(refs[1]["mass_percent"], {names[1]: 0.0})
        self.assertEqual(refs[2]["mass_percent"], {})
        svg = lsq._composition_profile_svg(cfg, refs, {
            "CV": np.array([0.0, 0.5, 1.0]),
            "component_g_L": np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]),
        })
        self.assertIn("Fraction midpoint [mL from run start]", svg)
        ui = {"tdm_ui_lsq_reference_count": 3, "tdm_ui_lsq_fraction_unit": "mL",
              "tdm_ui_lsq_r1_start": 0.4, "tdm_ui_lsq_r1_end": 0.8,
              "tdm_ui_lsq_r1_c1_mass_percent": 55.0,
              "tdm_ui_lsq_r2_start": 0.8, "tdm_ui_lsq_r2_end": 1.2,
              "tdm_ui_lsq_r3_start": 1.2, "tdm_ui_lsq_r3_end": 1.6,
              "tdm_ui_lsq_r3_c2_mass_percent": 0.0}
        with patch.dict(tdm_bridge.__dict__, ui):
            from_ui = tdm_bridge._composition_references_from_jmp(cfg)
        self.assertEqual([r["fraction_index"] for r in from_ui], [1, 2, 3])
        self.assertEqual(from_ui[0]["x_unit"], "ML")
        self.assertEqual(from_ui[2]["mass_percent"][names[1]], 0.0)
        with self.assertRaisesRegex(ValueError, "end must be greater"):
            lsq.validate_composition_references(cfg, [
                {"start": 1.0, "end": 1.0, "x_unit": "CV", "mass_percent": {names[0]: 30}}
            ])

    def test_residuals_and_graph_never_bridge_missing_fraction_values(self):
        cfg = make_config(MODEL_EDM_LANGMUIR, 1, multistep=False)
        names = [item["name"] for item in cfg["components"][:2]]
        trace = {"CV": np.array([0.0, 0.5, 1.0]),
                 "component_g_L": np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])}
        refs = lsq.validate_composition_references(cfg, [
            {"fraction_index": 1, "start": 0.1, "end": 0.2, "mass_percent": {names[0]: 55.0}},
            {"fraction_index": 2, "start": 0.2, "end": 0.3, "mass_percent": {names[0]: 60.0, names[1]: 30.0}},
            {"fraction_index": 3, "start": 0.3, "end": 0.4, "mass_percent": {}},
            {"fraction_index": 4, "start": 0.4, "end": 0.5, "mass_percent": {names[0]: 65.0, names[1]: 25.0}},
            {"fraction_index": 5, "start": 0.5, "end": 0.6, "mass_percent": {names[1]: 20.0}},
        ])
        residuals, rows = lsq._composition_residuals(cfg, refs, trace)
        self.assertEqual(len(residuals), 6)
        self.assertEqual({r["Fraction"] for r in rows}, {1, 2, 4, 5})
        self.assertEqual(rows[0]["Protein_Species"], names[0])
        self.assertAlmostEqual(rows[0]["Simulated_mass_percent"], 100 * 1.3 / 2.3)
        svg = lsq._composition_profile_svg(cfg, refs, trace)
        self.assertEqual(svg.count('stroke-width="2.3"'), 2)  # one connected segment per species
        self.assertIn("Blank: no fit or line", svg)
        self.assertIn('stroke-dasharray="1 6"', svg)
        self.assertNotIn("nan", svg.lower())

    def test_jmp_fraction_inputs_and_real_refinement(self):
        jsl = build_jsl()
        _validate_jsl(jsl)
        self.assertIn('lsqFractionUnit << Get Selected, Python Name("tdm_ui_lsq_fraction_unit")', jsl)
        self.assertIn('lsqR30End << Get, Python Name("tdm_ui_lsq_r30_end")', jsl)

        true = make_config(MODEL_EDM_LANGMUIR, 1, multistep=False)
        true["numerics"].update(time_steps=50, axial_positions=6)
        start = copy.deepcopy(true)
        start["components"][1]["b_L_g"] *= 2.0
        trace = simulate(true)["trace"]
        starting_uv = simulate(start)["trace"]["uv_mAU"]
        cv = np.asarray(trace["CV"])
        comp = np.asarray(trace["component_g_L"])
        names = [item["name"] for item in true["components"][:2]]
        fractions = []
        for index, (lo, hi) in enumerate(((0.19, 0.21), (0.25, 0.27), (0.27, 0.29), (0.29, 0.31)), 1):
            midpoint = (lo + hi) / 2
            observed = lsq._predicted_mass_percent_at_cv(cv, comp, midpoint, 2)
            values = {} if index == 2 else {names[0]: float(observed[0]), names[1]: float(observed[1])}
            fractions.append({"fraction_index": index, "start": lo, "end": hi,
                              "x_unit": "CV", "mass_percent": values})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uv = root / "measured.csv"
            with uv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["CV", "UV 280 [mAU]"])
                # UV alone favors the wrong starting value. Only the entered
                # fraction mass% observations can pull this parameter toward truth.
                writer.writerows(zip(cv, starting_uv))
            outputs = ("FIT_RESULTS_CSV", "FIT_TRACE_CSV", "FIT_COMPOSITION_CSV",
                       "FIT_REPORT_HTML", "FIT_CONFIG_JSON", "FIT_APPLY_JSL",
                       "FIT_SETTINGS_JSON", "FIT_HISTORY_CSV", "FIT_IMPROVED_CSV",
                       "FIT_IMPROVED_TXT", "CONFIG_PATH")
            with patch.multiple(lsq, **{name: root / getattr(lsq, name).name for name in outputs}):
                preview_path = root / "preview.html"
                preview = lsq.preview_lsq_alignment(
                    [{"config": start, "chromatogram_csv": uv,
                      "composition_references": fractions}], output_html=preview_path)
                self.assertEqual(len(preview["profiles"]), 1)
                preview_html = preview_path.read_text()
                self.assertIn("Fraction mass% at midpoints", preview_html)
                self.assertIn('stroke-width="2.3" stroke-dasharray="2 4"', preview_html)
                self.assertIn('stroke-dasharray="1 6"', preview_html)
                self.assertIn("Dotted: measured impurities", preview_html)
                fit = lsq.run_global_least_squares_refinement(
                    [{"config": start, "chromatogram_csv": uv,
                      "composition_references": fractions}],
                    mode=lsq.MODE_CHROMATOGRAM_AND_COMPOSITION,
                    unlocked_paths=["components[1].b_L_g"], max_nfev=5)
                with lsq.FIT_COMPOSITION_CSV.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 6)
                self.assertEqual({int(r["Fraction"]) for r in rows}, {1, 3, 4})
                self.assertIn("Fraction mass% at midpoints", lsq.FIT_REPORT_HTML.read_text())
            self.assertEqual(fit["parameters"], ["components[1].b_L_g"])
            self.assertLess(fit["final_objective"], fit["initial_objective"])
            self.assertLess(fit["fitted_config"]["components"][1]["b_L_g"],
                            start["components"][1]["b_L_g"])
