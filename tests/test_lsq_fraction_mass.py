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
from tdm_minimal_io import MODEL_EDM_LANGMUIR, MODEL_TDM_CPA, default_batch_row, load_lsq_mass_rows, normalize_config
from tdm_minimal_model import simulate
from verify_model import make_config


class FractionMassTests(unittest.TestCase):
    def test_cpa_empty_fraction_has_uv_anchored_species_gradient(self):
        cfg = make_config(MODEL_TDM_CPA, 1, multistep=False)
        names = [row["name"] for row in cfg["components"][:2]]
        ref = lsq.validate_composition_references(cfg, [{
            "start": 0.5, "end": 1.5, "x_unit": "CV",
            "mass_percent": {names[0]: 60.0, names[1]: 40.0},
        }])
        measured = lsq.ChromatogramReference(
            cv=np.array([0.0, 1.0, 2.0]), signal=np.array([100.0, 100.0, 100.0]),
            x_column="CV", signal_column="UV mAU", x_unit="CV", source_path="synthetic")
        empty = {"CV": np.array([0.0, 1.0, 2.0]),
                 "component_g_L": np.zeros((3, 2)),
                 "uv_response_factors_mAU_L_g": np.array([200.0, 200.0])}
        approaching = copy.deepcopy(empty)
        approaching["component_g_L"] = np.tile([0.3, 0.2], (3, 1))
        empty_anchor = lsq._cpa_fraction_uv_residuals(cfg, ref, empty, measured, 0.0)
        fitted_anchor = lsq._cpa_fraction_uv_residuals(cfg, ref, approaching, measured, 0.0)
        self.assertGreater(float(empty_anchor @ empty_anchor), 0.0)
        np.testing.assert_allclose(fitted_anchor, 0.0, atol=1e-12)
        partial = copy.deepcopy(ref)
        partial[0]["mass_percent"].pop(names[1])
        self.assertEqual(len(lsq._cpa_fraction_uv_residuals(cfg, partial, empty, measured, 0.0)), 0)

    def test_four_species_cpa_affinity_recovers_from_uv_and_fraction_mass(self):
        truth = make_config(MODEL_TDM_CPA, 3, multistep=False)
        truth["numerics"].update(time_steps=60, axial_positions=6)
        truth["feed"]["load_density_mg_mL_resin"] = 6.0
        truth["process"]["buffer_A"]["conductivity_mS_cm"] = 2.0
        truth["process"]["plw_steps"][0]["CV"] = 1.0
        truth["process"]["elution_steps"][0].update(CV=4.0, end_percent_B=100.0)
        truth["least_squares"]["objective"] = "NORMALIZED_MSE"
        measured = simulate(truth)
        audit = measured["simulation"]["diagnostics"]["CPA_adsorption_audit"]["species"]
        self.assertTrue(all(row["end_load_bound_percent_of_load"] > 95.0 for row in audit))
        self.assertTrue(all(row["peak_outlet_stage_kind"] == "ELUTION" for row in audit))
        self.assertTrue(all(row["elution_outlet_percent_of_load"] > 95.0 for row in audit))
        self.assertTrue(all(abs(error) < 1.0
                            for stage in measured["simulation"]["diagnostics"]["CPA_adsorption_audit"]["stages"]
                            for error in stage["cumulative_mass_balance_error_percent_of_load"]))
        names = [row["name"] for row in truth["components"][:4]]
        trace = measured["trace"]
        fractions = []
        for index in range(6):
            start = 4.75 + 0.1 * index
            end = start + 0.1
            values, quantifiable = lsq._predicted_fraction_mass_percent(
                truth, trace, {"fraction_index": index + 1, "start": start,
                               "end": end, "x_unit": "CV", "CV": (start + end) / 2})
            self.assertTrue(quantifiable)
            fractions.append({"fraction_index": index + 1, "start": start, "end": end,
                              "x_unit": "CV", "mass_percent": dict(zip(names, map(float, values)))})
        initial = copy.deepcopy(truth)
        initial["components"][0]["delta_ref"] = 0.002
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "reference.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["CV", "UV mAU"])
                writer.writerows(zip(trace["CV"], trace["uv_mAU"]))
            outputs = ("FIT_RESULTS_CSV", "FIT_TRACE_CSV", "FIT_COMPOSITION_CSV",
                       "FIT_CONDUCTIVITY_CSV", "FIT_REPORT_HTML", "FIT_CONFIG_JSON",
                       "FIT_APPLY_JSL", "FIT_SETTINGS_JSON", "FIT_HISTORY_CSV",
                       "FIT_IMPROVED_CSV", "FIT_IMPROVED_TXT", "CONFIG_PATH")
            with patch.multiple(lsq, **{name: root / getattr(lsq, name).name
                                        for name in outputs}):
                fit = lsq.run_global_least_squares_refinement(
                    [{"config": initial, "chromatogram_csv": source,
                      "composition_references": fractions}],
                    mode=lsq.MODE_CHROMATOGRAM_AND_COMPOSITION,
                    unlocked_paths=["components[0].delta_ref"],
                    max_nfev=8, max_model_evaluations=18)
                report = lsq.FIT_REPORT_HTML.read_text(encoding="utf-8")
            self.assertTrue(fit["accepted"])
            self.assertLess(fit["final_objective"], fit["initial_objective"] * 1e-6)
            self.assertAlmostEqual(fit["fitted_config"]["components"][0]["delta_ref"],
                                   truth["components"][0]["delta_ref"], delta=1e-4)
            self.assertTrue(all(row["final_rmse_percentage_points"] <
                                max(row["initial_rmse_percentage_points"] * 1e-4, 1e-4)
                                for row in fit["composition_profile_metrics"]))
            self.assertIn("complete mass% row also constrains", report)

    def test_fraction_mass_uses_collected_interval_with_interpolated_boundaries(self):
        cfg = make_config(MODEL_EDM_LANGMUIR, 1, multistep=False)
        cfg["column"]["volume_mL"] = 2.0
        names = [item["name"] for item in cfg["components"][:2]]
        trace = {"CV": np.array([0.0, 1.0, 2.0]),
                 "component_g_L": np.array([[0.0, 1.0], [10.0, 1.0], [0.0, 1.0]])}
        refs = lsq.validate_composition_references(cfg, [{
            "fraction_index": 1, "start": 1.0, "end": 3.0, "x_unit": "ML",
            "mass_percent": {names[0]: 100.0 * 7.5 / 8.5,
                             names[1]: 100.0 / 8.5},
        }])
        prediction, quantifiable = lsq._predicted_fraction_mass_percent(cfg, trace, refs[0])
        self.assertTrue(quantifiable)
        np.testing.assert_allclose(prediction, [100.0 * 7.5 / 8.5, 100.0 / 8.5])
        self.assertGreater(lsq._predicted_mass_percent_at_cv(trace["CV"], trace["component_g_L"], 1.0, 2)[0], prediction[0])
        residual, rows = lsq._composition_residuals(cfg, refs, trace)
        np.testing.assert_allclose(residual, 0.0, atol=1e-12)
        self.assertTrue(all(row["Model_Fraction_Quantifiable"] for row in rows))
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            lsq.validate_composition_references(cfg, [
                {"start": 0.0, "end": 1.0, "mass_percent": {names[0]: 50.0}},
                {"start": 0.8, "end": 1.5, "mass_percent": {names[0]: 50.0}},
            ])

    def test_tdm_cpa_four_species_uv_and_mass_fit_reproduces_saved_trace(self):
        truth = normalize_config(make_config(MODEL_TDM_CPA, 3, multistep=False))
        truth["numerics"].update(time_steps=50, axial_positions=6)
        truth["process"]["load_CV"] = 3.0
        truth["process"]["plw_count"] = 0
        truth["process"]["elution_steps"][0]["CV"] = 10.0
        truth["least_squares"]["objective"] = "NORMALIZED_MSE"
        measured = simulate(truth)["trace"]
        initial = copy.deepcopy(truth)
        initial["components"][3]["D_ax_mm2_s"] *= 5.0
        initial["conversion"]["uv_to_protein_mAU_L_g"] *= 0.65
        cv = np.asarray(measured["CV"])
        concentrations = np.asarray(measured["component_g_L"])
        names = [row["name"] for row in truth["components"][:4]]
        intervals = ((.35, .45), (.45, .55), (.55, .65), (.65, .75),
                     (.75, .85), (3.65, 3.75), (3.85, 3.95))
        fractions = []
        for number, (start, end) in enumerate(intervals, start=1):
            values, _ = lsq._predicted_fraction_mass_percent(truth, measured, {
                "fraction_index": number, "start": start, "end": end,
                "x_unit": "CV", "CV": (start + end) / 2})
            fractions.append({"fraction_index": number, "start": start, "end": end,
                              "x_unit": "CV", "mass_percent": dict(zip(names, map(float, values)))})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "measured.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["CV", "UV 280 [mAU]", "Conductivity [mS/cm]"])
                writer.writerows(zip(cv, measured["uv_mAU"], measured["conductivity_mS_cm"]))
            output_names = ("FIT_RESULTS_CSV", "FIT_TRACE_CSV", "FIT_COMPOSITION_CSV",
                            "FIT_CONDUCTIVITY_CSV", "FIT_REPORT_HTML", "FIT_CONFIG_JSON",
                            "FIT_APPLY_JSL", "FIT_SETTINGS_JSON", "FIT_HISTORY_CSV",
                            "FIT_IMPROVED_CSV", "FIT_IMPROVED_TXT", "CONFIG_PATH")
            with patch.multiple(lsq, **{name: root / getattr(lsq, name).name
                                        for name in output_names}):
                fit = lsq.run_global_least_squares_refinement(
                    [{"config": initial, "chromatogram_csv": source,
                      "composition_references": fractions, "fit_conductivity": True}],
                    mode=lsq.MODE_CHROMATOGRAM_AND_COMPOSITION,
                    unlocked_paths=["components[3].D_ax_mm2_s",
                                    "conversion.uv_to_protein_mAU_L_g"],
                    max_nfev=20, max_model_evaluations=60)
                with lsq.FIT_TRACE_CSV.open(newline="", encoding="utf-8") as stream:
                    uv_rows = list(csv.DictReader(stream))
                with lsq.FIT_COMPOSITION_CSV.open(newline="", encoding="utf-8") as stream:
                    mass_rows = list(csv.DictReader(stream))
                with lsq.FIT_CONDUCTIVITY_CSV.open(newline="", encoding="utf-8") as stream:
                    conductivity_rows = list(csv.DictReader(stream))
                with lsq.FIT_RESULTS_CSV.open(newline="", encoding="utf-8") as stream:
                    parameter_rows = list(csv.DictReader(stream))
                report_html = lsq.FIT_REPORT_HTML.read_text(encoding="utf-8")
            fitted = fit["fitted_config"]
            reproduced = simulate(fitted)["trace"]
            self.assertTrue(fit["accepted"])
            self.assertLess(fit["final_objective"], fit["initial_objective"] * 1e-5)
            self.assertAlmostEqual(fitted["components"][3]["D_ax_mm2_s"],
                                   truth["components"][3]["D_ax_mm2_s"], delta=0.01)
            self.assertAlmostEqual(fitted["conversion"]["uv_to_protein_mAU_L_g"],
                                   truth["conversion"]["uv_to_protein_mAU_L_g"], delta=0.01)
            np.testing.assert_allclose(
                [float(row["Predicted_Detector_Signal"]) for row in uv_rows],
                np.interp([float(row["CV"]) for row in uv_rows],
                          reproduced["CV"], reproduced["uv_mAU"]), rtol=1e-9, atol=1e-9)
            self.assertEqual(len(mass_rows), 28)
            self.assertEqual(len(conductivity_rows), len(cv))
            self.assertLess(fit["profile_metrics"][0]["conductivity_rmse_mS_cm"], 1e-8)
            self.assertLess(max(abs(float(row["Residual_mass_percent"]))
                                for row in mass_rows), 0.02)
            self.assertIn("UV chromatogram with collected fraction mass%", report_html)
            self.assertIn("Predicted impurity chromatogram after refinement", report_html)
            self.assertEqual(report_html.count('aria-label="Predicted impurity concentration chromatogram by CV"'), 1)
            self.assertIn("Mass% (right axis)", report_html)
            self.assertNotIn("Residual_DOF", parameter_rows[0])

    def test_tdm_cpa_four_species_fraction_profile_refines_wrong_parameter(self):
        truth = make_config(MODEL_TDM_CPA, 3, multistep=False)
        truth["numerics"].update(time_steps=60, axial_positions=6)
        truth["process"]["load_CV"] = 3.0
        truth["process"]["plw_count"] = 0
        truth["process"]["elution_steps"][0]["CV"] = 10.0
        truth["least_squares"]["objective"] = "NORMALIZED_MSE"
        starting = copy.deepcopy(truth)
        starting["components"][3]["D_ax_mm2_s"] *= 5.0
        true_trace = simulate(truth)["trace"]
        starting_trace = simulate(starting)["trace"]
        names = [row["name"] for row in truth["components"][:4]]
        cv = np.asarray(true_trace["CV"])
        concentrations = np.asarray(true_trace["component_g_L"])
        intervals = ((.35, .45), (.45, .55), (.55, .65), (.65, .75),
                     (.75, .85), (3.65, 3.75), (3.85, 3.95))
        fractions = []
        for index, (start, end) in enumerate(intervals, 1):
            predicted, _ = lsq._predicted_fraction_mass_percent(truth, true_trace, {
                "fraction_index": index, "start": start, "end": end,
                "x_unit": "CV", "CV": (start + end) / 2})
            fractions.append({
                "fraction_index": index, "start": start, "end": end,
                "x_unit": "CV",
                "mass_percent": {} if index == 4 else dict(zip(names, map(float, predicted))),
            })
        validated = lsq.validate_composition_references(starting, fractions)
        initial_mass_residuals, _ = lsq._composition_residuals(
            starting, validated, starting_trace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "measured.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["CV", "UV 280 [mAU]"])
                # UV already matches the wrong model. Movement toward truth
                # therefore has to come from the entered fraction mass% data.
                writer.writerows(zip(starting_trace["CV"], starting_trace["uv_mAU"]))
            outputs = ("FIT_RESULTS_CSV", "FIT_TRACE_CSV", "FIT_COMPOSITION_CSV",
                       "FIT_CONDUCTIVITY_CSV", "FIT_REPORT_HTML", "FIT_CONFIG_JSON",
                       "FIT_APPLY_JSL", "FIT_SETTINGS_JSON", "FIT_HISTORY_CSV",
                       "FIT_IMPROVED_CSV", "FIT_IMPROVED_TXT", "CONFIG_PATH")
            with patch.multiple(lsq, **{name: root / getattr(lsq, name).name
                                        for name in outputs}):
                fit = lsq.run_global_least_squares_refinement(
                    [{"config": starting, "chromatogram_csv": source,
                      "composition_references": fractions}],
                    mode=lsq.MODE_CHROMATOGRAM_AND_COMPOSITION,
                    unlocked_paths=["components[3].D_ax_mm2_s"], max_nfev=8)
                with lsq.FIT_COMPOSITION_CSV.open(newline="", encoding="utf-8") as stream:
                    fitted_rows = list(csv.DictReader(stream))
                report = lsq.FIT_REPORT_HTML.read_text()
            fitted_value = fit["fitted_config"]["components"][3]["D_ax_mm2_s"]
            true_value = truth["components"][3]["D_ax_mm2_s"]
            self.assertLess(abs(fitted_value - true_value), 0.15 * true_value)
            fitted_trace = simulate(fit["fitted_config"])["trace"]
            final_mass_residuals, _ = lsq._composition_residuals(
                fit["fitted_config"], validated, fitted_trace)
            self.assertLess(float(final_mass_residuals @ final_mass_residuals),
                            0.01 * float(initial_mass_residuals @ initial_mass_residuals))
            self.assertEqual(len(fitted_rows), 24)
            self.assertEqual({int(row["Fraction"]) for row in fitted_rows},
                             {1, 2, 3, 5, 6, 7})
            metrics = fit["composition_profile_metrics"]
            self.assertEqual({row["protein_species"] for row in metrics}, set(names))
            self.assertTrue(all(row["observed_fractions"] == 6 for row in metrics))
            impurity = next(row for row in metrics if row["protein_species"] == names[3])
            self.assertLess(impurity["final_rmse_percentage_points"],
                            0.1 * impurity["initial_rmse_percentage_points"])
            self.assertIn("Fraction mass% fit by protein species", report)

    def test_zero_feed_impurity_does_not_shift_later_fraction_columns(self):
        cfg = make_config(MODEL_EDM_LANGMUIR, 2, multistep=False)
        cfg["components"][0]["mass_percent"] = 60
        cfg["components"][1].update(name="Absent impurity", mass_percent=0)
        cfg["components"][2].update(name="Late impurity", mass_percent=40)
        with tempfile.TemporaryDirectory() as directory:
            rows = load_lsq_mass_rows(Path(directory) / "mass.csv")
            rows[0].update(Enabled="On", Fraction_Count="1", Unit="CV", Start="0.1",
                           End="0.2", Target_mass_percent="60", Impurity2_mass_percent="40")
            refs = tdm_bridge._composition_references_for_run(cfg, 1, rows)
            self.assertEqual(refs[0]["mass_percent"], {
                "Target protein": "60", "Late impurity": "40"})
            validated = lsq.validate_composition_references(cfg, refs)
            self.assertEqual(validated[0]["mass_percent"], {
                "Target protein": 60.0, "Late impurity": 40.0})
            with patch.dict(tdm_bridge.__dict__, {
                "tdm_ui_lsq_reference_count": 1, "tdm_ui_lsq_r1_start": 0.1,
                "tdm_ui_lsq_r1_end": 0.2, "tdm_ui_lsq_r1_c1_mass_percent": 60,
                "tdm_ui_lsq_r1_c2_mass_percent": None,
                "tdm_ui_lsq_r1_c3_mass_percent": 40,
            }):
                from_ui = tdm_bridge._composition_references_from_jmp(cfg)
            self.assertEqual(from_ui[0]["mass_percent"]["Late impurity"], 40)
            rows[0]["Impurity1_mass_percent"] = "10"
            with self.assertRaisesRegex(ValueError, "Absent impurity has zero feed mass"):
                tdm_bridge._composition_references_for_run(cfg, 1, rows)

    def test_duplicate_active_names_rejected_for_fraction_fit(self):
        cfg = make_config(MODEL_EDM_LANGMUIR, 1, multistep=False)
        cfg["components"][1]["name"] = cfg["components"][0]["name"]
        with self.assertRaisesRegex(ValueError, "names must be unique"):
            lsq.validate_composition_references(cfg, [{
                "fraction_index": 1, "start": 0.1, "end": 0.2,
                "mass_percent": {cfg["components"][0]["name"]: 50},
            }])

    def test_fraction_headers_follow_model_species_names_and_share_input_columns(self):
        cfg = make_config(MODEL_EDM_LANGMUIR, 5, multistep=False)
        cfg["components"][0]["name"] = "Product A"
        cfg["components"][1]["name"] = "Acidic variant"
        jsl = build_jsl(cfg)
        _validate_jsl(jsl)
        self.assertIn('c1Name = Text Edit Box("Product A", << Set Width(150), << Set Function', jsl)
        self.assertIn('c2Name = Text Edit Box("Acidic variant", << Set Width(150), << Set Function', jsl)
        for i in range(1, 7):
            self.assertIn(f'speciesName = Trim(Char(c{i}Name << Get Text));', jsl)
            self.assertIn(f'lsqMassHeaderC{i} << Set Text(speciesName || " mass%");', jsl)
            self.assertIn(f'lsqMassColumnC{i} = V List Box(lsqMassHeaderC{i} = Text Box', jsl)
            column = jsl.split(f'lsqMassColumnC{i} = V List Box(', 1)[1].split('\n)', 1)[0]
            self.assertIn(f'lsqR1C{i}Mass = Number Edit Box', column)
            self.assertIn(f'lsqR30C{i}Mass = Number Edit Box', column)
            self.assertLess(column.index(f'lsqMassHeaderC{i} = Text Box'),
                            column.index(f'lsqR1C{i}Mass = Number Edit Box'))
            field = "Target_mass_percent" if i == 1 else f"Impurity{i - 1}_mass_percent"
            self.assertIn(f'Column(dtLSQMass, "{field}")[massRow + 0] = lsqR1C{i}Mass << Get;', jsl)
        self.assertIn('lsqMassColumnC6 << Visibility(If(impurityCount >= 5, "Visible", "Collapse"))', jsl)
        with patch.dict(tdm_bridge.__dict__, {
            "tdm_ui_lsq_reference_count": 1, "tdm_ui_lsq_r1_start": 0.2,
            "tdm_ui_lsq_r1_end": 0.3, "tdm_ui_lsq_r1_c1_mass_percent": 70,
            "tdm_ui_lsq_r1_c2_mass_percent": 30,
        }):
            references = tdm_bridge._composition_references_from_jmp(cfg)
        self.assertEqual(references[0]["mass_percent"]["Product A"], 70)
        self.assertEqual(references[0]["mass_percent"]["Acidic variant"], 30)

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
            mass_rows = load_lsq_mass_rows(root / "mass.csv")
            mass_rows[0].update(Enabled="On", Fraction_Count="4", Unit="CV")
            ui = {"tdm_ui_action": "PREVIEW_LSQ_ALIGNMENT", "tdm_ui_lsq_target_run": 1,
                  "tdm_ui_lsq_run_count": 1, "tdm_ui_lsq_selected_run_1": 1,
                  "tdm_ui_lsq_mode": "Least-squares: chromatogram + species mass% by fraction"}
            for i, (lo, hi) in enumerate(((0.20, 0.22), (0.22, 0.24),
                                          (0.24, 0.26), (0.26, 0.28)), 1):
                mass_rows[i - 1].update(Start=str(lo), End=str(hi))
                if i != 3:
                    mass_rows[i - 1].update(Target_mass_percent="60", Impurity1_mass_percent="40")
            receipt = {"model_run_source": "INDEPENDENT_LSQ", "model_run_number": 1,
                       "model_run_name": "Run 1", "model_run_label": "Independent LSQ recipe Run 1"}
            with patch.dict(tdm_bridge.__dict__, ui), \
                 patch.object(tdm_bridge, "build_shared_config_from_jmp", return_value=cfg), \
                 patch.object(tdm_bridge, "_effective_run", return_value=cfg), \
                 patch.object(tdm_bridge, "_selected_lsq_profile_rows", return_value=[(1, row, str(uv))]), \
                 patch.object(tdm_bridge, "load_lsq_rows", return_value=[row]), \
                 patch.object(tdm_bridge, "load_lsq_mass_rows", return_value=mass_rows), \
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
        self.assertIn("Mass% [0–100]", svg)
        self.assertNotIn("log scale", svg)
        self.assertEqual(svg.count('stroke-width="2.3"'), 2)  # one connected segment per species
        self.assertIn("Blank: no fit or line", svg)
        self.assertIn('stroke-dasharray="1 6"', svg)
        self.assertNotIn("nan", svg.lower())

    def test_jmp_fraction_inputs_and_real_refinement(self):
        jsl = build_jsl()
        _validate_jsl(jsl)
        self.assertIn('Column(dtLSQMass, "Unit")[massRow] = lsqFractionUnit << Get Selected', jsl)
        self.assertIn('Column(dtLSQMass, "End")[massRow + 29] = lsqR30End << Get', jsl)

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
            observed, _ = lsq._predicted_fraction_mass_percent(true, trace, {
                "fraction_index": index, "start": lo, "end": hi,
                "x_unit": "CV", "CV": midpoint})
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
