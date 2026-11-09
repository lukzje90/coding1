"""Operating-condition search uses the unchanged production simulator."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import tdm_bridge
from tdm_bridge import import_lsq_parameters
from tdm_method_optimisation import (_elution_start, _metrics, _pool_masses,
                                     _recipe, optimise_method)
from tdm_minimal_io import (MODEL_EDM_CPA, MODEL_EDM_LANGMUIR, MODEL_TDM_CPA,
                            MODEL_TDM_LANGMUIR, MODEL_TDM_WANG, save_config,
                            validate_config)
from tdm_minimal_model import simulate
from tdm_jmp import build_jsl, _validate_jsl
from verify_model import make_config


def bounds():
    return {"loading_g_L": [.32, .48], "flow_mL_min": [.8, 1.2],
            "wash_CV": [.03, .09], "gradient_start_B": [0, 5],
            "gradient_end_B": [56, 84], "gradient_CV": [.144, .216],
            "pool_start_CV": [0, .05], "pool_end_CV": [.12, .18]}


class MethodOptimisationTests(unittest.TestCase):
    def test_jmp_route_exposes_method_settings_without_applying_recipe(self):
        jsl = build_jsl(make_config(MODEL_EDM_LANGMUIR, 0, multistep=False))
        _validate_jsl(jsl)
        self.assertIn('Button Box("Optimise selected run method", SendAndRun("OPTIMISE_METHOD"))', jsl)
        self.assertIn('Python Name("tdm_ui_method_max_trial_minutes")', jsl)
        self.assertIn('Button Box("Import parameters from LSQ JSON", ImportLSQParameters())', jsl)
        self.assertIn('Save Text File(importStagingFile, Load Text File(importFile));', jsl)
        self.assertIn('Python Send(importStagingFile, Python Name("tdm_ui_import_parameter_path"));', jsl)
        self.assertLess(jsl.index('Save Text File(importStagingFile, Load Text File(importFile));'),
                        jsl.index('Python Send(importStagingFile, Python Name("tdm_ui_import_parameter_path"));'))
        self.assertIn('plw1EndBRow << Visibility(If(stepMode == "SET_POINT", "Collapse", "Visible"))', jsl)
        self.assertIn('actionText == "OPTIMISE_METHOD"),', jsl)

    def test_bridge_method_action_writes_report_and_preserves_selected_run(self):
        config = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        config["numerics"].update(time_steps=50, axial_positions=6)
        original = copy.deepcopy(config)
        sent = {"tdm_ui_action": "OPTIMISE_METHOD", "tdm_ui_run_number": 1,
                "tdm_ui_batch_count": 1, "tdm_ui_method_min_purity_percent": 1,
                "tdm_ui_method_max_pool_volume_mL": 1,
                "tdm_ui_method_max_evaluations": 2,
                "tdm_ui_method_max_trial_minutes": 1}
        for key, pair in bounds().items():
            sent[f"tdm_ui_method_{key}_min"] = pair[0]
            sent[f"tdm_ui_method_{key}_max"] = pair[1]
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            statuses = []
            with patch.dict(tdm_bridge.__dict__, sent), \
                 patch.object(tdm_bridge, "PROJECT_DIR", folder), \
                 patch.object(tdm_bridge, "CONFIG_PATH", folder / "tdm_inputs.json"), \
                 patch.object(tdm_bridge, "BATCH_COUNT_FILE", folder / "batch_count.txt"), \
                 patch.object(tdm_bridge, "build_shared_config_from_jmp", return_value=config), \
                 patch.object(tdm_bridge, "_effective_run", return_value=config), \
                 patch.object(tdm_bridge, "_restore_fitted_model_if_ui_stale", side_effect=lambda x: (x, False)), \
                 patch.object(tdm_bridge, "_selected_run_context", return_value={"run_number": 1, "run_name": "Method audit"}), \
                 patch.object(tdm_bridge, "_write_status", side_effect=statuses.append):
                self.assertEqual(tdm_bridge.main(), 0)
            self.assertIn("METHOD_OPTIMISATION_READY:", statuses[-1])
            self.assertTrue((folder / "latest_method_optimisation_report.html").is_file())
            self.assertEqual(config, original)

    def test_pool_mass_is_integral_of_concentration_times_volume_with_partial_cells(self):
        trace = {"CV": np.array([0., 1., 2.]),
                 "column_component_g_L": np.array([[0., 2.], [2., 2.], [4., 2.]])}
        np.testing.assert_allclose(_pool_masses(trace, .25, 1.75, 10), [.03, .03])

    def test_candidate_recipe_changes_only_operating_conditions_across_all_models(self):
        for model in (MODEL_EDM_CPA, MODEL_EDM_LANGMUIR, MODEL_TDM_CPA,
                      MODEL_TDM_LANGMUIR, MODEL_TDM_WANG):
            with self.subTest(model=model):
                config = make_config(model, 0, multistep=False)
                config["numerics"].update(time_steps=50, axial_positions=6)
                original = copy.deepcopy(config)
                values = dict(zip(bounds(), [.42, 1.1, .07, 1, 72, .19, 0, .17]))
                candidate = _recipe(config, values)
                self.assertEqual(config, original)
                self.assertEqual(candidate["components"], config["components"])
                self.assertEqual(candidate["tdm"], config["tdm"])
                self.assertEqual(candidate["cpa"], config["cpa"])
                self.assertEqual(candidate["edm"], config["edm"])
                self.assertEqual(validate_config(candidate, strict=True), [])
                result = simulate(candidate)
                metrics = _metrics(result, candidate, values,
                                   {"min_purity_percent": 1, "max_pool_volume_mL": 1})
                self.assertGreaterEqual(metrics["pool_volume_mL"], 0)
                self.assertEqual(len(metrics["masses_g"]), 1)
                self.assertGreater(metrics["run_time_min"], 0)
                self.assertGreater(metrics["pool_end_min"], metrics["pool_start_min"])

    def test_pool_uses_selected_percent_b_gradient_after_explicit_elution(self):
        config = make_config(MODEL_EDM_LANGMUIR, 0, multistep=True)
        config["numerics"].update(time_steps=50, axial_positions=6)
        process = config["process"]
        process["elution_count"] = 2
        process["elution_steps"][0]["chemistry_control"] = "ENDPOINT_CHEMISTRY"
        process["elution_steps"][1].update(mode="LINEAR", chemistry_control="BUFFER_B_PERCENT",
                                           start_percent_B=20, end_percent_B=80)
        values = dict(zip(bounds(), [.4, 1, .06, 25, 75, .12, 0, .1]))
        candidate = _recipe(config, values)
        self.assertEqual(candidate["process"]["elution_steps"][0], config["process"]["elution_steps"][0] | {"flow_mL_min": 1})
        self.assertEqual(candidate["process"]["elution_steps"][1]["start_percent_B"], 25)
        result = simulate(candidate)
        expected_start = result["simulation"]["program"].stage_start_CV[4]
        self.assertAlmostEqual(_elution_start(result, candidate), expected_start)
        metrics = _metrics(result, candidate, values,
                           {"min_purity_percent": 1, "max_pool_volume_mL": 1})
        self.assertAlmostEqual(metrics["pool_start_CV"], expected_start)

    def test_search_has_hard_cap_history_and_does_not_change_input(self):
        config = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        config["numerics"].update(time_steps=50, axial_positions=6)
        original = copy.deepcopy(config)
        with tempfile.TemporaryDirectory() as d:
            info = optimise_method(config, {"bounds": bounds(), "min_purity_percent": 1,
                                            "max_pool_volume_mL": 1,
                                            "max_impurity_percent": {},
                                            "max_evaluations": 3}, output_root=d)
            self.assertLessEqual(info["evaluations"], 3)
            self.assertEqual(len(Path(info["history_csv"]).read_text().splitlines()),
                             info["history_rows"] + 1)
            self.assertEqual(info["total_model_evaluations"],
                             info["evaluations"] + info["robustness_model_evaluations"])
            self.assertIn("global optimality is not established", info["guarantee"])
            self.assertTrue(Path(info["report_html"]).is_file())
            self.assertLessEqual(info["retained_full_simulations"], info["simulation_cache_limit"] + 1)
            self.assertGreater(info["pool_screening_candidates_added"], 0)
            self.assertGreaterEqual(info["pool"]["yield_percent"],
                                    info["starting_candidate"]["metrics"]["yield_percent"])
            self.assertIn(info["search_status"], {"LOCAL_SEED_CONVERGED_SEARCH_CAPPED",
                                                   "LOCAL_SEED_CONVERGED", "EVALUATION_CAP_REACHED",
                                                   "LOCAL_NOT_CONVERGED", "GLOBAL_EXPLORATION_ONLY"})
        self.assertEqual(config, original)

    def test_impossible_pool_geometry_is_rejected_before_simulation(self):
        config = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        invalid = bounds()
        invalid["pool_end_CV"] = [2, 3]
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "no collection window"):
                optimise_method(config, {"bounds": invalid, "min_purity_percent": 90,
                                         "max_pool_volume_mL": 1, "max_evaluations": 3},
                                output_root=d)
            self.assertFalse(list(Path(d).iterdir()))

    def test_no_feasible_candidate_is_reported_honestly(self):
        config = make_config(MODEL_EDM_LANGMUIR, 1, multistep=False)
        config["numerics"].update(time_steps=50, axial_positions=6)
        with tempfile.TemporaryDirectory() as d:
            info = optimise_method(config, {
                "bounds": bounds(), "min_purity_percent": 100,
                "max_pool_volume_mL": 1,
                "max_impurity_percent": {"Impurity 1": 0},
                "max_evaluations": 2,
            }, output_root=d)
            self.assertFalse(info["best_feasible"])
            self.assertIn("No feasible candidate found", info["guarantee"])
            self.assertFalse((Path(info["folder"]) / "best_recipe.json").exists())

    def test_pressure_limit_without_any_allowed_flow_fails_before_simulation(self):
        config = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "Every allowed flow violates"):
                optimise_method(config, {
                    "bounds": bounds(), "min_purity_percent": 1,
                    "max_pool_volume_mL": 1,
                    "max_pressure_bar": .01,
                    "reference_pressure_bar": 1,
                    "reference_flow_mL_min": 1,
                    "max_evaluations": 2,
                }, output_root=d)
            self.assertFalse(list(Path(d).iterdir()))

    def test_duplicate_species_names_cannot_merge_impurity_constraints(self):
        config = make_config(MODEL_EDM_LANGMUIR, 2, multistep=False)
        config["components"][2]["name"] = config["components"][1]["name"]
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "distinct nonblank name"):
                optimise_method(config, {"bounds": bounds(), "min_purity_percent": 90,
                                         "max_pool_volume_mL": 1, "max_evaluations": 3},
                                output_root=d)

    def test_import_uses_only_model_parameters_and_checks_species(self):
        current = make_config(MODEL_EDM_LANGMUIR, 1, multistep=False)
        imported = copy.deepcopy(current)
        imported["components"][0]["b_L_g"] *= 1.2
        imported["process"]["plw_steps"][0]["CV"] = 99
        imported["column"]["length_mm"] = 999
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "least_squares_fitted_inputs.json"
            save_config(imported, file)
            with patch("tdm_least_squares.FIT_APPLY_JSL", Path(d) / "apply.jsl"):
                result, changed = import_lsq_parameters(current, str(file))
            self.assertIn("components[0].b_L_g", changed)
            self.assertAlmostEqual(result["components"][0]["b_L_g"], imported["components"][0]["b_L_g"])
            self.assertEqual(result["process"], current["process"])
            self.assertEqual(result["column"], current["column"])
            self.assertTrue((Path(d) / "apply.jsl").is_file())
            imported["components"][1]["name"] = "Different impurity"
            save_config(imported, file)
            with self.assertRaisesRegex(ValueError, "different product/impurity names"):
                import_lsq_parameters(current, str(file))

    def test_import_accepts_parameter_summary_only_with_companion_json(self):
        current = make_config(MODEL_EDM_LANGMUIR, 1, multistep=False)
        fitted = copy.deepcopy(current)
        fitted["components"][0]["b_L_g"] *= 1.1
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            summary = folder / "least_squares_improved_parameters.csv"
            summary.write_text("Parameter,Initial,Fitted\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "companion least_squares_fitted_inputs.json"):
                import_lsq_parameters(current, str(summary))
            save_config(fitted, folder / "least_squares_fitted_inputs.json")
            with patch("tdm_least_squares.FIT_APPLY_JSL", folder / "apply.jsl"):
                result, changed = import_lsq_parameters(current, str(summary))
            self.assertIn("components[0].b_L_g", changed)
            self.assertAlmostEqual(result["components"][0]["b_L_g"], fitted["components"][0]["b_L_g"])
            self.assertEqual(result["process"], current["process"])


if __name__ == "__main__":
    unittest.main()
