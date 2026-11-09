"""Multi-stage method search, ranking and process-metric regressions."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tdm_method_optimisation_extended import (
    OBJECTIVES, _metrics, _optimise_pool_window, _peak_resolution, _recipe,
    optimise_method, variable_names)
from tdm_minimal_io import (MODEL_EDM_CPA, MODEL_EDM_LANGMUIR, MODEL_TDM_CPA,
                            MODEL_TDM_LANGMUIR, MODEL_TDM_WANG, validate_config)
from tdm_minimal_model import simulate
from tdm_jmp import _validate_jsl, build_jsl
from verify_model import make_config


def setup(washes=1, elutions=1, style="GRADIENT", objective="RECOVERY"):
    config = make_config(MODEL_EDM_LANGMUIR, 1, multistep=False)
    config["numerics"].update(time_steps=50, axial_positions=6)
    bounds = {"loading_g_L": [.32, .48], "flow_mL_min": [.8, 1.2],
              "pool_start_CV": [0, .05], "pool_end_CV": [.12, .18]}
    for i in range(1, washes+1):
        bounds[f"plw{i}_CV"] = [.03, .09]
        bounds[f"plw{i}_B"] = [0, 20]
    for i in range(1, elutions+1):
        bounds[f"el{i}_CV"] = [.14, .22]
        if style == "GRADIENT" or (style == "BOTH" and i == 1):
            bounds[f"el{i}_start_B"] = [0, 5]
            bounds[f"el{i}_end_B"] = [56, 84]
        else:
            bounds[f"el{i}_B"] = [50, 90]
    options = {"plw_count": washes, "elution_count": elutions,
               "elution_style": style, "objective": objective, "bounds": bounds,
               "min_purity_percent": 1, "min_recovery_percent": 0,
               "min_product_mass_g": 0, "max_pool_volume_mL": 1,
               "max_evaluations": 2, "max_trial_minutes": 1}
    return config, options


class ExtendedMethodOptimisationTests(unittest.TestCase):
    def test_every_goal_has_a_distinct_reported_metric(self):
        self.assertEqual(len(OBJECTIVES), len({v[0] for v in OBJECTIVES.values()}))
        self.assertIn("PEAK_SEPARATION", OBJECTIVES)
        self.assertIn("PURITY", OBJECTIVES)
        self.assertIn("PRODUCTIVITY", OBJECTIVES)

    def test_peak_resolution_uses_worst_impurity_and_zero_for_unresolved(self):
        x = np.linspace(0, 10, 1001)
        g = lambda center: np.exp(-.5*((x-center)/.5)**2)
        resolution, detail = _peak_resolution(x, np.column_stack([g(4), g(6), g(5)]), 0)
        self.assertGreater(detail[0]["resolution"], detail[1]["resolution"])
        self.assertAlmostEqual(resolution, detail[1]["resolution"])
        self.assertEqual(_peak_resolution(x, np.column_stack([g(4), np.zeros_like(x)]), 0)[0], 0)

    def test_continuous_pool_cuts_improve_coarse_screen_without_another_solve(self):
        config, options = setup(0, 1)
        options.update(min_purity_percent=95, max_pool_volume_mL=2)
        values = {key: sum(pair)/2 for key, pair in options["bounds"].items()}
        candidate = _recipe(config, values, 0, 1, "GRADIENT")
        result = simulate(candidate)
        elution_start = float(result["simulation"]["program"].stage_start_CV[1])
        x = np.linspace(0, elution_start + values["el1_CV"], 1201)
        position = x - elution_start
        product = np.exp(-.5*((position-.105)/.03)**2)
        impurity = .8*np.exp(-.5*((position-.16)/.02)**2)
        result["trace"]["CV"] = x
        result["trace"]["column_component_g_L"] = np.column_stack([product, impurity])
        start_bounds, end_bounds = (0, .03), (.08, .18)
        screened = []
        for start in np.linspace(*start_bounds, 5):
            for end in np.linspace(*end_bounds, 5):
                if end - start <= 1e-4:
                    continue
                measured = _metrics(result, candidate, dict(values, pool_start_CV=start,
                                                           pool_end_CV=end), options)
                if measured["feasible"]:
                    screened.append(measured["yield_percent"])
        optimum = _optimise_pool_window(result, candidate, values, options,
                                        start_bounds, end_bounds, "yield_percent", grid_size=5)
        self.assertIsNotNone(optimum)
        cuts, measured = optimum
        self.assertTrue(measured["feasible"])
        self.assertGreater(measured["yield_percent"], max(screened) + .01)
        self.assertGreaterEqual(measured["purity_percent"], 95 - 1e-6)
        self.assertLess(cuts["pool_end_CV"], .18)

    def test_chosen_structure_builds_valid_candidate_without_changing_source(self):
        for washes, elutions, style in ((0, 1, "ISOCRATIC"), (2, 2, "BOTH"),
                                        (5, 5, "GRADIENT")):
            with self.subTest(washes=washes, elutions=elutions, style=style):
                config, options = setup(washes, elutions, style)
                original = copy.deepcopy(config)
                values = {key: sum(pair)/2 for key, pair in options["bounds"].items()}
                candidate = _recipe(config, values, washes, elutions, style)
                self.assertEqual(config, original)
                self.assertEqual(candidate["process"]["plw_count"], washes)
                self.assertEqual(candidate["process"]["elution_count"], elutions)
                modes = [s["mode"] for s in candidate["process"]["elution_steps"][:elutions]]
                expected = (["LINEAR"]*elutions if style == "GRADIENT" else
                            ["SET_POINT"]*elutions if style == "ISOCRATIC" else
                            ["LINEAR"]+["SET_POINT"]*(elutions-1))
                self.assertEqual(modes, expected)
                self.assertTrue(all(s["mode"] == "SET_POINT" for s in
                                    candidate["process"]["plw_steps"][:washes]))
                self.assertFalse(validate_config(candidate, strict=True))
                result = simulate(candidate)
                self.assertEqual(len(result["simulation"]["program"].stages), 1+washes+elutions)

    def test_multi_stage_recipe_preserves_every_model_family_parameters(self):
        for model in (MODEL_EDM_CPA, MODEL_EDM_LANGMUIR, MODEL_TDM_CPA,
                      MODEL_TDM_LANGMUIR, MODEL_TDM_WANG):
            with self.subTest(model=model):
                _, options = setup(2, 2, "BOTH")
                # Use the complete valid model-specific recipe and buffer-ion inputs.
                model_config = make_config(model, 1, multistep=False)
                values = {key: sum(pair)/2 for key, pair in options["bounds"].items()}
                candidate = _recipe(model_config, values, 2, 2, "BOTH")
                self.assertEqual(candidate["components"], model_config["components"])
                self.assertEqual(candidate["tdm"], model_config["tdm"])
                self.assertEqual(candidate["cpa"], model_config["cpa"])
                self.assertEqual(candidate["edm"], model_config["edm"])
                self.assertFalse(validate_config(candidate, strict=True))

    def test_end_to_end_isocratic_and_mixed_search_and_histories(self):
        for washes, elutions, style, goal in ((0, 1, "ISOCRATIC", "PURITY"),
                                              (2, 2, "BOTH", "PEAK_SEPARATION")):
            with self.subTest(style=style), tempfile.TemporaryDirectory() as directory:
                config, options = setup(washes, elutions, style, goal)
                original = copy.deepcopy(config)
                info = optimise_method(config, options, output_root=directory)
                self.assertEqual(config, original)
                self.assertEqual(info["evaluations"], 2)
                self.assertEqual(info["method_structure"]["elution_style"], style)
                self.assertEqual(info["objective"], goal)
                self.assertTrue(Path(info["report_html"]).is_file())
                with Path(info["history_csv"]).open(encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh))
                self.assertEqual(len(rows), info["history_rows"])
                self.assertEqual(sum(r["phase"] == "robustness" for r in rows),
                                 12 if info["best_feasible"] else 0)
                if info["best_feasible"]:
                    self.assertAlmostEqual(info["objective_value"],
                                           info["pool"][OBJECTIVES[goal][0]])
                    recipe = json.loads((Path(info["folder"])/"best_recipe.json").read_text())
                    self.assertEqual(recipe["process"]["elution_count"], elutions)
                    tradeoff = info["fixed_recipe_pool_tradeoff"]
                    self.assertTrue((Path(info["folder"])/"fixed_recipe_pool_tradeoff.csv").is_file())
                    self.assertEqual(tradeoff[0]["minimum_purity_percent"],
                                     options["min_purity_percent"])
                    self.assertTrue(tradeoff[0]["feasible"])
                    self.assertGreaterEqual(tradeoff[0]["achieved_purity_percent"],
                                            options["min_purity_percent"] - 1e-6)
                    self.assertEqual(info["total_model_evaluations"],
                                     info["evaluations"] + info["robustness_model_evaluations"])
                    self.assertEqual(info["robustness_summary"]["tested_variations"], 12)

    def test_ui_exposes_goals_stage_counts_styles_and_bounds(self):
        jsl = build_jsl(setup()[0])
        _validate_jsl(jsl)
        for text in ('"PEAK_SEPARATION"', '"PURITY"', '"BOTH"',
                     'Python Name("tdm_ui_method_plw_count")',
                     'Python Name("tdm_ui_method_elution_style")',
                     'method_el5_end_B_min', 'UpdateMethodStepVisibility'):
            self.assertIn(text, jsl)

    def test_invalid_mixed_single_elution_fails_before_creating_output(self):
        config, options = setup(0, 1, "BOTH")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "at least two elution"):
                optimise_method(config, options, output_root=directory)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
