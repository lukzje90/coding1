"""Full-budget LSQ restarts keep the best fit and use the selected cap."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import tdm_bridge
import tdm_least_squares as lsq
from tdm_jmp import _validate_jsl, build_jsl
from tdm_minimal_io import MODEL_EDM_LANGMUIR, normalize_config
from tdm_minimal_model import simulate
from verify_model import make_config


OUTPUTS = ("FIT_RESULTS_CSV", "FIT_TRACE_CSV", "FIT_COMPOSITION_CSV",
           "FIT_CONDUCTIVITY_CSV", "FIT_REPORT_HTML", "FIT_CONFIG_JSON",
           "FIT_APPLY_JSL", "FIT_SETTINGS_JSON", "FIT_HISTORY_CSV",
           "FIT_IMPROVED_CSV", "FIT_IMPROVED_TXT", "CONFIG_PATH")


class FullBudgetTests(unittest.TestCase):
    def fit(self, force: bool, cap: int = 8):
        truth = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        truth["numerics"].update(time_steps=50, axial_positions=6)
        trace = simulate(truth)["trace"]
        starting = copy.deepcopy(truth)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "measured.csv"
            with source.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["CV", "UV_mAU"])
                writer.writerows(zip(trace["CV"], trace["uv_mAU"]))
            patched = {key: root / getattr(lsq, key).name for key in OUTPUTS}
            with patch.multiple(lsq, **patched):
                result = lsq.run_global_least_squares_refinement(
                    [{"config": starting, "chromatogram_csv": source}],
                    unlocked_paths=["components[0].qmax_g_L"],
                    max_nfev=cap, max_model_evaluations=cap,
                    force_full_evaluation_budget=force)
            settings = json.loads(patched["FIT_SETTINGS_JSON"].read_text())
            with patched["FIT_HISTORY_CSV"].open(newline="", encoding="utf-8") as fh:
                history = list(csv.DictReader(fh))
            return starting, result, settings, history

    def test_full_cap_continues_after_exact_local_solution(self):
        starting, result, settings, history = self.fit(True)
        self.assertEqual(result["residual_evaluations"], 8)
        self.assertTrue(result["evaluation_cap_reached"])
        self.assertTrue(result["force_full_evaluation_budget"])
        self.assertGreater(len(result["local_search_history"]), 1)
        self.assertEqual(result["fitted_config"]["components"][0]["qmax_g_L"],
                         starting["components"][0]["qmax_g_L"])
        self.assertEqual(len(history), 9)
        self.assertEqual(history[-1]["phase"], "final verification")
        self.assertGreater(max(int(row["restart"]) for row in history), 0)
        self.assertTrue(settings["force_full_evaluation_budget"])

    def test_previous_convergence_policy_stops_before_cap(self):
        _, result, settings, history = self.fit(False)
        self.assertLess(result["residual_evaluations"], 8)
        self.assertFalse(result["evaluation_cap_reached"])
        self.assertFalse(settings["force_full_evaluation_budget"])
        self.assertEqual(len(history), result["residual_evaluations"] + 1)

    def test_full_budget_retains_best_tested_objective(self):
        truth = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        truth["numerics"].update(time_steps=50, axial_positions=6)
        trace = simulate(truth)["trace"]
        initial = copy.deepcopy(truth)
        initial["components"][0]["qmax_g_L"] *= 1.8
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "measured.csv"
            with source.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["CV", "UV_mAU"])
                writer.writerows(zip(trace["CV"], trace["uv_mAU"]))
            patched = {key: root / getattr(lsq, key).name for key in OUTPUTS}
            with patch.multiple(lsq, **patched):
                result = lsq.run_global_least_squares_refinement(
                    [{"config": initial, "chromatogram_csv": source}],
                    unlocked_paths=["components[0].qmax_g_L"],
                    max_nfev=12, max_model_evaluations=12,
                    force_full_evaluation_budget=True)
            self.assertEqual(result["residual_evaluations"], 12)
            self.assertLessEqual(result["final_objective"], result["initial_objective"])
            with patched["FIT_HISTORY_CSV"].open(newline="", encoding="utf-8") as fh:
                history = list(csv.DictReader(fh))
            self.assertAlmostEqual(result["final_objective"],
                                   min(float(row["objective"]) for row in history), places=8)

    def test_jmp_switch_round_trips_as_explicit_fit_setting(self):
        config = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        self.assertFalse(config["least_squares"]["force_full_evaluation_budget"])
        config["least_squares"]["force_full_evaluation_budget"] = True
        self.assertTrue(normalize_config(config)["least_squares"]["force_full_evaluation_budget"])
        jsl = build_jsl(config)
        _validate_jsl(jsl)
        self.assertIn('Python Name("tdm_ui_lsq_full_budget")', jsl)
        self.assertIn('lsqFullBudget = Combo Box({"Stop on convergence", "Use full cap"}, << Set(2)', jsl)
        with patch.dict(tdm_bridge.__dict__, {"tdm_ui_lsq_full_budget": "Use full cap"}):
            shared = tdm_bridge.build_shared_config_from_jmp()
        self.assertTrue(shared["least_squares"]["force_full_evaluation_budget"])


if __name__ == "__main__":
    unittest.main()
