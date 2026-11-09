"""A running LSQ fit picks up an externally edited evaluation cap."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import tdm_least_squares as lsq
from tdm_minimal_io import MODEL_EDM_LANGMUIR
from tdm_minimal_model import simulate
from verify_model import make_config


OUTPUTS = ("FIT_RESULTS_CSV", "FIT_TRACE_CSV", "FIT_COMPOSITION_CSV",
           "FIT_CONDUCTIVITY_CSV", "FIT_REPORT_HTML", "FIT_CONFIG_JSON",
           "FIT_APPLY_JSL", "FIT_SETTINGS_JSON", "FIT_HISTORY_CSV",
           "FIT_IMPROVED_CSV", "FIT_IMPROVED_TXT", "CONFIG_PATH")


class LiveEvaluationCapTests(unittest.TestCase):
    def fit_with_update(self, initial_cap, new_text, *, after_call=2, force_full=False):
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
            cap_file = root / "least_squares_evaluation_cap.txt"
            actual_objective = lsq._objective_parts
            calls = 0

            def change_after_first_trial(*args, **kwargs):
                nonlocal calls
                result = actual_objective(*args, **kwargs)
                calls += 1
                if calls == after_call:
                    cap_file.write_text(new_text + "\n", encoding="utf-8")
                return result

            patched = {key: root / getattr(lsq, key).name for key in OUTPUTS}
            with patch.multiple(lsq, **patched), patch.object(
                    lsq, "_objective_parts", side_effect=change_after_first_trial):
                result = lsq.run_global_least_squares_refinement(
                    [{"config": initial, "chromatogram_csv": source}],
                    unlocked_paths=["components[0].qmax_g_L"], max_nfev=initial_cap,
                    max_model_evaluations=initial_cap, live_evaluation_cap_path=cap_file,
                    force_full_evaluation_budget=force_full)
            settings = json.loads(patched["FIT_SETTINGS_JSON"].read_text())
            with patched["FIT_HISTORY_CSV"].open(newline="", encoding="utf-8") as fh:
                history = list(csv.DictReader(fh))
            active = (root / "least_squares_active_evaluation.txt").read_text()
            return result, settings, history, active

    def test_increasing_cap_allows_more_than_original_budget(self):
        result, settings, history, active = self.fit_with_update(3, "6")
        self.assertGreater(result["residual_evaluations"], 3)
        self.assertLessEqual(result["residual_evaluations"], 6)
        self.assertEqual(result["max_model_evaluations"], 6)
        self.assertEqual(settings["initial_model_evaluation_cap"], 3)
        self.assertEqual(settings["optimizer_candidate_limit"], 10000)
        self.assertEqual(settings["evaluation_cap_changes"][0]["effective_cap"], 6)
        self.assertEqual(len(history), result["residual_evaluations"] + 1)
        self.assertIn("Current evaluation cap: 6", active)

    def test_lowering_cap_stops_before_next_probe_and_keeps_final_check(self):
        result, settings, history, _ = self.fit_with_update(8, "3")
        self.assertEqual(result["residual_evaluations"], 3)
        self.assertTrue(result["evaluation_cap_reached"])
        self.assertEqual(result["max_model_evaluations"], 3)
        self.assertEqual(history[-1]["phase"], "final verification")
        self.assertEqual(settings["evaluation_cap_changes"][0]["effective_cap"], 3)

    def test_invalid_edit_keeps_prior_cap_and_is_reported(self):
        result, settings, _, _ = self.fit_with_update(3, "many")
        self.assertEqual(result["max_model_evaluations"], 3)
        self.assertTrue(any("Ignored invalid live cap" in x for x in settings["evaluation_cap_warnings"]))

    def test_cap_below_completed_work_preserves_one_final_evaluation(self):
        result, settings, _, _ = self.fit_with_update(8, "2", after_call=3)
        self.assertEqual(result["residual_evaluations"], 3)
        self.assertEqual(result["max_model_evaluations"], 3)
        self.assertTrue(any("effective cap is 3" in x for x in settings["evaluation_cap_warnings"]))

    def test_full_budget_mode_uses_live_increased_cap(self):
        result, settings, history, _ = self.fit_with_update(6, "8", force_full=True)
        self.assertEqual(result["residual_evaluations"], 8)
        self.assertEqual(result["max_model_evaluations"], 8)
        self.assertTrue(result["evaluation_cap_reached"])
        self.assertTrue(settings["force_full_evaluation_budget"])
        self.assertEqual(len(history), 9)


if __name__ == "__main__":
    unittest.main()
