"""A pathological TDM-CPA LSQ probe must not occupy the solver indefinitely."""
import csv
import math
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import tdm_bridge
import tdm_least_squares as lsq
import tdm_minimal_model as model
from tdm_jmp import build_jsl
from tdm_minimal_io import MODEL_TDM_CPA, validate_config
from verify_model import make_config


class TrialTimeoutTests(unittest.TestCase):
    def test_jmp_timeout_is_adjustable_and_validated(self):
        jsl = build_jsl()
        self.assertIn('Python Send(lsqMaxTrialMinutes << Get, Python Name("tdm_ui_lsq_max_trial_minutes"));', jsl)
        with patch.dict(tdm_bridge.__dict__, {"tdm_ui_lsq_max_trial_minutes": 25.0}):
            cfg = tdm_bridge.build_shared_config_from_jmp()
        self.assertEqual(cfg["least_squares"]["max_trial_minutes"], 25.0)
        complete = make_config(MODEL_TDM_CPA, 0, multistep=False)
        complete["least_squares"]["max_trial_minutes"] = 25.0
        self.assertFalse(validate_config(complete, strict=True))
        for invalid in (-1, 241, None):
            with self.subTest(invalid=invalid), patch.dict(
                    tdm_bridge.__dict__, {"tdm_ui_lsq_max_trial_minutes": invalid}):
                with self.assertRaisesRegex(ValueError, "Trial timeout must be"):
                    tdm_bridge.build_shared_config_from_jmp()

    def test_deadline_interrupts_both_tdm_cpa_solver_stages_only_when_requested(self):
        cfg = make_config(MODEL_TDM_CPA, 0, multistep=False)
        cfg["numerics"].update(time_steps=50, axial_positions=6)
        expired = time.perf_counter() - 1.0
        with self.assertRaisesRegex(TimeoutError, "buffer transport"):
            model.simulate(cfg, solver_deadline=expired)
        actual_transport = model.MobilePhaseTransport

        def transport_without_deadline(*args, **kwargs):
            kwargs["solver_deadline"] = None
            return actual_transport(*args, **kwargs)

        with patch.object(model, "MobilePhaseTransport", side_effect=transport_without_deadline):
            with self.assertRaisesRegex(TimeoutError, "protein integration"):
                model.simulate(cfg, solver_deadline=expired)
        ordinary = model.simulate(cfg)["trace"]
        with_future_deadline = model.simulate(
            cfg, solver_deadline=time.perf_counter() + 60.0)["trace"]
        self.assertEqual(len(ordinary["CV"]), 51)
        np.testing.assert_array_equal(ordinary["uv_mAU"], with_future_deadline["uv_mAU"])
        np.testing.assert_array_equal(ordinary["CV"], with_future_deadline["CV"])

    def test_timeout_probe_uses_one_reverse_solve_or_names_blocked_parameter(self):
        invalid = set()
        timed_out = set()
        probes = []

        def residual(vector):
            key = np.asarray(vector, dtype=float).tobytes()
            probes.append(float(vector[0]))
            if vector[0] > 0.5:
                invalid.add(key)
                timed_out.add(key)
                return np.array([1e6])
            return np.array([vector[0] ** 2])

        kwargs = {
            "trial_valid": lambda v: np.asarray(v, dtype=float).tobytes() not in invalid,
            "trial_timed_out": lambda v: np.asarray(v, dtype=float).tobytes() in timed_out,
            "parameter_labels": ["Target CPA radius"],
        }
        jac = lsq._bounded_jacobian(residual, np.zeros(1), np.ones(1), **kwargs)(np.array([0.5]))
        self.assertAlmostEqual(jac[0, 0], 1.0, delta=0.002)
        self.assertEqual(len(probes), 3)  # base, timed-out forward, valid reverse

        def both_directions_timeout(vector):
            key = np.asarray(vector, dtype=float).tobytes()
            if vector[0] != 0.5:
                invalid.add(key)
                timed_out.add(key)
                return np.array([1e6])
            return np.array([0.25])

        with self.assertRaisesRegex(TimeoutError, "Target CPA radius"):
            lsq._bounded_jacobian(both_directions_timeout, np.zeros(1), np.ones(1),
                                  **kwargs)(np.array([0.5]))

    def test_global_fit_records_deadline_and_current_probe(self):
        cfg = make_config(MODEL_TDM_CPA, 0, multistep=False)
        cfg["numerics"].update(time_steps=50, axial_positions=6)
        cfg["least_squares"]["max_trial_minutes"] = 5.0
        trace = model.simulate(cfg)["trace"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "measured.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["CV", "UV 280 [mAU]"])
                writer.writerows(zip(trace["CV"], trace["uv_mAU"]))
            outputs = ("FIT_RESULTS_CSV", "FIT_TRACE_CSV", "FIT_COMPOSITION_CSV",
                       "FIT_CONDUCTIVITY_CSV", "FIT_REPORT_HTML", "FIT_CONFIG_JSON",
                       "FIT_APPLY_JSL", "FIT_SETTINGS_JSON", "FIT_HISTORY_CSV",
                       "FIT_IMPROVED_CSV", "FIT_IMPROVED_TXT", "CONFIG_PATH")
            observed_deadlines = []
            original = lsq._objective_parts

            def observe(*args, **kwargs):
                observed_deadlines.append(kwargs.get("solver_deadline"))
                return original(*args, **kwargs)

            with patch.multiple(lsq, **{name: root / getattr(lsq, name).name for name in outputs}), \
                    patch.object(lsq, "_objective_parts", side_effect=observe):
                lsq.run_global_least_squares_refinement(
                    [{"config": cfg, "chromatogram_csv": source}],
                    unlocked_paths=[], max_nfev=2, max_model_evaluations=2)
            self.assertTrue(observed_deadlines)
            self.assertTrue(all(value is not None and math.isfinite(value)
                                for value in observed_deadlines))
            self.assertIn("fit completed", (root / "least_squares_active_evaluation.txt")
                          .read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
