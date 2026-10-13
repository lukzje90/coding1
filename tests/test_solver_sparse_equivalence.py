"""Sparse BDF Jacobians must preserve the chromatography calculations."""

import unittest
from unittest.mock import patch

import numpy as np

import tdm_minimal_model as model
from tdm_minimal_io import MODEL_EDM_LANGMUIR, MODEL_TDM_CPA, MODEL_TDM_LANGMUIR
from verify_model import make_config


class SparseSolverEquivalenceTests(unittest.TestCase):
    def test_two_species_elution_matches_dense_bdf(self):
        actual_solve_ivp = model.solve_ivp

        def without_sparsity(*args, **kwargs):
            kwargs.pop("jac_sparsity", None)
            return actual_solve_ivp(*args, **kwargs)

        for choice in (MODEL_TDM_CPA, MODEL_TDM_LANGMUIR, MODEL_EDM_LANGMUIR):
            with self.subTest(model=choice):
                config = make_config(choice, 1, multistep=False)
                # EDM keeps its original dense solve on small grids; exercise
                # its sparse path at the 50-cell size where it saves work.
                cells = 50 if choice == MODEL_EDM_LANGMUIR else 8
                config["numerics"].update(time_steps=50, axial_positions=cells)
                config["feed"]["load_density_mg_mL_resin"] = 0.2
                config["process"]["plw_count"] = 0
                elution = config["process"]["elution_steps"][0]
                elution.update(mode="LINEAR", CV=8.0, flow_mL_min=1.0)
                if choice in (MODEL_TDM_LANGMUIR, MODEL_EDM_LANGMUIR):
                    config["process"].update(load_start_percent_B=100.0,
                                             load_end_percent_B=100.0)
                    elution.update(start_percent_B=100.0, end_percent_B=0.0)
                else:
                    config["process"].update(load_start_percent_B=0.0,
                                             load_end_percent_B=0.0)
                    elution.update(start_percent_B=0.0, end_percent_B=100.0)

                optimized = model.simulate(config)
                with patch.object(model, "solve_ivp", side_effect=without_sparsity):
                    original = model.simulate(config)

                new_trace, old_trace = optimized["trace"], original["trace"]
                np.testing.assert_array_equal(new_trace["CV"], old_trace["CV"])
                for key in ("component_g_L", "total_g_L", "uv_mAU"):
                    new_values, old_values = np.asarray(new_trace[key]), np.asarray(old_trace[key])
                    self.assertTrue(np.isfinite(new_values).all())
                    self.assertGreater(float(np.max(old_values)), 0.0)
                    peak = float(np.max(np.abs(old_values)))
                    self.assertLessEqual(float(np.max(np.abs(new_values - old_values))),
                                         max(1e-6, peak * 2e-5))
                for species, old_balance in original["mass_balance"].items():
                    new_balance = optimized["mass_balance"][species]
                    self.assertAlmostEqual(new_balance["input_g"], old_balance["input_g"], places=12)
                    self.assertLess(abs(new_balance["closure_error_percent_of_input"]
                                        - old_balance["closure_error_percent_of_input"]), 0.01)


if __name__ == "__main__":
    unittest.main()
