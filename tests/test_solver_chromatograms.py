"""Representative elution checks across the four column/isotherm combinations."""
import unittest

import numpy as np

from verify_model import make_config
from tdm_minimal_io import (
    MODEL_EDM_CPA, MODEL_EDM_LANGMUIR, MODEL_TDM_CPA, MODEL_TDM_LANGMUIR,
)
from tdm_minimal_model import simulate


class SolverChromatogramTests(unittest.TestCase):
    def test_salt_gradient_elutes_protein_and_closes_mass_balance(self):
        for model in (MODEL_EDM_LANGMUIR, MODEL_TDM_LANGMUIR, MODEL_EDM_CPA, MODEL_TDM_CPA):
            with self.subTest(model=model):
                config = make_config(model, 0, multistep=False)
                config["numerics"].update(time_steps=160, axial_positions=8)
                config["feed"]["load_density_mg_mL_resin"] = 0.2
                config["process"]["plw_count"] = 0
                elution = config["process"]["elution_steps"][0]
                elution.update(mode="LINEAR", CV=8.0, flow_mL_min=1.0)
                if "LANGMUIR" in model:
                    config["chromatography_mode"] = "HIC"
                    config["components"][0]["salt_sensitivity_per_M"] = 5.0
                    config["process"].update(load_start_percent_B=100.0, load_end_percent_B=100.0)
                    elution.update(start_percent_B=100.0, end_percent_B=0.0)
                else:
                    config["process"].update(load_start_percent_B=0.0, load_end_percent_B=0.0)
                    elution.update(start_percent_B=0.0, end_percent_B=100.0)
                result = simulate(config)
                trace = result["trace"]
                peak = np.asarray(trace["total_g_L"])
                self.assertTrue(np.isfinite(peak).all())
                self.assertGreater(float(peak.max()), 0.01)
                self.assertLess(float(trace["CV"][np.argmax(peak)]), float(trace["CV"][-1]))
                balance = result["mass_balance"]["Target protein"]
                self.assertAlmostEqual(balance["outlet_g"] / balance["input_g"], 1.0, delta=0.02)
                self.assertLess(abs(balance["closure_error_percent_of_input"]), 0.1)


if __name__ == "__main__":
    unittest.main()
