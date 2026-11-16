"""Conservation and load sensitivity of the TDM + CPA counter-ion balance."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tdm_minimal_model import _build_cpa_components, _cpa_released_ion_pairs_M, simulate, write_outputs
from tests.test_cpa_binding_audit import binding_recipe


class CPACounterionExchangeTests(unittest.TestCase):
    def recipe(self, load_density=6.0):
        config = binding_recipe(0.012)
        config["numerics"].update(time_steps=50, axial_positions=6)
        config["feed"]["load_density_mg_mL_resin"] = load_density
        return config

    def test_projected_charge_and_ph_dependence_match_paper_equation(self):
        config = self.recipe()
        config["components"][0].update(pH_ref=6.0, Z_ref=40.0,
                                        Z1_per_pH=-10.0, Z2_per_pH2=0.0,
                                        delta_pH_slope_m2_C=0.0)
        component = _build_cpa_components(config, need_tdm_kinetics=True)[0]
        qv = np.array([[0.1], [0.1]])  # mol/m³ stationary solid
        pore_pH = np.array([6.0, 7.0])
        released = _cpa_released_ion_pairs_M(qv, pore_pH, [component], 0.5)
        np.testing.assert_allclose(released, [0.001, 0.00075], rtol=1e-12)

    def test_zero_protein_charge_leaves_buffer_profile_unchanged(self):
        config = self.recipe()
        config["components"][0]["Z_ref"] = 0.0
        result = simulate(config)
        baseline = result["simulation"]["mobile_phase"].baseline.outlet(result["trace"]["t_s"])
        np.testing.assert_allclose(result["trace"]["column_conductivity_mS_cm"],
                                   baseline[:, 3], atol=1e-9)
        self.assertEqual(result["simulation"]["diagnostics"]["counterion_exchange"]
                         ["cumulative_outlet_excess_umol"], 0.0)

    def test_loading_changes_ionic_strength_and_conductivity_with_ion_closure(self):
        low = simulate(self.recipe(load_density=0.5))
        high_config = self.recipe(load_density=6.0)
        high = simulate(high_config)

        def conductivity_shift(result):
            trace = result["trace"]
            baseline = result["simulation"]["mobile_phase"].baseline.outlet(trace["t_s"])
            factor = 0.01  # M/(mS/cm), set in binding_recipe
            np.testing.assert_allclose(trace["column_conductivity_mS_cm"] - baseline[:, 3],
                                       (trace["ionic_strength_M"] - baseline[:, 2]) / factor,
                                       atol=1e-8)
            return float(np.max(np.abs(trace["column_conductivity_mS_cm"] - baseline[:, 3])))

        self.assertGreater(conductivity_shift(high), conductivity_shift(low) * 5)
        self.assertGreater(conductivity_shift(high), 0.05)
        excess = np.asarray(high["trace"]["counterion_pair_excess_M"])
        self.assertGreater(float(np.max(excess)), 0.0)  # Release on binding.
        self.assertLess(float(np.min(excess)), 0.0)     # Uptake on elution.
        counterions = high["simulation"]["diagnostics"]["counterion_exchange"]
        self.assertLess(abs(counterions["closure_error_umol"]), 0.001)
        self.assertGreaterEqual(counterions["minimum_pore_ionic_strength_M"], 0.0)
        self.assertLess(abs(high["mass_balance"]["Target protein"]
                            ["closure_error_percent_of_input"]), 0.1)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_outputs(high_config, high, output_dir=directory)
            with Path(paths["csv"]).open(newline="", encoding="utf-8") as stream:
                self.assertIn("counterion_pair_excess_M", next(csv.reader(stream)))
            manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
            self.assertAlmostEqual(manifest["counterion_exchange"]["closure_error_umol"],
                                   counterions["closure_error_umol"])

    def test_ph_change_updates_bound_counterions_without_losing_ion_balance(self):
        config = self.recipe()
        config["process"]["buffer_A"]["pH"] = 5.5
        config["process"]["buffer_B"]["pH"] = 7.0
        config["components"][0].update(pH_ref=6.0, Z1_per_pH=-10.0,
                                        Z2_per_pH2=0.0, Z3_per_pH3=0.0,
                                        delta_pH_slope_m2_C=0.0)
        result = simulate(config)
        balance = result["simulation"]["diagnostics"]["counterion_exchange"]
        self.assertLess(abs(balance["closure_error_umol"]), 0.001)
        self.assertGreaterEqual(balance["minimum_pore_ionic_strength_M"], 0.0)


if __name__ == "__main__":
    unittest.main()
