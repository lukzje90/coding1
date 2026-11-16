"""Checks for the TDM + CPA equations in Briskot et al. (2020, 2021)."""

import math
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
from scipy.integrate import quad

import tdm_bridge
import tdm_least_squares as lsq
from tdm_jmp import build_jsl
from tdm_minimal_io import MODEL_LABELS, MODEL_TDM_CPA, required_parameter_paths, validate_config
from tdm_minimal_model import E_CHARGE, N_A, _build_cpa_components, _cpa_model, simulate
from verify_model import make_config


class CPAPaperEquationsTests(unittest.TestCase):
    def setUp(self):
        self.config = make_config(MODEL_TDM_CPA, 0, multistep=False)
        self.model = _cpa_model(self.config, _build_cpa_components(self.config, need_tdm_kinetics=True))

    def test_surface_charge_uses_surface_ph_and_grahame_balance(self):
        model = self.model
        pH, salt_M = 5.0, 0.05
        kappa = model.debye_kappa(salt_M * 1000.0)
        psi = model._adsorber_surface_potential(pH, kappa)
        surface_pH = pH + E_CHARGE * psi / (math.log(10.0) * model.kBT)
        ligand_sigma = -E_CHARGE * N_A * model.ligand_density_mol_m2 / (1.0 + 10.0 ** (model.ligand_pK - surface_pH))
        diffuse_sigma = model._grahame_scale(kappa) * math.sinh(E_CHARGE * psi / (2.0 * model.kBT))
        self.assertAlmostEqual(ligand_sigma / diffuse_sigma, 1.0, places=11)
        self.assertNotAlmostEqual(psi, model.surface_potential_linear(model.sigma_adsorber(pH), kappa))

    def test_energy_at_minimum_matches_integrated_plate_interaction(self):
        model, component = self.model, self.model.components[0]
        kappa = model.debye_kappa(0.05 * 1000.0)
        pa = model._adsorber_surface_potential(5.0, kappa)
        pp = model._protein_surface_potential(component.sigma(5.0), kappa)
        t = -2.0 * pa * pp / (pa * pa + pp * pp)
        minimum = -math.log(t) / kappa

        def plate_energy(z):
            x = math.exp(-kappa * z)
            return model.eps * kappa * (2.0 * pa * pp * x + (pa * pa + pp * pp) * x * x) / (1.0 - x * x)

        integral, _ = quad(lambda x: plate_energy(x / kappa) / kappa,
                           kappa * minimum, kappa * minimum + 80.0, epsabs=1e-14)
        expected = 2.0 * math.pi * component.radius_m * integral
        actual = model.u_adsorber(component, 5.0, 50.0)
        self.assertAlmostEqual(actual / expected, 1.0, places=7)

    def test_molecular_mass_is_required_and_independent_of_fitted_radius(self):
        config = self.config
        self.assertIn("components[0].molecular_weight_kDa", required_parameter_paths(config))
        self.assertIn("cpa.ligand_pK", required_parameter_paths(config))
        self.assertNotIn("cpa.system_specific_adsorption_parameter", required_parameter_paths(config))
        self.assertNotIn("components[0].accessible_particle_porosity", required_parameter_paths(config))
        self.assertIn("components[0].Z_lat", required_parameter_paths(config))
        config["components"][0]["molecular_weight_kDa"] = None
        self.assertTrue(any("molecular weight" in message for message in validate_config(config)))
        config["components"][0]["molecular_weight_kDa"] = 150.0
        first = _build_cpa_components(config, need_tdm_kinetics=True)[0]
        config["components"][0]["diameter_nm"] = 16.0
        second = _build_cpa_components(config, need_tdm_kinetics=True)[0]
        self.assertEqual(first.mw_kDa, second.mw_kDa)
        self.assertNotEqual(first.radius_m, second.radius_m)

    def test_adsorption_strength_responds_to_ionic_strength_and_not_old_multiplier(self):
        model = self.model
        q = np.zeros(1)
        low = model.equilibrium_factors(q, 5.0, 0.05)[0]
        high = model.equilibrium_factors(q, 5.0, 0.3)[0]
        model.system_scale = 1e6
        self.assertGreater(low, high)
        self.assertAlmostEqual(low, model.equilibrium_factors(q, 5.0, 0.05)[0])

    def test_jmp_and_bridge_transfer_fixed_molecular_weight(self):
        jsl = build_jsl(self.config)
        self.assertIn('c1MolecularWeight = Number Edit Box(150', jsl)
        self.assertIn('Python Name("tdm_ui_c1_molecular_weight_kDa")', jsl)
        self.assertIn('systemAdsorptionRow << Visibility', jsl)
        self.assertIn('ligandPKRow << Visibility', jsl)
        self.assertIn('Python Name("tdm_ui_ligand_pK")', jsl)
        self.assertIn('c1EpsPiRow << Visibility', jsl)
        self.assertIn('c1ZlatRow << Visibility', jsl)
        with patch.dict(tdm_bridge.__dict__, {
                "tdm_ui_model": MODEL_LABELS[MODEL_TDM_CPA],
                "tdm_ui_impurity_count": 0,
                "tdm_ui_c1_molecular_weight_kDa": 150.0,
                "tdm_ui_c1_Z_lat": 40.0,
                "tdm_ui_ligand_pK": 2.3,
        }):
            shared = tdm_bridge.build_shared_config_from_jmp()
        self.assertEqual(shared["components"][0]["molecular_weight_kDa"], 150.0)
        self.assertEqual(shared["components"][0]["Z_lat"], 40.0)
        self.assertEqual(shared["cpa"]["ligand_pK"], 2.3)

    def test_resin_ligand_pk_changes_regulated_adsorption(self):
        original = self.model.equilibrium_factors(np.zeros(1), 4.5, 0.05)[0]
        self.config["cpa"]["ligand_pK"] = 5.0
        changed = _cpa_model(self.config, _build_cpa_components(self.config, need_tdm_kinetics=True))
        self.assertLess(changed.equilibrium_factors(np.zeros(1), 4.5, 0.05)[0], original)

    def test_old_lsq_export_import_preserves_new_lateral_charge(self):
        current = self.config
        current["components"][0]["Z_lat"] = 40.0
        old_export = json.loads(json.dumps(current))
        old_export["components"][0].pop("Z_lat")
        old_export["components"][0]["Z_ref"] = 45.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "least_squares_fitted_inputs.json"
            path.write_text(json.dumps(old_export), encoding="utf-8")
            with patch.object(lsq, "_write_apply_jsl"):
                imported, changed = tdm_bridge.import_lsq_parameters(current, str(path))
        self.assertEqual(imported["components"][0]["Z_lat"], 40.0)
        self.assertEqual(imported["components"][0]["Z_ref"], 45.0)
        self.assertIn("components[0].Z_ref", changed)

    def test_paper_lateral_charge_is_independent_and_fittable(self):
        config = self.config
        config["components"][0]["Z_lat"] = 40.0
        model = _cpa_model(config, _build_cpa_components(config, need_tdm_kinetics=True))
        comp = model.components[0]
        ceiling = comp.As_m_inv / (math.pi * N_A * comp.radius_m ** 2)
        q = np.array([0.1 * ceiling])
        salt = 0.05
        kappa = model.debye_kappa(salt * 1000.0)
        surface_density = N_A * q[0] / comp.As_m_inv
        distance = math.sqrt(2.0 / (math.sqrt(3.0) * surface_density))
        beta = (E_CHARGE ** 2 * comp.Z_lat ** 2 * math.exp(2.0 * kappa * comp.radius_m)
                / (4.0 * math.pi * model.eps * (1.0 + kappa * comp.radius_m) ** 2))
        expected = (3.0 * math.sqrt(3.0) * distance * surface_density * math.exp(-kappa * distance)
                    * beta / (1.0 - math.exp(-3.0 * math.sqrt(3.0) * kappa * distance / (2.0 * math.pi))))
        self.assertAlmostEqual(model.lateral_interactions(q, salt * 1000.0)[0] / expected, 1.0)
        self.assertLess(model.equilibrium_factors(q, 5.0, salt)[0],
                        self.model.equilibrium_factors(q, 5.0, salt)[0])
        specs = lsq.fit_parameter_specs(config, unlocked_paths=["components[0].Z_lat"])
        self.assertEqual([item.path for item in specs], ["components[0].Z_lat"])

    def test_tdm_cpa_pore_balance_uses_resin_porosity(self):
        config = self.config
        config["numerics"].update(time_steps=50, axial_positions=6)
        baseline = simulate(config)["trace"]["uv_mAU"]
        # Legacy species porosity can remain in an old file; TDM + CPA uses
        # the single resin eps_p from the published pore balance instead.
        config["components"][0]["accessible_particle_porosity"] = 0.01
        changed = simulate(config)["trace"]["uv_mAU"]
        np.testing.assert_allclose(changed, baseline, rtol=0.0, atol=0.0)

    def test_nonzero_lateral_charge_simulation_conserves_mass(self):
        config = self.config
        config["components"][0]["Z_lat"] = 40.0
        config["numerics"].update(time_steps=50, axial_positions=6)
        output = simulate(config)
        self.assertLess(abs(output["mass_balance"]["Target protein"]["closure_error_percent_of_input"]), 0.1)
        self.assertLessEqual(output["simulation"]["diagnostics"]["maximum_surface_coverage_fraction"], 1.05)


if __name__ == "__main__":
    unittest.main()
