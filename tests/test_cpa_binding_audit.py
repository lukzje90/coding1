"""Check the CPA adsorption law and stage-resolved bind-and-elute diagnostics."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import tdm_least_squares as lsq
from tdm_minimal_io import MODEL_TDM_CPA
from tdm_minimal_model import _build_cpa_components, _cpa_model, simulate, write_outputs
from verify_model import make_config


def binding_recipe(delta: float) -> dict:
    cfg = make_config(MODEL_TDM_CPA, 0, multistep=False)
    cfg["numerics"].update(time_steps=90, axial_positions=8)
    cfg["feed"]["load_density_mg_mL_resin"] = 6.0  # 3 CV at 2 mg/mL
    cfg["process"]["buffer_A"]["conductivity_mS_cm"] = 2.0  # 0.02 M via saved conversion
    cfg["process"]["plw_steps"][0]["CV"] = 1.0
    cfg["process"]["elution_steps"][0].update(CV=4.0, end_percent_B=100.0)
    cfg["components"][0]["delta_ref"] = delta
    return cfg


class CPABindingAuditTests(unittest.TestCase):
    def test_cpa_units_equilibrium_and_kinetics(self):
        cfg = binding_recipe(0.012)
        comp = _build_cpa_components(cfg, need_tdm_kinetics=True)[0]
        cpa = _cpa_model(cfg, [comp])
        self.assertAlmostEqual(comp.radius_m, cfg["components"][0]["diameter_nm"] * 0.5e-9)
        self.assertAlmostEqual(cpa.ligand_density_mol_m2,
                               cfg["cpa"]["ligand_surface_density_umol_m2"] * 1e-6)
        q0 = np.zeros(1)
        low_salt_K = cpa.equilibrium_factors(q0, 6.0, 0.02)[0]
        high_salt_K = cpa.equilibrium_factors(q0, 6.0, 0.5)[0]
        self.assertGreater(low_salt_K, high_salt_K * 100)
        self.assertGreater(cpa.kinetic_rate(np.ones(1), q0, 6.0, 0.02)[0], 0)
        concentration, bound = cpa.equilibrium_from_total(np.ones(1), 0.5, 6.0, 0.02)
        np.testing.assert_allclose(concentration + 0.5 * bound, [1.0], rtol=1e-8)
        np.testing.assert_allclose(bound,
                                   cpa.equilibrium_factors(bound, 6.0, 0.02) * concentration,
                                   rtol=1e-7, atol=1e-8)

    def test_bound_and_flow_through_cases_are_distinguished(self):
        strong_cfg = binding_recipe(0.012)
        weak_cfg = binding_recipe(1e-7)
        strong = simulate(strong_cfg)
        weak = simulate(weak_cfg)
        target_strong = strong["simulation"]["diagnostics"]["CPA_adsorption_audit"]["species"][0]
        target_weak = weak["simulation"]["diagnostics"]["CPA_adsorption_audit"]["species"][0]
        self.assertGreater(target_strong["end_load_bound_percent_of_load"], 90)
        self.assertGreater(target_strong["elution_outlet_percent_of_load"], 90)
        self.assertLess(target_strong["load_outlet_percent_of_load"], 1)
        self.assertLess(target_weak["end_load_bound_percent_of_load"], 1)
        self.assertGreater(target_weak["load_outlet_percent_of_load"], 50)
        self.assertGreater(target_strong["zero_coverage_partition_load"],
                           target_weak["zero_coverage_partition_load"] * 1000)
        for cfg, result in ((strong_cfg, strong), (weak_cfg, weak)):
            audit = result["simulation"]["diagnostics"]["CPA_adsorption_audit"]
            stage_outlet = sum(stage["outlet_g"][0] for stage in audit["stages"])
            loaded = audit["species"][0]["loaded_g"]
            final_inventory = audit["stages"][-1]["column_inventory_g"][0]
            self.assertAlmostEqual(stage_outlet + final_inventory, loaded,
                                   delta=loaded * 0.002)
            with tempfile.TemporaryDirectory() as directory:
                paths = write_outputs(cfg, result, Path(directory))
                self.assertIn("CPA binding and elution audit",
                              Path(paths["html"]).read_text(encoding="utf-8"))
        strong_peak = strong["parameter_receipt"]["target_peak"]
        weak_peak = weak["parameter_receipt"]["target_peak"]
        self.assertGreater(strong_peak["column_outlet_peak_CV"],
                           strong_peak["first_elution_start_CV"])
        self.assertLess(weak_peak["column_outlet_peak_CV"],
                        weak_peak["first_elution_start_CV"])
        self.assertTrue(any("before the first elution stage" in warning
                            for warning in weak["simulation"]["diagnostics"]["warnings"]))

    def test_lsq_reports_binding_state_for_weak_initial_model(self):
        truth = binding_recipe(0.012)
        initial = binding_recipe(0.006)
        measured = simulate(truth)["trace"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "reference.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["CV", "UV [mAU]"])
                writer.writerows(zip(measured["CV"], measured["uv_mAU"]))
            output_names = ("FIT_RESULTS_CSV", "FIT_TRACE_CSV", "FIT_COMPOSITION_CSV",
                            "FIT_CONDUCTIVITY_CSV", "FIT_REPORT_HTML", "FIT_CONFIG_JSON",
                            "FIT_APPLY_JSL", "FIT_SETTINGS_JSON", "FIT_HISTORY_CSV",
                            "FIT_IMPROVED_CSV", "FIT_IMPROVED_TXT", "CONFIG_PATH")
            with patch.multiple(lsq, **{name: root / getattr(lsq, name).name
                                        for name in output_names}):
                fit = lsq.run_global_least_squares_refinement(
                    [{"config": initial, "chromatogram_csv": source}],
                    unlocked_paths=["components[0].delta_ref"],
                    max_nfev=8, max_model_evaluations=14)
                report = lsq.FIT_REPORT_HTML.read_text(encoding="utf-8")
            self.assertLess(fit["final_objective"], fit["initial_objective"])
            self.assertIn("CPA binding and elution audit", report)
            self.assertIn("Initial bound after load %", report)
            self.assertGreater(fit["fitted_config"]["components"][0]["delta_ref"],
                               initial["components"][0]["delta_ref"])

    def test_lsq_flags_a_flow_through_candidate_even_if_objective_is_accepted(self):
        truth = binding_recipe(0.012)
        weak = binding_recipe(1e-7)
        measured = simulate(truth)["trace"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "reference.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["CV", "UV [mAU]"])
                writer.writerows(zip(measured["CV"], measured["uv_mAU"]))
            output_names = ("FIT_RESULTS_CSV", "FIT_TRACE_CSV", "FIT_COMPOSITION_CSV",
                            "FIT_CONDUCTIVITY_CSV", "FIT_REPORT_HTML", "FIT_CONFIG_JSON",
                            "FIT_APPLY_JSL", "FIT_SETTINGS_JSON", "FIT_HISTORY_CSV",
                            "FIT_IMPROVED_CSV", "FIT_IMPROVED_TXT", "CONFIG_PATH")
            with patch.multiple(lsq, **{name: root / getattr(lsq, name).name
                                        for name in output_names}):
                fit = lsq.run_global_least_squares_refinement(
                    [{"config": weak, "chromatogram_csv": source}],
                    unlocked_paths=[], max_nfev=2, max_model_evaluations=2)
                report = lsq.FIT_REPORT_HTML.read_text(encoding="utf-8")
            self.assertTrue(fit["accepted"])
            self.assertIn("do not predict convincing target binding and elution",
                          fit["fit_quality_warning"])
            self.assertIn("Binding warning", report)


if __name__ == "__main__":
    unittest.main()
