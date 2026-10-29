"""Check the CPA adsorption law and stage-resolved bind-and-elute diagnostics."""
import csv
import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import tdm_least_squares as lsq
from tdm_minimal_io import MODEL_TDM_CPA
from tdm_minimal_model import _CoverageGuardedBDF, _build_cpa_components, _cpa_model, simulate, write_outputs
from verify_model import make_config


def binding_recipe(delta: float) -> dict:
    cfg = make_config(MODEL_TDM_CPA, 0, multistep=False)
    cfg["numerics"].update(time_steps=90, axial_positions=8)
    cfg["feed"]["load_density_mg_mL_resin"] = 6.0  # 3 CV at 2 mg/mL
    cfg["process"]["buffer_A"]["conductivity_mS_cm"] = 2.0
    for ion in cfg["process"]["buffer_A"]["ions"][:2]:
        ion["concentration_M"] = 0.02  # Match the former 0.02 M 1:1 salt proxy.
    cfg["process"]["plw_steps"][0]["CV"] = 1.0
    cfg["process"]["elution_steps"][0].update(CV=4.0, end_percent_B=100.0)
    cfg["components"][0]["delta_ref"] = delta
    return cfg


class CPABindingAuditTests(unittest.TestCase):
    def test_single_run_lsq_does_not_differentiate_cpa_penalty(self):
        cfg = binding_recipe(0.012)
        initial_delta = cfg["components"][0]["delta_ref"]
        evaluated = []

        def objective(candidate, *_args, **_kwargs):
            delta = candidate["components"][0]["delta_ref"]
            evaluated.append(delta)
            if delta > initial_delta * (1.0 + 1e-12):
                raise FloatingPointError("invalid forward CPA probe")
            return np.array([delta]), {}

        class ProbeComplete(Exception):
            pass

        def inspect_jacobian(_fun, x0, *, jac, **_kwargs):
            derivative = jac(x0)
            self.assertEqual(derivative.shape, (1, 1))
            self.assertAlmostEqual(derivative[0, 0],
                                   math.log(10.0) * initial_delta, delta=1e-4)
            raise ProbeComplete

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "reference.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["CV", "UV"])
                writer.writerows(((0.0, 0.0), (0.5, 1.0), (1.0, 0.0)))
            with patch.object(lsq, "_objective_parts", side_effect=objective), \
                    patch.object(lsq, "least_squares", side_effect=inspect_jacobian):
                with self.assertRaises(ProbeComplete):
                    lsq.run_least_squares_refinement(
                        cfg, mode="CHROMATOGRAM", chromatogram_csv=source,
                        unlocked_paths=["components[0].delta_ref"])
        self.assertTrue(any(value > initial_delta for value in evaluated))
        self.assertTrue(any(value < initial_delta for value in evaluated))

    def test_coverage_guard_checks_accepted_state_without_changing_bdf(self):
        def accumulating_rhs(_time, _state):
            return np.array([2.0])

        guarded = _CoverageGuardedBDF(
            accumulating_rhs, 0.0, np.zeros(1), 1.0,
            coverage_weights=np.ones(1), adsorbed_slice=slice(0, 1),
            axial_positions=1, species_count=1, max_step=0.1,
        )
        with self.assertRaisesRegex(FloatingPointError, "surface coverage"):
            while guarded.status == "running":
                guarded.step()
        self.assertLess(guarded.t, 1.0)

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
        self.assertEqual(cpa.kinetic_rate(q0, q0, 6.0, 0.02)[0], 0)
        concentration, bound = cpa.equilibrium_from_total(np.ones(1), 0.5, 6.0, 0.02)
        np.testing.assert_allclose(concentration + 0.5 * bound, [1.0], rtol=1e-8)
        np.testing.assert_allclose(bound,
                                   cpa.equilibrium_factors(bound, 6.0, 0.02) * concentration,
                                   rtol=1e-7, atol=1e-8)

    def test_charge_salt_saturation_and_kinetic_directions(self):
        cfg = binding_recipe(0.012)
        comp = _build_cpa_components(cfg, need_tdm_kinetics=True)[0]
        cpa = _cpa_model(cfg, [comp])
        empty = np.zeros(1)
        affinity = [cpa.equilibrium_factors(empty, 6.0, ionic)[0]
                    for ionic in (0.02, 0.1, 0.5)]
        self.assertGreater(affinity[0], affinity[1])
        self.assertGreater(affinity[1], affinity[2])
        neutral = copy.copy(comp); neutral.Z_ref = 0.0
        negative = copy.copy(comp); negative.Z_ref = -abs(comp.Z_ref)
        self.assertGreater(affinity[0], _cpa_model(cfg, [neutral]).equilibrium_factors(empty, 6.0, 0.02)[0])
        self.assertGreater(affinity[0], _cpa_model(cfg, [negative]).equilibrium_factors(empty, 6.0, 0.02)[0])
        ceiling = comp.As_m_inv / (np.pi * 6.02214076e23 * comp.radius_m**2)
        surfaces = [cpa.available_surface(0, np.array([ceiling * fraction]))
                    for fraction in (0.0, 0.2, 0.5, 0.9, 1.0)]
        self.assertTrue(all(a > b for a, b in zip(surfaces, surfaces[1:])))
        self.assertLess(surfaces[-1], 1e-4)
        self.assertGreater(cpa.kinetic_rate(np.ones(1), empty, 6.0, 0.02)[0], 0.0)
        self.assertLess(cpa.kinetic_rate(empty, np.array([ceiling * 0.2]), 6.0, 0.5)[0], 0.0)
        free, bound = cpa.equilibrium_from_total(np.ones(1), 0.5, 6.0, 0.02)
        self.assertAlmostEqual(cpa.kinetic_rate(free, bound, 6.0, 0.02)[0], 0.0, delta=1e-7)

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
            self.assertEqual(fit["acceptance"]["status"], "ACCEPTED")
            self.assertIn("CPA binding and elution audit", report)
            self.assertIn("Initial bound after load %", report)
            self.assertGreater(fit["fitted_config"]["components"][0]["delta_ref"],
                               initial["components"][0]["delta_ref"])

    def test_lsq_does_not_apply_a_flow_through_candidate(self):
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
                self.assertFalse(lsq.CONFIG_PATH.exists())
            self.assertFalse(fit["accepted"])
            self.assertEqual(fit["acceptance"]["status"], "REJECTED")
            self.assertIn("do not predict convincing target binding and elution",
                          fit["fit_quality_warning"])
            self.assertIn("Binding warning", report)

    def test_numerically_improved_flow_through_candidate_is_diagnostic_only(self):
        weak = binding_recipe(1e-7)
        result = simulate(weak)
        info = {"result": result}
        decision = lsq._fit_acceptance(weak, 10.0, 1.0, [info])
        self.assertEqual(decision["status"], "NUMERICALLY_IMPROVED_BUT_PHYSICALLY_INVALID")
        self.assertFalse(decision["accepted"])
        weak["least_squares"]["physical_acceptance_mode"] = "DIAGNOSTIC_ONLY"
        self.assertEqual(lsq._fit_acceptance(weak, 10.0, 1.0, [info])["status"], "ACCEPTED")

    def test_improved_but_unbound_fit_keeps_active_model_unchanged(self):
        weak = binding_recipe(1e-7)
        measured = simulate(weak)["trace"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "reference.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["CV", "UV [mAU]"])
                writer.writerows(zip(measured["CV"], np.asarray(measured["uv_mAU"]) * 1.5))
            names = ("FIT_RESULTS_CSV", "FIT_TRACE_CSV", "FIT_COMPOSITION_CSV",
                     "FIT_CONDUCTIVITY_CSV", "FIT_REPORT_HTML", "FIT_CONFIG_JSON",
                     "FIT_APPLY_JSL", "FIT_SETTINGS_JSON", "FIT_HISTORY_CSV",
                     "FIT_IMPROVED_CSV", "FIT_IMPROVED_TXT", "CONFIG_PATH")
            with patch.multiple(lsq, **{name: root / getattr(lsq, name).name for name in names}):
                fit = lsq.run_global_least_squares_refinement(
                    [{"config": weak, "chromatogram_csv": source}],
                    unlocked_paths=["conversion.uv_to_protein_mAU_L_g"],
                    max_nfev=5, max_model_evaluations=6)
                self.assertFalse(lsq.CONFIG_PATH.exists())
                self.assertFalse(lsq.FIT_APPLY_JSL.exists())
                self.assertTrue(lsq.FIT_CONFIG_JSON.exists())
            self.assertLess(fit["final_objective"], fit["initial_objective"])
            self.assertEqual(fit["acceptance"]["status"],
                             "NUMERICALLY_IMPROVED_BUT_PHYSICALLY_INVALID")

    def test_salt_input_and_stage_balance_are_auditable(self):
        cfg = binding_recipe(0.012)
        output = simulate(cfg)
        diagnostics = output["simulation"]["diagnostics"]
        audit = diagnostics["CPA_adsorption_audit"]
        for phase, minimum in diagnostics["minimum_solver_state_by_phase_mol_m3"].items():
            self.assertGreaterEqual(minimum,
                                    -diagnostics["negative_state_tolerance_by_phase_mol_m3"][phase])
        self.assertLessEqual(diagnostics["maximum_surface_coverage_fraction"], 1.05)
        self.assertEqual(diagnostics["mobile_phase_transport_inputs"]["axial_dispersion_mm2_s"],
                         cfg["system"]["salt_dispersion_mm2_s"])
        self.assertTrue(all(abs(row["cumulative_mass_balance_error_percent_of_load"][0]) < 1
                            for row in audit["stages"]))
        self.assertEqual(audit["species"][0]["peak_outlet_stage_kind"], "ELUTION")
        self.assertGreater(audit["species"][0]["elution_recovery_percent_of_end_load_retained"], 50)
        self.assertIn("zero_coverage_rate_at_feed_concentration_mol_m3_s", audit["species"][0])
        self.assertNotIn("initial_adsorption_rate_mol_m3_s", audit["species"][0])
        changed = copy.deepcopy(cfg)
        changed["system"]["salt_dispersion_mm2_s"] = 10.0
        changed_output = simulate(changed)
        self.assertEqual(changed_output["simulation"]["diagnostics"]["mobile_phase_transport_inputs"]
                         ["axial_dispersion_mm2_s"], 10.0)
        self.assertGreater(np.max(np.abs(np.asarray(output["trace"]["conductivity_mS_cm"])
                                         - np.asarray(changed_output["trace"]["conductivity_mS_cm"]))), 0.1)

    def test_bind_and_elute_metrics_converge_with_grid_and_time_resolution(self):
        metrics = []
        for cells, steps in ((8, 90), (16, 90), (16, 180)):
            cfg = binding_recipe(0.012)
            cfg["numerics"].update(axial_positions=cells, time_steps=steps)
            result = simulate(cfg)
            species = result["simulation"]["diagnostics"]["CPA_adsorption_audit"]["species"][0]
            metrics.append((result["parameter_receipt"]["target_peak"]["column_outlet_peak_CV"],
                            species["end_load_bound_percent_of_load"],
                            species["elution_recovery_percent_of_end_load_retained"]))
        for earlier, later in zip(metrics, metrics[1:]):
            self.assertLess(abs(earlier[0] - later[0]), 0.1)  # CV
            self.assertLess(abs(earlier[1] - later[1]), 1.0)  # percentage points
            self.assertLess(abs(earlier[2] - later[2]), 1.0)


if __name__ == "__main__":
    unittest.main()
