"""Paper-equation, JMP handoff, and UV-refinement checks for TDM–Wang."""
import copy
import csv
import math
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import tdm_bridge
import tdm_least_squares as lsq
import tdm_minimal_model as model
from tdm_jmp import _validate_jsl, build_jsl
from tdm_minimal_io import (
    MODEL_EDM_CPA, MODEL_EDM_LANGMUIR, MODEL_LABELS, MODEL_TDM_CPA,
    MODEL_TDM_LANGMUIR, MODEL_TDM_WANG, apply_batch_shared_parameters,
    batch_row_to_run_config, load_batch_rows, load_config,
    required_parameter_paths, validate_config,
)
from tdm_minimal_model import _wang_kinetic_rate, simulate


def wang_config():
    shared = load_config()
    shared["model"] = MODEL_TDM_WANG
    shared["chromatography_mode"] = "HIC"
    config, _ = apply_batch_shared_parameters(shared, batch_row_to_run_config(load_batch_rows()[0]))
    config["numerics"].update(time_steps=60, axial_positions=5)
    return config


class ModifiedWangTests(unittest.TestCase):
    def test_paper_kinetic_law_and_competitive_vacancy(self):
        cfg = wang_config()
        row = cfg["components"][0]
        row.update(wang_K_kin_s=3.0, wang_k_eq=5.0, wang_qmax_g_L=8.0,
                   wang_n=1.2, wang_beta0=0.3, wang_beta1_per_M=0.4,
                   wang_beta2_L_g=0.2, wang_beta3_per_pH=0.1)
        cfg["wang"].update(q0_g_L=6.0, eta=1.1)
        cp, q, salt, ph = 1.3, 2.0, 0.7, 6.1
        free = 1.0 - q / row["wang_qmax_g_L"]
        adsorption = row["wang_k_eq"] * free**row["wang_n"] * cp**cfg["wang"]["eta"]
        exponent = 1 + row["wang_n"] * row["wang_beta0"] * math.exp(
            row["wang_beta1_per_M"] * salt + row["wang_beta2_L_g"] * cp
            + row["wang_beta3_per_pH"] * ph)
        desorption = cfg["wang"]["q0_g_L"] ** (1 + row["wang_n"] * row["wang_beta0"]) * (
            q / cfg["wang"]["q0_g_L"]) ** exponent
        actual = _wang_kinetic_rate(np.array([[cp]]), np.array([[q]]),
                                    np.array([salt]), np.array([ph]), cfg)[0, 0]
        self.assertAlmostEqual(actual, (adsorption - desorption) / row["wang_K_kin_s"], places=12)
        saturated = _wang_kinetic_rate(np.array([[cp]]), np.array([[8.0]]),
                                       np.array([salt]), np.array([ph]), cfg)[0, 0]
        self.assertLess(saturated, 0.0)
        cfg["impurity_count"] = 1
        cfg["components"][1].update(row)
        cfg["components"][1]["name"] = "Impurity 1"
        alone = _wang_kinetic_rate(np.array([[cp, 0.0]]), np.array([[q, 0.0]]),
                                   np.array([salt]), np.array([ph]), cfg)[0, 0]
        competing = _wang_kinetic_rate(np.array([[cp, 0.0]]), np.array([[q, 2.0]]),
                                       np.array([salt]), np.array([ph]), cfg)[0, 0]
        self.assertLess(competing, alone)

    def test_existing_model_choices_remain_available(self):
        saved = load_config()
        jsl = build_jsl(saved)
        _validate_jsl(jsl)
        choices = json.loads("[" + re.search(
            r'modelBox\s*=\s*Combo Box\(\s*\{([^{}]*)\}', jsl).group(1) + "]")
        self.assertEqual(choices[0], MODEL_LABELS[saved["model"]])
        self.assertEqual(set(choices), {MODEL_LABELS[m] for m in (
            MODEL_EDM_LANGMUIR, MODEL_TDM_LANGMUIR, MODEL_EDM_CPA,
            MODEL_TDM_CPA, MODEL_TDM_WANG,
        )})
        self.assertIn('chromatographyMode = Combo Box({"HIC", "ION_EXCHANGE"}', jsl)
        self.assertIn('wangSharedPanel << Visibility(If(isWang, "Visible", "Collapse"))', jsl)
        self.assertIn('If(isWang & Uppercase(Char(chromatographyMode << Get Selected)) != "HIC", chromatographyMode << Set(1))', jsl)
        self.assertIn('c1QmaxFitLock = Combo Box({"Locked", "Unlocked"}, << Set(2)', jsl)
        self.assertIn('c1WangKeqFitLock = Combo Box({"Locked", "Unlocked"}, << Set(2)', jsl)

    def test_jmp_wang_parameter_handoff(self):
        cfg = wang_config()
        self.assertEqual(validate_config(cfg, strict=True), [])
        jsl = build_jsl(cfg)
        _validate_jsl(jsl)
        self.assertIn('modelBox = Combo Box({"TDM – Modified Wang",', jsl)
        paths = set(required_parameter_paths(cfg))
        for path in ("wang.q0_g_L", "wang.eta"):
            self.assertIn(path, paths)
        self.assertIn('Python Name("tdm_ui_wang_q0_g_L")', jsl)
        self.assertIn('Python Name("tdm_ui_wang_eta")', jsl)
        self.assertIn('wangQ0FitLock = Combo Box({"Locked", "Unlocked"}', jsl)
        self.assertIn('wangEtaFitLock = Combo Box({"Locked", "Unlocked"}', jsl)
        sent = {"tdm_ui_model": MODEL_LABELS[MODEL_TDM_WANG],
                "tdm_ui_chromatography_mode": "HIC",
                "tdm_ui_impurity_count": 0,
                "tdm_ui_wang_q0_g_L": 11.0,
                "tdm_ui_wang_eta": 1.25}
        fields = ("wang_K_kin_s", "wang_k_eq", "wang_qmax_g_L", "wang_n",
                  "wang_beta0", "wang_beta1_per_M", "wang_beta2_L_g", "wang_beta3_per_pH")
        values = (21.0, 9.0, 15.0, 1.4, 0.3, 0.8, 0.2, -0.05)
        for field, value in zip(fields, values):
            sent["tdm_ui_c1_" + field] = value
            self.assertIn("components[0]." + field, paths)
            self.assertIn('Python Name("tdm_ui_c1_' + field + '")', jsl)
        with patch.dict(tdm_bridge.__dict__, sent):
            received = tdm_bridge.build_shared_config_from_jmp()
        self.assertEqual(received["wang"], {"q0_g_L": 11.0, "eta": 1.25})
        for field, value in zip(fields, values):
            self.assertEqual(received["components"][0][field], value)
        eligible = {spec.path for spec in lsq.fit_parameter_lock_catalog(cfg)}
        self.assertTrue({"wang.q0_g_L", "wang.eta"}.issubset(eligible))
        self.assertTrue({"components[0]." + field for field in fields}.issubset(eligible))

    def test_elution_and_locked_uv_refinement(self):
        true = wang_config()
        true["components"][0]["wang_beta1_per_M"] = 1.2
        trace = simulate(true)["trace"]
        peak = int(np.argmax(trace["uv_mAU"]))
        self.assertGreater(float(trace["CV"][peak]), 5.0)
        self.assertLess(float(trace["CV"][peak]), float(trace["CV"][-1]))
        self.assertEqual(float(trace["CV"][-1]), 17.0)
        self.assertTrue(np.isfinite(trace["uv_mAU"]).all())

        start = copy.deepcopy(true)
        start["components"][0]["wang_beta1_per_M"] = 0.7
        baseline = simulate(start)
        self.assertGreater(np.max(np.abs(np.asarray(trace["uv_mAU"]) -
                                         np.asarray(baseline["trace"]["uv_mAU"]))), 1.0)
        self.assertLess(abs(baseline["mass_balance"]["Target protein"]["closure_error_percent_of_input"]), 0.2)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference = root / "measured_uv.csv"
            with reference.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["CV", "UV 280 [mAU]"])
                writer.writerows(zip(trace["CV"], trace["uv_mAU"]))
            outputs = ("FIT_RESULTS_CSV", "FIT_TRACE_CSV", "FIT_COMPOSITION_CSV",
                       "FIT_REPORT_HTML", "FIT_CONFIG_JSON", "FIT_APPLY_JSL",
                       "FIT_SETTINGS_JSON", "FIT_HISTORY_CSV", "FIT_IMPROVED_CSV",
                       "FIT_IMPROVED_TXT", "CONFIG_PATH")
            with patch.multiple(lsq, **{name: root / getattr(lsq, name).name for name in outputs}):
                fit = lsq.run_global_least_squares_refinement(
                    [{"config": start, "chromatogram_csv": reference}],
                    unlocked_paths=["components[0].wang_beta1_per_M"], max_nfev=7)
                self.assertIn("c1WangBeta1 << Set(", lsq.FIT_APPLY_JSL.read_text())
        self.assertEqual(fit["parameters"], ["components[0].wang_beta1_per_M"])
        self.assertLess(fit["final_objective"], fit["initial_objective"] * 0.05)
        self.assertAlmostEqual(fit["fitted_config"]["components"][0]["wang_beta1_per_M"], 1.2, delta=0.08)
        self.assertEqual(fit["fitted_config"]["components"][0]["wang_k_eq"], start["components"][0]["wang_k_eq"])
        self.assertEqual(fit["fitted_config"]["wang"], start["wang"])

    def test_three_cv_load_ten_cv_elution_reaches_thirteen_cv(self):
        cfg = wang_config()
        cfg["process"].update(load_CV=3.0, plw_count=0, elution_count=1)
        cfg["process"]["elution_steps"][0]["CV"] = 10.0
        result = simulate(cfg)
        trace = result["trace"]
        self.assertAlmostEqual(float(trace["CV"][-1]), 13.0)
        self.assertTrue(np.isfinite(trace["uv_mAU"]).all())
        self.assertGreater(float(np.max(trace["uv_mAU"])), 0.0)
        self.assertLess(abs(result["mass_balance"]["Target protein"]["closure_error_percent_of_input"]), 0.2)

    def test_competing_species_sparse_solver_matches_dense_solver(self):
        cfg = wang_config()
        cfg["impurity_count"] = 1
        cfg["components"][0]["mass_percent"] = 70.0
        target = cfg["components"][0]
        impurity = cfg["components"][1]
        impurity.update(mass_percent=30.0, D_ax_mm2_s=target["D_ax_mm2_s"],
                        k_eff_um_s=target["k_eff_um_s"],
                        accessible_particle_porosity=target["accessible_particle_porosity"],
                        wang_qmax_g_L=10.0, wang_beta1_per_M=0.6)
        self.assertEqual(validate_config(cfg, strict=True), [])
        sparse = simulate(cfg)
        actual_solve_ivp = model.solve_ivp

        def solve_without_sparsity(*args, **kwargs):
            kwargs.pop("jac_sparsity", None)
            return actual_solve_ivp(*args, **kwargs)

        with patch.object(model, "solve_ivp", side_effect=solve_without_sparsity):
            dense = simulate(cfg)
        np.testing.assert_allclose(sparse["trace"]["component_g_L"],
                                   dense["trace"]["component_g_L"], rtol=0.002, atol=1e-5)
        for component in ("Target protein", "Impurity 1"):
            self.assertLess(abs(sparse["mass_balance"][component]["closure_error_percent_of_input"]), 0.2)

    def test_pore_ph_changes_wang_chromatogram_when_beta3_is_active(self):
        cfg = wang_config()
        cfg["components"][0]["wang_beta3_per_pH"] = 0.1
        reference = simulate(cfg)["trace"]
        changed = copy.deepcopy(cfg)
        changed["process"]["buffer_A"]["pH"] = 8.0
        changed["process"]["buffer_B"]["pH"] = 8.0
        shifted = simulate(changed)["trace"]
        self.assertAlmostEqual(float(reference["pH"][0]), 6.0)
        self.assertAlmostEqual(float(shifted["pH"][0]), 8.0)
        self.assertGreater(np.max(np.abs(np.asarray(shifted["uv_mAU"]) -
                                         np.asarray(reference["uv_mAU"]))), 1.0)


if __name__ == "__main__":
    unittest.main()
