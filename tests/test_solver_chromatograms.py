"""Representative elution checks across the four column/isotherm combinations."""
import unittest
import copy
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from verify_model import make_config
from tdm_minimal_io import (
    MODEL_EDM_CPA, MODEL_EDM_LANGMUIR, MODEL_TDM_CPA, MODEL_TDM_LANGMUIR,
    MODEL_LABELS, apply_batch_shared_parameters, batch_row_to_run_config,
    load_batch_rows, load_config, required_parameter_values,
)
from tdm_minimal_model import check_result_freshness, simulate, write_outputs
import tdm_bridge


class SolverChromatogramTests(unittest.TestCase):
    def test_packaged_run_jmp_handoff_preserves_binding_and_chromatogram(self):
        saved = load_config()
        recipe = batch_row_to_run_config(load_batch_rows()[0])
        expected, _ = apply_batch_shared_parameters(saved, recipe)
        expected["numerics"].update(time_steps=300, axial_positions=20)
        component = expected["components"][0]
        sent = {
            "tdm_ui_model": MODEL_LABELS[expected["model"]],
            "tdm_ui_impurity_count": 0,
            "tdm_ui_chromatography_mode": expected["chromatography_mode"],
            "tdm_ui_column_volume_mL": expected["column"]["volume_mL"],
            "tdm_ui_column_length_mm": expected["column"]["length_mm"],
            "tdm_ui_edm_porosity": expected["edm"]["total_bed_porosity"],
            "tdm_ui_uv_to_protein_factor": expected["conversion"]["uv_to_protein_mAU_L_g"],
            "tdm_ui_cond_to_salt_factor": expected["conversion"]["conductivity_to_salt_M_per_mS_cm"],
            "tdm_ui_c1_name": component["name"],
            "tdm_ui_c1_mass_percent": component["mass_percent"],
            "tdm_ui_c1_D_app_mm2_s": component["D_app_mm2_s"],
            "tdm_ui_c1_qmax_g_L": component["qmax_g_L"],
            "tdm_ui_c1_b_L_g": component["b_L_g"],
            "tdm_ui_c1_salt_sensitivity_per_M": component["salt_sensitivity_per_M"],
        }
        for field in ("buffer_dispersion_mL", "salt_dispersion_mm2_s", "uv_dead_volume_mL", "conductivity_dead_volume_mL"):
            sent["tdm_ui_system_" + field] = expected["system"][field]
        for field in ("buffer_dispersion_enabled", "salt_dispersion_enabled", "show_percent_B", "show_conductivity", "show_flow_rate"):
            sent["tdm_ui_system_" + field] = "ON" if expected["system"][field] else "OFF"
        with patch.dict(tdm_bridge.__dict__, sent):
            shared_from_jmp = tdm_bridge.build_shared_config_from_jmp()
        received, _ = apply_batch_shared_parameters(shared_from_jmp, recipe)
        received["numerics"].update(time_steps=300, axial_positions=20)
        self.assertEqual(required_parameter_values(received), required_parameter_values(expected))
        self.assertEqual(received["system"], expected["system"])
        original_trace = simulate(expected)["trace"]
        jmp_trace = simulate(received)["trace"]
        np.testing.assert_allclose(jmp_trace["uv_mAU"], original_trace["uv_mAU"], rtol=1e-12, atol=1e-12)

    def test_result_freshness_ignores_fit_controls_but_not_binding(self):
        config = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        config["numerics"].update(time_steps=50, axial_positions=6)
        context = {"run_number": 1, "run_name": "Forward model"}
        with tempfile.TemporaryDirectory() as directory:
            result = simulate(config)
            write_outputs(config, result, Path(directory), context)
            import json
            receipt = json.loads((Path(directory) / "latest_results_manifest.json").read_text())
            self.assertEqual(receipt["target_peak"], result["parameter_receipt"]["target_peak"])
            updated_fit = copy.deepcopy(config)
            updated_fit["least_squares"]["max_nfev"] = 240
            self.assertTrue(check_result_freshness(updated_fit, context, directory)[0])
            updated_binding = copy.deepcopy(updated_fit)
            updated_binding["components"][0]["b_L_g"] *= 2
            self.assertFalse(check_result_freshness(updated_binding, context, directory)[0])

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
                target_peak = result["parameter_receipt"]["target_peak"]
                self.assertTrue(np.isfinite(peak).all())
                self.assertGreater(float(peak.max()), 0.01)
                self.assertLess(float(trace["CV"][np.argmax(peak)]), float(trace["CV"][-1]))
                self.assertAlmostEqual(target_peak["detector_peak_CV"], float(trace["CV"][np.argmax(peak)]))
                self.assertAlmostEqual(
                    target_peak["CV_after_elution_start"],
                    target_peak["detector_peak_CV"] - target_peak["first_elution_start_CV"],
                )
                balance = result["mass_balance"]["Target protein"]
                self.assertAlmostEqual(balance["outlet_g"] / balance["input_g"], 1.0, delta=0.02)
                self.assertLess(abs(balance["closure_error_percent_of_input"]), 0.1)


if __name__ == "__main__":
    unittest.main()
