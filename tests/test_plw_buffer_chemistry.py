"""Fixed PLW setpoints resolve the current Buffer A/B chemistry for TDM + CPA."""

import json
import re
import unittest
from unittest.mock import patch

import numpy as np
import tdm_bridge

from tdm_jmp import build_jsl
from tdm_minimal_io import (MODEL_EDM_CPA, MODEL_EDM_LANGMUIR,
                            MODEL_TDM_CPA, MODEL_TDM_LANGMUIR,
                            MODEL_TDM_WANG, SELECTABLE_MODELS,
                            normalize_config, process_pH_span,
                            validate_config)
from tdm_minimal_model import _build_program, simulate
from verify_model import make_config


class PLWBufferChemistryTests(unittest.TestCase):
    def make_cpa_run(self):
        config = make_config(MODEL_TDM_CPA, 0, multistep=False)
        config["numerics"].update(time_steps=50, axial_positions=6)
        process = config["process"]
        process["buffer_A"]["pH"] = 5.8
        process["buffer_B"]["pH"] = 6.2
        process["plw_steps"][0].update(
            mode="LINEAR", chemistry_control="ENDPOINT_CHEMISTRY",
            start_percent_B=25.0, end_percent_B=90.0,
            start_source="DIRECT", start_pH=12.0,
            start_conductivity_mS_cm=999.0,
            end_source="DIRECT", end_pH=2.0,
            end_conductivity_mS_cm=999.0,
        )
        config["components"][0].update(
            pH_ref=6.0, Z1_per_pH=-6.0, Z2_per_pH2=0.0,
            delta_pH_slope_m2_C=0.0,
        )
        return config

    def test_old_explicit_wash_migrates_to_start_percent_b_and_blends_chemistry(self):
        original = self.make_cpa_run()
        config = normalize_config(original)
        wash = config["process"]["plw_steps"][0]
        self.assertEqual(wash["mode"], "SET_POINT")
        self.assertEqual(wash["chemistry_control"], "BUFFER_B_PERCENT")
        self.assertEqual((wash["start_percent_B"], wash["end_percent_B"]), (25.0, 25.0))
        self.assertAlmostEqual(process_pH_span(original), 0.28)
        self.assertEqual(validate_config(original), [])
        stage = _build_program(original, np.array([1.0])).stages[1]
        self.assertEqual(stage.mode, "SET_POINT")
        self.assertAlmostEqual(stage.start_pH, 5.9)
        self.assertAlmostEqual(stage.end_pH, 5.9)
        self.assertAlmostEqual(stage.start_conductivity_mS_cm, 16.25)
        self.assertAlmostEqual(stage.start_salt_M, 0.1625)
        self.assertAlmostEqual(stage.start_I_M, 0.1625)
        self.assertEqual(stage.start_percent_B, 25.0)

    def test_cpa_solver_reports_blended_plw_and_mass_balance(self):
        output = simulate(self.make_cpa_run())
        wash = next(row for row in output["parameter_receipt"]["stage_chemistry_used"]
                    if row["kind"] == "PLW")
        self.assertEqual(wash["chemistry_control"], "BUFFER_B_PERCENT")
        self.assertEqual(wash["start_salt_source"], "BUFFER_BLEND")
        self.assertAlmostEqual(wash["start_pH"], 5.9)
        self.assertAlmostEqual(wash["start_conductivity_mS_cm"], 16.25)
        self.assertAlmostEqual(wash["start_ionic_strength_M"], 0.1625)
        self.assertLess(abs(output["mass_balance"]["Target protein"]["closure_error_percent_of_input"]), 0.1)

    def test_cpa_prediction_responds_to_plw_setpoint(self):
        config = self.make_cpa_run()
        low = simulate(config)
        config["process"]["plw_steps"][0]["start_percent_B"] = 75.0
        high = simulate(config)
        low_trace = np.asarray(low["trace"]["total_g_L"])
        high_trace = np.asarray(high["trace"]["total_g_L"])
        self.assertGreater(float(np.max(np.abs(low_trace - high_trace))), 1e-6)
        high_wash = next(row for row in high["parameter_receipt"]["stage_chemistry_used"]
                         if row["kind"] == "PLW")
        self.assertAlmostEqual(high_wash["start_pH"], 6.1)
        self.assertAlmostEqual(high_wash["start_conductivity_mS_cm"], 38.75)

    def test_ui_displays_plw_blend_and_offers_only_tdm_models(self):
        jsl = build_jsl(self.make_cpa_run())
        choices = json.loads("[" + re.search(
            r'modelBox\s*=\s*Combo Box\(\s*\{([^{}]*)\}', jsl).group(1) + "]")
        self.assertEqual(set(SELECTABLE_MODELS),
                         {MODEL_TDM_LANGMUIR, MODEL_TDM_CPA, MODEL_TDM_WANG})
        self.assertEqual(len(choices), 3)
        self.assertFalse(any("EDM" in label for label in choices))
        self.assertIn('plw1ChemPanel << Visibility("Collapse")', jsl)
        self.assertIn('plw1BlendDerived << Set Text("Buffer A/B at "', jsl)
        self.assertIn('Column(dtRuns, "PLW1_Chemistry_Control")[currentRun] = "BUFFER_B_PERCENT";', jsl)
        for legacy in (MODEL_EDM_CPA, MODEL_EDM_LANGMUIR):
            saved = make_config(legacy, 0, multistep=False)
            migrated_ui = build_jsl(saved)
            self.assertIn("This saved project uses an EDM model that is no longer selectable.", migrated_ui)
            selected = json.loads("[" + re.search(
                r'modelBox\s*=\s*Combo Box\(\s*\{([^{}]*)\}', migrated_ui).group(1) + "]")[0]
            self.assertEqual(selected, "TDM – CPA" if legacy == MODEL_EDM_CPA
                             else "TDM – Competitive Langmuir")

    def test_retired_edm_models_cannot_run(self):
        for legacy in (MODEL_EDM_CPA, MODEL_EDM_LANGMUIR):
            with self.subTest(model=legacy):
                config = make_config(legacy, 0, multistep=False)
                self.assertIn("Select a supported model combination.", validate_config(config))
                with self.assertRaisesRegex(ValueError, "Select a supported model combination"):
                    simulate(config)
        with patch.dict(tdm_bridge.__dict__, {"tdm_ui_model": "EDM – CPA"}):
            with self.assertRaisesRegex(ValueError, "Only TDM model combinations"):
                tdm_bridge.build_shared_config_from_jmp()


if __name__ == "__main__":
    unittest.main()
