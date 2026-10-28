"""The visible load %B controls feed the saved recipe and column solver."""
import copy
import unittest

import numpy as np

import tdm_bridge
from tdm_jmp import _validate_jsl, build_jsl
from tdm_minimal_io import (
    MODEL_EDM_CPA, MODEL_EDM_LANGMUIR, MODEL_TDM_CPA,
    MODEL_TDM_LANGMUIR, MODEL_TDM_WANG,
    LSQ_MODEL_SOURCE_BATCH, apply_batch_shared_parameters,
    batch_row_to_run_config, default_batch_row, validate_config,
)
from tdm_minimal_model import _build_program, simulate
from verify_model import make_config


class LoadPercentBTests(unittest.TestCase):
    def test_load_percent_b_is_the_visible_authority(self):
        jsl = build_jsl()
        _validate_jsl(jsl)
        self.assertIn('loadChemPanel << Visibility("Collapse")', jsl)
        self.assertIn('loadMaterialModeRow << Visibility("Collapse")', jsl)
        self.assertIn('loadMaterialMode << Set(1); loadChemControl << Set(1); SaveCurrentRun()', jsl)
        self.assertIn('loadMaterialStatus << Set Text("Load chemistry from Buffer A/B:', jsl)

    def test_percent_b_load_chemistry_reaches_every_model_and_selected_lsq_run(self):
        models = (MODEL_TDM_CPA, MODEL_EDM_CPA, MODEL_TDM_LANGMUIR,
                  MODEL_EDM_LANGMUIR, MODEL_TDM_WANG)
        for model in models:
            with self.subTest(model=model):
                shared = make_config(model, 0, multistep=False)
                shared["process"]["buffer_A"].update(pH=6.0, conductivity_mS_cm=5.0)
                shared["process"]["buffer_B"].update(pH=6.4, conductivity_mS_cm=35.0)
                if model in (MODEL_TDM_CPA, MODEL_EDM_CPA):
                    shared["components"][0].update(pH_ref=6.1, Z1_per_pH=0.0,
                                                    Z2_per_pH2=0.0, Z3_per_pH3=0.0,
                                                    delta_pH_slope_m2_C=0.0)
                row = default_batch_row(5)
                row.update(Load_Mode="LINEAR", Load_Start_Percent_B=25.0,
                           Load_End_Percent_B=75.0,
                           Load_Material_Chemistry_Mode="BUFFER_RECIPE",
                           Load_Chemistry_Control="BUFFER_B_PERCENT",
                           BufferA_pH=6.0, BufferA_Conductivity_mS_cm=5.0,
                           BufferB_pH=6.4, BufferB_Conductivity_mS_cm=35.0)
                recipe, _ = apply_batch_shared_parameters(shared, batch_row_to_run_config(row))
                self.assertEqual(validate_config(recipe, strict=True), [])
                load = _build_program(recipe, np.zeros(1)).stages[0]
                self.assertAlmostEqual(load.start_pH, 6.1)
                self.assertAlmostEqual(load.end_pH, 6.3)
                self.assertAlmostEqual(load.start_conductivity_mS_cm, 12.5)
                self.assertAlmostEqual(load.end_conductivity_mS_cm, 27.5)
                self.assertAlmostEqual(load.start_salt_M, 0.125)
                lsq_row = default_batch_row(5)
                lsq_row.update(LSQ_Model_Source=LSQ_MODEL_SOURCE_BATCH,
                               LSQ_Model_Run_Number=5)
                batch = [default_batch_row(i) for i in range(1, 6)]
                batch[4] = row
                selected, context = tdm_bridge._lsq_model_run_config(shared, 5, lsq_row, batch)
                selected_load = _build_program(selected, np.zeros(1)).stages[0]
                self.assertEqual(context["model_run_name"], row["Run_Name"])
                self.assertAlmostEqual(selected_load.start_conductivity_mS_cm, 12.5)
                self.assertAlmostEqual(selected_load.end_pH, 6.3)

    def test_load_percent_b_changes_tdm_cpa_prediction_and_legacy_explicit_is_preserved(self):
        config = make_config(MODEL_TDM_CPA, 0, multistep=False)
        config["numerics"].update(time_steps=50, axial_positions=6)
        config["process"]["buffer_A"].update(pH=6.0, conductivity_mS_cm=5.0)
        config["process"]["buffer_B"].update(pH=6.4, conductivity_mS_cm=35.0)
        config["process"].update(load_mode="SET_POINT",
                                 load_start_percent_B=25.0,
                                 load_end_percent_B=25.0)
        config["components"][0].update(pH_ref=6.1, Z1_per_pH=0.0,
                                        Z2_per_pH2=0.0, Z3_per_pH3=0.0,
                                        delta_pH_slope_m2_C=0.0)
        first = simulate(config)
        changed = copy.deepcopy(config)
        changed["process"].update(load_start_percent_B=75.0,
                                  load_end_percent_B=75.0)
        second = simulate(changed)
        self.assertAlmostEqual(first["trace"]["binding_salt_concentration_M"][0], 0.125)
        self.assertAlmostEqual(second["trace"]["binding_salt_concentration_M"][0], 0.275)
        self.assertGreater(np.max(np.abs(first["trace"]["uv_mAU"] -
                                         second["trace"]["uv_mAU"])), 0.01)
        legacy = copy.deepcopy(config)
        legacy["process"].update(load_material_chemistry_mode="EXPLICIT",
                                 load_chemistry_control="ENDPOINT_CHEMISTRY",
                                 load_source="DIRECT", load_pH=7.0,
                                 load_conductivity_mS_cm=20.0)
        stage = _build_program(legacy, np.zeros(1)).stages[0]
        self.assertAlmostEqual(stage.start_pH, 7.0)
        self.assertAlmostEqual(stage.start_conductivity_mS_cm, 20.0)


if __name__ == "__main__":
    unittest.main()
