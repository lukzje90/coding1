"""The visible load %B controls feed the saved recipe and column solver."""
import copy
import unittest

import numpy as np

import tdm_bridge
from tdm_jmp import _validate_jsl, build_jsl
from tdm_minimal_io import (
    MODEL_EDM_CPA, MODEL_EDM_LANGMUIR, MODEL_TDM_CPA,
    MODEL_TDM_LANGMUIR, MODEL_TDM_WANG,
    LSQ_MODEL_SOURCE_BATCH, LSQ_MODEL_SOURCE_INDEPENDENT, apply_batch_shared_parameters,
    batch_row_to_run_config, default_batch_row, validate_config,
)
from tdm_minimal_model import _build_program, simulate
from verify_model import make_config


class LoadPercentBTests(unittest.TestCase):
    def test_load_percent_b_and_measured_load_are_visible_choices(self):
        jsl = build_jsl()
        _validate_jsl(jsl)
        self.assertIn('loadMaterialModeRow << Visibility("Visible")', jsl)
        self.assertIn('loadChemPanel << Visibility(If(Uppercase(Char(loadMaterialMode << Get Selected)) == "EXPLICIT", "Visible", "Collapse"))', jsl)
        self.assertIn('Measured load material pH and conductivity', jsl)
        self.assertIn('loadPH << Set(BlendBufferPH(loadStartB << Get));', jsl)
        self.assertIn('loadCond << Set(BlendBufferCond(loadStartB << Get));', jsl)
        self.assertIn('loadSource << Set(1); loadSaltMode << Set(1); SaveCurrentRun(); UpdateUI()', jsl)
        self.assertIn('Column(dtRuns, "Load_pH")[currentRun] = loadPH << Get;', jsl)
        self.assertIn('Column(dtRuns, "Load_Conductivity_mS_cm")[currentRun] = loadCond << Get;', jsl)
        self.assertIn('loadMaterialMode << Set(1); loadChemControl << Set(1); SaveCurrentRun()', jsl)
        self.assertIn('loadMaterialStatus << Set Text("Load chemistry from Buffer A/B:', jsl)

    def test_measured_load_reaches_all_models_and_lsq_recipe_sources(self):
        models = (MODEL_EDM_CPA, MODEL_TDM_CPA, MODEL_EDM_LANGMUIR,
                  MODEL_TDM_LANGMUIR, MODEL_TDM_WANG)
        for model in models:
            with self.subTest(model=model):
                shared = make_config(model, 0, multistep=False)
                shared["numerics"].update(time_steps=50, axial_positions=6)
                if model in (MODEL_EDM_CPA, MODEL_TDM_CPA):
                    shared["components"][0].update(
                        pH_ref=6.0, Z1_per_pH=0.0, Z2_per_pH2=0.0,
                        Z3_per_pH3=0.0, delta_pH_slope_m2_C=0.0)
                row = default_batch_row(1)
                row.update(Load_Material_Chemistry_Mode="EXPLICIT",
                           Load_Chemistry_Control="ENDPOINT_CHEMISTRY",
                           Load_Source="DIRECT", Load_pH=5.4,
                           Load_Conductivity_mS_cm=18.0,
                           Load_Salt_Input_Mode="CONDUCTIVITY",
                           Time_Steps=50, Axial_Positions=6)
                if model in (MODEL_EDM_CPA, MODEL_TDM_CPA):
                    for prefix, concentration in (("BufferA", 0.05), ("BufferB", 0.50)):
                        row[f"{prefix}_Ionic_Strength_Source"] = "ION_COMPOSITION"
                        for index, species, charge in ((1, "Na+", 1), (2, "Cl-", -1)):
                            row[f"{prefix}_Ion{index}_Species"] = species
                            row[f"{prefix}_Ion{index}_Concentration_M"] = concentration
                            row[f"{prefix}_Ion{index}_Valency"] = charge
                normal, _ = apply_batch_shared_parameters(shared, batch_row_to_run_config(row))
                independent = copy.deepcopy(row)
                independent["LSQ_Model_Source"] = LSQ_MODEL_SOURCE_INDEPENDENT
                selected, _ = tdm_bridge._lsq_model_run_config(shared, 1, independent, [row])
                batch_ref = copy.deepcopy(row)
                batch_ref["LSQ_Model_Source"] = LSQ_MODEL_SOURCE_BATCH
                batch_ref["LSQ_Model_Run_Number"] = 1
                linked, _ = tdm_bridge._lsq_model_run_config(shared, 1, batch_ref, [row])
                for cfg in (normal, selected, linked):
                    self.assertEqual(validate_config(cfg, strict=True), [])
                    load = _build_program(cfg, np.zeros(1)).stages[0]
                    self.assertAlmostEqual(load.start_pH, 5.4)
                    self.assertAlmostEqual(load.end_pH, 5.4)
                    self.assertAlmostEqual(load.start_conductivity_mS_cm, 18.0)
                    self.assertAlmostEqual(load.start_I_M, 0.18)
                trace = simulate(normal)["trace"]
                self.assertAlmostEqual(float(trace["pH"][0]), 5.4)
                self.assertAlmostEqual(float(trace["conductivity_mS_cm"][0]), 18.0)
                self.assertAlmostEqual(float(trace["binding_ionic_strength_M"][0]), 0.18)

    def test_measured_load_changes_predictions_where_binding_law_uses_it(self):
        cpa = make_config(MODEL_TDM_CPA, 0, multistep=False)
        cpa["numerics"].update(time_steps=50, axial_positions=6)
        cpa["components"][0].update(
            pH_ref=6.0, Z1_per_pH=0.0, Z2_per_pH2=0.0,
            Z3_per_pH3=0.0, delta_pH_slope_m2_C=0.0)
        cpa["process"].update(load_material_chemistry_mode="EXPLICIT",
                              load_source="DIRECT", load_pH=5.4,
                              load_conductivity_mS_cm=18.0,
                              load_salt_input_mode="CONDUCTIVITY")
        baseline = np.asarray(simulate(cpa)["trace"]["uv_mAU"])
        changed_ph = copy.deepcopy(cpa)
        changed_ph["process"]["load_pH"] = 6.0
        changed_conductivity = copy.deepcopy(cpa)
        changed_conductivity["process"]["load_conductivity_mS_cm"] = 5.0
        self.assertGreater(np.max(np.abs(baseline - simulate(changed_ph)["trace"]["uv_mAU"])), 1e-4)
        self.assertGreater(np.max(np.abs(baseline - simulate(changed_conductivity)["trace"]["uv_mAU"])), 1.0)

        wang = make_config(MODEL_TDM_WANG, 0, multistep=False)
        wang["numerics"].update(time_steps=50, axial_positions=6)
        wang["components"][0]["wang_beta3_per_pH"] = 0.1
        wang["process"].update(load_material_chemistry_mode="EXPLICIT",
                               load_source="DIRECT", load_pH=5.4,
                               load_conductivity_mS_cm=18.0,
                               load_salt_input_mode="CONDUCTIVITY")
        wang_baseline = np.asarray(simulate(wang)["trace"]["uv_mAU"])
        wang_changed = copy.deepcopy(wang)
        wang_changed["process"]["load_pH"] = 6.0
        self.assertGreater(np.max(np.abs(wang_baseline - simulate(wang_changed)["trace"]["uv_mAU"])), 0.001)

        langmuir = make_config(MODEL_TDM_LANGMUIR, 0, multistep=False)
        langmuir["numerics"].update(time_steps=50, axial_positions=6)
        langmuir["process"].update(load_material_chemistry_mode="EXPLICIT",
                                   load_source="DIRECT", load_pH=5.4,
                                   load_conductivity_mS_cm=18.0,
                                   load_salt_input_mode="CONDUCTIVITY")
        langmuir_baseline = np.asarray(simulate(langmuir)["trace"]["uv_mAU"])
        langmuir_changed = copy.deepcopy(langmuir)
        langmuir_changed["process"]["load_conductivity_mS_cm"] = 5.0
        self.assertGreater(np.max(np.abs(langmuir_baseline - simulate(langmuir_changed)["trace"]["uv_mAU"])), 0.1)

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
                if model in (MODEL_TDM_CPA, MODEL_EDM_CPA):
                    for prefix, concentration in (("BufferA", 0.05), ("BufferB", 0.35)):
                        row[f"{prefix}_Ionic_Strength_Source"] = "ION_COMPOSITION"
                        for index, species, charge in ((1, "Na+", 1), (2, "Cl-", -1)):
                            row[f"{prefix}_Ion{index}_Species"] = species
                            row[f"{prefix}_Ion{index}_Concentration_M"] = concentration
                            row[f"{prefix}_Ion{index}_Valency"] = charge
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
