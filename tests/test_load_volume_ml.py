"""Per-run mL loading must survive CSV storage and reach simulation and LSQ."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tdm_bridge

from tdm_jmp import _validate_jsl, build_jsl
from tdm_minimal_io import (
    MODEL_EDM_CPA, MODEL_EDM_LANGMUIR, MODEL_TDM_CPA,
    MODEL_TDM_LANGMUIR, MODEL_TDM_WANG,
    LSQ_MODEL_SOURCE_BATCH, LSQ_MODEL_SOURCE_INDEPENDENT,
    apply_batch_shared_parameters, batch_row_to_run_config,
    batch_run_fields, batch_run_numeric_fields, batch_shared_parameter_snapshot,
    default_batch_row, derived_input_summary,
    ensure_batch_runs_csv, load_batch_rows, load_config, normalize_config,
    normalize_load_amount_basis, sanitize_config_for_selected_model,
    save_batch_rows, validate_operating_conditions,
)
from tdm_minimal_model import _build_program, simulate
from verify_model import make_config


class LoadVolumeMLTests(unittest.TestCase):
    def test_all_models_consume_equivalent_saved_load_and_salt_inputs(self):
        models = (MODEL_TDM_CPA, MODEL_TDM_WANG, MODEL_TDM_LANGMUIR,
                  MODEL_EDM_CPA, MODEL_EDM_LANGMUIR)
        load_inputs = (
            ("CAPACITY", {"Load_Density_mg_mL_resin": 6.0,
                          "Load_CV": 99.0, "Load_Volume_mL": 99.0}),
            ("LOAD_VOLUME", {"Load_Density_mg_mL_resin": 99.0,
                             "Load_CV": 3.0, "Load_Volume_mL": 99.0}),
            ("VOLUME_ML", {"Load_Density_mg_mL_resin": 99.0,
                           "Load_CV": 99.0, "Load_Volume_mL": 6.0}),
        )
        rows = []
        for index, (basis, amounts) in enumerate(load_inputs, start=1):
            row = default_batch_row(index)
            row.update(Load_Amount_Basis=basis, Feed_Concentration_mg_mL=2.0,
                       Load_Material_Chemistry_Mode="EXPLICIT",
                       Load_Chemistry_Control="ENDPOINT_CHEMISTRY",
                       Load_Source="DIRECT", Load_pH=7.2,
                       Load_Salt_Input_Mode="SALT_M", Load_Salt_M=0.2,
                       Load_Conductivity_mS_cm=20.0,
                       PLW_Count=0, Elution_Count=1, Elution1_CV=0.2,
                       Time_Steps=50, Axial_Positions=6, **amounts)
            rows.append(row)
        conductivity_row = dict(rows[0], Run_Number=4,
                                Load_Salt_Input_Mode="CONDUCTIVITY",
                                Load_Salt_M=0.9)
        rows.append(conductivity_row)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.csv"
            save_batch_rows(rows, path)
            saved = load_batch_rows(path)[:4]
            for model in models:
                with self.subTest(model=model):
                    shared = make_config(model, 0, multistep=False)
                    shared["column"]["volume_mL"] = 2.0
                    if model in (MODEL_TDM_CPA, MODEL_EDM_CPA):
                        shared["components"][0].update(
                            pH_ref=6.0, Z1_per_pH=0.0, Z2_per_pH2=0.0,
                            Z3_per_pH3=0.0, delta_pH_slope_m2_C=0.0)
                    if model == MODEL_TDM_WANG:
                        shared["chromatography_mode"] = "HIC"
                    configs = [apply_batch_shared_parameters(
                        shared, batch_row_to_run_config(row))[0] for row in saved]
                    for cfg in configs:
                        self.assertEqual(validate_operating_conditions(cfg), [])
                        self.assertAlmostEqual(cfg["process"]["load_CV"], 3.0)
                        self.assertAlmostEqual(cfg["process"]["load_volume_mL"], 6.0)
                        self.assertAlmostEqual(cfg["feed"]["load_density_mg_mL_resin"], 6.0)
                        program = _build_program(cfg, np.zeros(1))
                        self.assertAlmostEqual(program.stages[0].duration_CV, 3.0)
                        self.assertAlmostEqual(program.stages[0].start_pH, 7.2)
                        self.assertAlmostEqual(program.stages[0].start_salt_M, 0.2)
                        self.assertAlmostEqual(program.stages[0].start_conductivity_mS_cm, 20.0)
                    results = [simulate(cfg) for cfg in configs]
                    for result in results:
                        used = result["parameter_receipt"]["load_input_used"]
                        self.assertAlmostEqual(used["required_feed_volume_CV"], 3.0)
                        self.assertAlmostEqual(used["required_feed_volume_mL"], 6.0)
                        self.assertAlmostEqual(used["capacity_g_L_resin"], 6.0)
                        self.assertAlmostEqual(used["material_pH"], 7.2)
                        self.assertAlmostEqual(used["total_protein_input_g"], 0.012)
                        self.assertAlmostEqual(used["material_salt_M"], 0.2)
                        self.assertAlmostEqual(used["material_conductivity_mS_cm"], 20.0)
                    outputs = [result["trace"] for result in results]
                    for trace in outputs[1:]:
                        for key in ("CV", "uv_mAU", "conductivity_mS_cm",
                                    "binding_salt_concentration_M"):
                            np.testing.assert_allclose(trace[key], outputs[0][key], rtol=1e-10, atol=1e-10)

    def test_cleared_selected_load_input_does_not_reuse_stale_other_basis(self):
        selected = (
            ("CAPACITY", "Load_Density_mg_mL_resin", "Load capacity"),
            ("LOAD_VOLUME", "Load_CV", "Load volume [CV]"),
            ("VOLUME_ML", "Load_Volume_mL", "Load volume [mL]"),
        )
        for basis, field, label in selected:
            with self.subTest(basis=basis), tempfile.TemporaryDirectory() as directory:
                row = default_batch_row(1)
                row["Load_Amount_Basis"] = basis
                row[field] = ""
                path = Path(directory) / "runs.csv"
                save_batch_rows([row], path)
                saved = load_batch_rows(path)[0]
                self.assertEqual(saved[field], "")
                with self.assertRaisesRegex(ValueError, label.replace("[", r"\[").replace("]", r"\]")):
                    batch_row_to_run_config(saved)

    def test_saved_load_material_selector_controls_salt_source(self):
        shared = make_config(MODEL_TDM_CPA, 0, multistep=False)
        row = default_batch_row(1)
        row.update(Load_Amount_Basis="LOAD_VOLUME", Load_CV=1.0,
                   Load_Chemistry_Control="ENDPOINT_CHEMISTRY",
                   Load_Material_Chemistry_Mode="BUFFER_RECIPE",
                   Load_Source="DIRECT", Load_Salt_Input_Mode="SALT_M",
                   Load_Salt_M=0.2, Load_Conductivity_mS_cm=20.0,
                   Load_Start_Percent_B=0.0, Load_End_Percent_B=0.0)
        recipe, _ = apply_batch_shared_parameters(shared, batch_row_to_run_config(row))
        self.assertAlmostEqual(_build_program(recipe, np.zeros(1)).stages[0].start_salt_M, 0.05)
        row["Load_Material_Chemistry_Mode"] = "EXPLICIT"
        explicit, _ = apply_batch_shared_parameters(shared, batch_row_to_run_config(row))
        self.assertAlmostEqual(_build_program(explicit, np.zeros(1)).stages[0].start_salt_M, 0.2)
        row.pop("Load_Material_Chemistry_Mode")
        legacy, _ = apply_batch_shared_parameters(shared, batch_row_to_run_config(row))
        self.assertAlmostEqual(_build_program(legacy, np.zeros(1)).stages[0].start_salt_M, 0.2)
        jsl = build_jsl(shared)
        self.assertIn('If(loadMaterialMode == "EXPLICIT",', jsl)
        self.assertIn('If(Uppercase(Char(this << Get Selected)) == "BUFFER_RECIPE", loadChemControl << Set(1), loadChemControl << Set(2))', jsl)

    def test_salt_molarity_without_conductivity_still_reaches_load_stage(self):
        shared = make_config(MODEL_TDM_CPA, 0, multistep=False)
        row = default_batch_row(1)
        row.update(Load_Material_Chemistry_Mode="EXPLICIT",
                   Load_Source="DIRECT", Load_Salt_Input_Mode="SALT_M",
                   Load_Salt_M=0.2, Load_Conductivity_mS_cm="")
        config, _ = apply_batch_shared_parameters(shared, batch_row_to_run_config(row))
        self.assertEqual(validate_operating_conditions(config), [])
        stage = _build_program(config, np.zeros(1)).stages[0]
        self.assertAlmostEqual(stage.start_salt_M, 0.2)
        self.assertAlmostEqual(stage.start_conductivity_mS_cm, 20.0)

    def test_lsq_selected_batch_or_independent_run_keeps_its_own_load_and_salt(self):
        shared = make_config(MODEL_TDM_CPA, 0, multistep=False)
        shared["column"]["volume_mL"] = 2.0
        batch = [default_batch_row(i) for i in range(1, 3)]
        for row, salt in zip(batch, (0.2, 0.3)):
            row.update(Feed_Concentration_mg_mL=2.0,
                       Load_Material_Chemistry_Mode="EXPLICIT", Load_Source="DIRECT",
                       Load_Salt_Input_Mode="SALT_M", Load_Salt_M=salt,
                       Load_Conductivity_mS_cm=salt / 0.01)
        batch[0].update(Load_Amount_Basis="CAPACITY", Load_Density_mg_mL_resin=6.0)
        batch[1].update(Load_Amount_Basis="VOLUME_ML", Load_Volume_mL=8.0)
        lsq_batch = default_batch_row(1)
        lsq_batch.update(LSQ_Model_Source=LSQ_MODEL_SOURCE_BATCH,
                         LSQ_Model_Run_Number=2)
        lsq_independent = default_batch_row(2)
        lsq_independent.update(LSQ_Model_Source=LSQ_MODEL_SOURCE_INDEPENDENT,
                               Feed_Concentration_mg_mL=2.0,
                               Load_Amount_Basis="LOAD_VOLUME", Load_CV=5.0,
                               Load_Material_Chemistry_Mode="EXPLICIT",
                               Load_Source="DIRECT", Load_Salt_Input_Mode="SALT_M",
                               Load_Salt_M=0.4, Load_Conductivity_mS_cm=40.0)
        for profile, row, expected in ((1, lsq_batch, (2, 4.0, 0.3)),
                                       (2, lsq_independent, (2, 5.0, 0.4))):
            cfg, source = tdm_bridge._lsq_model_run_config(shared, profile, row, batch)
            stage = _build_program(cfg, np.zeros(1)).stages[0]
            self.assertEqual(source["model_run_number"], expected[0])
            self.assertAlmostEqual(stage.duration_CV, expected[1])
            self.assertAlmostEqual(stage.start_salt_M, expected[2])
            self.assertAlmostEqual(cfg["feed"]["load_density_mg_mL_resin"], expected[1] * 2)
            self.assertAlmostEqual(cfg["process"]["load_volume_mL"], expected[1] * 2)

    def test_independent_run_recipes_share_model_volume_and_parameters(self):
        shared = make_config(MODEL_TDM_CPA, 1, multistep=False)
        shared["column"]["volume_mL"] = 2.5
        batch = [default_batch_row(i) for i in range(1, 31)]
        batch[0].update(Load_Amount_Basis="VOLUME_ML", Load_Volume_mL=7.5,
                        Load_CV=99, PLW_Count=0, Elution_Count=1, Elution1_CV=4)
        batch[6].update(Load_Amount_Basis="VOLUME_ML", Load_Volume_mL=12.5,
                        Load_CV=99, PLW_Count=0, Elution_Count=1, Elution1_CV=6)
        lsq = [default_batch_row(i) for i in range(1, 31)]
        lsq[1].update(LSQ_Model_Source=LSQ_MODEL_SOURCE_BATCH, LSQ_Model_Run_Number=7)
        lsq[3].update(LSQ_Model_Source=LSQ_MODEL_SOURCE_INDEPENDENT,
                      Load_Amount_Basis="VOLUME_ML", Load_Volume_mL=17.5,
                      Load_CV=99, PLW_Count=0, Elution_Count=1, Elution1_CV=8)

        normal, _ = apply_batch_shared_parameters(shared, batch_row_to_run_config(batch[0]))
        fit2, source2 = tdm_bridge._lsq_model_run_config(shared, 2, lsq[1], batch)
        fit4, source4 = tdm_bridge._lsq_model_run_config(shared, 4, lsq[3], batch)
        self.assertEqual([source2["model_run_number"], source4["model_run_number"]], [7, 4])
        self.assertEqual([normal["process"]["load_CV"], fit2["process"]["load_CV"],
                          fit4["process"]["load_CV"]], [3.0, 5.0, 7.0])
        self.assertEqual([_build_program(cfg, np.zeros(2)).total_CV
                          for cfg in (normal, fit2, fit4)], [7.0, 11.0, 15.0])
        expected_shared = batch_shared_parameter_snapshot(sanitize_config_for_selected_model(shared))
        for cfg in (normal, fit2, fit4):
            self.assertEqual(cfg["column"]["volume_mL"], 2.5)
            self.assertEqual(batch_shared_parameter_snapshot(cfg), expected_shared)
        shared["column"]["volume_mL"] = 5.0
        changed = [apply_batch_shared_parameters(shared, batch_row_to_run_config(row))[0]
                   for row in (batch[0], batch[6], lsq[3])]
        self.assertEqual([cfg["process"]["load_CV"] for cfg in changed], [1.5, 2.5, 3.5])
        self.assertTrue(all(cfg["column"]["volume_mL"] == 5.0 for cfg in changed))

    def test_jmp_process_tables_type_every_field_before_saving(self):
        jsl = build_jsl()
        _validate_jsl(jsl)
        numeric = set(batch_run_numeric_fields())
        for field in batch_run_fields():
            kind = "Numeric" if field in numeric else "Character"
            for table in ("dtBatchRuns", "dtLSQRuns"):
                self.assertIn(f'Column({table}, "{field}") << Data Type({kind});', jsl)
        self.assertIn('Column(dtRuns, "Load_Salt_M")[currentRun] = loadSalt << Get;', jsl)

    def test_milliliters_are_authoritative_for_each_run_and_column_volume(self):
        shared = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        shared["column"]["volume_mL"] = 2.0
        rows = [default_batch_row(1), default_batch_row(2)]
        rows[0].update(Load_Amount_Basis="VOLUME_ML", Load_Volume_mL=6.0,
                       Load_CV=99.0, Load_Density_mg_mL_resin=999.0,
                       Feed_Concentration_mg_mL=2.0, PLW_Count=0,
                       Elution_Count=1, Elution1_CV=10.0)
        rows[1].update(Load_Amount_Basis="VOLUME_ML", Load_Volume_mL=2.0,
                       Load_CV=99.0, Load_Density_mg_mL_resin=999.0,
                       Feed_Concentration_mg_mL=2.0, PLW_Count=0,
                       Elution_Count=1, Elution1_CV=10.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.csv"
            save_batch_rows(rows, path)
            saved = load_batch_rows(path)
            self.assertEqual(saved[0]["Load_Volume_mL"], "6.0")
            self.assertEqual(saved[1]["Load_Volume_mL"], "2.0")
            first, _ = apply_batch_shared_parameters(shared, batch_row_to_run_config(saved[0]))
            second, _ = apply_batch_shared_parameters(shared, batch_row_to_run_config(saved[1]))
        self.assertAlmostEqual(first["process"]["load_CV"], 3.0)
        self.assertAlmostEqual(second["process"]["load_CV"], 1.0)
        self.assertAlmostEqual(first["feed"]["load_density_mg_mL_resin"], 6.0)
        self.assertAlmostEqual(second["feed"]["load_density_mg_mL_resin"], 2.0)
        self.assertEqual(validate_operating_conditions(first), [])
        program = _build_program(first, np.zeros(1))
        self.assertEqual(program.stage_end_CV, [3.0, 13.0])
        self.assertAlmostEqual(derived_input_summary(first)["required_load_volume_mL"], 6.0)
        first["column"]["volume_mL"] = 3.0
        changed = normalize_config(first)
        self.assertAlmostEqual(changed["process"]["load_CV"], 2.0)
        self.assertAlmostEqual(changed["feed"]["load_density_mg_mL_resin"], 4.0)
        self.assertAlmostEqual(changed["process"]["load_volume_mL"], 6.0)

    def test_existing_cv_and_capacity_recipes_keep_their_authority(self):
        cfg = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
        cfg["column"]["volume_mL"] = 2.0
        cfg["feed"]["total_concentration_mg_mL"] = 2.0
        cfg["process"].update(load_amount_basis="LOAD_VOLUME", load_CV=3.0,
                              load_volume_mL=999.0)
        cv = normalize_config(cfg)
        self.assertAlmostEqual(cv["process"]["load_volume_mL"], 6.0)
        self.assertAlmostEqual(cv["feed"]["load_density_mg_mL_resin"], 6.0)
        cfg["process"]["load_amount_basis"] = "CAPACITY"
        cfg["feed"]["load_density_mg_mL_resin"] = 8.0
        capacity = normalize_config(cfg)
        self.assertAlmostEqual(capacity["process"]["load_CV"], 4.0)
        self.assertAlmostEqual(capacity["process"]["load_volume_mL"], 8.0)

    def test_legacy_table_migrates_and_lsq_recipe_uses_its_own_milliliters(self):
        with tempfile.TemporaryDirectory() as directory:
            normal_path = Path(directory) / "normal.csv"
            lsq_path = Path(directory) / "lsq.csv"
            legacy_row = default_batch_row(1)
            legacy_row.pop("Load_Volume_mL")
            with normal_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(legacy_row))
                writer.writeheader(); writer.writerow(legacy_row)
            ensure_batch_runs_csv(normal_path)
            self.assertEqual(float(load_batch_rows(normal_path)[0]["Load_Volume_mL"]), 10.0)
            normal = load_batch_rows(normal_path)
            normal[0].update(Load_Amount_Basis="LOAD_VOLUME", Load_CV=3.0)
            save_batch_rows(normal, normal_path)
            lsq = load_batch_rows(normal_path)
            lsq[0].update(Load_Amount_Basis="VOLUME_ML", Load_Volume_mL=4.0,
                          Load_CV=999.0, PLW_Count=0, Elution_Count=1,
                          Elution1_CV=10.0)
            save_batch_rows(lsq, lsq_path)
            shared = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
            shared["column"]["volume_mL"] = 2.0
            normal_cfg, _ = apply_batch_shared_parameters(shared, batch_row_to_run_config(load_batch_rows(normal_path)[0]))
            lsq_cfg, _ = apply_batch_shared_parameters(shared, batch_row_to_run_config(load_batch_rows(lsq_path)[0]))
            self.assertAlmostEqual(normal_cfg["process"]["load_CV"], 3.0)
            self.assertAlmostEqual(lsq_cfg["process"]["load_CV"], 2.0)
            self.assertAlmostEqual(lsq_cfg["process"]["load_volume_mL"], 4.0)
            self.assertAlmostEqual(_build_program(lsq_cfg, np.zeros(1)).total_CV, 12.0)

    def test_wang_milliliter_recipe_reaches_solver_and_jmp_controls(self):
        shared = load_config()
        shared["model"] = MODEL_TDM_WANG
        shared["chromatography_mode"] = "HIC"
        cfg, _ = apply_batch_shared_parameters(shared, batch_row_to_run_config(default_batch_row(1)))
        cfg["column"]["volume_mL"] = 2.0
        cfg["process"].update(load_amount_basis="VOLUME_ML", load_volume_mL=6.0,
                              load_CV=99.0, plw_count=0, elution_count=1)
        cfg["process"]["elution_steps"][0]["CV"] = 10.0
        cfg["numerics"].update(time_steps=60, axial_positions=5)
        result = simulate(cfg)
        self.assertEqual(cfg["model"], MODEL_TDM_WANG)
        self.assertAlmostEqual(result["trace"]["CV"][-1], 13.0)
        self.assertTrue(np.all(np.isfinite(result["trace"]["uv_mAU"])))
        self.assertGreater(float(np.max(result["trace"]["uv_mAU"])), 0.0)
        self.assertEqual(normalize_load_amount_basis("Load volume [mL]"), "VOLUME_ML")
        jsl = build_jsl(cfg)
        _validate_jsl(jsl)
        self.assertNotIn('loadAmountBasis = Combo Box', jsl)
        self.assertIn('Column(dtRuns, "Load_Amount_Basis")[currentRun] = "CAPACITY";', jsl)
        self.assertIn('Calculated load volume [CV]', jsl)
        self.assertIn('Calculated load volume [mL]', jsl)
        self.assertIn('loadCV = Number Edit Box(., 9, << Set Width(92), << Enable(0))', jsl)
        self.assertIn('loadVolumeML = Number Edit Box(., 9, << Set Width(92), << Enable(0))', jsl)
        self.assertIn('loadBasis == "LOAD_VOLUME"', jsl)
        self.assertIn('loadBasis == "VOLUME_ML"', jsl)
        self.assertIn('cv = If(Is Missing(capacity) | Is Missing(concentration)', jsl)
        self.assertIn('loadChemPanel << Visibility("Collapse")', jsl)
        self.assertIn('loadMaterialStatus << Set Text("Load chemistry from Buffer A/B:', jsl)
        self.assertIn('loadMaterialMode << Set(1); loadChemControl << Set(1); SaveCurrentRun()', jsl)
        self.assertIn('Column(dtRuns, "Load_Volume_mL")[currentRun] = loadVolumeML << Get;', jsl)
        self.assertIn('loadVolumeML << Set(Column(dtRuns, "Load_Volume_mL")[currentRun]);', jsl)
        self.assertIn('Return(Char(Round(volume / columnVol, 0.0001)))', jsl)
        self.assertIn('": load " || LSQRunLoadCVText(r) || " CV, "', jsl)


if __name__ == "__main__":
    unittest.main()
