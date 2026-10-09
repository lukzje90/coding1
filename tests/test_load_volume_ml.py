"""Per-run mL loading must survive CSV storage and reach simulation and LSQ."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tdm_jmp import _validate_jsl, build_jsl
from tdm_minimal_io import (
    MODEL_EDM_LANGMUIR, MODEL_TDM_WANG, apply_batch_shared_parameters,
    batch_row_to_run_config, default_batch_row, derived_input_summary,
    ensure_batch_runs_csv, load_batch_rows, load_config, normalize_config,
    normalize_load_amount_basis, save_batch_rows, validate_operating_conditions,
)
from tdm_minimal_model import _build_program, simulate
from verify_model import make_config


class LoadVolumeMLTests(unittest.TestCase):
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
        self.assertIn('loadAmountBasis = Combo Box({"Capacity [g/L resin]", "Load volume [CV]", "Load volume [mL]"}', jsl)
        self.assertIn('Column(dtRuns, "Load_Volume_mL")[currentRun] = loadVolumeML << Get;', jsl)
        self.assertIn('loadVolumeML << Set(Column(dtRuns, "Load_Volume_mL")[currentRun]);', jsl)


if __name__ == "__main__":
    unittest.main()
