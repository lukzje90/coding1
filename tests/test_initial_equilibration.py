"""Initial column chemistry is run-specific and feeds every column solver."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tdm_jmp import _validate_jsl, build_jsl
from tdm_minimal_io import (
    MODEL_EDM_CPA, MODEL_EDM_LANGMUIR, MODEL_TDM_CPA,
    MODEL_TDM_LANGMUIR, MODEL_TDM_WANG,
    LSQ_MODEL_SOURCE_BATCH, LSQ_MODEL_SOURCE_INDEPENDENT,
    apply_batch_shared_parameters, batch_row_to_run_config,
    batch_run_fields, default_batch_row, load_batch_rows,
    process_pH_span, save_batch_rows, validate_config,
)
from tdm_minimal_model import simulate, write_outputs
import tdm_least_squares as lsq
import tdm_bridge
from verify_model import make_config


MODELS = (MODEL_TDM_CPA, MODEL_TDM_WANG, MODEL_TDM_LANGMUIR,
          MODEL_EDM_CPA, MODEL_EDM_LANGMUIR)


def small_config(model):
    config = make_config(model, 0, multistep=False)
    config["numerics"].update(time_steps=50, axial_positions=6)
    if model == MODEL_TDM_WANG:
        config["chromatography_mode"] = "HIC"
    return config


class InitialEquilibrationTests(unittest.TestCase):
    def test_buffer_b_initial_liquid_reaches_all_five_models_and_changes_chromatogram(self):
        for model in MODELS:
            with self.subTest(model=model):
                baseline = small_config(model)
                original = simulate(baseline)
                changed = copy.deepcopy(baseline)
                changed["process"]["equilibration_source"] = "BUFFER_B"
                self.assertEqual(validate_config(changed, strict=True), [])
                result = simulate(changed)
                initial = result["parameter_receipt"]["initial_equilibration_used"]
                self.assertEqual(initial["source"], "BUFFER_B")
                self.assertAlmostEqual(initial["salt_concentration_M"], 0.5)
                self.assertAlmostEqual(result["trace"]["binding_salt_concentration_M"][0], 0.5)
                self.assertAlmostEqual(original["trace"]["binding_salt_concentration_M"][0], 0.05)
                self.assertGreater(np.max(np.abs(result["trace"]["uv_mAU"] - original["trace"]["uv_mAU"])), 1e-4)

    def test_direct_input_uses_only_selected_salt_authority_and_initial_pH(self):
        config = small_config(MODEL_TDM_CPA)
        process = config["process"]
        process.update(equilibration_source="DIRECT", equilibration_pH=6.0,
                       equilibration_salt_input_mode="SALT_M",
                       equilibration_salt_concentration_M=0.2,
                       equilibration_conductivity_mS_cm=80.0)
        result = simulate(config)
        initial = result["parameter_receipt"]["initial_equilibration_used"]
        self.assertAlmostEqual(initial["salt_concentration_M"], 0.2)
        self.assertAlmostEqual(initial["ionic_strength_M"], 0.2)
        self.assertAlmostEqual(initial["conductivity_mS_cm"], 80.0)
        self.assertAlmostEqual(result["trace"]["binding_salt_concentration_M"][0], 0.2)
        with tempfile.TemporaryDirectory() as directory:
            exported = write_outputs(config, result, output_dir=directory)
            self.assertIn("Initial column equilibration", Path(exported["html"]).read_text())
            manifest = json.loads(Path(exported["manifest"]).read_text())
            self.assertEqual(manifest["initial_equilibration"]["source"], "DIRECT")
            self.assertAlmostEqual(manifest["initial_equilibration"]["salt_concentration_M"], 0.2)
        process["equilibration_salt_input_mode"] = "CONDUCTIVITY"
        result = simulate(config)
        initial = result["parameter_receipt"]["initial_equilibration_used"]
        self.assertAlmostEqual(initial["salt_concentration_M"], 0.8)
        self.assertAlmostEqual(result["trace"]["binding_salt_concentration_M"][0], 0.8)

    def test_initial_pH_extends_cpa_parameter_contract(self):
        config = small_config(MODEL_TDM_CPA)
        config["process"].update(equilibration_source="DIRECT", equilibration_pH=7.5,
                                 equilibration_salt_input_mode="SALT_M",
                                 equilibration_salt_concentration_M=0.05)
        self.assertAlmostEqual(process_pH_span(config), 1.5)
        errors = validate_config(config, strict=True)
        self.assertTrue(any("Z3" in error for error in errors))
        config["components"][0].update(pH_ref=6.0, Z1_per_pH=0.0,
                                        Z2_per_pH2=0.0, Z3_per_pH3=0.0,
                                        delta_pH_slope_m2_C=0.0)
        result = simulate(config)
        self.assertAlmostEqual(result["parameter_receipt"]["initial_equilibration_used"]["pH"], 7.5)
        self.assertAlmostEqual(result["trace"]["pH"][0], 7.5)

    def test_per_run_csv_migration_and_jmp_round_trip_fields(self):
        fields = [name for name in batch_run_fields() if not name.startswith("Equilibration_")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy_runs.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow({name: default_batch_row(1)[name] for name in fields})
            migrated = load_batch_rows(path)
            self.assertEqual(migrated[0]["Equilibration_Source"], "MATCH_LOAD")
            migrated[0]["Equilibration_Source"] = "BUFFER_A"
            migrated[1]["Equilibration_Source"] = "BUFFER_B"
            migrated[2].update(Equilibration_Source="DIRECT", Equilibration_pH=6.4,
                               Equilibration_Salt_Input_Mode="SALT_M",
                               Equilibration_Salt_M=0.3)
            save_batch_rows(migrated, path)
            rows = load_batch_rows(path)
            self.assertEqual([batch_row_to_run_config(row)["process"]["equilibration_source"]
                              for row in rows[:3]], ["BUFFER_A", "BUFFER_B", "DIRECT"])
            self.assertAlmostEqual(batch_row_to_run_config(rows[2])["process"]["equilibration_salt_concentration_M"], 0.3)
            shared = small_config(MODEL_TDM_CPA)
            configs = [apply_batch_shared_parameters(shared, batch_row_to_run_config(row))[0]
                       for row in rows[:3]]
            self.assertEqual([config["process"]["equilibration_source"] for config in configs],
                             ["BUFFER_A", "BUFFER_B", "DIRECT"])
            lsq_row = dict(rows[0])
            lsq_row.update(LSQ_Model_Source=LSQ_MODEL_SOURCE_BATCH, LSQ_Model_Run_Number=2)
            model_run, _ = tdm_bridge._lsq_model_run_config(shared, 1, lsq_row, rows)
            self.assertEqual(model_run["process"]["equilibration_source"], "BUFFER_B")
            lsq_row["LSQ_Model_Source"] = LSQ_MODEL_SOURCE_INDEPENDENT
            model_run, _ = tdm_bridge._lsq_model_run_config(shared, 1, lsq_row, rows)
            self.assertEqual(model_run["process"]["equilibration_source"], "BUFFER_A")

        jsl = build_jsl(small_config(MODEL_TDM_CPA))
        _validate_jsl(jsl)
        for field in ("Equilibration_Source", "Equilibration_pH", "Equilibration_Salt_M",
                      "Equilibration_Conductivity_mS_cm", "Equilibration_Salt_Input_Mode"):
            self.assertIn(f'Column(dtRuns, "{field}")[currentRun] =', jsl)
            self.assertIn(f'Column(dtRuns, "{field}")[currentRun]', jsl)
        for field in ("Equilibration_pH", "Equilibration_Salt_M", "Equilibration_Conductivity_mS_cm"):
            self.assertIn(f'Column(dtLSQRuns, "{field}") << Data Type(Numeric);', jsl)

    def test_direct_initial_chemistry_must_be_complete(self):
        config = small_config(MODEL_EDM_LANGMUIR)
        config["process"]["equilibration_source"] = "DIRECT"
        errors = validate_config(config, strict=True)
        self.assertTrue(any("Initial equilibration pH" in error for error in errors))
        self.assertTrue(any("Initial equilibration conductivity" in error for error in errors))

    def test_lsq_preview_and_fit_keep_each_runs_initial_chemistry(self):
        shared = small_config(MODEL_EDM_LANGMUIR)
        rows = [default_batch_row(1), default_batch_row(2)]
        for number, row in enumerate(rows, start=1):
            row.update(Load_CV=0.4, PLW_Count=0, Elution_Count=1,
                       Elution1_CV=0.2, Time_Steps=50, Axial_Positions=6,
                       Equilibration_Source="BUFFER_A" if number == 1 else "BUFFER_B")
        profiles = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for number, row in enumerate(rows, start=1):
                config, context = tdm_bridge._lsq_model_run_config(shared, number, row, rows)
                trace = simulate(config)["trace"]
                reference = root / f"run_{number}.csv"
                with reference.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["CV", "UV 280 [mAU]"])
                    writer.writerows(zip(trace["CV"], trace["uv_mAU"]))
                profiles.append({"config": config, "chromatogram_csv": reference,
                                 "run_context": {"run_number": number,
                                                 "run_name": f"LSQ Run {number}", **context}})
            preview = lsq.preview_lsq_alignment(profiles, output_html=root / "alignment.html")
            self.assertEqual([item["initial_equilibration"]["source"] for item in preview["profiles"]],
                             ["BUFFER_A", "BUFFER_B"])
            self.assertTrue(all(item["uv_rmse_mAU"] < 1e-5 for item in preview["profiles"]))
            self.assertIn("Initial column equilibration", (root / "alignment.html").read_text())

            output_paths = {name: root / getattr(lsq, name).name for name in (
                "FIT_RESULTS_CSV", "FIT_TRACE_CSV", "FIT_COMPOSITION_CSV",
                "FIT_REPORT_HTML", "FIT_CONFIG_JSON", "FIT_APPLY_JSL",
                "FIT_SETTINGS_JSON", "FIT_HISTORY_CSV", "FIT_IMPROVED_CSV",
                "FIT_IMPROVED_TXT", "CONFIG_PATH")}
            with patch.multiple(lsq, **output_paths):
                result = lsq.run_global_least_squares_refinement(
                    profiles, unlocked_paths=set(), max_nfev=2)
            settings = json.loads(Path(result["fit_settings_json"]).read_text())
            self.assertEqual([item["recipe"]["process"]["equilibration_source"]
                              for item in settings["profiles"]], ["BUFFER_A", "BUFFER_B"])
            self.assertEqual(result["nfev"], 0)
            self.assertAlmostEqual(result["initial_objective"], 0.0, delta=1e-8)

    def test_tdm_cpa_fits_ph_parameter_required_only_by_later_run(self):
        first = small_config(MODEL_TDM_CPA)
        first["components"][0].update(pH_ref=6.0, Z1_per_pH=5.0,
                                      Z2_per_pH2=0.0, Z3_per_pH3=0.0,
                                      delta_pH_slope_m2_C=0.0)
        second = copy.deepcopy(first)
        second["process"].update(equilibration_source="DIRECT", equilibration_pH=7.5,
                                 equilibration_salt_input_mode="SALT_M",
                                 equilibration_salt_concentration_M=0.05)
        fit_path = "components[0].Z1_per_pH"
        from tdm_minimal_io import required_parameter_paths
        self.assertNotIn(fit_path, required_parameter_paths(first))
        self.assertIn(fit_path, required_parameter_paths(second))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = []
            for number, truth in enumerate((first, second), start=1):
                trace = simulate(truth)["trace"]
                source = root / f"measured_{number}.csv"
                with source.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["CV", "UV 280 [mAU]"])
                    writer.writerows(zip(trace["CV"], trace["uv_mAU"]))
                initial = copy.deepcopy(truth)
                initial["components"][0]["Z1_per_pH"] = 4.0
                profiles.append({"config": initial, "chromatogram_csv": source,
                                 "run_context": {"run_number": number,
                                                 "run_name": f"LSQ Run {number}"}})
            output_names = ("FIT_RESULTS_CSV", "FIT_TRACE_CSV", "FIT_COMPOSITION_CSV",
                            "FIT_CONDUCTIVITY_CSV", "FIT_REPORT_HTML", "FIT_CONFIG_JSON",
                            "FIT_APPLY_JSL", "FIT_SETTINGS_JSON", "FIT_HISTORY_CSV",
                            "FIT_IMPROVED_CSV", "FIT_IMPROVED_TXT", "CONFIG_PATH")
            with patch.multiple(lsq, **{name: root / getattr(lsq, name).name
                                        for name in output_names}):
                result = lsq.run_global_least_squares_refinement(
                    profiles, unlocked_paths=[fit_path], max_nfev=12,
                    max_model_evaluations=30)
                apply_jsl = lsq.FIT_APPLY_JSL.read_text(encoding="utf-8")
            self.assertEqual(result["parameters"], [fit_path])
            self.assertTrue(result["accepted"])
            self.assertLess(result["final_objective"], result["initial_objective"] * 0.01)
            self.assertAlmostEqual(result["fitted_config"]["components"][0]["Z1_per_pH"],
                                   5.0, delta=0.2)
            self.assertIn("c1Z1 << Set(", apply_jsl)
            self.assertLess(result["profile_metrics"][1]["raw_rmse"], 0.1)


if __name__ == "__main__":
    unittest.main()
