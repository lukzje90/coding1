"""JMP CPA area units are m²/mL solid; saved recipes and LSQ use m⁻¹."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import tdm_bridge
import tdm_least_squares as lsq
from tdm_jmp import build_jsl, _validate_jsl
from tdm_minimal_io import MODEL_LABELS, MODEL_TDM_CPA, load_config, save_config
from tdm_minimal_model import simulate
from verify_model import make_config


class CPAAreaUnitTests(unittest.TestCase):
    def test_old_saved_area_displays_in_new_jmp_unit_without_changing_solver_value(self):
        config = make_config(MODEL_TDM_CPA, 1, multistep=False)
        config["components"][0]["As_m_inv"] = 200_000_000.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "saved.json"
            save_config(config, path)
            loaded = load_config(path)
            jsl = build_jsl(loaded)
        _validate_jsl(jsl)
        self.assertEqual(loaded["components"][0]["As_m_inv"], 200_000_000.0)
        self.assertIn('c1As = Number Edit Box(200, 9', jsl)
        self.assertIn('Accessible area A_s,i [m²/mL solid]', jsl)
        self.assertIn('Python Name("tdm_ui_c1_As_m2_mL_solid")', jsl)

    def test_new_jmp_area_converts_once_and_old_jmp_field_remains_compatible(self):
        common = {"tdm_ui_model": MODEL_LABELS[MODEL_TDM_CPA], "tdm_ui_impurity_count": 0}
        with patch.dict(tdm_bridge.__dict__, {**common, "tdm_ui_c1_As_m2_mL_solid": 200.0,
                                               "tdm_ui_c1_As_m_inv": 1.0}):
            new_recipe = tdm_bridge.build_shared_config_from_jmp()
        self.assertEqual(new_recipe["components"][0]["As_m_inv"], 200_000_000.0)
        with patch.dict(tdm_bridge.__dict__, {**common, "tdm_ui_c1_As_m_inv": 200_000_000.0}):
            tdm_bridge.__dict__.pop("tdm_ui_c1_As_m2_mL_solid", None)
            old_recipe = tdm_bridge.build_shared_config_from_jmp()
        self.assertEqual(old_recipe["components"][0]["As_m_inv"], 200_000_000.0)

    def test_lsq_still_fits_stored_area_but_applies_and_reports_ui_unit(self):
        config = make_config(MODEL_TDM_CPA, 0, multistep=False)
        config["components"][0]["As_m_inv"] = 200_000_000.0
        path = "components[0].As_m_inv"
        specs = lsq.fit_parameter_specs(config, unlocked_paths=[path])
        spec = next(item for item in specs if item.path == path)
        self.assertAlmostEqual(spec.initial, 8.301029995663981)
        self.assertIn("[m²/mL stationary solid]", spec.label)
        self.assertEqual(lsq._display_fit_value(path, 200_000_000.0), 200.0)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "apply.jsl"
            with patch.object(lsq, "FIT_APPLY_JSL", target):
                lsq._write_apply_jsl(config, [spec])
            script = target.read_text(encoding="utf-8")
        self.assertIn("c1As << Set(200);", script)
        self.assertEqual(config["components"][0]["As_m_inv"], 200_000_000.0)

    def test_lsq_fit_report_shows_area_in_ui_units_while_fit_config_stays_legacy(self):
        config = make_config(MODEL_TDM_CPA, 0, multistep=False)
        config["numerics"].update(time_steps=50, axial_positions=6)
        config["components"][0]["As_m_inv"] = 200_000_000.0
        trace = simulate(config)["trace"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "reference.csv"
            with source.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["CV", "UV [mAU]"])
                writer.writerows(zip(trace["CV"], trace["uv_mAU"]))
            outputs = ("FIT_RESULTS_CSV", "FIT_TRACE_CSV", "FIT_CONDUCTIVITY_CSV",
                       "FIT_COMPOSITION_CSV", "FIT_REPORT_HTML", "FIT_CONFIG_JSON",
                       "FIT_APPLY_JSL", "FIT_SETTINGS_JSON", "FIT_HISTORY_CSV",
                       "FIT_IMPROVED_CSV", "FIT_IMPROVED_TXT", "CONFIG_PATH")
            with patch.multiple(lsq, **{name: root / getattr(lsq, name).name for name in outputs}):
                fit = lsq.run_global_least_squares_refinement(
                    [{"config": config, "chromatogram_csv": source,
                      "run_context": {"run_number": 1, "run_name": "Area test"}}],
                    unlocked_paths=["components[0].As_m_inv"],
                    max_nfev=3, max_model_evaluations=6)
                with lsq.FIT_RESULTS_CSV.open(newline="", encoding="utf-8") as fh:
                    area = next(row for row in csv.DictReader(fh)
                                if row["Parameter_Path"] == "components[0].As_m_inv")
                report = lsq.FIT_REPORT_HTML.read_text(encoding="utf-8")
                fitted = load_config(lsq.FIT_CONFIG_JSON)
        self.assertEqual(float(area["Initial_Value"]), 200.0)
        self.assertIn("m²/mL stationary solid", area["Parameter"])
        self.assertIn("m²/mL stationary solid", report)
        self.assertEqual(fitted["components"][0]["As_m_inv"],
                         fit["fitted_config"]["components"][0]["As_m_inv"])


if __name__ == "__main__":
    unittest.main()
