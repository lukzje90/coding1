"""Saved washes migrate to one fixed setpoint while elution modes remain selectable."""
import csv
from pathlib import Path
import tempfile
import unittest

from tdm_minimal_io import (batch_row_to_run_config, batch_run_fields,
                            default_batch_row, load_batch_rows, save_batch_rows)


class PLWSetpointOnlyTests(unittest.TestCase):
    def test_existing_gradient_and_step_washes_use_their_active_setpoint(self):
        row = default_batch_row(3)
        row.update(PLW_Count=2, PLW1_Mode="LINEAR", PLW1_Start_Percent_B=20.0,
                   PLW1_End_Percent_B=80.0, PLW2_Mode="STEP",
                   PLW2_Start_Percent_B="", PLW2_End_Percent_B=65.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=batch_run_fields())
                writer.writeheader()
                writer.writerow(row)
            saved = load_batch_rows(path)[2]
            for number, setpoint in ((1, 20.0), (2, 65.0)):
                self.assertEqual(saved[f"PLW{number}_Mode"], "SET_POINT")
                self.assertEqual(float(saved[f"PLW{number}_Start_Percent_B"]), setpoint)
                self.assertEqual(float(saved[f"PLW{number}_End_Percent_B"]), setpoint)
            recipe = batch_row_to_run_config(saved)["process"]
            self.assertEqual([step["mode"] for step in recipe["plw_steps"][:2]],
                             ["SET_POINT", "SET_POINT"])
            self.assertEqual([step["start_percent_B"] for step in recipe["plw_steps"][:2]],
                             [20.0, 65.0])
            self.assertEqual(saved["Elution1_Mode"], "LINEAR")

    def test_explicit_step_chemistry_moves_end_values_to_one_wash_setpoint(self):
        row = default_batch_row(1)
        row.update(PLW1_Mode="STEP", PLW1_Chemistry_Control="ENDPOINT_CHEMISTRY",
                   PLW1_Start_Source="DIRECT", PLW1_Start_pH="",
                   PLW1_End_Source="BUFFER_B", PLW1_End_pH=5.3,
                   PLW1_End_Conductivity_mS_cm=42.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.csv"
            save_batch_rows([row], path)
            saved = load_batch_rows(path)[0]
            self.assertEqual(saved["PLW1_Mode"], "SET_POINT")
            self.assertEqual(saved["PLW1_Start_Source"], "BUFFER_B")
            self.assertEqual(float(saved["PLW1_Start_pH"]), 5.3)
            self.assertEqual(float(saved["PLW1_Start_Conductivity_mS_cm"]), 42.0)
            self.assertEqual(saved["PLW1_Start_Source"], saved["PLW1_End_Source"])


if __name__ == "__main__":
    unittest.main()
