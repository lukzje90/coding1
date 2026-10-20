"""Saved JMP tables remain readable while workers open the model."""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import tdm_minimal_io as io


class SavedTableAtomicityTests(unittest.TestCase):
    def test_worker_import_does_not_rewrite_an_unchanged_run_table(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.csv"
            io.ensure_batch_runs_csv(path)
            original = path.read_bytes()
            with patch.object(io.os, "replace", side_effect=AssertionError("unexpected rewrite")):
                io.ensure_batch_runs_csv(path)
                self.assertEqual(len(io.load_batch_rows(path)), io.MAX_RUNS)
            self.assertEqual(path.read_bytes(), original)

    def test_concurrent_saves_never_expose_a_partial_run_table(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.csv"
            rows = io.load_batch_rows(path)

            def save() -> None:
                for number in range(40):
                    rows[0]["Run_Name"] = f"run {number}"
                    io.save_batch_rows(rows, path)

            def read() -> None:
                for _ in range(120):
                    with path.open("r", encoding="utf-8", newline="") as stream:
                        reader = csv.DictReader(stream)
                        self.assertIn("Run_Number", reader.fieldnames or ())
                        observed = list(reader)
                    self.assertEqual(len(observed), io.MAX_RUNS)
                    self.assertEqual([int(row["Run_Number"]) for row in observed],
                                     list(range(1, io.MAX_RUNS + 1)))

            with ThreadPoolExecutor(max_workers=2) as pool:
                future_save = pool.submit(save)
                future_read = pool.submit(read)
                future_save.result()
                future_read.result()


if __name__ == "__main__":
    unittest.main()
