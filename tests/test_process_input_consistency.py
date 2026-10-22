"""Check saved process counts, geometry units, and variable-flow fractions."""
import copy
import math
import unittest

import numpy as np

import tdm_least_squares as lsq
from tdm_minimal_io import (
    MODEL_EDM_LANGMUIR, batch_row_to_run_config, default_batch_row,
    validate_operating_conditions,
)
from tdm_minimal_model import simulate
from verify_model import make_config


class ProcessInputConsistencyTests(unittest.TestCase):
    def test_truncated_or_fractional_active_stages_are_rejected(self):
        config = make_config(MODEL_EDM_LANGMUIR, 1, multistep=False)
        config["process"]["plw_count"] = 2
        config["process"]["plw_steps"] = config["process"]["plw_steps"][:1]
        self.assertIn("declares 2 active stages", " ".join(validate_operating_conditions(config)))
        with self.assertRaisesRegex(ValueError, "declares 2 active stages"):
            simulate(config)

        fractional = copy.deepcopy(config)
        fractional["process"]["plw_count"] = 1.5
        self.assertIn("whole number", " ".join(validate_operating_conditions(fractional)))
        for invalid in ("1.5", "6", "NaN", "wrong"):
            row = default_batch_row(1)
            row["Elution_Count"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "Elution_Count must be a whole number"):
                batch_row_to_run_config(row)

    def test_geometry_receipt_uses_entered_bed_length_and_column_volume(self):
        config = make_config(MODEL_EDM_LANGMUIR, 1, multistep=False)
        config["numerics"].update(time_steps=50, axial_positions=6)
        config["column"].update(volume_mL=2.0, length_mm=100.0, flow_mL_min=4.0)
        result = simulate(config)
        geometry = result["parameter_receipt"]["column_geometry_used"]
        self.assertAlmostEqual(geometry["bed_length_mm"], 100.0)
        self.assertAlmostEqual(geometry["column_volume_mL"], 2.0)
        self.assertAlmostEqual(geometry["cross_sectional_area_mm2"], 20.0)
        self.assertAlmostEqual(geometry["derived_internal_diameter_mm"], 2 * math.sqrt(20 / math.pi))
        self.assertAlmostEqual(geometry["load_residence_time_min"], 0.5)

    def test_fraction_cv_integral_matches_flow_times_concentration(self):
        config = make_config(MODEL_EDM_LANGMUIR, 1, multistep=False)
        config["column"]["volume_mL"] = 2.0
        # 0–1 CV at 2 mL/min, then 1–2 CV at 4 mL/min.
        times_min = np.array([0.0, 1.0, 1.5])
        cv = np.array([0.0, 1.0, 2.0])
        concentrations = np.array([[0.0, 1.0], [10.0, 1.0], [0.0, 1.0]])
        trace = {"CV": cv, "component_g_L": concentrations}
        fraction = {"fraction_index": 1, "start": 0.5, "end": 1.5,
                    "x_unit": "CV", "CV": 1.0}
        predicted, quantifiable = lsq._predicted_fraction_mass_percent(config, trace, fraction)
        self.assertTrue(quantifiable)
        boundary_cv = np.array([0.5, 1.0, 1.5])
        boundary_time = np.interp(boundary_cv, cv, times_min)
        boundary_conc = np.column_stack([
            np.interp(boundary_cv, cv, concentrations[:, index]) for index in range(2)
        ])
        flow = np.array([2.0, 4.0])
        recovered = np.sum((boundary_conc[:-1] + boundary_conc[1:]) / 2
                           * np.diff(boundary_time)[:, None] * flow[:, None], axis=0)
        np.testing.assert_allclose(predicted, 100 * recovered / recovered.sum(), atol=1e-12)

    def test_fraction_percentages_cannot_exceed_recovered_mass(self):
        config = make_config(MODEL_EDM_LANGMUIR, 1, multistep=False)
        names = [species["name"] for species in config["components"][:2]]
        with self.assertRaisesRegex(ValueError, "mass% total exceeds 100%"):
            lsq.validate_composition_references(config, [{
                "start": 0.1, "end": 0.2,
                "mass_percent": {names[0]: 70.0, names[1]: 31.0},
            }])
        partial = lsq.validate_composition_references(config, [{
            "start": 0.1, "end": 0.2, "mass_percent": {names[0]: 70.0},
        }])
        self.assertEqual(partial[0]["mass_percent"], {names[0]: 70.0})


if __name__ == "__main__":
    unittest.main()
