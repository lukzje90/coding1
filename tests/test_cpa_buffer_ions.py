"""CPA run recipes must provide the ions used for their ionic-strength calculation."""
import copy
import unittest

import numpy as np

from tdm_jmp import build_jsl, _validate_jsl
from tdm_minimal_io import (
    MODEL_EDM_CPA, MODEL_TDM_CPA, apply_batch_shared_parameters,
    batch_row_to_run_config, default_batch_row, validate_config,
)
from tdm_minimal_model import _build_program
from verify_model import make_config


class CPABufferIonTests(unittest.TestCase):
    def test_both_cpa_models_require_complete_oppositely_charged_buffer_ions(self):
        for model in (MODEL_EDM_CPA, MODEL_TDM_CPA):
            with self.subTest(model=model):
                cfg = make_config(model, 0, multistep=False)
                self.assertEqual(validate_config(cfg), [])
                proxy = copy.deepcopy(cfg)
                proxy["process"]["buffer_B"]["ionic_strength_source"] = "SALT_PROXY"
                self.assertTrue(any("Buffer B: CPA requires ION_COMPOSITION" in error
                                    for error in validate_config(proxy)))
                missing = copy.deepcopy(cfg)
                missing["process"]["buffer_A"]["ions"][1].update(
                    species="", concentration_M=None, valency=None)
                self.assertTrue(any("Buffer A: CPA needs at least one positive and one negative ion"
                                    in error for error in validate_config(missing)))
                partial = copy.deepcopy(cfg)
                partial["process"]["buffer_A"]["ions"][1]["valency"] = None
                self.assertTrue(any("Buffer A ion 2 valency" in error
                                    for error in validate_config(partial)))

    def test_each_saved_run_uses_its_own_buffer_ion_concentrations(self):
        shared = make_config(MODEL_EDM_CPA, 0, multistep=False)
        rows = []
        for run_number, salt in ((2, 0.10), (4, 0.30)):
            row = default_batch_row(run_number)
            for prefix, concentration in (("BufferA", salt), ("BufferB", 0.50)):
                row[f"{prefix}_Ionic_Strength_Source"] = "ION_COMPOSITION"
                for index, species, charge in ((1, "Na+", 1), (2, "Cl-", -1)):
                    row[f"{prefix}_Ion{index}_Species"] = species
                    row[f"{prefix}_Ion{index}_Concentration_M"] = concentration
                    row[f"{prefix}_Ion{index}_Valency"] = charge
            rows.append(row)
        for row, expected in zip(rows, (0.10, 0.30)):
            config, _ignored = apply_batch_shared_parameters(shared, batch_row_to_run_config(row))
            self.assertEqual(validate_config(config), [])
            program = _build_program(config, np.zeros(1))
            self.assertAlmostEqual(program.stages[0].start_I_M, expected)
            self.assertAlmostEqual(program.stages[0].start_salt_M, 0.05)

    def test_cpa_ion_inputs_are_visible_and_explained_in_process_setup(self):
        jsl = build_jsl(make_config(MODEL_TDM_CPA, 0, multistep=False))
        _validate_jsl(jsl)
        self.assertIn('bufferAIonSection << Visibility(If(isCPA |', jsl)
        self.assertIn('bufferBIonSection << Visibility(If(isCPA |', jsl)
        self.assertIn('CPA requires ion composition for Buffer A and B.', jsl)
        self.assertIn('CPA calculates I = 0.5 × sum(c × z²)', jsl)


if __name__ == "__main__":
    unittest.main()
