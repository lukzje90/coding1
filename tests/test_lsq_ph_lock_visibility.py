"""Regression checks for pH fit-lock visibility in the generated JMP interface."""

import unittest

from tdm_jmp import _validate_jsl, build_jsl


class LSQPHLockVisibilityTests(unittest.TestCase):
    def test_selected_lsq_recipes_control_ph_lock_visibility(self):
        jsl = build_jsl()
        _validate_jsl(jsl)
        span_code = jsl.split("ComputePHSpanForRuns = Function", 1)[1].split("UpdateUI = Function", 1)[0]
        self.assertIn("Num(lsqRunCount << Get Selected)", span_code)
        self.assertIn("r = If(useSelectedLSQ, LSQSelectedRun(slot), slot);", span_code)
        self.assertIn('Column(runsTable, "Load_Material_Chemistry_Mode")[r]', span_code)
        self.assertIn("fitPHSpan = ComputePHSpanForRuns(dtLSQRuns, 1);", jsl)
        self.assertIn("fitPHSpan = Max(fitPHSpan, batchPHSpan);", jsl)
        self.assertIn("PHPanel << Visibility(If(isCPA & !isHIC & fitPHSpan > 1e-9", jsl)
        self.assertIn("Z3Row << Visibility(If(isCPA & !isHIC & fitPHSpan > 1", jsl)
        slot_code = jsl.split("ChangeLSQSlot = Function", 1)[1].split("SwitchProcessScope = Function", 1)[0]
        self.assertIn("LoadLSQReference(slot, r);\n    UpdateUI();", slot_code)


if __name__ == "__main__":
    unittest.main()
