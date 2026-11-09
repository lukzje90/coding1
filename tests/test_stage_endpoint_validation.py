"""A saved STEP uses its end setpoint even when the hidden start cell is blank."""
import copy
from pathlib import Path
import tempfile
import unittest

from tdm_minimal_io import MODEL_TDM_CPA, validate_config
from tdm_minimal_model import simulate, write_outputs
from verify_model import make_config


class StageEndpointValidationTests(unittest.TestCase):
    def config(self):
        config = make_config(MODEL_TDM_CPA, 0, multistep=False)
        config["numerics"].update(time_steps=50, axial_positions=6)
        config["process"]["plw_count"] = 0
        return config

    def test_step_buffer_percent_uses_end_without_hidden_start(self):
        config = self.config()
        step = config["process"]["elution_steps"][0]
        step.update(mode="STEP", start_percent_B=None, end_percent_B=80.0)
        self.assertEqual(validate_config(config, strict=True), [])
        result = simulate(config)
        self.assertAlmostEqual(float(result["trace"]["percent_B"][-1]), 80.0)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_outputs(config, result, output_dir=Path(directory))
            html = Path(paths["html"]).read_text(encoding="utf-8")
            self.assertIn("<td>Isocratic</td>", html)

        step["end_percent_B"] = None
        self.assertTrue(any("Elution 1 end Buffer B [%] is required" in error
                            for error in validate_config(config, strict=True)))

    def test_step_explicit_chemistry_uses_end_without_hidden_start(self):
        config = self.config()
        step = config["process"]["elution_steps"][0]
        step.update(mode="STEP", chemistry_control="ENDPOINT_CHEMISTRY",
                    start_source="DIRECT", start_pH=None,
                    start_salt_concentration_M=None,
                    start_conductivity_mS_cm=None, end_source="BUFFER_B")
        self.assertEqual(validate_config(config, strict=True), [])
        self.assertEqual(len(simulate(config)["trace"]["CV"]), 51)

    def test_set_point_and_linear_require_only_their_visible_endpoints(self):
        config = self.config()
        step = config["process"]["elution_steps"][0]
        step.update(mode="SET_POINT", start_percent_B=35.0, end_percent_B=None)
        self.assertEqual(validate_config(config, strict=True), [])

        linear = copy.deepcopy(config)
        linear["process"]["elution_steps"][0]["mode"] = "LINEAR"
        self.assertTrue(any("Elution 1 end Buffer B [%] is required" in error
                            for error in validate_config(linear, strict=True)))


if __name__ == "__main__":
    unittest.main()
