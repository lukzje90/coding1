"""Reproduce the R18 PDF's entered chemistry and loading/capacity ratio."""
from __future__ import annotations

import math
import json
import tempfile
from pathlib import Path

import numpy as np

from tdm_minimal_io import MODEL_EDM_LANGMUIR, validate_config
from tdm_minimal_model import _build_program, _nominal_langmuir_saturation, simulate, write_outputs
from verify_r13 import make_hic_config


def pdf_setup() -> dict:
    config = make_hic_config(MODEL_EDM_LANGMUIR, steps=50, cells=12)
    config["column"].update(volume_mL=1.0, length_mm=25.0, flow_mL_min=0.115)
    config["feed"].update(total_concentration_mg_mL=0.39, load_density_mg_mL_resin=1.3)
    config["process"].update(
        load_amount_basis="CAPACITY", load_CV=1.3 / 0.39,
        load_mode="SET_POINT", load_start_percent_B=50.0, load_end_percent_B=50.0,
        plw_count=1, elution_count=1,
    )
    config["process"]["buffer_A"].update(pH=7.0, conductivity_mS_cm=5.12, salt_input_mode="CONDUCTIVITY")
    config["process"]["buffer_B"].update(pH=6.0, conductivity_mS_cm=86.38, salt_input_mode="CONDUCTIVITY")
    config["process"]["plw_steps"][0].update(
        mode="STEP", CV=2.0, flow_mL_min=0.2,
        start_percent_B=60.0, end_percent_B=60.0,
    )
    config["process"]["elution_steps"][0].update(
        mode="STEP", CV=6.0, flow_mL_min=0.2,
        start_percent_B=10.0, end_percent_B=10.0,
    )
    config["edm"]["total_bed_porosity"] = 0.7
    config["components"][0].update(qmax_g_L=2.0, b_L_g=0.0029,
                                    salt_sensitivity_per_M=5.0, D_app_mm2_s=0.033)
    config["conversion"].update(uv_to_protein_mAU_L_g=300.0,
                                conductivity_to_salt_M_per_mS_cm=1000.0)
    config["system"].update(buffer_dispersion_mL=0.03, salt_dispersion_mm2_s=0.03,
                            uv_dead_volume_mL=0.4, conductivity_dead_volume_mL=0.4)
    return config


def main() -> None:
    config = pdf_setup()
    program = _build_program(config, np.array([0.39]))
    assert [round(s.start_salt_M, 4) for s in program.stages] == [45750.0, 53876.0, 13246.0]
    assert [round(s.duration_CV, 6) for s in program.stages] == [3.333333, 2.0, 6.0]
    saturation = _nominal_langmuir_saturation(config)
    assert saturation is not None
    assert math.isclose(saturation["target_nominal_saturation_g_L_bed"], 0.6)
    assert math.isclose(saturation["target_feed_load_g_L_bed"], 1.3)
    errors = validate_config(config)
    assert any("Buffer A resolves to 5120 M salt" in error for error in errors), errors
    assert any("Buffer B resolves to 86380 M salt" in error for error in errors), errors
    try:
        simulate(config)
    except ValueError as exc:
        assert "86380 M salt" in str(exc)
    else:
        raise AssertionError("The attached R18 setup must not generate a new HIC chromatogram.")
    print("PASS R18 PDF: factor 1000 reaches the solver; impossible chemistry is rejected")

    # At a valid fixed high salt, the same overload still produces an early
    # protein peak. This distinguishes finite resin capacity from desorption.
    config["process"]["buffer_A"].update(salt_input_mode="SALT_M", salt_concentration_M=2.0)
    config["process"]["buffer_B"].update(salt_input_mode="SALT_M", salt_concentration_M=2.0)
    result = simulate(config)
    diagnostics = result["simulation"]["diagnostics"]
    assert diagnostics["protein_outlet_before_elution_percent_of_load"] > 30.0, diagnostics
    assert any("Target load exceeds nominal stationary-phase saturation" in w for w in diagnostics["warnings"])
    assert any("breakthrough/wash loss, not an elution-stage peak" in w for w in diagnostics["warnings"])
    assert all(math.isclose(s["start_salt_M"], 2.0) and math.isclose(s["end_salt_M"], 2.0)
               for s in result["parameter_receipt"]["stage_chemistry_used"])
    assert math.isclose(result["parameter_receipt"]["load_input_used"]["nominal_langmuir_saturation"]["target_nominal_saturation_g_L_bed"], 0.6)
    with tempfile.TemporaryDirectory() as td:
        files = write_outputs(config, result, Path(td), {"run_number": 1, "run_name": "Overloaded fixed-salt run"})
        manifest = json.loads(Path(files["manifest"]).read_text(encoding="utf-8"))
        html = Path(files["html"]).read_text(encoding="utf-8")
        assert math.isclose(manifest["nominal_langmuir_saturation"]["target_nominal_saturation_g_L_bed"], 0.6)
        assert any("Target load exceeds nominal" in w for w in manifest["warnings"])
        assert "Load versus nominal Langmuir saturation" in html
    print("PASS fixed 2 M salt: overload appears as pre-elution breakthrough, with capacity in the receipt")


if __name__ == "__main__":
    main()
