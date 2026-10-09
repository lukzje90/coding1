"""Check the JMP b/k_s handoff and their effect on a resolved HIC elution peak."""
from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

import numpy as np

import tdm_bridge
from tdm_jmp import _validate_jsl, build_jsl
from tdm_minimal_io import RESULT_GUARD_BUILD, apply_batch_shared_parameters
from tdm_minimal_model import simulate, write_outputs
from verify_jmp_interface_inputs import _set_jmp_globals
from verify_r19_counterexample import pdf_setup

ROOT = Path(__file__).resolve().parent


def effective_run(config: dict, *, b: float, k_s: float) -> dict:
    sent = copy.deepcopy(config)
    sent["components"][0]["b_L_g"] = b
    sent["components"][0]["salt_sensitivity_per_M"] = k_s
    _set_jmp_globals(tdm_bridge, sent)
    shared = tdm_bridge.build_shared_config_from_jmp()
    run, _ignored = apply_batch_shared_parameters(shared, config)
    assert math.isclose(run["components"][0]["b_L_g"], b)
    assert math.isclose(run["components"][0]["salt_sensitivity_per_M"], k_s)
    return run


def main() -> None:
    config = pdf_setup()
    config["conversion"]["conductivity_to_salt_M_per_mS_cm"] = 0.02
    config["feed"]["load_density_mg_mL_resin"] = 0.20
    config["process"].update(load_start_percent_B=100.0, load_end_percent_B=100.0)
    config["process"]["plw_steps"][0].update(start_percent_B=100.0, end_percent_B=100.0)
    config["process"]["elution_steps"][0].update(mode="LINEAR", start_percent_B=100.0, end_percent_B=10.0)
    config["numerics"].update(time_steps=300, axial_positions=50)

    jsl = build_jsl(config)
    _validate_jsl(jsl)
    for name, control in (("b_L_g", "c1B"), ("salt_sensitivity_per_M", "c1SaltSensitivity")):
        pattern = rf'Python Send\({control} << Get, Python Name\("tdm_ui_c1_{name}"\)\)'
        assert re.search(pattern, jsl), pattern

    profiles = []
    for b, k_s in ((0.0029, 5.0), (0.0058, 5.0), (0.0029, 7.0)):
        run = effective_run(config, b=b, k_s=k_s)
        result = simulate(run)
        receipt = result["parameter_receipt"]
        assert math.isclose(receipt["required_inputs_used"]["components[0].b_L_g"], b)
        assert math.isclose(receipt["required_inputs_used"]["components[0].salt_sensitivity_per_M"], k_s)
        affinity = receipt["langmuir_affinity_used"][0]
        assert math.isclose(affinity["b_L_g"], b) and math.isclose(affinity["salt_sensitivity_per_M"], k_s)
        stage = affinity["stage_endpoints"][-1]
        assert math.isclose(stage["start_b_eff_L_g"], b * math.exp(k_s * stage["start_salt_M"]))
        assert math.isclose(stage["end_b_eff_L_g"], b * math.exp(k_s * stage["end_salt_M"]))
        salt = result["trace"]["binding_salt_concentration_M"]
        np.testing.assert_allclose(result["trace"]["component_effective_b_L_g"][:, 0], b * np.exp(k_s * salt), rtol=1e-12)
        peak = float(result["trace"]["CV"][np.argmax(result["trace"]["uv_mAU"])])
        profiles.append({"b_L_g": b, "k_s_per_M": k_s, "peak_CV": peak,
                         "elution_start_b_eff_L_g": stage["start_b_eff_L_g"],
                         "elution_end_b_eff_L_g": stage["end_b_eff_L_g"]})
        if len(profiles) == 1:
            from tempfile import TemporaryDirectory
            with TemporaryDirectory() as directory:
                paths = write_outputs(run, result, Path(directory), {"run_number": 1, "run_name": "Affinity audit"})
                import csv
                with Path(paths["csv"]).open(newline="", encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh))
                assert "Target protein_effective_b_L_g" in rows[0]
                assert math.isclose(float(rows[-1]["Target protein_effective_b_L_g"]),
                                    float(result["trace"]["component_effective_b_L_g"][-1, 0]))
                html = Path(paths["html"]).read_text(encoding="utf-8")
                assert "Competitive Langmuir affinity used" in html
                manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
                assert manifest["langmuir_affinity_used"][0]["b_L_g"] == b

    assert profiles[1]["peak_CV"] > profiles[0]["peak_CV"] + 0.2, profiles
    assert profiles[2]["peak_CV"] > profiles[0]["peak_CV"] + 1.0, profiles
    report = {"build_id": RESULT_GUARD_BUILD, "model": "EDM_LANGMUIR", "status": "PASS",
              "scope": "Generated JMP sends, bridge shared parameters, solver local affinity, result receipt/CSV, and elution peak sensitivity",
              "profiles": profiles}
    output = ROOT / "verification" / "r20_langmuir_affinity.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PASS b and k_s: JMP → bridge → local salt-dependent affinity → CSV/HTML; gradient peaks separate")
    for p in profiles:
        print(f"  b={p['b_L_g']:g} L/g, k_s={p['k_s_per_M']:g} M^-1, peak={p['peak_CV']:.3f} CV")


if __name__ == "__main__":
    main()
