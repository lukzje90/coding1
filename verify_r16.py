"""R16 buffer-to-solver regression tests. All protein cases are synthetic.

Run: python verify_r16.py
The JMP application is not required; generated UI structure and saved-run
handoff are tested, but this does not replace an interactive JMP smoke test.
"""
from __future__ import annotations

import copy
import csv
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from tdm_minimal_io import (
    MODELS, MODEL_EDM_CPA, MODEL_TDM_CPA, RESULT_GUARD_BUILD,
    apply_batch_shared_parameters, batch_row_to_run_config, buffer_blend_chemistry,
    default_batch_row, load_batch_rows, load_config, normalize_config,
    save_batch_rows, save_config, validate_config,
)
from tdm_minimal_model import (
    _build_program, _scalar_transport, simulate, write_outputs,
    _build_cpa_components, _cpa_model,
)
from tdm_mobile_phase import MobilePhaseTransport
from tdm_jmp import build_jsl, _validate_jsl
from verify_r13 import make_hic_config, before_elution
from verify_model import make_config

ROOT = Path(__file__).resolve().parent


def direct_buffers(config, low=0.0, high=2.0, conductivity=50.0):
    result = copy.deepcopy(config)
    for key, salt in (("buffer_A", low), ("buffer_B", high)):
        result["process"][key].update(salt_input_mode="SALT_M", salt_concentration_M=salt,
                                       conductivity_mS_cm=conductivity)
    return result


def ions_for_sulfate(salt):
    return [{"species": "NH4+", "concentration_M": 2 * salt, "valency": 1},
            {"species": "SO4--", "concentration_M": salt, "valency": -2}]


def main():
    started = time.perf_counter()
    checks, metrics = [], []

    def passed(name):
        checks.append(name)
        print("PASS", name, flush=True)

    for model in sorted(MODELS):
        cfg = direct_buffers(make_hic_config(model), low=.1, high=1.5)
        for key in ("buffer_A", "buffer_B"):
            buf = cfg["process"][key]
            buf.update(ionic_strength_source="ION_COMPOSITION", ions=ions_for_sulfate(buf["salt_concentration_M"]))
        for percent in (0.0, 25.0, 100.0):
            chem = buffer_blend_chemistry(cfg, percent)
            salt = .1 + percent / 100.0 * 1.4
            np.testing.assert_allclose([chem["salt_concentration_M"], chem["ionic_strength_M"], chem["conductivity_mS_cm"]],
                                       [salt, 3 * salt, 50.0])
        program = _build_program(cfg, np.ones(1))
        np.testing.assert_allclose(program.chemistry_at_cv(7.0)[1:4], [.8, 2.4, 50.0])
        mobile = MobilePhaseTransport(cfg, program, _scalar_transport)
        np.testing.assert_allclose(mobile.local(0.0, pore=True)[:, 1:4],
                                   np.tile([1.5, 4.5, 50.0], (cfg["numerics"]["axial_positions"], 1)))
        transported = mobile.local(program.cv_to_time(8), pore=True)
        np.testing.assert_allclose(transported[:, 2], 3 * transported[:, 1], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(transported[:, 3], 50.0, atol=1e-8)
    passed("All four models transport salt, ionic strength and conductivity independently; sulfate stoichiometry is preserved")

    with tempfile.TemporaryDirectory(dir=ROOT) as td:
        folder = Path(td)
        row = default_batch_row(1)
        row.update(BufferB_Salt_M=1.5, BufferB_Salt_Input_Mode="SALT_M", BufferB_Conductivity_mS_cm="",
                   Load_Salt_Input_Mode="SALT_M", Load_Salt_M=1.7, Load_Conductivity_mS_cm=5.0,
                   Load_Chemistry_Control="ENDPOINT_CHEMISTRY", Load_Source="DIRECT",
                   PLW1_Chemistry_Control="ENDPOINT_CHEMISTRY", PLW1_Start_Source="DIRECT",
                   PLW1_Start_Salt_Input_Mode="SALT_M", PLW1_Start_Salt_M=1.6,
                   PLW1_Start_Conductivity_mS_cm=5.0)
        row2 = {**row, "BufferB_Salt_M": .25, "Run_Number": 2}
        path = folder / "runs.csv"
        save_batch_rows([row, row2], path)
        restored = load_batch_rows(path)
        assert restored[0]["BufferB_Conductivity_mS_cm"] == ""
        for index, expected in ((0, 1.5), (1, .25)):
            cfg, _ = apply_batch_shared_parameters(make_hic_config(), batch_row_to_run_config(restored[index]))
            assert cfg["process"]["buffer_B"]["salt_concentration_M"] == expected
            assert buffer_blend_chemistry(cfg, 100)["salt_concentration_M"] == expected
            program = _build_program(cfg, np.ones(1))
            assert program.stages[0].start_salt_M == 1.7
            assert program.stages[1].start_salt_M == 1.6
        bad = folder / "bad.csv"
        bad.write_text("not a valid saved run table\n")
        original = bad.read_bytes()
        try:
            load_batch_rows(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid CSV must not be replaced by defaults")
        assert bad.read_bytes() == original
        legacy_row = {key: value for key, value in row.items() if not key.endswith("_Salt_Input_Mode")}
        save_batch_rows([legacy_row], folder / "legacy_runs.csv")
        migrated = load_batch_rows(folder / "legacy_runs.csv")[0]
        assert migrated["BufferB_Salt_Input_Mode"] == "SALT_M"
        passed("Saved-run round trip preserves blanks, chosen salt inputs, direct endpoints and distinct per-run buffers")

        cfg = direct_buffers(make_hic_config(), conductivity=None)
        assert not validate_config(cfg), validate_config(cfg)
        cfg["process"]["buffer_B"]["salt_concentration_M"] = None
        assert any("Buffer B salt concentration" in error for error in validate_config(cfg))
        cfg["process"]["buffer_B"].update(salt_input_mode="CONDUCTIVITY", salt_concentration_M=2.0)
        assert any("Buffer B conductivity" in error for error in validate_config(cfg))
        legacy = direct_buffers(make_hic_config(), conductivity=None)
        for buf in (legacy["process"]["buffer_A"], legacy["process"]["buffer_B"]):
            buf.pop("salt_input_mode")
        save_config(legacy, folder / "legacy.json")
        restored_config = load_config(folder / "legacy.json")
        assert restored_config["process"]["buffer_B"]["salt_input_mode"] == "SALT_M"
        compatibility = make_hic_config()
        compatibility["process"]["buffer_B"]["salt_concentration_M"] = .25
        assert buffer_blend_chemistry(compatibility, 100)["salt_concentration_M"] == 2.0
        for model in (MODEL_EDM_CPA, MODEL_TDM_CPA):
            cfg = make_hic_config(model)
            cfg["process"]["buffer_B"].update(ionic_strength_source="ION_COMPOSITION", ions=[
                {"species": "NH4+", "concentration_M": None, "valency": 1}])
            assert any("ion 1 concentration" in error for error in validate_config(cfg))
        passed("Selected chemistry is required with no silent fallback; legacy conductivity priority and salt-only migration work")

        cfg = direct_buffers(make_hic_config())
        step = cfg["process"]["elution_steps"][0]
        step.update(chemistry_control="ENDPOINT_CHEMISTRY", start_source="DIRECT", end_source="DIRECT",
                    start_pH=6., end_pH=6., start_salt_input_mode="SALT_M", end_salt_input_mode="SALT_M",
                    start_salt_concentration_M=1.7, end_salt_concentration_M=.2,
                    start_conductivity_mS_cm=50., end_conductivity_mS_cm=50.)
        for mode, expected in (("SET_POINT", 1.7), ("STEP", .2), ("LINEAR", .95)):
            step["mode"] = mode
            np.testing.assert_allclose(_build_program(cfg, np.ones(1)).chemistry_at_cv(7.)[1], expected)
        passed("SET_POINT, STEP and LINEAR endpoint programs use the selected analytical salt")

        jsl = build_jsl(cfg)
        _validate_jsl(jsl)
        for column, variable in (("BufferA_Salt_M", "bufferASalt"), ("BufferB_Salt_M", "bufferBSalt"),
                                 ("Load_Salt_M", "loadSalt"), ("Elution1_Start_Salt_M", "el1StartSalt")):
            assert f'Column(dtRuns, "{column}")[currentRun] = {variable} << Get;' in jsl
            assert f'Column(dtRuns, "{column}")[currentRun] = .;' not in jsl
        assert "SaveCurrentRun(); UpdateLSQRunSummary(); SendSharedModel();" in jsl
        passed("Generated JMP controls save salt and mode before calling the bridge; JSL delimiters are balanced")

    for model in sorted(MODELS):
        original = make_hic_config(model, steps=500, cells=16)
        conductivity_run = simulate(original)
        cfg = direct_buffers(original)
        gradient = simulate(cfg)
        uv_error = np.max(np.abs(gradient["trace"]["uv_mAU"] - conductivity_run["trace"]["uv_mAU"]))
        assert uv_error < .05, (model, uv_error)
        np.testing.assert_allclose(gradient["trace"]["conductivity_mS_cm"], 50., atol=1e-6)
        assert before_elution(gradient) < .01, (model, before_elution(gradient))
        mb = gradient["mass_balance"]["Target protein"]
        recovery = 100 * mb["outlet_g"] / mb["input_g"]
        assert recovery > 99.5, (model, recovery)
        assert abs(mb["closure_error_percent_of_input"]) < .05, (model, mb)
        low = simulate(direct_buffers(original, high=0.0))
        hold = simulate(direct_buffers(original, low=2.0))
        held_mb = hold["mass_balance"]["Target protein"]
        assert before_elution(low) > 90., (model, before_elution(low))
        held_recovery = 100 * held_mb["outlet_g"] / held_mb["input_g"]
        assert held_recovery < .01, (model, held_recovery)
        assert abs(held_mb["closure_error_percent_of_input"]) < .05
        output = ROOT / "verification" / "R16" / model
        output.mkdir(parents=True, exist_ok=True)
        save_config(cfg, output / "hic_direct_salt.json")
        write_outputs(cfg, gradient, output, {"run_number": 1, "run_name": "R16 independent salt regression"})
        with (output / "latest_results.csv").open(newline="") as f:
            first = next(csv.DictReader(f))
        assert float(first["binding_salt_concentration_M"]) == 2.0
        assert float(first["programmed_salt_concentration_M"]) == 2.0
        metrics.append({"model": model, "before_elution_percent": before_elution(gradient),
                        "gradient_recovery_percent": recovery,
                        "high_salt_hold_recovery_percent": held_recovery,
                        "low_salt_before_elution_percent": before_elution(low),
                        "peak_CV": float(gradient["trace"]["CV"][np.argmax(gradient["trace"]["uv_mAU"])]),
                        "closure_error_percent": mb["closure_error_percent_of_input"],
                        "equivalent_input_max_UV_difference_mAU": float(uv_error)})
        passed(model + ": high-salt retention, gradient recovery and low-salt breakthrough respond with conductivity held fixed")

    for model in (MODEL_EDM_CPA, MODEL_TDM_CPA):
        cfg = make_hic_config(model)
        cfg["chromatography_mode"] = "ION_EXCHANGE"
        cfg["cpa"].update(ligand_surface_density_umol_m2=2.89, system_specific_adsorption_parameter=.03)
        cfg["components"][0]["Z_ref"] = 40.
        native = _cpa_model(cfg, _build_cpa_components(cfg, need_tdm_kinetics=model == MODEL_TDM_CPA))
        assert native.equilibrium_factors(np.zeros(1), 6., .1)[0] > native.equilibrium_factors(np.zeros(1), 6., .5)[0]
        cfg = make_config(model, 0, multistep=False)
        for key in ("buffer_A", "buffer_B"):
            cfg["process"][key].update(ionic_strength_source="ION_COMPOSITION", ions=ions_for_sulfate(.05))
        first = simulate(cfg)
        for key in ("buffer_A", "buffer_B"):
            cfg["process"][key]["ions"] = ions_for_sulfate(.2)
        second = simulate(cfg)
        np.testing.assert_allclose(first["trace"]["salt_concentration_M"], second["trace"]["salt_concentration_M"], atol=2e-6)
        # The selected ion rows must reach the CPA binding calculation. The
        # synthetic recipe can be non-binding for some parameter combinations,
        # so the observable assertion is on local binding chemistry plus the
        # model's equilibrium factor, rather than on UV alone.
        first_I = first["trace"]["binding_ionic_strength_M"]
        second_I = second["trace"]["binding_ionic_strength_M"]
        assert np.max(np.abs(first_I - second_I)) > .1
        np.testing.assert_allclose(first_I[0], .15, atol=1e-7)
        np.testing.assert_allclose(second_I[0], .6, atol=1e-7)
    passed("Native CPA ion-exchange salt screening is retained independently of the HIC extension")
    report = {"build": RESULT_GUARD_BUILD, "checks": checks, "metrics": metrics,
              "elapsed_seconds": time.perf_counter() - started,
              "scope": "Synthetic numerical and saved-run/JSL checks; no experimental calibration or native JMP execution"}
    (ROOT / "verification" / "r16_verification.json").write_text(json.dumps(report, indent=2))
    print("ALL R16 CHEMISTRY CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
