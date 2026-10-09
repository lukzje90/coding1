"""Paper Eq. 4, JMP buffer inputs, and %B/salt independence regression."""
from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import tdm_bridge
from tdm_jmp import _validate_jsl, build_jsl
from tdm_minimal_io import (
    MODEL_EDM_LANGMUIR, MODEL_TDM_LANGMUIR, RESULT_GUARD_BUILD,
    apply_batch_shared_parameters, batch_row_to_run_config, default_batch_row,
    load_batch_rows, save_batch_rows, validate_config,
)
from tdm_minimal_model import _langmuir_c_q_from_total, _langmuir_q_jac, simulate
from verify_jmp_interface_inputs import _set_jmp_globals
from verify_r13 import make_hic_config

ROOT = Path(__file__).resolve().parent


def check_paper_equation() -> dict:
    # Mutavdzin & Seidel-Morgenstern (2024), Eq. 4: each component's qmax*b*c
    # shares exactly one denominator. The empirical HIC extension applies the
    # same local salt multiplier to b and H=qmax*b, preserving that equation.
    qmax = np.array([2.0, 3.0])
    b = np.array([0.04, 0.10])
    k_s = np.array([2.0, 1.0])
    c = np.array([0.6, 0.3])
    H = qmax * b
    for salt in (0.0, 0.5, 2.0):
        b_eff = b * np.exp(k_s * salt)
        reference = qmax * b_eff * c / (1.0 + np.dot(b_eff, c))
        q, jac = _langmuir_q_jac(c, H, b, salt_M=salt, salt_sensitivity_per_M=k_s)
        np.testing.assert_allclose(q, reference, rtol=1e-13, atol=1e-14)
        assert jac[0, 1] < 0 and jac[1, 0] < 0  # shared-site competition
        for phase_ratio in (0.3, 1.5):
            totals = c + phase_ratio * reference
            recovered, bound = _langmuir_c_q_from_total(
                totals, H, b, salt_M=salt, salt_sensitivity_per_M=k_s,
                phase_ratio=phase_ratio,
            )
            np.testing.assert_allclose(recovered, c, rtol=1e-11, atol=1e-13)
            np.testing.assert_allclose(bound, reference, rtol=1e-11, atol=1e-13)

    # Previously, a residual normalized by an enormous affinity denominator
    # could accept a materially wrong free concentration near saturation.
    qmax = np.array([68.17302568, 23.81230291])
    b = np.array([0.06557271, 0.0001042])
    k_s = np.array([23.38614197, 4.04043706])
    c = np.array([1e-8, 6.736e-7])
    salt = 2.943821369341098
    phase_ratio = 2.2790337770615676
    q, _ = _langmuir_q_jac(c, qmax * b, b, salt_M=salt, salt_sensitivity_per_M=k_s)
    recovered, bound = _langmuir_c_q_from_total(
        c + phase_ratio * q, qmax * b, b,
        salt_M=salt, salt_sensitivity_per_M=k_s, phase_ratio=phase_ratio,
    )
    np.testing.assert_allclose(recovered, c, rtol=1e-4, atol=1e-13)
    np.testing.assert_allclose(bound, q, rtol=1e-8, atol=1e-13)
    return {"equation": "q_i=qmax_i*b_i*exp(k_s_i*m)*c_i/(1+sum_j b_j*exp(k_s_j*m)*c_j)",
            "salt_values_M": [0.0, 0.5, 2.0], "high_affinity_inversion": "PASS"}


def _saved_run(**changes: object) -> dict:
    row = default_batch_row(1)
    row.update({
        "Load_Amount_Basis": "LOAD_VOLUME", "Load_CV": 0.5,
        "Feed_Concentration_mg_mL": 0.2, "Load_Density_mg_mL_resin": 0.1,
        "Load_Mode": "SET_POINT", "Load_Start_Percent_B": 100.0,
        "Load_End_Percent_B": 100.0, "PLW_Count": 1, "PLW1_CV": 1.0,
        "PLW1_Start_Percent_B": 100.0, "PLW1_End_Percent_B": 100.0,
        "Elution_Count": 1, "Elution1_Mode": "STEP", "Elution1_CV": 3.0,
        "Elution1_Start_Percent_B": 100.0, "Elution1_End_Percent_B": 10.0,
        "BufferA_Salt_Input_Mode": "SALT_M", "BufferA_Salt_M": 2.0,
        "BufferB_Salt_Input_Mode": "SALT_M", "BufferB_Salt_M": 2.0,
        "Time_Steps": 240, "Axial_Positions": 12,
    })
    row.update(changes)
    with TemporaryDirectory() as directory:
        csv = Path(directory) / "jmp_batch_runs.csv"
        save_batch_rows([row], csv)
        return batch_row_to_run_config(load_batch_rows(csv)[0])


def check_solver(model: str) -> dict:
    seed = make_hic_config(model, steps=240, cells=12)
    jsl = build_jsl(seed)
    _validate_jsl(jsl)
    for token in (
        'Python Send(c1B << Get, Python Name("tdm_ui_c1_b_L_g"))',
        'Python Send(c1SaltSensitivity << Get, Python Name("tdm_ui_c1_salt_sensitivity_per_M"))',
        'Column(dtRuns, "BufferA_Salt_M")[currentRun] = bufferASalt << Get;',
        'Column(dtRuns, "BufferB_Salt_M")[currentRun] = bufferBSalt << Get;',
        'TargetLangmuirBEff = Function', 'Pump salt used (',
    ):
        assert token in jsl, token
    _set_jmp_globals(tdm_bridge, seed)
    shared = tdm_bridge.build_shared_config_from_jmp()
    assert shared["components"][0]["b_L_g"] == seed["components"][0]["b_L_g"]
    assert shared["components"][0]["salt_sensitivity_per_M"] == seed["components"][0]["salt_sensitivity_per_M"]

    effective, _ = apply_batch_shared_parameters(shared, _saved_run())
    assert not validate_config(effective), validate_config(effective)
    stage = simulate(effective)
    assert math.isclose(stage["parameter_receipt"]["stage_chemistry_used"][-1]["end_salt_M"], 2.0)
    np.testing.assert_allclose(stage["trace"]["binding_salt_concentration_M"], 2.0, atol=1e-8)
    b_eff = seed["components"][0]["b_L_g"] * math.exp(seed["components"][0]["salt_sensitivity_per_M"] * 2.0)
    np.testing.assert_allclose(stage["trace"]["component_effective_b_L_g"][:, 0], b_eff, rtol=1e-8)

    at_100, _ = apply_batch_shared_parameters(shared, _saved_run(Elution1_End_Percent_B=100.0))
    hold = simulate(at_100)
    np.testing.assert_allclose(stage["trace"]["uv_mAU"], hold["trace"]["uv_mAU"], rtol=1e-4, atol=1e-8)
    assert stage["mass_balance"]["Target protein"]["outlet_g"] < 0.001 * stage["mass_balance"]["Target protein"]["input_g"]

    # Reducing %B with a *low-salt A* genuinely reduces delivered salt.
    diluted, _ = apply_batch_shared_parameters(shared, _saved_run(BufferA_Salt_M=0.0))
    elution = simulate(diluted)
    assert math.isclose(elution["parameter_receipt"]["stage_chemistry_used"][-1]["end_salt_M"], 0.2)
    assert elution["mass_balance"]["Target protein"]["outlet_g"] > 0.8 * elution["mass_balance"]["Target protein"]["input_g"]
    assert elution["trace"]["binding_salt_concentration_M"][-1] < 0.25
    return {"model": model, "both_buffers_2M_low_B_end_salt_M": 2.0,
            "both_buffers_2M_low_B_recovery_percent": 100 * stage["mass_balance"]["Target protein"]["outlet_g"] / stage["mass_balance"]["Target protein"]["input_g"],
            "A_0M_B_2M_low_B_end_salt_M": 0.2,
            "A_0M_B_2M_low_B_recovery_percent": 100 * elution["mass_balance"]["Target protein"]["outlet_g"] / elution["mass_balance"]["Target protein"]["input_g"],
            "target_b_eff_at_2M_L_g": b_eff}


def main() -> None:
    report = {"build_id": RESULT_GUARD_BUILD, "paper_equation": check_paper_equation(),
              "JMP_to_solver": [check_solver(m) for m in (MODEL_EDM_LANGMUIR, MODEL_TDM_LANGMUIR)],
              "status": "PASS"}
    (ROOT / "verification" / "r22_paper_salt_equilibrium.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PASS paper Eq. 4 and high-salt inversion; both JMP-fed Langmuir solvers use actual blended salt")
    for row in report["JMP_to_solver"]:
        print(f"  {row['model']}: A=B=2 M, 10% B recovery={row['both_buffers_2M_low_B_recovery_percent']:.6g}%; "
              f"A=0/B=2 M, 10% B recovery={row['A_0M_B_2M_low_B_recovery_percent']:.3f}%")


if __name__ == "__main__":
    main()
