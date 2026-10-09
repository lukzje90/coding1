"""Audit generated JMP fields through the Python bridge and solver receipt.

Run from this directory with ``python verify_jmp_interface_inputs.py``.
The test uses synthetic HIC inputs and does not require a JMP installation.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

from tdm_jmp import _validate_jsl, build_jsl
from tdm_minimal_io import (
    MODELS,
    MODEL_LABELS,
    MAX_HIC_SALT_M,
    RESULT_GUARD_BUILD,
    _get_path,
    apply_batch_shared_parameters,
    batch_row_to_run_config,
    batch_run_fields,
    default_batch_row,
    load_batch_rows,
    required_parameter_values,
    save_batch_rows,
    uses_langmuir,
    validate_config,
)
from tdm_minimal_model import _build_program, simulate
from verify_r13 import make_hic_config

ROOT = Path(__file__).resolve().parent

COMPONENT_FIELDS = (
    "name", "mass_percent", "uv_response_factor_mAU_L_g", "D_ax_mm2_s",
    "k_eff_um_s", "accessible_particle_porosity", "D_app_mm2_s", "qmax_g_L",
    "b_L_g", "salt_sensitivity_per_M", "diameter_nm", "As_m_inv", "Z_ref",
    "delta_ref", "kkin_star_s", "pH_ref", "Z1_per_pH", "Z2_per_pH2",
    "Z3_per_pH3", "delta_pH_slope_m2_C",
)


def _set_jmp_globals(bridge, config: dict) -> None:
    """Populate bridge globals using the exact names emitted by Python Send."""
    values = {
        "tdm_ui_model": MODEL_LABELS[config["model"]],
        "tdm_ui_impurity_count": config["impurity_count"],
        "tdm_ui_chromatography_mode": "HIC",
        "tdm_ui_column_volume_mL": config["column"]["volume_mL"],
        "tdm_ui_column_length_mm": config["column"]["length_mm"],
        "tdm_ui_bead_radius": config["tdm"]["bead_radius_um"],
        "tdm_ui_void_fraction": config["tdm"]["void_fraction"],
        "tdm_ui_particle_porosity": config["tdm"]["particle_porosity"],
        "tdm_ui_salt_Dax": config["tdm"]["salt_axial_dispersion_mm2_s"],
        "tdm_ui_edm_porosity": config["edm"]["total_bed_porosity"],
        "tdm_ui_uv_to_protein_factor": config["conversion"]["uv_to_protein_mAU_L_g"],
        "tdm_ui_cond_to_salt_factor": config["conversion"]["conductivity_to_salt_M_per_mS_cm"],
        "tdm_ui_ligand_surface_density": config["cpa"]["ligand_surface_density_umol_m2"],
        "tdm_ui_system_adsorption_parameter": config["cpa"]["system_specific_adsorption_parameter"],
        "tdm_ui_lsq_objective": "RAW_SSE",
        "tdm_ui_lsq_baseline_mode": "NONE",
        "tdm_ui_lsq_detector_baseline": 0.0,
    }
    for field, value in config["system"].items():
        values[f"tdm_ui_system_{field}"] = ("ON" if value else "OFF") if isinstance(value, bool) else value
    for index, component in enumerate(config["components"], start=1):
        for field in COMPONENT_FIELDS:
            values[f"tdm_ui_c{index}_{field}"] = component.get(field)
    for name, value in values.items():
        setattr(bridge, name, value)


def _check_generated_contract(config: dict) -> dict:
    jsl = build_jsl(config)
    _validate_jsl(jsl)
    sent_names = set(re.findall(r'Python Name\("([^"]+)"\)', jsl))
    bridge_source = (ROOT / "tdm_bridge.py").read_text(encoding="utf-8")
    bridge_literals = set(re.findall(r'"(tdm_ui_[^"]+)"', bridge_source))
    required = {
        "tdm_ui_model", "tdm_ui_chromatography_mode", "tdm_ui_cond_to_salt_factor",
        "tdm_ui_system_salt_dispersion_mm2_s", "tdm_ui_system_buffer_dispersion_mL",
        "tdm_ui_c1_b_L_g", "tdm_ui_c2_b_L_g",
        "tdm_ui_c1_salt_sensitivity_per_M", "tdm_ui_c2_salt_sensitivity_per_M",
    }
    missing = sorted(required - sent_names)
    if missing:
        raise AssertionError(f"Generated JMP script omits bridge inputs: {missing}")
    for name in sent_names:
        if not name.startswith("tdm_ui_") or name in bridge_literals:
            continue
        dynamic_reader = (
            re.fullmatch(r"tdm_ui_c[1-6]_[A-Za-z0-9_]+", name)
            or re.fullmatch(r"tdm_ui_system_(?:buffer_dispersion_mL|salt_dispersion_mm2_s|uv_dead_volume_mL|conductivity_dead_volume_mL|buffer_dispersion_enabled|salt_dispersion_enabled|show_percent_B|show_conductivity|show_flow_rate)", name)
            or re.fullmatch(r"tdm_ui_lsq_selected_run_[1-5]", name)
            or re.fullmatch(r"tdm_ui_lsq_r(?:[1-9]|10)_CV", name)
            or re.fullmatch(r"tdm_ui_lsq_r(?:[1-9]|10)_c[1-6]_mass_percent", name)
        )
        if not dynamic_reader:
            raise AssertionError(f"Generated JMP value has no bridge reader: {name}")

    schema = set(batch_run_fields())
    saved = set(re.findall(r'Column\(dtRuns,\s*"([^"]+)"\)\[currentRun\]\s*=', jsl))
    loaded = set(re.findall(r'Column\(dtRuns,\s*"([^"]+)"\)\[currentRun\]\)', jsl))
    # Run_Number is assigned from the selected row index, rather than read back.
    if schema - saved:
        raise AssertionError(f"JMP fields not saved to the run table: {sorted(schema - saved)}")
    if (schema - {"Run_Number"}) - loaded:
        raise AssertionError(f"JMP fields not restored from the run table: {sorted((schema - {'Run_Number'}) - loaded)}")
    if "SaveCurrentRun(); UpdateLSQRunSummary(); SendSharedModel();" not in jsl:
        raise AssertionError("Run action must save the current profile before sending shared model inputs")
    if "saltFactorStatus << Set Text" not in jsl or "In CONDUCTIVITY mode, a 0 mS/cm endpoint stays at 0 M" not in jsl:
        raise AssertionError("The factor's active modes and resolved salt must be visible in JMP")
    if "loadSaturationStatus << Set Text" not in jsl or "Above saturation: protein can break through" not in jsl:
        raise AssertionError("The load must show stationary-phase saturation and overload in JMP")
    return {"generated_python_send_names": len(sent_names), "saved_run_fields": len(schema),
            "restored_run_fields": len(schema - {"Run_Number"})}


def _run_row() -> dict:
    row = default_batch_row(1)
    row.update({
        "Run_Name": "JMP interface handoff audit",
        "Load_Flow_mL_min": 1.3,
        "Feed_Concentration_mg_mL": 1.0,
        "Load_Chemistry_Control": "ENDPOINT_CHEMISTRY",
        "Load_Material_Chemistry_Mode": "EXPLICIT",
        "Load_Source": "DIRECT",
        "Load_pH": 7.1,
        "Load_Salt_M": "",
        "Load_Salt_Input_Mode": "CONDUCTIVITY",
        "Load_Conductivity_mS_cm": 40.0,
        "Load_CV": 0.5,
        "Time_Steps": 50,
        "Axial_Positions": 6,
        "BufferA_pH": 5.7,
        "BufferA_Conductivity_mS_cm": 10.0,
        "BufferA_Salt_Input_Mode": "CONDUCTIVITY",
        "BufferB_pH": 7.4,
        "BufferB_Conductivity_mS_cm": 70.0,
        "BufferB_Salt_M": 1.4,
        "BufferB_Salt_Input_Mode": "SALT_M",
        "PLW_Count": 1,
        "PLW1_CV": 0.2,
        "PLW1_Flow_mL_min": 0.8,
        "PLW1_Chemistry_Control": "BUFFER_B_PERCENT",
        "PLW1_Start_Percent_B": 100.0,
        "PLW1_End_Percent_B": 100.0,
        "Elution_Count": 1,
        "Elution1_Mode": "LINEAR",
        "Elution1_CV": 0.4,
        "Elution1_Flow_mL_min": 2.4,
        "Elution1_Chemistry_Control": "BUFFER_B_PERCENT",
        "Elution1_Start_Percent_B": 100.0,
        "Elution1_End_Percent_B": 0.0,
    })
    return row


def main() -> None:
    import tdm_bridge

    interface = _check_generated_contract(make_hic_config(impurities=1, steps=50, cells=6))
    row = _run_row()
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "jmp_runs.csv"
        save_batch_rows([row], csv_path)
        saved_row = load_batch_rows(csv_path)[0]

    per_model = []
    for model in sorted(MODELS):
        seed = make_hic_config(model, impurities=1, steps=50, cells=6)
        _set_jmp_globals(tdm_bridge, seed)
        shared = tdm_bridge.build_shared_config_from_jmp()
        for section in ("tdm", "edm", "conversion", "cpa", "system"):
            if shared[section] != seed[section]:
                raise AssertionError(f"{model} bridge changed or dropped the {section} inputs")
        for path in ("column.volume_mL", "column.length_mm", "chromatography_mode", "impurity_count", "model"):
            if _get_path(shared, path) != _get_path(seed, path):
                raise AssertionError(f"{model} bridge mismatch at {path}")
        for index in range(2):
            for field in COMPONENT_FIELDS:
                # The target uses the shared conversion. The UI deliberately
                # does not send a target-specific UV response row.
                if index == 0 and field == "uv_response_factor_mAU_L_g":
                    continue
                if shared["components"][index].get(field) != seed["components"][index].get(field):
                    raise AssertionError(f"{model} component {index + 1} bridge mismatch at {field}")
        for path, expected in required_parameter_values(seed).items():
            actual = _get_path(shared, path)
            if actual != expected:
                raise AssertionError(f"{model} bridge mismatch at {path}: {expected!r} != {actual!r}")

        run_config = batch_row_to_run_config(saved_row)
        effective, _ = apply_batch_shared_parameters(shared, run_config)
        effective["numerics"].update(time_steps=50, axial_positions=6)
        result = simulate(effective)
        receipt = result["parameter_receipt"]
        stages = receipt["stage_chemistry_used"]
        assert receipt["load_input_used"]["material_chemistry_source"] == "DIRECT"
        assert receipt["load_input_used"]["material_salt_input_mode"] == "CONDUCTIVITY"
        assert receipt["load_input_used"]["material_pH"] == 7.1
        assert receipt["load_input_used"]["feed_concentration_mg_mL"] == 1.0
        assert receipt["load_input_used"]["capacity_g_L_resin"] == 0.5
        assert receipt["load_input_used"]["material_conductivity_mS_cm"] == 40.0
        assert abs(receipt["load_input_used"]["material_salt_M"] - 0.8) < 1e-12
        assert abs(receipt["buffer_chemistry_used"]["buffer_A"]["salt_concentration_M"] - 0.2) < 1e-12
        assert receipt["buffer_chemistry_used"]["buffer_B"]["salt_concentration_M"] == 1.4
        assert [stages[i]["flow_mL_min"] for i in range(3)] == [1.3, 0.8, 2.4]
        assert stages[2]["start_salt_M"] == 1.4 and abs(stages[2]["end_salt_M"] - 0.2) < 1e-12
        assert stages[2]["start_salt_source"] == "BUFFER_BLEND"
        assert stages[2]["start_salt_input_mode"] == "Buffer A=CONDUCTIVITY; Buffer B=SALT_M"
        assert effective["numerics"]["time_steps"] == 50 and effective["numerics"]["axial_positions"] == 6
        assert receipt["derived_species_feed_concentration_mg_mL"] == {"Target protein": 0.5, "Impurity 1": 0.5}

        # Changing the JMP conductivity conversion must alter conductivity-mode
        # salt, while SALT_M remains authoritative even when conductivity exists.
        tdm_bridge.tdm_ui_cond_to_salt_factor = 0.01
        changed_shared = tdm_bridge.build_shared_config_from_jmp()
        changed, _ = apply_batch_shared_parameters(changed_shared, run_config)
        changed["numerics"].update(time_steps=50, axial_positions=6)
        changed_result = simulate(changed)
        changed_receipt = changed_result["parameter_receipt"]
        assert abs(changed_receipt["load_input_used"]["material_salt_M"] - 0.4) < 1e-12
        assert abs(changed_receipt["buffer_chemistry_used"]["buffer_A"]["salt_concentration_M"] - 0.1) < 1e-12
        assert changed_receipt["buffer_chemistry_used"]["buffer_B"]["salt_concentration_M"] == 1.4

        # Change the JMP k_s value independently and require it in the solver
        # receipt and a changed HIC prediction.
        tdm_bridge.tdm_ui_cond_to_salt_factor = 0.02
        tdm_bridge.tdm_ui_c1_salt_sensitivity_per_M = 0.0
        affinity_shared = tdm_bridge.build_shared_config_from_jmp()
        affinity, _ = apply_batch_shared_parameters(affinity_shared, run_config)
        affinity["numerics"].update(time_steps=50, axial_positions=6)
        affinity_result = simulate(affinity)
        delta_uv = float(np.max(np.abs(result["trace"]["uv_mAU"] - affinity_result["trace"]["uv_mAU"])))
        assert affinity_result["parameter_receipt"]["required_inputs_used"]["components[0].salt_sensitivity_per_M"] == 0.0
        assert delta_uv > 1e-8, (model, delta_uv)

        # The competitive Langmuir b field must also survive the JMP bridge
        # and alter the local binding affinity, independently of k_s.
        delta_b_uv = None
        tdm_bridge.tdm_ui_c1_salt_sensitivity_per_M = seed["components"][0]["salt_sensitivity_per_M"]
        if uses_langmuir(model):
            tdm_bridge.tdm_ui_c1_b_L_g = 2.0 * seed["components"][0]["b_L_g"]
            b_shared = tdm_bridge.build_shared_config_from_jmp()
            b_effective, _ = apply_batch_shared_parameters(b_shared, run_config)
            b_effective["numerics"].update(time_steps=50, axial_positions=6)
            b_result = simulate(b_effective)
            b_receipt = b_result["parameter_receipt"]["langmuir_affinity_used"][0]
            assert b_receipt["b_L_g"] == 2.0 * seed["components"][0]["b_L_g"]
            assert b_receipt["salt_sensitivity_per_M"] == seed["components"][0]["salt_sensitivity_per_M"]
            np.testing.assert_allclose(b_result["trace"]["component_effective_b_L_g"][:, 0],
                                       2.0 * result["trace"]["component_effective_b_L_g"][:, 0], rtol=1e-11)
            delta_b_uv = float(np.max(np.abs(result["trace"]["uv_mAU"] - b_result["trace"]["uv_mAU"])))
            assert delta_b_uv > 1e-8, (model, delta_b_uv)
            tdm_bridge.tdm_ui_c1_b_L_g = seed["components"][0]["b_L_g"]

        # Also verify the explicit SALT_M load route retains entered conductivity
        # for its output overlay while binding uses the entered salt molarity.
        salt_row = dict(saved_row, Load_Salt_M=1.25, Load_Salt_Input_Mode="SALT_M", Load_pH=8.2)
        salt_run = batch_row_to_run_config(salt_row)
        salt_config, _ = apply_batch_shared_parameters(shared, salt_run)
        salt_config["numerics"].update(time_steps=50, axial_positions=6)
        salt_result = simulate(salt_config)
        load_used = salt_result["parameter_receipt"]["load_input_used"]
        assert load_used["material_pH"] == 8.2
        assert load_used["material_salt_input_mode"] == "SALT_M"
        assert load_used["material_salt_M"] == 1.25
        assert load_used["material_conductivity_mS_cm"] == 40.0

        # The deliberately extreme JMP factor must be visible in the resolved
        # chemistry and rejected before an unphysical HIC chromatogram is made.
        tdm_bridge.tdm_ui_cond_to_salt_factor = 1000.0
        extreme_shared = tdm_bridge.build_shared_config_from_jmp()
        extreme, _ = apply_batch_shared_parameters(extreme_shared, run_config)
        extreme_errors = validate_config(extreme)
        assert any("Load material resolves to 40000 M salt" in error for error in extreme_errors), extreme_errors
        assert any("Buffer A resolves to 10000 M salt" in error for error in extreme_errors), extreme_errors
        assert not any("Buffer B resolves to" in error for error in extreme_errors), extreme_errors

        # If all salt inputs are SALT_M, the factor is inactive for binding;
        # changing it must neither invent a 1000 M salt nor alter the program.
        direct_row = dict(salt_row, BufferA_Salt_Input_Mode="SALT_M", BufferA_Salt_M=0.2)
        direct_run = batch_row_to_run_config(direct_row)
        direct_baseline, _ = apply_batch_shared_parameters(shared, direct_run)
        direct_extreme, _ = apply_batch_shared_parameters(extreme_shared, direct_run)
        assert not validate_config(direct_extreme), validate_config(direct_extreme)
        before = _build_program(direct_baseline, np.ones(2))
        after = _build_program(direct_extreme, np.ones(2))
        assert [(s.start_salt_M, s.end_salt_M) for s in before.stages] == [
            (s.start_salt_M, s.end_salt_M) for s in after.stages]

        per_model.append({
            "model": model,
            "shared_inputs_checked": len(required_parameter_values(seed)),
            "active_components_checked": 2,
            "load_conductivity_mode_salt_M": receipt["load_input_used"]["material_salt_M"],
            "load_salt_M_mode_salt_M": load_used["material_salt_M"],
            "buffer_A_salt_M": receipt["buffer_chemistry_used"]["buffer_A"]["salt_concentration_M"],
            "buffer_B_salt_M": receipt["buffer_chemistry_used"]["buffer_B"]["salt_concentration_M"],
            "max_uv_change_when_k_s_zeroed_mAU": delta_uv,
            "max_uv_change_when_b_doubled_mAU": delta_b_uv,
            "stage_flows_mL_min": [stages[0]["flow_mL_min"], stages[1]["flow_mL_min"], stages[2]["flow_mL_min"]],
            "factor_1000_rejected": True,
            "factor_1000_errors": extreme_errors,
            "all_SALT_M_modes_ignore_factor": True,
        })
        print(f"PASS {model}: JMP fields → bridge → solver receipt; salt modes and k_s respond")

    report = {
        "build_id": RESULT_GUARD_BUILD,
        "status": "PASS",
        "scope": "Generated JMP source contract, saved run table, Python bridge, HIC solver receipt and factor-1000 rejection; synthetic cases; no native JMP session.",
        "max_supported_hic_salt_M": MAX_HIC_SALT_M,
        "interface_contract": interface,
        "models": per_model,
    }
    output = ROOT / "verification" / "jmp_interface_handoff.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: all {len(per_model)} model handoffs; report written to {output}")


if __name__ == "__main__":
    main()
