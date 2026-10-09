"""Focused load chemistry and flow-output regression check, extended in R18."""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from tdm_minimal_io import (
    MODEL_EDM_LANGMUIR,
    MODEL_LABELS,
    apply_batch_shared_parameters,
    batch_row_to_run_config,
    default_batch_row,
    normalize_config,
    validate_config,
)
from tdm_minimal_model import _build_program, simulate, write_outputs
from tdm_jmp import _validate_jsl, build_jsl
from verify_r13 import make_hic_config

ROOT = Path(__file__).resolve().parent


def main() -> None:
    config = make_hic_config("EDM_LANGMUIR")
    process = config["process"]
    process.update(
        load_chemistry_control="BUFFER_B_PERCENT",
        load_material_chemistry_mode="EXPLICIT",
        load_source="DIRECT",
        load_pH=8.25,
        load_conductivity_mS_cm=35.0,
        load_salt_input_mode="CONDUCTIVITY",
        load_start_percent_B=100.0,
        load_end_percent_B=100.0,
    )
    config = normalize_config(config)
    errors = validate_config(config)
    if errors:
        raise AssertionError(errors)
    program = _build_program(config, np.ones(1))
    load = program.stages[0]
    assert abs(load.start_pH - 8.25) < 1e-12
    assert abs(load.start_salt_M - 0.70) < 1e-12
    assert abs(load.start_conductivity_mS_cm - 35.0) < 1e-12
    assert load.start_percent_B is None

    config["column"]["flow_mL_min"] = 1.7
    config["process"]["plw_count"] = 1
    config["process"]["plw_steps"][0].update(CV=0.2, flow_mL_min=0.8)
    config["process"]["elution_count"] = 1
    config["process"]["elution_steps"][0].update(CV=0.3, flow_mL_min=2.4)
    config["numerics"].update(time_steps=80, axial_positions=8)
    result = simulate(config)
    assert {round(float(v), 6) for v in result["trace"]["flow_mL_min"]} >= {0.8, 1.7, 2.4}
    assert result["parameter_receipt"]["stage_chemistry_used"][0]["chemistry_control"] == "EXPLICIT_LOAD_MATERIAL"

    # A legacy/manual row with populated DIRECT load fields but no new mode
    # column must also be promoted to explicit chemistry rather than silently
    # following the unrelated %B recipe.
    legacy_row = default_batch_row(1)
    legacy_row.update(
        Load_Chemistry_Control="BUFFER_B_PERCENT",
        Load_Material_Chemistry_Mode="",
        Load_Source="DIRECT",
        Load_pH=8.25,
        Load_Conductivity_mS_cm=35.0,
    )
    legacy_cfg = batch_row_to_run_config(legacy_row)
    assert legacy_cfg["process"]["load_material_chemistry_mode"] == "EXPLICIT"

    # Exercise the same Python Name values sent by the generated JMP UI, then
    # combine them with a saved run row and execute the effective solver input.
    # This catches a broken UI/bridge handoff even when the chemistry helpers
    # themselves are correct.
    import tdm_bridge

    bridge_seed = make_hic_config(MODEL_EDM_LANGMUIR)
    bridge_values = {
        "tdm_ui_model": MODEL_LABELS[MODEL_EDM_LANGMUIR],
        "tdm_ui_impurity_count": 0,
        "tdm_ui_chromatography_mode": "HIC",
        "tdm_ui_column_volume_mL": bridge_seed["column"]["volume_mL"],
        "tdm_ui_column_length_mm": bridge_seed["column"]["length_mm"],
        "tdm_ui_bead_radius": bridge_seed["tdm"]["bead_radius_um"],
        "tdm_ui_void_fraction": bridge_seed["tdm"]["void_fraction"],
        "tdm_ui_particle_porosity": bridge_seed["tdm"]["particle_porosity"],
        "tdm_ui_salt_Dax": bridge_seed["tdm"]["salt_axial_dispersion_mm2_s"],
        "tdm_ui_edm_porosity": bridge_seed["edm"]["total_bed_porosity"],
        "tdm_ui_uv_to_protein_factor": bridge_seed["conversion"]["uv_to_protein_mAU_L_g"],
        "tdm_ui_cond_to_salt_factor": bridge_seed["conversion"]["conductivity_to_salt_M_per_mS_cm"],
        "tdm_ui_ligand_surface_density": None,
        "tdm_ui_system_adsorption_parameter": None,
    }
    for key, value in bridge_seed["system"].items():
        bridge_values[f"tdm_ui_system_{key}"] = ("ON" if value else "OFF") if isinstance(value, bool) else value
    component = bridge_seed["components"][0]
    for source, target in {
        "name": "name",
        "mass_percent": "mass_percent",
        "D_ax_mm2_s": "D_ax_mm2_s",
        "k_eff_um_s": "k_eff_um_s",
        "accessible_particle_porosity": "accessible_particle_porosity",
        "D_app_mm2_s": "D_app_mm2_s",
        "qmax_g_L": "qmax_g_L",
        "b_L_g": "b_L_g",
        "salt_sensitivity_per_M": "salt_sensitivity_per_M",
        "diameter_nm": "diameter_nm",
        "As_m_inv": "As_m_inv",
        "Z_ref": "Z_ref",
        "delta_ref": "delta_ref",
        "kkin_star_s": "kkin_star_s",
        "pH_ref": "pH_ref",
        "Z1_per_pH": "Z1_per_pH",
        "Z2_per_pH2": "Z2_per_pH2",
        "Z3_per_pH3": "Z3_per_pH3",
        "delta_pH_slope_m2_C": "delta_pH_slope_m2_C",
    }.items():
        bridge_values[f"tdm_ui_c1_{target}"] = component[source]
    for name, value in bridge_values.items():
        setattr(tdm_bridge, name, value)

    jsl = build_jsl(bridge_seed)
    _validate_jsl(jsl)
    assert jsl.count("loadMaterialMode << Set(2); loadSource << Set(1); SaveCurrentRun(); UpdateUI()") >= 4

    bridge_row = default_batch_row(1)
    bridge_row.update(
        Load_Start_Percent_B=100.0,
        Load_End_Percent_B=100.0,
        PLW1_Start_Percent_B=100.0,
        PLW1_End_Percent_B=100.0,
        Elution1_Mode="LINEAR",
        Elution1_Start_Percent_B=100.0,
        Elution1_End_Percent_B=0.0,
    )
    bridge_run = batch_row_to_run_config(bridge_row)

    def bridge_result():
        shared = tdm_bridge.build_shared_config_from_jmp()
        effective, _ = apply_batch_shared_parameters(shared, bridge_run)
        effective = normalize_config(effective)
        effective["numerics"].update(time_steps=70, axial_positions=8)
        return effective, simulate(effective)

    bridge_base_config, bridge_base = bridge_result()
    tdm_bridge.tdm_ui_cond_to_salt_factor = 0.01
    bridge_factor_config, bridge_factor = bridge_result()
    tdm_bridge.tdm_ui_cond_to_salt_factor = 0.02
    tdm_bridge.tdm_ui_c1_salt_sensitivity_per_M = 0.0
    bridge_affinity_config, bridge_affinity = bridge_result()
    assert bridge_base_config["components"][0]["salt_sensitivity_per_M"] == 6.0
    assert bridge_affinity_config["components"][0]["salt_sensitivity_per_M"] == 0.0
    assert bridge_base["parameter_receipt"]["stage_chemistry_used"][0]["start_salt_M"] == 2.0
    assert bridge_factor["parameter_receipt"]["stage_chemistry_used"][0]["start_salt_M"] == 1.0
    assert np.max(np.abs(bridge_base["trace"]["salt_concentration_M"] - bridge_factor["trace"]["salt_concentration_M"])) > 0.5
    assert np.max(np.abs(bridge_base["trace"]["uv_mAU"] - bridge_affinity["trace"]["uv_mAU"])) > 1e-8

    # This is the row the load edit callbacks save after promoting a prior
    # BUFFER_A recipe to DIRECT + EXPLICIT.
    edited_load_row = dict(bridge_row)
    edited_load_row.update(
        Load_Material_Chemistry_Mode="EXPLICIT",
        Load_Source="DIRECT",
        Load_pH=8.25,
        Load_Conductivity_mS_cm=35.0,
        Load_Salt_Input_Mode="CONDUCTIVITY",
        Load_Start_Percent_B=100.0,
        Load_End_Percent_B=100.0,
    )
    edited_load = batch_row_to_run_config(edited_load_row)
    edited_effective, _ = apply_batch_shared_parameters(
        tdm_bridge.build_shared_config_from_jmp(), edited_load
    )
    edited_program = _build_program(normalize_config(edited_effective), np.ones(1))
    assert abs(edited_program.stages[0].start_pH - 8.25) < 1e-12
    assert abs(edited_program.stages[0].start_salt_M - 0.70) < 1e-12
    assert edited_program.stages[0].start_percent_B is None

    # The three chemistry/affinity knobs must have a measurable path to the
    # solver for the actual HIC Langmuir calculation.
    def peak_for(candidate):
        candidate = normalize_config(candidate)
        candidate["numerics"].update(time_steps=70, axial_positions=8)
        return simulate(candidate)

    baseline = peak_for(config)
    changed_salt = normalize_config(config)
    changed_salt["process"]["buffer_B"]["conductivity_mS_cm"] = 180.0
    changed_salt_result = peak_for(changed_salt)
    changed_factor = normalize_config(config)
    changed_factor["conversion"]["conductivity_to_salt_M_per_mS_cm"] = 0.01
    changed_factor_result = peak_for(changed_factor)
    changed_affinity = normalize_config(config)
    changed_affinity["components"][0]["salt_sensitivity_per_M"] = 0.0
    changed_affinity_result = peak_for(changed_affinity)
    base_stage_salts = np.asarray([
        value for stage in baseline["parameter_receipt"]["stage_chemistry_used"]
        for value in (stage["start_salt_M"], stage["end_salt_M"])
    ])
    changed_stage_salts = np.asarray([
        value for stage in changed_salt_result["parameter_receipt"]["stage_chemistry_used"]
        for value in (stage["start_salt_M"], stage["end_salt_M"])
    ])
    assert np.max(np.abs(base_stage_salts - changed_stage_salts)) > 0.5
    assert np.max(np.abs(baseline["trace"]["salt_concentration_M"] - changed_factor_result["trace"]["salt_concentration_M"])) > 0.1
    assert np.max(np.abs(baseline["trace"]["uv_mAU"] - changed_affinity_result["trace"]["uv_mAU"])) > 1e-8

    with tempfile.TemporaryDirectory() as temp_dir:
        output = write_outputs(config, result, temp_dir, {"run_number": 17, "run_name": "R17"})
        html = Path(output["html"]).read_text(encoding="utf-8")
        svg = Path(output["svg"]).read_text(encoding="utf-8")
        assert "Show flow rate" in html
        assert 'data-overlay="flow-rate"' in svg
        with Path(output["csv"]).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert {round(float(row["flow_mL_min"]), 6) for row in rows} >= {0.8, 1.7, 2.4}
    report = {
        "build_id": "2026-10-07-R18",
        "purpose": "Verify buffer and explicit load chemistry plus HIC affinity through the generated JMP field names and bridge.",
        "bridge_sensitivity": {
            "model": "EDM_LANGMUIR",
            "buffer_B_conductivity_mS_cm": 100.0,
            "factor_M_per_mS_cm_baseline": 0.02,
            "factor_M_per_mS_cm_changed": 0.01,
            "load_salt_M_baseline": bridge_base["parameter_receipt"]["stage_chemistry_used"][0]["start_salt_M"],
            "load_salt_M_changed": bridge_factor["parameter_receipt"]["stage_chemistry_used"][0]["start_salt_M"],
            "k_s_per_M_baseline": bridge_base_config["components"][0]["salt_sensitivity_per_M"],
            "k_s_per_M_changed": bridge_affinity_config["components"][0]["salt_sensitivity_per_M"],
            "max_uv_difference_from_k_s_change_mAU": float(np.max(np.abs(bridge_base["trace"]["uv_mAU"] - bridge_affinity["trace"]["uv_mAU"]))),
        },
        "edited_load_from_prior_buffer_A_row": {
            "saved_source_after_edit": edited_load_row["Load_Source"],
            "material_chemistry_mode": edited_load_row["Load_Material_Chemistry_Mode"],
            "pH_used": edited_program.stages[0].start_pH,
            "conductivity_mS_cm_used": edited_program.stages[0].start_conductivity_mS_cm,
            "salt_M_used": edited_program.stages[0].start_salt_M,
            "retained_load_percent_B": edited_load_row["Load_Start_Percent_B"],
            "programmed_load_percent_B": edited_program.stages[0].start_percent_B,
        },
        "flow_overlay_values_mL_min": [0.8, 1.7, 2.4],
        "status": "PASS",
    }
    (ROOT / "verification" / "r18_buffer_load_chemistry.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("R18 JMP bridge, salt/affinity sensitivity, load-material chemistry and flow overlay checks passed")


if __name__ == "__main__":
    main()
