from __future__ import annotations

import csv
import html
import importlib
import json
import math
import os
import shutil
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def _resolve_project_dir() -> Path:
    """Locate the extracted model folder even when JMP executes this file without __file__."""
    candidates = [
        globals().get("tdm_project_dir"),
        globals().get("tdm_project_dir_posix"),
        os.environ.get("TDM_PROJECT_DIR"),
        Path.cwd(),
        globals().get("__file__"),
    ]
    checked: list[str] = []
    for raw in candidates:
        if raw in (None, ""):
            continue
        try:
            candidate = Path(str(raw)).expanduser()
            # __file__ is a file path; the other candidates are directories.
            if raw is globals().get("__file__") and candidate.suffix:
                candidate = candidate.parent
            candidate = candidate.resolve()
        except Exception:
            continue
        checked.append(str(candidate))
        if (candidate / "tdm_minimal_model.py").is_file() and (candidate / "tdm_jmp.py").is_file():
            return candidate
    raise RuntimeError("Could not locate the extracted model folder. Checked: " + "; ".join(checked))


PROJECT_DIR = _resolve_project_dir()


def _prepare_project_imports(project: Path) -> None:
    """Reload project modules from this extraction in JMP's persistent Python."""
    project = project.resolve()
    kept_paths: list[str] = []
    for entry in sys.path:
        try:
            resolved = Path(entry or Path.cwd()).expanduser().resolve()
        except Exception:
            kept_paths.append(entry)
            continue
        if resolved != project:
            kept_paths.append(entry)
    sys.path[:] = [str(project), *kept_paths]
    importlib.invalidate_caches()
    # JMP's Python interpreter survives Python Submit File calls. Discard cached
    # project modules before importing so edits and newly extracted copies take
    # effect on the very next Run or Attach action.
    for module_name in (
        "tdm_jmp",
        "tdm_least_squares",
        "tdm_minimal_io",
        "tdm_minimal_model",
        "tdm_mobile_phase",
    ):
        sys.modules.pop(module_name, None)


_prepare_project_imports(PROJECT_DIR)

from tdm_minimal_io import (
    BATCH_COUNT_FILE,
    BATCH_GALLERY_HTML,
    BATCH_RUNS_CSV,
    LSQ_RUNS_CSV,
    BATCH_SUMMARY_CSV,
    CONFIG_PATH,
    LABEL_TO_MODEL,
    MAX_LSQ_PROFILES,
    MAX_RUNS,
    MODEL_TDM_CPA,
    MODEL_TDM_WANG,
    STATUS_PATH,
    active_components,
    apply_batch_shared_parameters,
    batch_max_pH_span,
    batch_row_to_run_config,
    blank_config,
    load_batch_rows,
    load_lsq_rows,
    normalize_config,
    canonical_fit_parameter_path,
    required_parameter_values,
    sanitize_config_for_selected_model,
    save_batch_rows,
    save_lsq_rows,
    save_config,
    validate_config,
    validate_mechanistic_config,
)

BATCH_RESULTS_ROOT = PROJECT_DIR / "batch_results"


def _new_batch_results_dir() -> Path:
    """Give each batch a fresh directory so Windows/OneDrive locks cannot block reruns."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return BATCH_RESULTS_ROOT / f"batch_{stamp}_{uuid.uuid4().hex[:8]}"


def _g(name: str, default: Any = None) -> Any:
    return globals().get(name, default)


def _num(name: str) -> float | None:
    value = _g(name)
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _text(name: str) -> str:
    return str(_g(name, "") or "").strip()


def _write_status(text: str) -> None:
    STATUS_PATH.write_text(text.rstrip() + "\n", encoding="utf-8")
    print(text)


def build_shared_config_from_jmp() -> dict[str, Any]:
    config = blank_config()
    config["model"] = LABEL_TO_MODEL.get(_text("tdm_ui_model"), MODEL_TDM_CPA)
    n = _num("tdm_ui_impurity_count")
    config["impurity_count"] = int(n) if n is not None else 0
    config["chromatography_mode"] = _text("tdm_ui_chromatography_mode") or ("HIC" if config["model"] == MODEL_TDM_WANG or "LANGMUIR" in config["model"] else "ION_EXCHANGE")
    for field, ui in (("q0_g_L", "tdm_ui_wang_q0_g_L"), ("eta", "tdm_ui_wang_eta")):
        if ui in globals():
            config["wang"][field] = _num(ui)
    for field in ("buffer_dispersion_mL", "salt_dispersion_mm2_s", "uv_dead_volume_mL", "conductivity_dead_volume_mL"):
        value = _num("tdm_ui_system_" + field)
        if value is not None:
            config["system"][field] = value
    for field in ("buffer_dispersion_enabled", "salt_dispersion_enabled", "show_percent_B", "show_conductivity", "show_flow_rate"):
        value = _text("tdm_ui_system_" + field)
        if value:
            config["system"][field] = value.upper() == "ON"
    config["least_squares"]["objective"] = _text("tdm_ui_lsq_objective") or "RAW_SSE"
    config["least_squares"]["baseline_mode"] = _text("tdm_ui_lsq_baseline_mode") or "NONE"
    config["least_squares"]["detector_baseline"] = _num("tdm_ui_lsq_detector_baseline") or 0.0
    # Older bridge callers may not send this field. Once the JMP box exists,
    # reject an empty, fractional or out-of-range cap instead of resetting it.
    if "tdm_ui_lsq_max_nfev" in globals():
        cap = _num("tdm_ui_lsq_max_nfev")
        if cap is None or not cap.is_integer() or not 2 <= cap <= 10000:
            raise ValueError("Safety cap must be a whole number from 2 to 10000 optimizer evaluations.")
        config["least_squares"]["max_nfev"] = int(cap)
    locks = _lsq_unlocked_paths_from_jmp()
    if locks is not None:
        config["least_squares"]["unlocked_paths"] = sorted(locks)
    if _num("tdm_ui_lsq_run_count") is not None:
        config["least_squares"]["selected_run_numbers"] = _selected_lsq_run_numbers()
    config["column"]["volume_mL"] = _num("tdm_ui_column_volume_mL")
    config["column"]["length_mm"] = _num("tdm_ui_column_length_mm")
    config["tdm"].update(
        bead_radius_um=_num("tdm_ui_bead_radius"),
        void_fraction=_num("tdm_ui_void_fraction"),
        particle_porosity=_num("tdm_ui_particle_porosity"),
        salt_axial_dispersion_mm2_s=_num("tdm_ui_salt_Dax"),
    )
    if _num("tdm_ui_system_salt_dispersion_mm2_s") is None and _num("tdm_ui_salt_Dax") is not None:
        config["system"]["salt_dispersion_mm2_s"] = _num("tdm_ui_salt_Dax")
    config["edm"]["total_bed_porosity"] = _num("tdm_ui_edm_porosity")
    config["conversion"].update(
        uv_to_protein_mAU_L_g=_num("tdm_ui_uv_to_protein_factor"),
        conductivity_to_salt_M_per_mS_cm=_num("tdm_ui_cond_to_salt_factor"),
    )
    config["cpa"].update(
        ligand_surface_density_umol_m2=_num("tdm_ui_ligand_surface_density"),
        system_specific_adsorption_parameter=_num("tdm_ui_system_adsorption_parameter"),
    )
    for i, row in enumerate(config["components"], start=1):
        p = f"tdm_ui_c{i}_"
        name = _text(p + "name")
        if name:
            row["name"] = name
        row.update(
            mass_percent=_num(p + "mass_percent"),
            uv_response_factor_mAU_L_g=_num(p + "uv_response_factor_mAU_L_g") if i > 1 else row.get("uv_response_factor_mAU_L_g"),
            D_ax_mm2_s=_num(p + "D_ax_mm2_s"),
            k_eff_um_s=_num(p + "k_eff_um_s"),
            accessible_particle_porosity=_num(p + "accessible_particle_porosity"),
            D_app_mm2_s=_num(p + "D_app_mm2_s"),
            qmax_g_L=_num(p + "qmax_g_L") if p + "qmax_g_L" in globals() else _num(p + "qmax_mg_ml"),
            b_L_g=_num(p + "b_L_g"),
            salt_sensitivity_per_M=_num(p + "salt_sensitivity_per_M"),
            diameter_nm=_num(p + "diameter_nm"),
            As_m_inv=_num(p + "As_m_inv"),
            Z_ref=_num(p + "Z_ref"),
            delta_ref=_num(p + "delta_ref"),
            kkin_star_s=_num(p + "kkin_star_s"),
            pH_ref=_num(p + "pH_ref"),
            Z1_per_pH=_num(p + "Z1_per_pH"),
            Z2_per_pH2=_num(p + "Z2_per_pH2"),
            Z3_per_pH3=_num(p + "Z3_per_pH3"),
            delta_pH_slope_m2_C=_num(p + "delta_pH_slope_m2_C"),
        )
        for field in ("wang_K_kin_s", "wang_k_eq", "wang_qmax_g_L", "wang_n",
                      "wang_beta0", "wang_beta1_per_M", "wang_beta2_L_g", "wang_beta3_per_pH"):
            if p + field in globals():
                row[field] = _num(p + field)
    return normalize_config(config)


def _run_number() -> int:
    raw = _num("tdm_ui_run_number")
    n = int(raw) if raw is not None else 1
    return min(max(n, 1), MAX_RUNS)


def _lsq_target_run(default: int | None = None) -> int:
    raw = _num("tdm_ui_lsq_target_run")
    n = int(raw) if raw is not None else int(default or _run_number())
    return min(max(n, 1), MAX_RUNS)


def _batch_count() -> int:
    raw = _num("tdm_ui_batch_count")
    n = int(raw) if raw is not None else 1
    return min(max(n, 1), MAX_RUNS)


def _effective_run(shared: dict[str, Any], run_number: int) -> dict[str, Any]:
    rows = load_batch_rows()
    run_cfg = batch_row_to_run_config(rows[run_number - 1])
    effective, _ignored = apply_batch_shared_parameters(shared, run_cfg)
    return effective


def _selected_run_context(run_number: int) -> dict[str, Any]:
    row = load_batch_rows()[run_number - 1]
    return {
        "run_number": run_number,
        "run_name": str(row.get("Run_Name") or f"Run {run_number}"),
    }


def _fit_run_context(config: dict[str, Any], run_number: int) -> dict[str, Any]:
    rows = load_lsq_rows()
    row = rows[run_number - 1]
    return {
        "run_number": run_number,
        "run_name": str(row.get("Run_Name") or f"Run {run_number}"),
        "chromatogram_csv": str(row.get("LSQ_Chromatogram_CSV") or ""),
        "model": config["model"],
        "column": config["column"],
        "feed": config["feed"],
        "numerics": config["numerics"],
        "process": config["process"],
    }


def _lsq_reference_column_options(row: dict[str, Any]) -> dict[str, str | None]:
    """Column choices are independent for every LSQ process recipe."""
    return {
        "sheet_name": str(row.get("LSQ_Sheet") or "").strip() or None,
        "x_column": str(row.get("LSQ_X_Column") or "").strip() or None,
        "signal_column": str(row.get("LSQ_Signal_Column") or "").strip() or None,
        "x_unit": str(row.get("LSQ_X_Unit") or "AUTO").strip() or "AUTO",
        "x_origin": str(row.get("LSQ_X_Origin") or "RUN_START").strip() or "RUN_START",
    }


def _lsq_profile_csv_path(run_number: int, entered_path: str = "") -> str:
    """Resolve the independent fit profile's chromatogram file."""
    entered = str(entered_path or "").strip().strip('"').strip("'")
    if entered:
        return entered
    rows = load_lsq_rows()
    return str(rows[run_number - 1].get("LSQ_Chromatogram_CSV") or "").strip()


def _attached_lsq_profile_rows(rows: list[dict[str, Any]] | None = None) -> list[tuple[int, dict[str, Any], str]]:
    rows = rows if rows is not None else load_lsq_rows()
    profiles = []
    for index, row in enumerate(rows[:MAX_LSQ_PROFILES], start=1):
        path = str(row.get("LSQ_Chromatogram_CSV") or "").strip().strip('"').strip("'")
        if path:
            profiles.append((index, row, path))
    return profiles


def _selected_lsq_run_numbers() -> list[int]:
    count = _num("tdm_ui_lsq_run_count")
    if count is None or count != int(count) or not 1 <= count <= MAX_LSQ_PROFILES:
        raise ValueError("Select an integer number of least-squares runs from 1 to 5.")
    selected = []
    for slot in range(1, int(count) + 1):
        number = _num(f"tdm_ui_lsq_selected_run_{slot}")
        if number is None or number != int(number) or not 1 <= number <= MAX_LSQ_PROFILES:
            raise ValueError(f"Choose an independent LSQ run 1–{MAX_LSQ_PROFILES} for fit slot {slot}.")
        selected.append(int(number))
    if len(selected) != len(set(selected)):
        raise ValueError("Select each least-squares run only once.")
    return selected


def _selected_lsq_profile_rows(rows=None):
    rows = rows if rows is not None else load_lsq_rows()
    if _num("tdm_ui_lsq_run_count") is None:
        return _attached_lsq_profile_rows(rows)  # Compatibility with older callers.
    selected = []
    for number in _selected_lsq_run_numbers():
        row = rows[number - 1]
        path = str(row.get("LSQ_Chromatogram_CSV") or "").strip()
        if not path:
            raise ValueError(f"Selected least-squares Run {number} has no attached chromatogram CSV.")
        selected.append((number, row, path))
    return selected


def _lsq_unlocked_paths_from_jmp() -> set[str] | None:
    """Read parameter lock states sent by the per-parameter JMP selectors."""
    raw = _text("tdm_ui_lsq_parameter_lock_states")
    if not raw:
        return None  # Direct Python/older UI calls keep the established defaults.
    unlocked: set[str] = set()
    for item in raw.split(";"):
        if "=" not in item:
            continue
        path, state = item.rsplit("=", 1)
        if path.strip() and state.strip().upper() == "UNLOCKED":
            unlocked.add(canonical_fit_parameter_path(path.strip()))
    return unlocked


def _save_lsq_profile_csv_path(run_number: int, path: str) -> None:
    rows = load_lsq_rows()
    rows[run_number - 1]["LSQ_Chromatogram_CSV"] = str(path)
    save_lsq_rows(rows)


def _lsq_mode() -> str:
    label = _text("tdm_ui_lsq_mode").upper()
    return "CHROMATOGRAM_AND_COMPOSITION" if ("MASS" in label or "COMPOSITION" in label) else "CHROMATOGRAM"


def _composition_references_from_jmp(config: dict[str, Any]) -> list[dict[str, Any]]:
    count_raw = _num("tdm_ui_lsq_reference_count")
    count = min(max(int(count_raw) if count_raw is not None else 0, 0), 10)
    names = [str(r["name"]) for r in active_components(config)]
    refs: list[dict[str, Any]] = []
    for k in range(1, count + 1):
        cv = _num(f"tdm_ui_lsq_r{k}_CV")
        mass = {name: _num(f"tdm_ui_lsq_r{k}_c{i}_mass_percent") for i, name in enumerate(names, start=1)}
        refs.append({"CV": cv, "mass_percent": mass})
    return refs


def _save_batch_summary_to_root(batch_info: dict[str, Any]) -> None:
    src = Path(batch_info["summary_csv"])
    if src.is_file():
        shutil.copyfile(src, BATCH_SUMMARY_CSV)
    gallery = Path(batch_info.get("gallery_html", ""))
    if gallery.is_file():
        gallery_html = gallery.read_text(encoding="utf-8")
        relative_assets = Path(os.path.relpath(gallery.parent, BATCH_GALLERY_HTML.parent)).as_posix().rstrip("/") + "/"
        base_tag = f'<base href="{html.escape(relative_assets, quote=True)}">'
        if "<base " not in gallery_html.lower():
            meta_end = gallery_html.lower().find("<meta charset=")
            if meta_end >= 0:
                tag_end = gallery_html.find(">", meta_end)
                gallery_html = gallery_html[:tag_end + 1] + base_tag + gallery_html[tag_end + 1:]
            else:
                head_start = gallery_html.lower().find("<head")
                if head_start >= 0:
                    tag_end = gallery_html.find(">", head_start)
                    gallery_html = gallery_html[:tag_end + 1] + base_tag + gallery_html[tag_end + 1:]
                else:
                    gallery_html = base_tag + gallery_html
        BATCH_GALLERY_HTML.write_text(gallery_html, encoding="utf-8")


def _result_recipe_line(manifest_path: str | Path) -> str:
    """Summarize the recipe persisted with a chromatogram result."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    def show(value: Any) -> str:
        if value is None or value == "":
            return "—"
        try:
            return f"{float(value):g}"
        except (TypeError, ValueError):
            return str(value)

    stage_text = []
    for stage in manifest.get("stages", []):
        item = (
            f"{stage['name']} {float(stage['duration_CV']):g} CV "
            f"@ {float(stage['flow_mL_min']):g} mL/min "
            f"[{float(stage['start_CV']):g}–{float(stage['end_CV']):g} CV]"
        )
        if str(stage.get("chemistry_control", "")).upper() == "BUFFER_B_PERCENT":
            item += f" (%B {show(stage.get('start_percent_B'))}→{show(stage.get('end_percent_B'))})"
        item += (
            f" (salt {show(stage.get('start_salt_M'))}→{show(stage.get('end_salt_M'))} M; "
            f"conductivity {show(stage.get('start_conductivity_mS_cm'))}→"
            f"{show(stage.get('end_conductivity_mS_cm'))} mS/cm)"
        )
        stage_text.append(item)
    stages = ", ".join(stage_text)
    load_volume = manifest.get("required_load_volume_mL")
    load_capacity = manifest.get("load_capacity_g_L_resin")
    return (
        f"Run {manifest.get('run_number')} — {manifest.get('run_name')}: "
        f"load capacity {show(load_capacity)} g/L resin; required feed volume {show(load_volume)} mL; "
        f"{float(manifest['total_CV']):g} CV total ({stages})"
    )


def main() -> int:
    action = _text("tdm_ui_action").upper() or "SAVE_MODEL"
    try:
        shared_raw = build_shared_config_from_jmp()
        current_run_number = _run_number()
        run_number = _lsq_target_run(current_run_number) if action in {"FIT_LSQ", "ATTACH_LSQ_CSV"} else current_run_number
        batch_count = _batch_count()
        BATCH_COUNT_FILE.write_text(str(batch_count), encoding="utf-8")
        current_effective = _effective_run(shared_raw, run_number)
        shared = sanitize_config_for_selected_model(current_effective)
        # Preserve only the shared mechanistic part in the active file; the batch
        # CSV remains authoritative for the per-run operating conditions.
        save_config(shared, CONFIG_PATH)

        if action == "ATTACH_LSQ_CSV":
            from tdm_least_squares import attach_chromatogram_csv, load_chromatogram_reference
            from tdm_minimal_model import _build_program
            import numpy as np
            source = _lsq_profile_csv_path(run_number, _text("tdm_ui_lsq_csv_path"))
            if not source:
                raise ValueError(f"Enter or choose a chromatogram CSV for LSQ profile Run {run_number}.")
            rows = load_lsq_rows()
            options = _lsq_reference_column_options(rows[run_number - 1])
            info = attach_chromatogram_csv(source, require_uv=True, **options)
            run_cfg = batch_row_to_run_config(rows[run_number - 1])
            profile_config, _ = apply_batch_shared_parameters(shared_raw, run_cfg)
            errors = validate_config(profile_config, strict=True)
            if errors:
                raise ValueError(f"LSQ Run {run_number} process recipe must be complete before attaching:\n- " + "\n- ".join(errors))
            reference = load_chromatogram_reference(info["source_path"], profile_config, require_uv=True, **options)
            program = _build_program(profile_config, np.zeros(max(1, len(active_components(profile_config)))))
            cv_min, cv_max = float(np.min(reference.cv)), float(np.max(reference.cv))
            if cv_min < -1e-9 or cv_max > program.total_CV + 1e-9:
                raise ValueError(
                    f"Measured X converts to {cv_min:.4g}–{cv_max:.4g} CV, outside LSQ Run {run_number}'s "
                    f"0–{program.total_CV:.4g} CV recipe. Check mL/CV units, column volume, and X=0 origin."
                )
            rows[run_number - 1]["LSQ_Chromatogram_CSV"] = info["source_path"]
            _save_lsq_profile_csv_path(run_number, info["source_path"])
            run_name = load_lsq_rows()[run_number - 1].get("Run_Name") or f"LSQ Run {run_number}"
            x_warning = f"\nAxis assumption: {info['x_axis_assumption']}" if info.get("x_axis_assumption") else ""
            x_receipt = (
                f"Measured mL divided by column volume ({shared_raw['column']['volume_mL']} mL) to align with solver CV."
                if info["x_unit"] == "ML" else "Measured CV aligned directly with solver CV."
                if info["x_unit"] == "CV" else f"Measured {info['x_unit']} converted to solver CV using the LSQ recipe."
            )
            if info["x_origin"] == "ELUTION_START":
                x_receipt += " X=0 is the start of elution; load and wash CV are added."
            _write_status(
                f"Chromatogram attached to LSQ profile Run {run_number} — {run_name}.\n"
                f"Active model folder: {PROJECT_DIR}\n"
                f"Selected chromatogram: {info['source_path']}\n"
                f"X: {info['x_column']} ({info['x_unit']}, zero at {info['x_origin']}); Y: {info['signal_column']} (UV compared in mAU; AU-labelled values converted); points: {info['rows']}.\n"
                f"Aligned measured range: {cv_min:.4g}–{cv_max:.4g} CV; saved recipe: 0–{program.total_CV:.4g} CV. {x_receipt}{x_warning}\n"
                f"Independent process setup: {LSQ_RUNS_CSV}"
            )
            return 0

        if action == "SAVE_MODEL":
            span = batch_max_pH_span(batch_count, shared)
            errors = validate_mechanistic_config(shared, pH_span_override=span)
            if errors:
                _write_status("Model parameters saved, but required fields are incomplete:\n- " + "\n- ".join(errors))
            else:
                run_errors = validate_config(current_effective, strict=True)
                if run_errors:
                    _write_status("Model parameters saved, but the selected run cannot be solved:\n- " + "\n- ".join(run_errors))
                else:
                    _write_status(
                        "Model parameters saved. Direct input boxes are authoritative; no templates are used.\n"
                        f"Selected batch: {batch_count} run(s). Shared mechanistic values are locked across those runs."
                    )
            return 0

        if action == "CHECK_RESULT_CURRENT":
            from tdm_minimal_model import check_result_freshness
            context = _selected_run_context(current_run_number)
            valid, message = check_result_freshness(current_effective, context)
            _write_status(("RESULT_CURRENT: " if valid else "RESULT_STALE: ") + message)
            return 0

        if action == "FIT_LSQ":
            from tdm_least_squares import inspect_chromatogram_csv, run_global_least_squares_refinement
            profile_rows = _selected_lsq_profile_rows()
            if not profile_rows:
                raise ValueError("Attach at least one chromatogram CSV to a saved run before fitting.")
            if len(profile_rows) > MAX_LSQ_PROFILES:
                run_list = ", ".join(str(n) for n, _row, _path in profile_rows)
                raise ValueError(f"Global least-squares fitting supports at most {MAX_LSQ_PROFILES} attached chromatograms; currently attached to runs {run_list}. Clear extra CSV paths before fitting.")
            mode = _lsq_mode()
            refs = _composition_references_from_jmp(current_effective) if mode == "CHROMATOGRAM_AND_COMPOSITION" else []
            profiles = []
            lsq_rows = load_lsq_rows()
            for profile_run_number, row, source in profile_rows:
                run_cfg = batch_row_to_run_config(lsq_rows[profile_run_number - 1])
                profile_config, _ = apply_batch_shared_parameters(shared_raw, run_cfg)
                errors = validate_config(profile_config, strict=True)
                if errors:
                    raise ValueError(f"Least-squares profile Run {profile_run_number} must be complete:\n- " + "\n- ".join(errors))
                options = _lsq_reference_column_options(row)
                info = inspect_chromatogram_csv(source, require_uv=True, **options)
                # inspect_chromatogram_csv() reports the normalized source as
                # `path`; `source_path` is only returned by the attach helper.
                source = info["path"]
                context = _fit_run_context(profile_config, profile_run_number)
                context["chromatogram_csv"] = source
                profile = {"config": profile_config, "chromatogram_csv": source, "run_context": context,
                           "weight": row.get("LSQ_Weight") or 1.0, **options}
                if mode == "CHROMATOGRAM_AND_COMPOSITION" and profile_run_number == run_number:
                    profile["composition_references"] = refs
                profiles.append(profile)
            if mode == "CHROMATOGRAM_AND_COMPOSITION" and refs and run_number not in {p["run_context"]["run_number"] for p in profiles}:
                raise ValueError("Select an attached chromatogram profile before adding species-composition reference points.")
            unlocked_paths = _lsq_unlocked_paths_from_jmp()
            from tdm_least_squares import FIT_HISTORY_CSV, fit_parameter_specs
            fitted_paths = fit_parameter_specs(profiles[0]["config"], unlocked_paths=unlocked_paths)
            _write_status(
                f"Least-squares refinement running: {len(profiles)} profile(s), "
                f"{len(fitted_paths)} unlocked parameters, "
                f"{shared_raw['least_squares']['max_nfev']} optimizer-evaluation safety cap.\n"
                f"Each numerical Jacobian needs about {len(profiles) * (len(fitted_paths) + 1)} column solves. "
                f"Progress: {FIT_HISTORY_CSV}"
            )
            fit = run_global_least_squares_refinement(
                profiles,
                mode=mode,
                unlocked_paths=unlocked_paths,
                max_nfev=int(shared_raw["least_squares"]["max_nfev"]),
            )
            if fit["accepted"]:
                fitted_shared = sanitize_config_for_selected_model(fit["fitted_config"])
                save_config(fitted_shared, CONFIG_PATH)
            profile_summary = "\n".join(
                f"LSQ Run {p['run_number']} — {p['run_name']}: {p['overlap_points']} points, weighted RMSE {p['weighted_rmse']:.6g}; run weight {p['run_weight']:.6g}"
                for p in fit["profile_metrics"]
            )
            weighted_line = (f"Initial weighted RMSE: {fit['initial_weighted_rmse']:.7g}; final weighted RMSE: {fit['final_weighted_rmse']:.7g}\n"
                             if fit["initial_weighted_rmse"] is not None else "")
            with open(fit["improved_parameters_txt"], encoding="utf-8") as fh:
                improved_list = fh.read()
            _write_status(
                "Least-squares refinement completed.\n"
                f"Global profiles fitted: {len(fit['profile_metrics'])} (objective: {shared_raw['least_squares']['objective']}).\n{profile_summary}\n"
                f"Mode: {mode}\nInitial objective: {fit['initial_objective']:.7g}\nFinal objective: {fit['final_objective']:.7g}\n"
                f"{weighted_line}"
                f"Objective improvement: {fit['improvement_percent']:.4g}%\n"
                f"Optimizer function evaluations: {fit['nfev']}; actual residual evaluations: {fit['residual_evaluations']}\n"
                f"Optimizer converged: {fit['success']}; {fit['message']}\n"
                f"Accepted/applied to shared batch parameters: {fit['accepted']}\n"
                f"Improved parameters:\n{improved_list}\n"
                f"Report: {fit['fit_report_html']}\nParameter list CSV: {fit['improved_parameters_csv']}\nObjective history CSV: {fit['latest_fit_history_csv']}"
            )
            return 0

        if action == "RUN_SELECTED":
            errors = validate_config(current_effective, strict=True)
            if errors:
                raise ValueError("Run validation failed; no new chromatogram was generated. "
                                 "Any existing result belongs to earlier settings:\n- " + "\n- ".join(errors))
            from tdm_minimal_model import run_and_write
            context = _selected_run_context(run_number)
            paths = run_and_write(current_effective, run_context=context)
            result_manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
            run_warnings = result_manifest.get("warnings") or []
            peak = result_manifest.get("target_peak") or {}
            peak_line = ""
            if peak.get("detector_peak_CV") is not None:
                peak_line = (
                    f"Target detector peak: {peak['detector_peak_CV']:.3f} total CV "
                    f"({peak['CV_after_elution_start']:.3f} CV from elution start); "
                    f"column outlet peak {peak['column_outlet_peak_CV']:.3f} CV; "
                    f"binding salt at peak {peak['binding_salt_M_at_detector_peak']:.4g} M.\n"
                )
            affinity = result_manifest.get("langmuir_affinity_used") or []
            affinity_line = ""
            if affinity:
                target = affinity[0]
                load_endpoint = target["stage_endpoints"][0]
                elution_endpoint = next((stage for stage in target["stage_endpoints"] if stage["stage"].startswith("Elution")), None)
                affinity_line = (
                    f"Target affinity used: b={target['b_L_g']:g} L/g, "
                    f"k_s={target['salt_sensitivity_per_M']:g} M^-1; "
                    f"load b_eff={load_endpoint['start_b_eff_L_g']:g} L/g"
                    + (f"; elution b_eff={elution_endpoint['start_b_eff_L_g']:g}"
                       f"→{elution_endpoint['end_b_eff_L_g']:g} L/g" if elution_endpoint else "")
                    + ".\n"
                )
            _write_status(
                f"Run {run_number} — {context['run_name']} completed successfully.\n"
                f"Recipe used: {_result_recipe_line(paths['manifest'])}\n"
                f"Active mechanistic inputs consumed: {len(required_parameter_values(current_effective))}\n"
                + peak_line
                + affinity_line
                + ("Run warnings:\n- " + "\n- ".join(run_warnings) + "\n" if run_warnings else "") +
                f"HTML: {paths['html']}\nCSV: {paths['csv']}\nSVG: {paths['svg']}\nRun receipt: {paths['manifest']}"
            )
            return 0

        if action == "RUN_BATCH":
            span = batch_max_pH_span(batch_count, shared)
            mech_errors = validate_mechanistic_config(shared, pH_span_override=span)
            if mech_errors:
                raise ValueError("Shared model parameters are incomplete for the selected batch:\n- " + "\n- ".join(mech_errors))
            rows = load_batch_rows()[:batch_count]
            run_configs: list[dict[str, Any]] = []
            for idx, row in enumerate(rows, start=1):
                run_cfg = batch_row_to_run_config(row)
                run_cfg["_run_name"] = str(row.get("Run_Name") or f"Run {idx}")
                effective, _ignored = apply_batch_shared_parameters(shared, run_cfg)
                errors = validate_config(effective, strict=True)
                if errors:
                    raise ValueError(f"Run {idx} validation failed:\n- " + "\n- ".join(errors))
                run_configs.append(run_cfg)
            from tdm_minimal_model import run_batch_and_write
            batch_output_dir = _new_batch_results_dir()
            info = run_batch_and_write(shared, run_configs, batch_output_dir)
            _save_batch_summary_to_root(info)
            recipe_lines = "\n".join(
                _result_recipe_line(run["manifest"]) for run in info.get("runs", [])
            )
            _write_status(
                f"Batch completed: {batch_count} run(s).\n"
                f"Recipes used:\n{recipe_lines}\n"
                f"Shared model fingerprint: {info['shared_parameter_fingerprint']}\n"
                f"Batch result folder: {batch_output_dir}\n"
                f"Summary: {BATCH_SUMMARY_CSV}\nGallery: {BATCH_GALLERY_HTML}"
            )
            return 0

        raise ValueError(f"Unknown action: {action}")
    except Exception as exc:
        _write_status(f"ERROR: {exc}\n\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__" or "tdm_ui_action" in globals():
    RETURN_CODE = main()
    # JMP's SendAndRun callback reads tdm_status.txt after every action, so let
    # UI errors return normally and display their full message in the window.
    # Keep command-line runs nonzero when they fail.
    if RETURN_CODE != 0 and "tdm_ui_action" not in globals():
        raise RuntimeError("Chromatography model action failed. See tdm_status.txt and the Run / Results tab.")
