from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "tdm_inputs.json"
STATUS_PATH = PROJECT_DIR / "tdm_status.txt"
LATEST_RESULTS_HTML = PROJECT_DIR / "latest_results.html"
LATEST_RESULTS_CSV = PROJECT_DIR / "latest_results.csv"
LATEST_RESULTS_MANIFEST = PROJECT_DIR / "latest_results_manifest.json"
RESULT_GUARD_BUILD = "2026-10-08-R22"
BATCH_RUNS_CSV = PROJECT_DIR / "tdm_batch_runs.csv"
LSQ_RUNS_CSV = PROJECT_DIR / "tdm_lsq_process_runs.csv"
BATCH_SUMMARY_CSV = PROJECT_DIR / "tdm_batch_summary.csv"
BATCH_GALLERY_HTML = PROJECT_DIR / "tdm_batch_gallery.html"
BATCH_COUNT_FILE = PROJECT_DIR / "tdm_batch_count.txt"

MODEL_EDM_LANGMUIR = "EDM_LANGMUIR"
MODEL_TDM_LANGMUIR = "TDM_LANGMUIR"
MODEL_EDM_CPA = "EDM_CPA"
MODEL_TDM_CPA = "TDM_CPA"
MODELS = {MODEL_EDM_LANGMUIR, MODEL_TDM_LANGMUIR, MODEL_EDM_CPA, MODEL_TDM_CPA}
MODEL_LABELS = {
    MODEL_EDM_LANGMUIR: "EDM – Competitive Langmuir",
    MODEL_TDM_LANGMUIR: "TDM – Competitive Langmuir",
    MODEL_EDM_CPA: "EDM – CPA",
    MODEL_TDM_CPA: "TDM – CPA",
}
LABEL_TO_MODEL = {v: k for k, v in MODEL_LABELS.items()}

MAX_IMPURITIES = 5
MAX_RUNS = 30
MAX_LSQ_PROFILES = 5
MAX_PLW_STEPS = 5
MAX_ELUTION_STEPS = 5
MAX_BUFFER_IONS = 4
TARGET_COMPONENT_NAME = "Target protein"
IMPURITY_COMPONENT_NAMES = [f"Impurity {i}" for i in range(1, MAX_IMPURITIES + 1)]
COMPONENT_NAMES = [TARGET_COMPONENT_NAME] + IMPURITY_COMPONENT_NAMES
PROCESS_MODES = ("SET_POINT", "LINEAR", "STEP")
CHEMISTRY_SOURCES = ("DIRECT", "BUFFER_A", "BUFFER_B")
CHEMISTRY_CONTROLS = ("BUFFER_B_PERCENT", "ENDPOINT_CHEMISTRY")
LOAD_MATERIAL_CHEMISTRY_MODES = ("BUFFER_RECIPE", "EXPLICIT")
SALT_INPUT_MODES = ("CONDUCTIVITY", "SALT_M")
# The empirical HIC salt laws are not valid at arbitrarily large molarity.
# This deliberately generous domain guard also catches unit mistakes such as
# entering an mM conversion into a field labelled M/(mS/cm).
MAX_HIC_SALT_M = 10.0
# Both HIC affinity implementations clip exp(k_s * salt) at 80. Reject a run
# before that numerical safeguard can silently flatten its parameter response.
MAX_HIC_LOG_AFFINITY = 80.0


def uses_tdm(model: str) -> bool:
    return model in {MODEL_TDM_LANGMUIR, MODEL_TDM_CPA}


def uses_edm(model: str) -> bool:
    return model in {MODEL_EDM_LANGMUIR, MODEL_EDM_CPA}


def uses_cpa(model: str) -> bool:
    return model in {MODEL_EDM_CPA, MODEL_TDM_CPA}


def uses_langmuir(model: str) -> bool:
    return model in {MODEL_EDM_LANGMUIR, MODEL_TDM_LANGMUIR}


def _blank_component(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "mass_percent": None,
        "uv_response_factor_mAU_L_g": None,
        "D_ax_mm2_s": None,
        "k_eff_um_s": None,
        "accessible_particle_porosity": None,
        "D_app_mm2_s": None,
        # Saturation capacity in g/L of non-mobile stationary phase.
        # Legacy mg/mL values migrate with factor 1 because 1 mg/mL = 1 g/L.
        "qmax_g_L": None,
        # Derived Henry coefficient: H = qmax * b. Kept in saved configs for
        # reporting and backward compatibility; it is not independently fitted.
        "H": None,
        "b_L_g": None,
        # Initial value for the exponentially modified HIC Langmuir affinity.
        # It must be calibrated for the protein/resin/salt system.
        "salt_sensitivity_per_M": 1.0,
        "diameter_nm": None,
        "As_m_inv": None,
        "Z_ref": None,
        "delta_ref": None,
        "kkin_star_s": None,
        "pH_ref": None,
        "Z1_per_pH": None,
        "Z2_per_pH2": None,
        "Z3_per_pH3": None,
        "delta_pH_slope_m2_C": None,
    }


def _blank_ion() -> dict[str, Any]:
    return {"species": "", "concentration_M": None, "valency": None}


def _blank_buffer(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "pH": 6.0,
        "conductivity_mS_cm": None,
        # Selected input is authoritative; the other value may be retained.
        "salt_input_mode": "CONDUCTIVITY",
        "salt_concentration_M": None,
        # Salt/conductivity is the default ionic-strength source. Explicit ion
        # rows are only used when the user selects ION_COMPOSITION.
        "ionic_strength_source": "SALT_PROXY",
        "ions": [_blank_ion() for _ in range(MAX_BUFFER_IONS)],
    }


def _blank_step() -> dict[str, Any]:
    return {
        "mode": "SET_POINT",
        "CV": 1.0,
        "flow_mL_min": 1.0,
        # BUFFER_B_PERCENT uses the run-level Buffer A (0% B) and Buffer B
        # (100% B) definitions. ENDPOINT_CHEMISTRY preserves the explicit
        # pH/conductivity endpoint path for unusual programs.
        "chemistry_control": "BUFFER_B_PERCENT",
        "start_percent_B": 0.0,
        "end_percent_B": 0.0,
        "start_source": "DIRECT",
        "end_source": "DIRECT",
        "start_pH": 6.0,
        "end_pH": 6.0,
        "start_conductivity_mS_cm": None,
        "end_conductivity_mS_cm": None,
        "start_salt_concentration_M": None,
        "end_salt_concentration_M": None,
        "start_salt_input_mode": "CONDUCTIVITY",
        "end_salt_input_mode": "CONDUCTIVITY",
        # Effective ionic strength consumed by CPA after resolving direct or
        # buffer chemistry. These are derived, never user-entered inputs.
        "start_ionic_strength_M": None,
        "end_ionic_strength_M": None,
    }


def is_hic(config: dict[str, Any]) -> bool:
    return str(config.get("chromatography_mode") or ("HIC" if uses_langmuir(str(config.get("model"))) else "ION_EXCHANGE")).upper() == "HIC"


def system_defaults() -> dict[str, Any]:
    return {
        "buffer_dispersion_enabled": True,
        "salt_dispersion_enabled": True,
        "buffer_dispersion_mL": 0.0,
        "salt_dispersion_mm2_s": 0.0,
        "uv_dead_volume_mL": 0.0,
        "conductivity_dead_volume_mL": 0.0,
        "show_percent_B": True,
        "show_conductivity": True,
        "show_flow_rate": True,
    }


def blank_config() -> dict[str, Any]:
    """Shared mechanistic inputs plus one effective run's operating conditions."""
    return {
        "model": MODEL_TDM_CPA,
        "chromatography_mode": "ION_EXCHANGE",
        "system": system_defaults(),
        "least_squares": {"objective": "RAW_SSE", "baseline_mode": "NONE", "detector_baseline": 0.0, "max_nfev": 120,
                          "unlocked_paths": None, "selected_run_numbers": None},
        "impurity_count": 0,
        "column": {
            "volume_mL": None,
            "length_mm": None,
            # Run-specific. Filled from tdm_batch_runs.csv before simulation.
            "flow_mL_min": None,
        },
        "feed": {
            "total_concentration_mg_mL": None,
            "load_density_mg_mL_resin": None,
        },
        "numerics": {
            # Run-specific numerical resolution. The requested time-step count
            # sets the result grid and caps the adaptive integrator step size;
            # axial_positions is the number of finite-volume axial cells.
            "time_steps": 700,
            "axial_positions": 40,
        },
        "process": {
            "buffer_A": _blank_buffer("Buffer A"),
            "buffer_B": _blank_buffer("Buffer B"),
            "load_source": "DIRECT",
            "load_chemistry_control": "BUFFER_B_PERCENT",
            "load_material_chemistry_mode": "BUFFER_RECIPE",
            "load_amount_basis": "LOAD_VOLUME",
            "load_CV": None,
            "load_mode": "LINEAR",
            "load_start_percent_B": 0.0,
            "load_end_percent_B": 0.0,
            "load_pH": None,
            "load_conductivity_mS_cm": None,
            "load_salt_concentration_M": None,
            "load_salt_input_mode": "CONDUCTIVITY",
            "load_ionic_strength_M": None,
            "plw_count": 0,
            "elution_count": 1,
            "plw_steps": [_blank_step() for _ in range(MAX_PLW_STEPS)],
            "elution_steps": [_blank_step() for _ in range(MAX_ELUTION_STEPS)],
        },
        "tdm": {
            "bead_radius_um": None,
            "void_fraction": None,
            "particle_porosity": None,
            "salt_axial_dispersion_mm2_s": None,
        },
        "edm": {"total_bed_porosity": None},
        "conversion": {
            "uv_to_protein_mAU_L_g": None,
            "conductivity_to_salt_M_per_mS_cm": None,
        },
        "cpa": {
            "ligand_surface_density_umol_m2": None,
            "system_specific_adsorption_parameter": None,
        },
        "components": [_blank_component(name) for name in COMPONENT_NAMES],
    }


def deep_merge(base: Any, incoming: Any) -> Any:
    if isinstance(base, dict) and isinstance(incoming, dict):
        out = dict(base)
        for key, value in incoming.items():
            if key in out:
                out[key] = deep_merge(out[key], value)
        return out
    if isinstance(base, list) and isinstance(incoming, list):
        out = list(base)
        for i, value in enumerate(incoming[: len(out)]):
            out[i] = deep_merge(out[i], value)
        return out
    return incoming


def canonical_fit_parameter_path(path: Any) -> str:
    """Keep legacy saved locks and direct fitting calls attached to qmax."""
    value = str(path)
    if value.startswith("components[") and value.endswith(".qmax_mg_ml"):
        return value[:-len("qmax_mg_ml")] + "qmax_g_L"
    return value


def synchronize_langmuir_parameters(config: dict[str, Any]) -> dict[str, Any]:
    """Make qmax and b authoritative, migrating legacy H/b inputs if needed."""
    if str(config.get("model") or "") not in {MODEL_EDM_LANGMUIR, MODEL_TDM_LANGMUIR}:
        return config
    components = config.get("components", [])
    for row in components if isinstance(components, list) else []:
        if not isinstance(row, dict):
            continue
        try:
            b = float(row.get("b_L_g"))
            if not math.isfinite(b):
                b = None
        except (TypeError, ValueError):
            b = None
        try:
            qmax = float(row.get("qmax_g_L"))
            if not math.isfinite(qmax):
                qmax = None
        except (TypeError, ValueError):
            qmax = None
        if qmax is None and b is not None and b > 0.0:
            # Legacy project files stored H and b but no explicit capacity.
            try:
                legacy_h = float(row.get("H"))
                if math.isfinite(legacy_h):
                    qmax = legacy_h / b
            except (TypeError, ValueError):
                pass
        if qmax is not None:
            row["qmax_g_L"] = qmax
            if b is not None:
                row["H"] = qmax * b
    return config


def normalize_config(config: Any) -> Any:
    if isinstance(config, dict):
        out = {k: normalize_config(v) for k, v in config.items()}
        if "qmax_mg_ml" in out:
            # Identical numerical units; no factor of 1000 and no phase-basis
            # conversion. An explicitly supplied new key takes precedence.
            if "qmax_g_L" not in out:
                out["qmax_g_L"] = out["qmax_mg_ml"]
            out.pop("qmax_mg_ml")
        if isinstance(out.get("unlocked_paths"), list):
            out["unlocked_paths"] = sorted({canonical_fit_parameter_path(p)
                                             for p in out["unlocked_paths"]})
        out = synchronize_langmuir_parameters(out)
        if "model" in out and "column" in out:
            out.setdefault("chromatography_mode", "HIC" if uses_langmuir(str(out["model"])) else "ION_EXCHANGE")
            legacy_salt_d = _as_float(out.get("tdm", {}).get("salt_axial_dispersion_mm2_s"))
            defaults = system_defaults()
            if str(out["model"]) == MODEL_TDM_CPA and legacy_salt_d is not None:
                defaults["salt_dispersion_mm2_s"] = legacy_salt_d
            out["system"] = deep_merge(defaults, out.get("system", {}))
            out["least_squares"] = deep_merge({"objective": "RAW_SSE", "baseline_mode": "NONE", "detector_baseline": 0.0, "max_nfev": 120,
                                                "unlocked_paths": None, "selected_run_numbers": None}, out.get("least_squares", {}))
        # Migrate legacy projects and initialize new impurity factors from
        # the shared target response. Each impurity remains independently
        # editable after this default is written into its component row.
        components = out.get("components")
        conversion = out.get("conversion")
        global_response = _as_float(conversion.get("uv_to_protein_mAU_L_g")) if isinstance(conversion, dict) else None
        if isinstance(components, list) and global_response is not None and global_response > 0:
            for row in components[1:]:
                if isinstance(row, dict) and _as_float(row.get("uv_response_factor_mAU_L_g")) is None:
                    row["uv_response_factor_mAU_L_g"] = global_response
        if isinstance(out.get("feed"), dict) and isinstance(out.get("process"), dict):
            feed = out["feed"]
            process = out["process"]
            cv = _as_float(process.get("load_CV"))
            conc = _as_float(feed.get("total_concentration_mg_mL"))
            density = _as_float(feed.get("load_density_mg_mL_resin"))
            basis = normalize_load_amount_basis(process.get("load_amount_basis"))
            process["load_amount_basis"] = basis
            if basis == "CAPACITY":
                if density is None and cv is not None and conc is not None and conc > 0:
                    density = cv * conc
                    feed["load_density_mg_mL_resin"] = density
                if density is not None and density > 0 and conc is not None and conc > 0:
                    process["load_CV"] = density / conc
            else:
                # Legacy rows use load CV as the authority. Capacity is kept
                # consistent for mass balance and result receipts.
                if cv is None and conc is not None and conc > 0 and density is not None:
                    cv = density / conc
                    process["load_CV"] = cv
                if cv is not None and cv > 0 and conc is not None and conc > 0:
                    feed["load_density_mg_mL_resin"] = cv * conc
            mode = str(process.get("load_mode") or "LINEAR").strip().upper()
            process["load_mode"] = mode if mode in PROCESS_MODES else "LINEAR"
            control = str(process.get("load_chemistry_control") or "BUFFER_B_PERCENT").strip().upper()
            process["load_chemistry_control"] = control if control in CHEMISTRY_CONTROLS else "BUFFER_B_PERCENT"
            raw_load_material_mode = process.get("load_material_chemistry_mode")
            if (not str(raw_load_material_mode or "").strip()
                    and _chemistry_source(process.get("load_source")) == "DIRECT"
                    and any(_as_float(process.get(key)) is not None for key in (
                        "load_pH", "load_salt_concentration_M", "load_conductivity_mS_cm"))):
                # A pre-R17 direct load recipe had no separate mode column.
                # Preserve its intended endpoint chemistry instead of silently
                # discarding the populated load material fields.
                process["load_material_chemistry_mode"] = "EXPLICIT"
            else:
                process["load_material_chemistry_mode"] = normalize_load_material_chemistry_mode(
                    raw_load_material_mode, legacy_control=process["load_chemistry_control"]
                )
            for key in ("buffer_A", "buffer_B"):
                buffer = process.get(key)
                if isinstance(buffer, dict):
                    buffer["salt_input_mode"] = salt_input_mode(buffer)
                    source = str(buffer.get("ionic_strength_source") or "SALT_PROXY").strip().upper()
                    buffer["ionic_strength_source"] = source if source in {"SALT_PROXY", "ION_COMPOSITION"} else "SALT_PROXY"
            process["load_salt_input_mode"] = salt_input_mode(process, "load_")
            for key in ("plw_steps", "elution_steps"):
                for step in process.get(key, []):
                    for side in ("start_", "end_"):
                        step[side + "salt_input_mode"] = salt_input_mode(step, side)
        return out
    if isinstance(config, list):
        return [normalize_config(v) for v in config]
    if isinstance(config, float) and (math.isnan(config) or math.isinf(config)):
        return None
    return config


def configuration_fingerprint(config: Any) -> str:
    """Hash the complete normalized configuration used by a single run."""
    payload = json.dumps(
        normalize_config(config), sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    config = blank_config()
    p = Path(path)
    if p.is_file():
        with p.open("r", encoding="utf-8-sig") as fh:
            # Migrate incoming keys before merging new-schema blank defaults;
            # otherwise a default qmax_g_L=None would shadow a legacy value.
            config = deep_merge(config, normalize_config(json.load(fh)))
    return normalize_config(config)


def save_config(config: dict[str, Any], path: Path | str = CONFIG_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(normalize_config(config), fh, indent=2, ensure_ascii=False, allow_nan=False)


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(math.isnan(float(value)))
    except Exception:
        return False


def _number(value: Any, label: str, errors: list[str], *, positive: bool = False,
            nonnegative: bool = False, lower: float | None = None,
            upper: float | None = None) -> float | None:
    if _missing(value):
        errors.append(f"{label} is required.")
        return None
    try:
        out = float(value)
    except Exception:
        errors.append(f"{label} must be numeric.")
        return None
    if not math.isfinite(out):
        errors.append(f"{label} must be finite.")
        return None
    if positive and out <= 0:
        errors.append(f"{label} must be > 0.")
    if nonnegative and out < 0:
        errors.append(f"{label} must be >= 0.")
    if lower is not None and out < lower:
        errors.append(f"{label} must be >= {lower}.")
    if upper is not None and out > upper:
        errors.append(f"{label} must be <= {upper}.")
    return out


def impurity_count(config: dict[str, Any]) -> int:
    try:
        value = int(float(config.get("impurity_count", 0)))
    except Exception:
        return 0
    return min(max(value, 0), MAX_IMPURITIES)


def selected_components(config: dict[str, Any]) -> list[dict[str, Any]]:
    return list(config.get("components", []))[: 1 + impurity_count(config)]


def active_components(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in selected_components(config):
        try:
            if float(row.get("mass_percent")) > 0:
                rows.append(row)
        except Exception:
            pass
    return rows


def selected_process_steps(config: dict[str, Any], key: str) -> list[dict[str, Any]]:
    process = config.get("process", {})
    count_key = "plw_count" if key == "plw_steps" else "elution_count"
    max_count = MAX_PLW_STEPS if key == "plw_steps" else MAX_ELUTION_STEPS
    try:
        count = int(float(process.get(count_key, 0)))
    except Exception:
        count = 0
    count = min(max(count, 0), max_count)
    return list(process.get(key, []))[:count]


def _chemistry_source(value: Any) -> str:
    source = str(value or "DIRECT").strip().upper()
    return source if source in CHEMISTRY_SOURCES else "DIRECT"


def _chemistry_control(value: Any) -> str:
    control = str(value or "BUFFER_B_PERCENT").strip().upper()
    return control if control in CHEMISTRY_CONTROLS else "BUFFER_B_PERCENT"


def normalize_load_material_chemistry_mode(value: Any, *, legacy_control: Any = None) -> str:
    """Normalize the load-material chemistry selector.

    ``BUFFER_RECIPE`` follows the load %B program. ``EXPLICIT`` resolves the
    saved load source/pH/salt/conductivity fields and is independent of the
    load %B display values. Old projects with endpoint chemistry retain their
    previous behavior by migrating to ``EXPLICIT``.
    """
    mode = str(value or "").strip().upper()
    if not mode:
        return "EXPLICIT" if _chemistry_control(legacy_control) == "ENDPOINT_CHEMISTRY" else "BUFFER_RECIPE"
    return mode if mode in LOAD_MATERIAL_CHEMISTRY_MODES else "BUFFER_RECIPE"


def load_material_chemistry_is_explicit(process: dict[str, Any]) -> bool:
    """Return whether the load stage consumes its explicit chemistry fields."""
    return (
        normalize_load_material_chemistry_mode(
            process.get("load_material_chemistry_mode"),
            legacy_control=process.get("load_chemistry_control"),
        ) == "EXPLICIT"
        or _chemistry_control(process.get("load_chemistry_control")) == "ENDPOINT_CHEMISTRY"
    )


def buffer_ionic_strength_M(buffer: dict[str, Any]) -> float | None:
    """Return I = 0.5*sum(c_i*z_i^2) from complete buffer-ion rows.

    Blank ion rows are ignored. A partially completed row is left for validation
    to flag and is not silently used in the calculation.
    """
    terms: list[float] = []
    for ion in list(buffer.get("ions", []))[:MAX_BUFFER_IONS]:
        species = str(ion.get("species") or "").strip()
        c = _as_float(ion.get("concentration_M")) if "_as_float" in globals() else None
        z = _as_float(ion.get("valency")) if "_as_float" in globals() else None
        # _as_float is defined later in this module; use a local fallback while
        # keeping this helper callable during normal runtime after import.
        if c is None:
            try:
                c = float(ion.get("concentration_M"))
                if not math.isfinite(c): c = None
            except Exception:
                c = None
        if z is None:
            try:
                z = float(ion.get("valency"))
                if not math.isfinite(z): z = None
            except Exception:
                z = None
        if species and c is not None and z is not None and c >= 0:
            terms.append(0.5 * c * z * z)
    return float(sum(terms)) if terms else None


def buffer_salt_concentration_M(config: dict[str, Any], buffer: dict[str, Any]) -> float | None:
    """Resolve analytical salt molarity from the explicitly selected input."""
    return _resolve_salt_and_conductivity(config, buffer)[0]


def salt_input_mode(chemistry: dict[str, Any], prefix: str = "") -> str:
    """Infer salt-only legacy inputs, but keep old conductivity recipes stable.

    When both values exist in an older file, conductivity remains authoritative.
    An explicit selection never falls back to the other (possibly stale) field.
    """
    mode = str(chemistry.get(prefix + "salt_input_mode") or "").strip().upper()
    if mode:
        return mode
    if (_as_float(chemistry.get(prefix + "conductivity_mS_cm")) is None
            and _as_float(chemistry.get(prefix + "salt_concentration_M")) is not None):
        return "SALT_M"
    return "CONDUCTIVITY"


def _resolve_salt_and_conductivity(config: dict[str, Any], chemistry: dict[str, Any]) -> tuple[float | None, float | None]:
    mode = salt_input_mode(chemistry)
    cond = _as_float(chemistry.get("conductivity_mS_cm"))
    factor = _as_float(config.get("conversion", {}).get("conductivity_to_salt_M_per_mS_cm"))
    if mode == "SALT_M":
        salt = _as_float(chemistry.get("salt_concentration_M"))
        # An entered conductivity is independent of binding chemistry. If it
        # is absent, the calibrated factor supplies an approximate overlay.
        if cond is None and salt is not None and factor is not None and factor > 0:
            cond = salt / factor
    elif mode == "CONDUCTIVITY":
        salt = cond * factor if cond is not None and factor is not None and factor > 0 else None
    else:
        salt = None
    return (salt if salt is not None and salt >= 0 else None,
            cond if cond is not None and cond >= 0 else None)


def buffer_chemistry(config: dict[str, Any], buffer: dict[str, Any]) -> dict[str, Any]:
    salt, cond = _resolve_salt_and_conductivity(config, buffer)
    ion_source = str(buffer.get("ionic_strength_source") or "SALT_PROXY").upper()
    ionic = buffer_ionic_strength_M(buffer) if ion_source == "ION_COMPOSITION" else salt
    return {"pH": buffer.get("pH"), "salt_input_mode": salt_input_mode(buffer),
            "salt_concentration_M": salt, "conductivity_mS_cm": cond,
            "ionic_strength_source": ion_source, "ionic_strength_M": ionic,
            "conductivity_source": ("SALT_FACTOR_ESTIMATE" if salt_input_mode(buffer) == "SALT_M"
                                    and _as_float(buffer.get("conductivity_mS_cm")) is None else "ENTERED")}


def buffer_blend_chemistry(config: dict[str, Any], percent_B: Any) -> dict[str, Any]:
    """Resolve a Buffer A/B blend at the requested %B.

    Buffer A is the 0% B endpoint and Buffer B is the 100% B endpoint. Salt
    concentration and conductivity are ideal volumetric blends of the two
    endpoint values. The entered endpoint pH values are linearly interpolated;
    this is an explicit process-program approximation, not a full acid/base
    equilibrium calculation. Ionic strength is resolved for every model from
    each buffer's selected source. Salt molarity is distinct from ionic strength.
    """
    try:
        pct = float(percent_B)
    except Exception:
        pct = float("nan")
    if not math.isfinite(pct):
        return {"source": "BUFFER_BLEND", "percent_B": None, "pH": None,
                "conductivity_mS_cm": None, "salt_concentration_M": None,
                "ionic_strength_M": None}
    f = min(max(pct, 0.0), 100.0) / 100.0
    p = config.get("process", {})
    a = buffer_chemistry(config, p.get("buffer_A", {}))
    b = buffer_chemistry(config, p.get("buffer_B", {}))

    def _blend(x: Any, y: Any) -> float | None:
        try:
            xv = float(x); yv = float(y)
            if math.isfinite(xv) and math.isfinite(yv):
                return (1.0 - f) * xv + f * yv
        except Exception:
            pass
        return None

    pH_value = _blend(a.get("pH"), b.get("pH"))
    salt_a = a["salt_concentration_M"]
    salt_b = b["salt_concentration_M"]
    salt_value = _blend(salt_a, salt_b)
    cond_value = _blend(a.get("conductivity_mS_cm"), b.get("conductivity_mS_cm"))
    ionic_value = _blend(a["ionic_strength_M"], b["ionic_strength_M"])
    return {
        "source": "BUFFER_BLEND",
        "percent_B": pct,
        "pH": pH_value,
        "conductivity_mS_cm": cond_value,
        "salt_concentration_M": salt_value,
        "ionic_strength_M": ionic_value,
    }


def resolve_process_chemistry(
    config: dict[str, Any], *, source: Any, pH: Any, salt_M: Any, conductivity_mS_cm: Any,
    salt_input_mode: Any = None,
) -> dict[str, Any]:
    """Resolve one process chemistry endpoint into pH, salt, conductivity and CPA ionic strength.

    BUFFER_A/B references are resolved from the current buffer definitions.
    Direct endpoints accept conductivity or analytical salt molarity. Ion rows
    determine I=0.5*sum(c_i*z_i^2) independently of the transport model.
    """
    src = _chemistry_source(source)
    if src in {"BUFFER_A", "BUFFER_B"}:
        buffer = config.get("process", {}).get("buffer_A" if src == "BUFFER_A" else "buffer_B", {})
    else:
        buffer = {"pH": pH, "conductivity_mS_cm": conductivity_mS_cm,
                  "salt_concentration_M": salt_M, "salt_input_mode": salt_input_mode}
    return {"source": src, **buffer_chemistry(config, buffer)}


def process_pH_span(config: dict[str, Any]) -> float:
    """pH span actually consumed by the selected CPA process program."""
    if not uses_cpa(str(config.get("model") or "")):
        return 0.0
    vals: list[float] = []
    p = config.get("process", {})
    if not load_material_chemistry_is_explicit(p):
        load_sides = ("start", "end") if str(p.get("load_mode") or "LINEAR").upper() == "LINEAR" else (("end",) if str(p.get("load_mode")).upper() == "STEP" else ("start",))
        for side in load_sides:
            load = buffer_blend_chemistry(config, p.get(f"load_{side}_percent_B"))
            try:
                vals.append(float(load.get("pH")))
            except Exception:
                pass
    else:
        load = resolve_process_chemistry(
            config, source=p.get("load_source"), pH=p.get("load_pH"),
            salt_M=p.get("load_salt_concentration_M"), conductivity_mS_cm=p.get("load_conductivity_mS_cm"),
            salt_input_mode=p.get("load_salt_input_mode")
        )
        try:
            vals.append(float(load.get("pH")))
        except Exception:
            pass
    for step in selected_process_steps(config, "plw_steps") + selected_process_steps(config, "elution_steps"):
        mode = str(step.get("mode") or "SET_POINT").upper()
        sides = ("start", "end") if mode == "LINEAR" else (("end",) if mode == "STEP" else ("start",))
        control = _chemistry_control(step.get("chemistry_control"))
        for side in sides:
            if control == "BUFFER_B_PERCENT":
                resolved = buffer_blend_chemistry(config, step.get(f"{side}_percent_B"))
            else:
                resolved = resolve_process_chemistry(
                    config, source=step.get(f"{side}_source"), pH=step.get(f"{side}_pH"),
                    salt_M=step.get(f"{side}_salt_concentration_M"),
                    conductivity_mS_cm=step.get(f"{side}_conductivity_mS_cm"),
                    salt_input_mode=step.get(f"{side}_salt_input_mode"),
                )
            try:
                vals.append(float(resolved.get("pH")))
            except Exception:
                pass
    return max(vals) - min(vals) if vals else 0.0


# Exact mechanistic input contract. Operating conditions are intentionally kept
# separate because they are edited per run in the classic Process Setup tab.
MECH_COMMON_FIELDS = ("column.volume_mL", "column.length_mm", "conversion.uv_to_protein_mAU_L_g")
TDM_INPUT_FIELDS = ("tdm.bead_radius_um", "tdm.void_fraction", "tdm.particle_porosity")
EDM_INPUT_FIELDS = ("edm.total_bed_porosity",)
CPA_COMMON_INPUT_FIELDS = (
    "conversion.conductivity_to_salt_M_per_mS_cm",
    "cpa.ligand_surface_density_umol_m2",
    "cpa.system_specific_adsorption_parameter",
)
CPA_TDM_ONLY_INPUT_FIELDS = ("tdm.salt_axial_dispersion_mm2_s",)
COMPONENT_COMMON_FIELDS = ("mass_percent",)
COMPONENT_TDM_FIELDS = ("D_ax_mm2_s", "k_eff_um_s", "accessible_particle_porosity")
COMPONENT_EDM_FIELDS = ("D_app_mm2_s",)
COMPONENT_LANGMUIR_FIELDS = ("qmax_g_L", "b_L_g", "salt_sensitivity_per_M")
COMPONENT_CPA_FIELDS = ("diameter_nm", "As_m_inv", "Z_ref", "delta_ref")
COMPONENT_TDM_CPA_FIELDS = ("kkin_star_s",)
COMPONENT_CPA_PH_FIELDS = ("pH_ref", "Z1_per_pH", "Z2_per_pH2", "delta_pH_slope_m2_C")
COMPONENT_CPA_PH_LARGE_SPAN_FIELDS = ("Z3_per_pH3",)


def required_global_fields(model: str) -> tuple[str, ...]:
    fields = list(MECH_COMMON_FIELDS)
    if uses_tdm(model):
        fields.extend(TDM_INPUT_FIELDS)
        if model == MODEL_TDM_CPA:
            fields.extend(CPA_TDM_ONLY_INPUT_FIELDS)
    else:
        fields.extend(EDM_INPUT_FIELDS)
    used_buffers: set[str] = set()
    if uses_cpa(model):
        fields.extend(CPA_COMMON_INPUT_FIELDS)
    elif uses_langmuir(model):
        fields.append("conversion.conductivity_to_salt_M_per_mS_cm")
    return tuple(fields)


def required_component_fields(model: str, pH_span: float = 0.0) -> tuple[str, ...]:
    fields = list(COMPONENT_COMMON_FIELDS)
    fields.extend(COMPONENT_TDM_FIELDS if uses_tdm(model) else COMPONENT_EDM_FIELDS)
    if uses_langmuir(model):
        fields.extend(COMPONENT_LANGMUIR_FIELDS)
    else:
        fields.extend(COMPONENT_CPA_FIELDS)
        if model == MODEL_TDM_CPA:
            fields.extend(COMPONENT_TDM_CPA_FIELDS)
        if uses_cpa(model) and pH_span > 1e-9:
            fields.extend(COMPONENT_CPA_PH_FIELDS)
            if pH_span > 1.0:
                fields.extend(COMPONENT_CPA_PH_LARGE_SPAN_FIELDS)
    return tuple(fields)


def _parse_path(path: str) -> list[str | int]:
    import re
    out: list[str | int] = []
    for name, index in re.findall(r"([^\.\[\]]+)|\[(\d+)\]", path):
        out.append(name if name else int(index))
    return out


def _get_path(data: Any, path: str) -> Any:
    cur = data
    for token in _parse_path(path):
        cur = cur[token]
    return cur


def _set_path(data: Any, path: str, value: Any) -> None:
    tokens = _parse_path(path)
    cur = data
    for token in tokens[:-1]:
        cur = cur[token]
    cur[tokens[-1]] = value


def required_parameter_paths(config: dict[str, Any]) -> list[str]:
    model = str(config.get("model") or "")
    paths = list(required_global_fields(model))
    paths = [p for p in paths if p != "tdm.salt_axial_dispersion_mm2_s"]
    system = config.get("system", system_defaults())
    paths += ["system.uv_dead_volume_mL", "system.conductivity_dead_volume_mL"]
    if system.get("buffer_dispersion_enabled", True):
        paths.append("system.buffer_dispersion_mL")
    if system.get("salt_dispersion_enabled", True):
        paths.append("system.salt_dispersion_mm2_s")
    span = process_pH_span(config) if uses_cpa(model) else 0.0
    fields = required_component_fields(model, span)
    if uses_cpa(model) and is_hic(config):
        fields = tuple(f for f in fields if f not in {"Z_ref", *COMPONENT_CPA_PH_FIELDS, *COMPONENT_CPA_PH_LARGE_SPAN_FIELDS}) + ("salt_sensitivity_per_M",)
        paths = [p for p in paths if not p.startswith("cpa.")]
    for i, _row in enumerate(selected_components(config)):
        paths.extend(f"components[{i}].{field}" for field in fields)
        if i > 0:
            paths.append(f"components[{i}].uv_response_factor_mAU_L_g")
    return paths


def required_parameter_values(config: dict[str, Any]) -> dict[str, Any]:
    return {p: _get_path(config, p) for p in required_parameter_paths(config)}


def _copy_run_conditions(src: dict[str, Any], dst: dict[str, Any]) -> None:
    dst["column"]["flow_mL_min"] = src.get("column", {}).get("flow_mL_min")
    dst["feed"] = deep_merge(dst["feed"], src.get("feed", {}))
    dst["numerics"] = deep_merge(dst["numerics"], src.get("numerics", {}))
    dst["process"] = deep_merge(dst["process"], src.get("process", {}))


def sanitize_config_for_selected_model(config: dict[str, Any]) -> dict[str, Any]:
    config = normalize_config(config)
    clean = blank_config()
    clean["model"] = str(config.get("model") or MODEL_TDM_CPA)
    clean["impurity_count"] = impurity_count(config)
    clean["chromatography_mode"] = config["chromatography_mode"]
    clean["system"] = dict(config["system"])
    clean["least_squares"] = dict(config["least_squares"])
    _copy_run_conditions(config, clean)
    for i, row in enumerate(selected_components(config)):
        clean["components"][i]["name"] = str(row.get("name") or clean["components"][i]["name"])
    for path in required_parameter_paths(config):
        try:
            _set_path(clean, path, _get_path(config, path))
        except Exception:
            pass
    # Preserve the optional CPA pH parameterisation even when the current
    # run is isocratic. A different run in the same selected batch may contain
    # a pH program; these fields remain hidden unless such a run exists.
    if uses_cpa(clean["model"]):
        for i, _row in enumerate(selected_components(config)):
            for field in COMPONENT_CPA_PH_FIELDS + COMPONENT_CPA_PH_LARGE_SPAN_FIELDS:
                try:
                    clean["components"][i][field] = config["components"][i].get(field)
                except Exception:
                    pass
    return normalize_config(clean)


def batch_shared_parameter_paths(config: dict[str, Any]) -> list[str]:
    config = normalize_config(config)
    paths = ["model", "impurity_count", "chromatography_mode"] + list(required_global_fields(str(config.get("model") or "")))
    paths += ["system." + field for field in config["system"]]
    paths += ["least_squares." + field for field in config["least_squares"]]
    span = process_pH_span(config) if uses_cpa(str(config.get("model") or "")) else 0.0
    fields = list(required_component_fields(str(config.get("model") or ""), span))
    if uses_cpa(str(config.get("model"))) and is_hic(config):
        fields.append("salt_sensitivity_per_M")
    if uses_cpa(str(config.get("model") or "")):
        for f in COMPONENT_CPA_PH_FIELDS + COMPONENT_CPA_PH_LARGE_SPAN_FIELDS:
            if f not in fields:
                fields.append(f)
    for i, _row in enumerate(selected_components(config)):
        paths.append(f"components[{i}].name")
        paths.extend(f"components[{i}].{f}" for f in fields)
        if i > 0:
            paths.append(f"components[{i}].uv_response_factor_mAU_L_g")
    return paths


def batch_shared_parameter_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_config(config)
    return {p: _get_path(normalized, p) for p in batch_shared_parameter_paths(normalized)}


def batch_parameter_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(normalize_config(batch_shared_parameter_snapshot(config)), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def apply_batch_shared_parameters(base_config: dict[str, Any], run_config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    base_config = normalize_config(base_config)
    run_config = normalize_config(run_config)
    effective = deep_merge(blank_config(), base_config)
    effective = deep_merge(effective, run_config)
    overridden: list[str] = []
    for path in batch_shared_parameter_paths(base_config):
        base_value = _get_path(base_config, path)
        try:
            candidate = _get_path(effective, path)
        except Exception:
            candidate = None
        if normalize_config(candidate) != normalize_config(base_value):
            overridden.append(path)
        _set_path(effective, path, base_value)
    return sanitize_config_for_selected_model(effective), overridden


def requirement_text(model: str, pH_span: float = 0.0, selected_impurities: int = 0, chromatography_mode: str | None = None) -> str:
    n_imp = min(max(int(selected_impurities), 0), MAX_IMPURITIES)
    hic = chromatography_mode == "HIC"
    lines = [MODEL_LABELS.get(model, model), "", "MODEL / COLUMN PARAMETERS (shared by the selected batch)",
             "• Column volume V_col [mL]", "• Column length L_C [mm]",
             "• Target UV → protein response factor [mAU·L/g]"]
    if uses_tdm(model):
        lines += ["• Bead radius r_p [µm]", "• Void fraction ε_v [-]", "• Particle porosity ε_p [-]"]
        if model == MODEL_TDM_CPA:
            lines += ["• Salt axial dispersion D_ax,salt [mm²/s]"]
    else:
        lines += ["• Total bed porosity ε [-]"]
    if uses_cpa(model):
        lines += ["• Conductivity → salt concentration factor [M per (mS/cm)]",
                  "• Ligand surface density Γ_L [µmol/m²]", "• System-specific adsorption parameter [-]"]
    elif uses_langmuir(model):
        lines += ["• Conductivity → salt concentration factor [M per (mS/cm)]"]
    lines += ["", f"PROTEIN SPECIES: target + {n_imp} impurit{'y' if n_imp == 1 else 'ies'}",
              "The following are required for every selected species:",
              "• Feed composition mass% of total protein [%]"]
    if n_imp:
        lines += ["• UV → protein response factor [mAU·L/g] for every selected impurity"]
    if uses_tdm(model):
        lines += ["• D_ax,i [mm²/s]", "• k_eff,i [µm/s]", "• Accessible particle porosity ε_p,i [-] (F_acc,i is derived)"]
    else:
        lines += ["• D_app,i [mm²/s]"]
    if uses_langmuir(model):
        lines += ["• Saturation capacity qmax,i [g/L stationary phase]",
                  "• Competitive Langmuir b_i [L/g]",
                  "• HIC salt sensitivity k_s,i [M⁻¹]",
                  "H_i = qmax,i × b_i is calculated automatically; b_i is multiplied by exp(k_s,i × salt concentration) in the HIC isotherm."]
    else:
        lines += ["• Diameter [nm] (a_i = diameter/2)", "• A_s,i [m⁻¹]", "• Z_i [-]", "• Δ_i [-]"]
        if model == MODEL_TDM_CPA:
            lines += ["• k*kin,i [s⁻¹]"]
        if uses_cpa(model) and pH_span > 1e-9:
            lines += ["• pH_ref, Z1,i, Z2,i and Δ1,i because the selected batch contains a pH-changing run"]
            if pH_span > 1.0:
                lines += ["• Z3,i because the selected batch pH span exceeds 1 pH unit"]
    lines += ["", "OPERATING CONDITIONS are entered separately for each run in Process Setup.",
              "Only the selected PLW and elution rows are used by that run."]
    if hic and uses_cpa(model):
        lines = [line for line in lines if "Ligand surface density" not in line and "System-specific adsorption" not in line and "because the selected batch" not in line]
        lines = [line for line in lines if line != "• Z_i [-]"]
        lines += ["• HIC salt sensitivity k_s,i [M^-1]", "CPA HIC: Delta_i × available surface × exp(k_s,i × salt_M); empirical HIC extension."]
    lines += ["", "SYSTEM RESPONSE: upstream buffer dispersion volume, column salt dispersion, independent UV and conductivity dead volumes.", "Buffer and salt dispersion and the %B/conductivity overlays have independent On/Off selectors."]
    return "\n".join(lines)


def validate_mechanistic_config(config: dict[str, Any], *, pH_span_override: float | None = None) -> list[str]:
    config = normalize_config(config)
    errors: list[str] = []
    model = str(config.get("model") or "")
    if model not in MODELS:
        return ["Select one of the four supported model combinations."]
    try:
        n_imp_raw = float(config.get("impurity_count", 0))
        if n_imp_raw != int(n_imp_raw) or not 0 <= int(n_imp_raw) <= MAX_IMPURITIES:
            errors.append(f"Number of impurities must be an integer from 0 to {MAX_IMPURITIES}.")
    except Exception:
        errors.append(f"Number of impurities must be an integer from 0 to {MAX_IMPURITIES}.")

    col = config["column"]
    _number(col.get("volume_mL"), "Column volume V_col [mL]", errors, positive=True)
    _number(col.get("length_mm"), "Column length L_C [mm]", errors, positive=True)
    if config.get("chromatography_mode") not in {"HIC", "ION_EXCHANGE"}:
        errors.append("Chromatography mode must be HIC or ION_EXCHANGE.")
    system = config["system"]
    for field in ("buffer_dispersion_mL", "salt_dispersion_mm2_s", "uv_dead_volume_mL", "conductivity_dead_volume_mL"):
        _number(system.get(field), field, errors, nonnegative=True)
    for field in ("buffer_dispersion_enabled", "salt_dispersion_enabled", "show_percent_B", "show_conductivity", "show_flow_rate"):
        if not isinstance(system.get(field), bool):
            errors.append(f"{field} must be true or false.")
    lsq = config["least_squares"]
    if lsq.get("objective") not in {"RAW_SSE", "NORMALIZED_MSE", "WEIGHTED_RMSE"}:
        errors.append("Least-squares objective must be RAW_SSE, NORMALIZED_MSE or WEIGHTED_RMSE.")
    if lsq.get("baseline_mode") not in {"NONE", "INITIAL_MEDIAN"}:
        errors.append("Baseline mode must be NONE or INITIAL_MEDIAN.")
    _number(lsq.get("detector_baseline"), "Detector baseline", errors)
    _number(lsq.get("max_nfev"), "Least-squares maximum solver evaluations", errors, lower=2, upper=10000)
    conv = config.get("conversion", {})
    _number(conv.get("uv_to_protein_mAU_L_g"), "Target UV → protein response factor [mAU·L/g]", errors, positive=True)
    epsp: float | None = None
    if uses_tdm(model):
        t = config["tdm"]
        _number(t.get("bead_radius_um"), "Bead radius r_p [µm]", errors, positive=True)
        _number(t.get("void_fraction"), "Void fraction ε_v", errors, lower=1e-9, upper=0.999999)
        epsp = _number(t.get("particle_porosity"), "Particle porosity ε_p", errors, lower=1e-9, upper=0.999999)
        
    else:
        _number(config["edm"].get("total_bed_porosity"), "EDM total bed porosity ε", errors, lower=1e-9, upper=0.999999)
    if uses_cpa(model):
        _number(conv.get("conductivity_to_salt_M_per_mS_cm"), "Conductivity → salt concentration factor [M per (mS/cm)]", errors, positive=True)
        cp = config["cpa"]
        if not is_hic(config):
            _number(cp.get("ligand_surface_density_umol_m2"), "CPA ligand surface density Γ_L", errors, positive=True)
            _number(cp.get("system_specific_adsorption_parameter"), "CPA system-specific adsorption parameter", errors, positive=True)
    elif uses_langmuir(model):
        _number(conv.get("conductivity_to_salt_M_per_mS_cm"), "Conductivity → salt concentration factor [M per (mS/cm)]", errors, positive=True)

    pH_span = process_pH_span(config) if pH_span_override is None else float(pH_span_override)
    total = 0.0
    for idx, row in enumerate(selected_components(config), start=1):
        name = str(row.get("name") or (TARGET_COMPONENT_NAME if idx == 1 else f"Impurity {idx-1}"))
        mf = _number(row.get("mass_percent"), f"{name} feed composition mass% [%]", errors, positive=True, upper=100.0)
        if mf is not None:
            total += mf
        if idx > 1:
            _number(row.get("uv_response_factor_mAU_L_g"), f"{name} UV → protein response factor [mAU·L/g]", errors, positive=True)
        if uses_tdm(model):
            _number(row.get("D_ax_mm2_s"), f"{name} D_ax,i [mm²/s]", errors, nonnegative=True)
            _number(row.get("k_eff_um_s"), f"{name} k_eff,i [µm/s]", errors, positive=True)
            epspi = _number(row.get("accessible_particle_porosity"), f"{name} ε_p,i", errors, lower=1e-12, upper=0.999999)
            if epsp is not None and epspi is not None and epspi > epsp + 1e-12:
                errors.append(f"{name} ε_p,i must be <= ε_p.")
        else:
            _number(row.get("D_app_mm2_s"), f"{name} D_app,i [mm²/s]", errors, nonnegative=True)
        if uses_langmuir(model):
            _number(row.get("qmax_g_L"), f"{name} qmax,i [g/L stationary phase]", errors, positive=True)
            _number(row.get("b_L_g"), f"{name} b_i [L/g]", errors, positive=True)
            _number(row.get("salt_sensitivity_per_M"), f"{name} HIC salt sensitivity k_s,i [M⁻¹]", errors, lower=(0.0 if is_hic(config) else -25.0), upper=25.0)
        else:
            _number(row.get("diameter_nm"), f"{name} diameter [nm]", errors, positive=True)
            _number(row.get("As_m_inv"), f"{name} A_s,i [m⁻¹]", errors, positive=True)
            if is_hic(config):
                _number(row.get("salt_sensitivity_per_M"), f"{name} HIC salt sensitivity k_s,i", errors, lower=0.0, upper=25.0)
            else:
                _number(row.get("Z_ref"), f"{name} Z_i", errors)
            _number(row.get("delta_ref"), f"{name} Δ_i", errors, positive=True)
            if model == MODEL_TDM_CPA:
                _number(row.get("kkin_star_s"), f"{name} k*kin,i [s⁻¹]", errors, positive=True)
            if uses_cpa(model) and not is_hic(config) and pH_span > 1e-9:
                _number(row.get("pH_ref"), f"{name} pH_ref", errors, lower=0, upper=14)
                _number(row.get("Z1_per_pH"), f"{name} Z1,i", errors)
                _number(row.get("Z2_per_pH2"), f"{name} Z2,i", errors)
                _number(row.get("delta_pH_slope_m2_C"), f"{name} Δ1,i", errors)
                if pH_span > 1.0:
                    _number(row.get("Z3_per_pH3"), f"{name} Z3,i", errors)
    if selected_components(config) and abs(total - 100.0) > 1e-6:
        errors.append(f"Target + selected impurity mass fractions must sum to 100; current total is {total:g}.")
    return errors


def _validate_salt_input(chemistry: dict[str, Any], prefix: str, label: str, errors: list[str]) -> None:
    mode = salt_input_mode(chemistry, prefix)
    if mode not in SALT_INPUT_MODES:
        errors.append(f"{label} salt input must be CONDUCTIVITY or SALT_M.")
    elif mode == "SALT_M":
        _number(chemistry.get(prefix + "salt_concentration_M"), f"{label} salt concentration [M]", errors, nonnegative=True)
        if not _missing(chemistry.get(prefix + "conductivity_mS_cm")):
            _number(chemistry.get(prefix + "conductivity_mS_cm"), f"{label} conductivity [mS/cm]", errors, nonnegative=True)
    else:
        _number(chemistry.get(prefix + "conductivity_mS_cm"), f"{label} conductivity [mS/cm]", errors, nonnegative=True)


def _salt_chemistry_from_fields(chemistry: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    return {
        "salt_input_mode": chemistry.get(prefix + "salt_input_mode"),
        "salt_concentration_M": chemistry.get(prefix + "salt_concentration_M"),
        "conductivity_mS_cm": chemistry.get(prefix + "conductivity_mS_cm"),
    }


def _validate_hic_salt_domain(
    config: dict[str, Any], sources: list[tuple[str, dict[str, Any]]], errors: list[str],
) -> None:
    """Check effective, selected salt inputs before HIC affinity is evaluated."""
    resolved: list[tuple[str, float]] = []
    factor = _as_float(config.get("conversion", {}).get("conductivity_to_salt_M_per_mS_cm"))
    for label, chemistry in sources:
        salt = buffer_salt_concentration_M(config, chemistry)
        if salt is None:
            continue  # The selected input already has its own required-field error.
        mode = salt_input_mode(chemistry)
        if salt > MAX_HIC_SALT_M:
            origin = (
                f"conductivity {chemistry.get('conductivity_mS_cm')} mS/cm × factor {factor:g} M/(mS/cm)"
                if mode == "CONDUCTIVITY" and factor is not None else "entered SALT_M"
            )
            errors.append(
                f"{label} resolves to {salt:g} M salt from {origin}. "
                f"The HIC model limit is {MAX_HIC_SALT_M:g} M; check the input mode and units."
            )
        else:
            resolved.append((label, salt))
    if not resolved:
        return
    label, maximum = max(resolved, key=lambda entry: entry[1])
    for row in selected_components(config):
        sensitivity = _as_float(row.get("salt_sensitivity_per_M"))
        if sensitivity is not None and sensitivity * maximum > MAX_HIC_LOG_AFFINITY:
            errors.append(
                f"{row.get('name') or 'Protein'} HIC affinity would reach "
                f"k_s × salt = {sensitivity * maximum:g} at {label}, above the "
                f"numerical limit {MAX_HIC_LOG_AFFINITY:g}. Check k_s and salt units."
            )


def _validate_buffer_definition(config: dict[str, Any], key: str, label: str, errors: list[str], *, require_pH: bool = True) -> None:
    buffer = config.get("process", {}).get(key, {})
    if require_pH:
        _number(buffer.get("pH"), f"{label} pH", errors, lower=0, upper=14)
    _validate_salt_input(buffer, "", label, errors)
    if str(buffer.get("ionic_strength_source", "SALT_PROXY")).upper() == "ION_COMPOSITION":
        for i, ion in enumerate(list(buffer.get("ions", []))[:MAX_BUFFER_IONS], start=1):
            species = str(ion.get("species") or "").strip()
            has_c = not _missing(ion.get("concentration_M"))
            has_z = not _missing(ion.get("valency"))
            if species or has_c or has_z:
                if not species:
                    errors.append(f"{label} ion {i} species is required when the ion row is used.")
                _number(ion.get("concentration_M"), f"{label} ion {i} concentration [M]", errors, nonnegative=True)
                z = _number(ion.get("valency"), f"{label} ion {i} valency", errors)
                if z is not None and abs(z) < 1e-15:
                    errors.append(f"{label} ion {i} valency must be non-zero.")


def validate_operating_conditions(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    model = str(config.get("model") or "")
    _number(config["column"].get("flow_mL_min"), "Load flow [mL/min]", errors, positive=True)
    feed = config["feed"]
    c = _number(feed.get("total_concentration_mg_mL"), "Feed concentration [mg/mL]", errors, positive=True)
    p = config["process"]
    basis = normalize_load_amount_basis(p.get("load_amount_basis"))
    if basis == "CAPACITY":
        density = _number(feed.get("load_density_mg_mL_resin"), "Load capacity [g/L resin]", errors, positive=True)
        load_cv_value = (density / c) if density is not None and c is not None and c > 0 else None
    else:
        load_cv_value = p.get("load_CV")
        if _missing(load_cv_value):
            density = _number(feed.get("load_density_mg_mL_resin"), "Load capacity [g/L resin]", errors, positive=True)
            load_cv_value = (density / c) if density is not None and c is not None and c > 0 else None
    load_cv = _number(load_cv_value, "Load volume [CV]", errors, positive=True)
    numerics = config.get("numerics", {})
    nt = _number(numerics.get("time_steps"), "Number of time steps", errors, lower=50, upper=20000)
    nx = _number(numerics.get("axial_positions"), "Number of axial positions", errors, lower=5, upper=250)
    if nt is not None and abs(nt - round(nt)) > 1e-9:
        errors.append("Number of time steps must be an integer.")
    if nx is not None and abs(nx - round(nx)) > 1e-9:
        errors.append("Number of axial positions must be an integer.")
    needs_process_salt = uses_cpa(model) or uses_langmuir(model)
    hic_salt_sources: list[tuple[str, dict[str, Any]]] = []
    if needs_process_salt:
        used_buffers: set[str] = set()
        load_control = _chemistry_control(p.get("load_chemistry_control"))
        if not load_material_chemistry_is_explicit(p):
            for side, label in (("start", "start"), ("end", "end")):
                _number(p.get(f"load_{side}_percent_B"), f"Load {label} Buffer B [%]", errors, lower=0, upper=100)
            used_buffers.update({"BUFFER_A", "BUFFER_B"})
        else:
            load_source = _chemistry_source(p.get("load_source"))
            if load_source == "DIRECT":
                if uses_cpa(model):
                    _number(p.get("load_pH"), "Load/process pH", errors, lower=0, upper=14)
                _validate_salt_input(p, "load_", "Load", errors)
                if is_hic(config):
                    hic_salt_sources.append(("Load material", _salt_chemistry_from_fields(p, "load_")))
            else:
                key = "buffer_A" if load_source == "BUFFER_A" else "buffer_B"
                used_buffers.add(load_source)
    for key, label, max_count in (("plw_steps", "PLW", MAX_PLW_STEPS), ("elution_steps", "Elution", MAX_ELUTION_STEPS)):
        count_key = "plw_count" if key == "plw_steps" else "elution_count"
        try:
            count = int(float(p.get(count_key, 0)))
            if not 0 <= count <= max_count:
                errors.append(f"{label} step count must be 0-{max_count}.")
                count = 0
        except Exception:
            errors.append(f"{label} step count must be 0-{max_count}.")
            count = 0
        for i, step in enumerate(list(p.get(key, []))[:count], start=1):
            mode = str(step.get("mode") or "").upper()
            if mode not in PROCESS_MODES:
                errors.append(f"{label} {i} mode must be SET_POINT, LINEAR or STEP.")
            _number(step.get("CV"), f"{label} {i} volume [CV]", errors, positive=True)
            _number(step.get("flow_mL_min"), f"{label} {i} flow [mL/min]", errors, positive=True)
            if needs_process_salt:
                control = _chemistry_control(step.get("chemistry_control"))
                if control == "BUFFER_B_PERCENT":
                    used_buffers.update({"BUFFER_A", "BUFFER_B"})
                    for side in ("start", "end"):
                        _number(step.get(f"{side}_percent_B"), f"{label} {i} {side} Buffer B [%]", errors, lower=0, upper=100)
                else:
                    for side in ("start", "end"):
                        src = _chemistry_source(step.get(f"{side}_source"))
                        if src == "DIRECT":
                            if uses_cpa(model):
                                _number(step.get(f"{side}_pH"), f"{label} {i} {side} pH", errors, lower=0, upper=14)
                            _validate_salt_input(step, side + "_", f"{label} {i} {side}", errors)
                            if is_hic(config):
                                hic_salt_sources.append((f"{label} {i} {side}", _salt_chemistry_from_fields(step, side + "_")))
                        else:
                            used_buffers.add(src)
    if needs_process_salt:
        for src in sorted(used_buffers):
            key = "buffer_A" if src == "BUFFER_A" else "buffer_B"
            _validate_buffer_definition(config, key, "Buffer A" if key == "buffer_A" else "Buffer B", errors,
                                        require_pH=uses_cpa(model))
            buffer = p.get(key, {})
            if is_hic(config):
                hic_salt_sources.append(("Buffer A" if key == "buffer_A" else "Buffer B", buffer))
            if (str(buffer.get("ionic_strength_source", "SALT_PROXY")).upper() == "ION_COMPOSITION"
                    and buffer_ionic_strength_M(buffer) is None):
                errors.append(f"{'Buffer A' if key == 'buffer_A' else 'Buffer B'} is set to ION_COMPOSITION but has no complete ion rows.")
    if is_hic(config):
        _validate_hic_salt_domain(config, hic_salt_sources, errors)
    if load_cv is not None and load_cv <= 0:
        errors.append("Load volume must be positive.")
    return errors


def validate_config(config: dict[str, Any], *, strict: bool = True) -> list[str]:
    return validate_mechanistic_config(config) + validate_operating_conditions(config)


def derived_input_summary(config: dict[str, Any]) -> dict[str, Any]:
    feed = config.get("feed", {})
    process = config.get("process", {})
    load_cv = _as_float(process.get("load_CV"))
    concentration = _as_float(feed.get("total_concentration_mg_mL"))
    capacity = _as_float(feed.get("load_density_mg_mL_resin"))
    if normalize_load_amount_basis(process.get("load_amount_basis")) == "CAPACITY":
        try:
            load_cv = capacity / concentration
        except Exception:
            load_cv = None
    elif load_cv is None:
        try:
            load_cv = capacity / concentration
        except Exception:
            pass
    try:
        load_volume_mL = float(load_cv) * float(config["column"]["volume_mL"])
    except Exception:
        load_volume_mL = None
    return {
        "model": MODEL_LABELS.get(str(config.get("model")), str(config.get("model"))),
        "impurity_count": impurity_count(config),
        "load_CV": load_cv,
        "load_capacity_g_L_resin": capacity,
        "required_load_volume_mL": load_volume_mL,
        "time_steps": config.get("numerics", {}).get("time_steps"),
        "axial_positions": config.get("numerics", {}).get("axial_positions"),
        "pH_span": process_pH_span(config),
        "buffers": {
            key: {
                "name": config.get("process", {}).get(key, {}).get("name"),
                "pH": config.get("process", {}).get(key, {}).get("pH"),
                "resolved_salt_concentration_M": buffer_salt_concentration_M(config, config.get("process", {}).get(key, {})),
                "ion_derived_ionic_strength_M": buffer_ionic_strength_M(config.get("process", {}).get(key, {})),
            }
            for key in ("buffer_A", "buffer_B")
        } if uses_cpa(str(config.get("model") or "")) else {},
        "species_feed_concentration_mg_mL": {
            str(r.get("name")): (float(feed.get("total_concentration_mg_mL")) * float(r.get("mass_percent")) / 100.0)
            for r in selected_components(config)
            if feed.get("total_concentration_mg_mL") is not None and r.get("mass_percent") is not None
        },
        "required_parameter_values": required_parameter_values(config),
    }


# ----------------------- batch run table -----------------------
def _migrate_row_salt_inputs(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    for prefix in ["Load", "BufferA", "BufferB"] + [
        f"{group}{i}_{side}" for group, count in (("PLW", MAX_PLW_STEPS), ("Elution", MAX_ELUTION_STEPS))
        for i in range(1, count + 1) for side in ("Start", "End")
    ]:
        if not str(row.get(prefix + "_Salt_Input_Mode") or "").strip():
            row[prefix + "_Salt_Input_Mode"] = salt_input_mode({
                "salt_concentration_M": row.get(prefix + "_Salt_M"),
                "conductivity_mS_cm": row.get(prefix + "_Conductivity_mS_cm")})
    return row


def batch_run_fields() -> list[str]:
    fields = [
        "Run_Number", "Run_Name", "LSQ_Chromatogram_CSV", "LSQ_Weight", "LSQ_Sheet", "LSQ_X_Column", "LSQ_Signal_Column", "LSQ_X_Unit", "Load_Flow_mL_min", "Feed_Concentration_mg_mL",
        "Load_Amount_Basis", "Load_CV", "Load_Density_mg_mL_resin", "Load_Mode", "Load_Start_Percent_B", "Load_End_Percent_B",
        "Load_Chemistry_Control", "Load_Material_Chemistry_Mode", "Time_Steps", "Axial_Positions",
        "Load_Source", "Load_pH", "Load_Salt_M", "Load_Conductivity_mS_cm",
        "Load_Salt_Input_Mode",
        "PLW_Count", "Elution_Count",
    ]
    for prefix in ("BufferA", "BufferB"):
        fields += [f"{prefix}_Name", f"{prefix}_pH", f"{prefix}_Salt_M", f"{prefix}_Conductivity_mS_cm", f"{prefix}_Ionic_Strength_Source"]
        fields.append(f"{prefix}_Salt_Input_Mode")
        for i in range(1, MAX_BUFFER_IONS + 1):
            fields += [f"{prefix}_Ion{i}_Species", f"{prefix}_Ion{i}_Concentration_M", f"{prefix}_Ion{i}_Valency"]
    for prefix, count in (("PLW", MAX_PLW_STEPS), ("Elution", MAX_ELUTION_STEPS)):
        for i in range(1, count + 1):
            fields += [
                f"{prefix}{i}_Mode", f"{prefix}{i}_CV", f"{prefix}{i}_Flow_mL_min",
                f"{prefix}{i}_Chemistry_Control", f"{prefix}{i}_Start_Percent_B", f"{prefix}{i}_End_Percent_B",
                f"{prefix}{i}_Start_Source", f"{prefix}{i}_Start_pH", f"{prefix}{i}_Start_Salt_M",
                f"{prefix}{i}_Start_Conductivity_mS_cm",
                f"{prefix}{i}_End_Source", f"{prefix}{i}_End_pH", f"{prefix}{i}_End_Salt_M",
                f"{prefix}{i}_End_Conductivity_mS_cm",
                f"{prefix}{i}_Start_Salt_Input_Mode", f"{prefix}{i}_End_Salt_Input_Mode",
            ]
    return fields


def default_batch_row(run_number: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "Run_Number": run_number,
        "Run_Name": f"Run {run_number}",
        "LSQ_Chromatogram_CSV": "",
        "LSQ_Weight": 1.0,
        "LSQ_Sheet": "",
        "LSQ_X_Column": "",
        "LSQ_Signal_Column": "",
        "LSQ_X_Unit": "AUTO",
        "Load_Flow_mL_min": 1.0,
        "Feed_Concentration_mg_mL": 1.0,
        "Load_Amount_Basis": "LOAD_VOLUME",
        "Load_CV": 10.0,
        "Load_Density_mg_mL_resin": 10.0,
        "Load_Mode": "LINEAR",
        "Load_Start_Percent_B": 0.0,
        "Load_End_Percent_B": 0.0,
        "Load_Chemistry_Control": "BUFFER_B_PERCENT",
        "Load_Material_Chemistry_Mode": "BUFFER_RECIPE",
        "Time_Steps": 700,
        "Axial_Positions": 40,
        "Load_Source": "BUFFER_A",
        "Load_pH": 6.0,
        "Load_Salt_M": "",
        "Load_Salt_Input_Mode": "CONDUCTIVITY",
        "Load_Conductivity_mS_cm": 5.0,
        "PLW_Count": 1,
        "Elution_Count": 1,
        "BufferA_Name": "Buffer A", "BufferA_pH": 6.0, "BufferA_Salt_M": "", "BufferA_Conductivity_mS_cm": 5.0,
        "BufferB_Name": "Buffer B", "BufferB_pH": 6.0, "BufferB_Salt_M": "", "BufferB_Conductivity_mS_cm": 100.0,
    }
    for prefix in ("BufferA", "BufferB"):
        row[f"{prefix}_Salt_Input_Mode"] = "CONDUCTIVITY"
        row[f"{prefix}_Ionic_Strength_Source"] = "SALT_PROXY"
        for i in range(1, MAX_BUFFER_IONS + 1):
            row[f"{prefix}_Ion{i}_Species"] = ""
            row[f"{prefix}_Ion{i}_Concentration_M"] = ""
            row[f"{prefix}_Ion{i}_Valency"] = ""
    for i in range(1, MAX_PLW_STEPS + 1):
        row.update({
            f"PLW{i}_Mode": "SET_POINT", f"PLW{i}_CV": 2.0 if i == 1 else 1.0,
            f"PLW{i}_Flow_mL_min": 1.0,
            f"PLW{i}_Chemistry_Control": "BUFFER_B_PERCENT", f"PLW{i}_Start_Percent_B": 0.0, f"PLW{i}_End_Percent_B": 0.0,
            f"PLW{i}_Start_Source": "BUFFER_A", f"PLW{i}_Start_pH": 6.0, f"PLW{i}_Start_Salt_M": "",
            f"PLW{i}_Start_Conductivity_mS_cm": 5.0,
            f"PLW{i}_End_Source": "BUFFER_A", f"PLW{i}_End_pH": 6.0, f"PLW{i}_End_Salt_M": "",
            f"PLW{i}_End_Conductivity_mS_cm": 5.0,
            f"PLW{i}_Start_Salt_Input_Mode": "CONDUCTIVITY", f"PLW{i}_End_Salt_Input_Mode": "CONDUCTIVITY",
        })
    for i in range(1, MAX_ELUTION_STEPS + 1):
        row.update({
            f"Elution{i}_Mode": "LINEAR" if i == 1 else "SET_POINT",
            f"Elution{i}_CV": 10.0 if i == 1 else 1.0,
            f"Elution{i}_Flow_mL_min": 1.0,
            f"Elution{i}_Chemistry_Control": "BUFFER_B_PERCENT", f"Elution{i}_Start_Percent_B": 0.0, f"Elution{i}_End_Percent_B": (100.0 if i == 1 else 0.0),
            f"Elution{i}_Start_Source": "BUFFER_A", f"Elution{i}_Start_pH": 6.0, f"Elution{i}_Start_Salt_M": "",
            f"Elution{i}_Start_Conductivity_mS_cm": 5.0,
            f"Elution{i}_End_Source": "BUFFER_B" if i == 1 else "BUFFER_A", f"Elution{i}_End_pH": 6.0, f"Elution{i}_End_Salt_M": "",
            f"Elution{i}_End_Conductivity_mS_cm": 100.0 if i == 1 else 5.0,
            f"Elution{i}_Start_Salt_Input_Mode": "CONDUCTIVITY", f"Elution{i}_End_Salt_Input_Mode": "CONDUCTIVITY",
        })
    return row


def ensure_batch_runs_csv(path: Path | str = BATCH_RUNS_CSV) -> Path:
    p = Path(path)
    fields = batch_run_fields()
    existing: dict[int, dict[str, Any]] = {}
    if p.is_file():
        try:
            with p.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                if not reader.fieldnames or "Run_Number" not in reader.fieldnames:
                    raise ValueError("Missing Run_Number column")
                for raw in reader:
                    try:
                        n = int(float(raw.get("Run_Number", 0)))
                    except Exception as exc:
                        raise ValueError("Invalid Run_Number in saved table") from exc
                    if n in existing or not 1 <= n <= MAX_RUNS:
                        raise ValueError(f"Duplicate or out-of-range Run_Number: {n}")
                    existing[n] = raw
        except Exception as exc:
            raise ValueError(f"Cannot read saved run table {p.name}; it was left unchanged: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for n in range(1, MAX_RUNS + 1):
        base = default_batch_row(n)
        raw = _migrate_row_salt_inputs(existing[n]) if n in existing else {}
        for f in fields:
            # Existing cells, including deliberately cleared cells, are input.
            # Defaults only fill columns absent from a legacy schema.
            if f in raw:
                base[f] = raw[f]
        if raw:
            # Older saved runs directly entered load CV. Preserve that basis
            # when the new capacity/volume selector is not present in the CSV.
            if not str(raw.get("Load_Amount_Basis", "")).strip():
                base["Load_Amount_Basis"] = "LOAD_VOLUME"
            # Legacy rows did not store a direct load CV. Preserve their old
            # load amount by deriving CV from density / feed concentration.
            if not str(raw.get("Load_CV", "")).strip():
                old_conc = _as_float(raw.get("Feed_Concentration_mg_mL"))
                old_density = _as_float(raw.get("Load_Density_mg_mL_resin"))
                if old_conc is not None and old_conc > 0 and old_density is not None:
                    base["Load_CV"] = old_density / old_conc
            if not str(raw.get("Load_Chemistry_Control", "")).strip():
                old_source = _chemistry_source(raw.get("Load_Source"))
                if old_source == "DIRECT":
                    base["Load_Chemistry_Control"] = "ENDPOINT_CHEMISTRY"
                    base["Load_Material_Chemistry_Mode"] = "EXPLICIT"
                else:
                    pct = 100.0 if old_source == "BUFFER_B" else 0.0
                    base["Load_Chemistry_Control"] = "BUFFER_B_PERCENT"
                    base["Load_Start_Percent_B"] = pct
                    base["Load_End_Percent_B"] = pct
            if not str(raw.get("Load_Material_Chemistry_Mode", "")).strip():
                base["Load_Material_Chemistry_Mode"] = (
                    "EXPLICIT" if _chemistry_control(base.get("Load_Chemistry_Control")) == "ENDPOINT_CHEMISTRY"
                    else "BUFFER_RECIPE"
                )
            for prefix in ("BufferA", "BufferB"):
                mode_col = f"{prefix}_Ionic_Strength_Source"
                if not str(raw.get(mode_col, "")).strip():
                    # Earlier releases prefilled 0.05 M / 1 M NaCl ion examples.
                    # Preserve a custom ion recipe as an explicit selection;
                    # default profiles now use the editable salt/conductivity.
                    ions = []
                    for ion_i in range(1, MAX_BUFFER_IONS + 1):
                        species = str(raw.get(f"{prefix}_Ion{ion_i}_Species", "") or "").strip()
                        conc = _as_float(raw.get(f"{prefix}_Ion{ion_i}_Concentration_M"))
                        valency = _as_float(raw.get(f"{prefix}_Ion{ion_i}_Valency"))
                        if species and conc is not None and valency is not None:
                            ions.append((species, conc, valency))
                    default_conc = 0.05 if prefix == "BufferA" else 1.0
                    canonical = [("Na+", default_conc, 1.0), ("Cl-", default_conc, -1.0)]
                    base[mode_col] = "ION_COMPOSITION" if ions and ions != canonical else "SALT_PROXY"
        # Migrate rows saved by the immediately preceding UI. Exact Buffer A/B
        # endpoint selections map naturally to 0/100 %B; explicit DIRECT
        # chemistry stays on the endpoint-chemistry path.
        if raw:
            for prefix, max_count in (("PLW", MAX_PLW_STEPS), ("Elution", MAX_ELUTION_STEPS)):
                for i in range(1, max_count + 1):
                    control_col = f"{prefix}{i}_Chemistry_Control"
                    if str(raw.get(control_col, "")).strip():
                        continue
                    ss = str(raw.get(f"{prefix}{i}_Start_Source", "") or "").strip().upper()
                    es = str(raw.get(f"{prefix}{i}_End_Source", "") or "").strip().upper()
                    if ss in {"BUFFER_A", "BUFFER_B"} and es in {"BUFFER_A", "BUFFER_B"}:
                        base[control_col] = "BUFFER_B_PERCENT"
                        base[f"{prefix}{i}_Start_Percent_B"] = 100.0 if ss == "BUFFER_B" else 0.0
                        base[f"{prefix}{i}_End_Percent_B"] = 100.0 if es == "BUFFER_B" else 0.0
                    else:
                        base[control_col] = "ENDPOINT_CHEMISTRY"
        rows.append(base)
    with p.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    return p


def load_batch_rows(path: Path | str = BATCH_RUNS_CSV) -> list[dict[str, str]]:
    p = ensure_batch_runs_csv(path)
    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def save_batch_rows(rows: list[dict[str, Any]], path: Path | str = BATCH_RUNS_CSV) -> Path:
    """Persist the run table, including per-run LSQ chromatogram associations."""
    p = Path(path)
    fields = batch_run_fields()
    normalized: list[dict[str, Any]] = []
    for index in range(MAX_RUNS):
        row = default_batch_row(index + 1)
        if index < len(rows):
            row.update({key: value for key, value in _migrate_row_salt_inputs(rows[index]).items() if key in fields})
        row["Run_Number"] = index + 1
        normalized.append(row)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(normalized)
    return p


def ensure_lsq_runs_csv() -> Path:
    """Own persisted LSQ process recipes, independent of the production batch.

    Existing projects start from a *copy* of their first five batch recipes,
    after which changing either table never mutates the other.
    """
    if not LSQ_RUNS_CSV.is_file():
        source_rows = load_batch_rows()
        initial = []
        for i in range(MAX_LSQ_PROFILES):
            row = dict(source_rows[i])
            row["Run_Name"] = f"LSQ Run {i + 1}"
            row["LSQ_Chromatogram_CSV"] = ""  # No accidental use of stale references.
            row["LSQ_Weight"] = 1.0
            initial.append(row)
        save_batch_rows(initial, LSQ_RUNS_CSV)
    return ensure_batch_runs_csv(LSQ_RUNS_CSV)


def load_lsq_rows() -> list[dict[str, str]]:
    return load_batch_rows(ensure_lsq_runs_csv())


def save_lsq_rows(rows: list[dict[str, Any]]) -> Path:
    return save_batch_rows(rows, ensure_lsq_runs_csv())


def _as_float(raw: Any, default: float | None = None) -> float | None:
    try:
        out = float(raw)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def normalize_load_amount_basis(value: Any) -> str:
    """Canonicalize saved JMP selector labels and API values."""
    text = str(value or "").strip().upper().replace("_", " ")
    if text == "CAPACITY" or text.startswith("CAPACITY "):
        return "CAPACITY"
    if text in {"LOAD VOLUME", "LOAD VOLUME [CV]", "VOLUME", "CV"} or text.startswith("LOAD VOLUME "):
        return "LOAD_VOLUME"
    return "LOAD_VOLUME"


def _as_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return default


def batch_row_to_run_config(row: dict[str, Any]) -> dict[str, Any]:
    cfg = blank_config()
    cfg["column"]["flow_mL_min"] = _as_float(row.get("Load_Flow_mL_min"))
    cfg["feed"]["total_concentration_mg_mL"] = _as_float(row.get("Feed_Concentration_mg_mL"))
    p = cfg["process"]
    p["load_CV"] = _as_float(row.get("Load_CV"))
    cfg["feed"]["load_density_mg_mL_resin"] = _as_float(row.get("Load_Density_mg_mL_resin"))
    # Accept display labels written by previous JMP builds as well as tokens.
    p["load_amount_basis"] = normalize_load_amount_basis(row.get("Load_Amount_Basis"))
    conc = cfg["feed"]["total_concentration_mg_mL"]
    density = cfg["feed"]["load_density_mg_mL_resin"]
    if p["load_amount_basis"] == "CAPACITY":
        if density is not None and conc is not None and conc > 0:
            p["load_CV"] = density / conc
        elif p["load_CV"] is not None and conc is not None:
            cfg["feed"]["load_density_mg_mL_resin"] = p["load_CV"] * conc
    else:
        if p["load_CV"] is None and conc is not None and conc > 0 and density is not None:
            p["load_CV"] = density / conc
        if p["load_CV"] is not None and conc is not None:
            cfg["feed"]["load_density_mg_mL_resin"] = p["load_CV"] * conc
    cfg["numerics"]["time_steps"] = _as_int(row.get("Time_Steps"), 700)
    cfg["numerics"]["axial_positions"] = _as_int(row.get("Axial_Positions"), 40)
    load_mode = str(row.get("Load_Mode") or "LINEAR").strip().upper()
    p["load_mode"] = load_mode if load_mode in PROCESS_MODES else "LINEAR"
    p["load_start_percent_B"] = _as_float(row.get("Load_Start_Percent_B"), 0.0)
    p["load_end_percent_B"] = _as_float(row.get("Load_End_Percent_B"), 0.0)
    p["load_source"] = _chemistry_source(row.get("Load_Source"))
    p["load_pH"] = _as_float(row.get("Load_pH"))
    p["load_salt_concentration_M"] = _as_float(row.get("Load_Salt_M"))
    p["load_conductivity_mS_cm"] = _as_float(row.get("Load_Conductivity_mS_cm"))
    p["load_salt_input_mode"] = salt_input_mode({
        "salt_input_mode": row.get("Load_Salt_Input_Mode"),
        "salt_concentration_M": p["load_salt_concentration_M"],
        "conductivity_mS_cm": p["load_conductivity_mS_cm"],
    })
    p["load_chemistry_control"] = _chemistry_control(row.get("Load_Chemistry_Control"))
    raw_load_material_mode = row.get("Load_Material_Chemistry_Mode")
    if (not str(raw_load_material_mode or "").strip()
            and p["load_source"] == "DIRECT"
            and any(_as_float(value) is not None for value in (
                p["load_pH"], p["load_salt_concentration_M"], p["load_conductivity_mS_cm"]))):
        p["load_material_chemistry_mode"] = "EXPLICIT"
    else:
        p["load_material_chemistry_mode"] = normalize_load_material_chemistry_mode(
            raw_load_material_mode, legacy_control=p["load_chemistry_control"]
        )
    for prefix, key in (("BufferA", "buffer_A"), ("BufferB", "buffer_B")):
        buf = p[key]
        buf["name"] = str(row.get(f"{prefix}_Name") or ("Buffer A" if key == "buffer_A" else "Buffer B")).strip()
        buf["pH"] = _as_float(row.get(f"{prefix}_pH"))
        buf["salt_concentration_M"] = _as_float(row.get(f"{prefix}_Salt_M"))
        buf["conductivity_mS_cm"] = _as_float(row.get(f"{prefix}_Conductivity_mS_cm"))
        buf["salt_input_mode"] = salt_input_mode({
            "salt_input_mode": row.get(f"{prefix}_Salt_Input_Mode"),
            "salt_concentration_M": buf["salt_concentration_M"],
            "conductivity_mS_cm": buf["conductivity_mS_cm"],
        })
        source = str(row.get(f"{prefix}_Ionic_Strength_Source") or "SALT_PROXY").strip().upper()
        buf["ionic_strength_source"] = source if source in {"SALT_PROXY", "ION_COMPOSITION"} else "SALT_PROXY"
        for i in range(1, MAX_BUFFER_IONS + 1):
            ion = buf["ions"][i-1]
            ion["species"] = str(row.get(f"{prefix}_Ion{i}_Species") or "").strip()
            ion["concentration_M"] = _as_float(row.get(f"{prefix}_Ion{i}_Concentration_M"))
            ion["valency"] = _as_float(row.get(f"{prefix}_Ion{i}_Valency"))
    p["plw_count"] = min(max(_as_int(row.get("PLW_Count"), 0), 0), MAX_PLW_STEPS)
    p["elution_count"] = min(max(_as_int(row.get("Elution_Count"), 0), 0), MAX_ELUTION_STEPS)
    for prefix, key, max_count in (("PLW", "plw_steps", MAX_PLW_STEPS), ("Elution", "elution_steps", MAX_ELUTION_STEPS)):
        for i in range(1, max_count + 1):
            step = p[key][i-1]
            mode = str(row.get(f"{prefix}{i}_Mode", "SET_POINT") or "SET_POINT").upper()
            step["mode"] = mode if mode in PROCESS_MODES else "SET_POINT"
            step["CV"] = _as_float(row.get(f"{prefix}{i}_CV"))
            step["flow_mL_min"] = _as_float(row.get(f"{prefix}{i}_Flow_mL_min"))
            step["chemistry_control"] = _chemistry_control(row.get(f"{prefix}{i}_Chemistry_Control"))
            step["start_percent_B"] = _as_float(row.get(f"{prefix}{i}_Start_Percent_B"))
            step["end_percent_B"] = _as_float(row.get(f"{prefix}{i}_End_Percent_B"))
            for side, cap in (("start", "Start"), ("end", "End")):
                step[f"{side}_source"] = _chemistry_source(row.get(f"{prefix}{i}_{cap}_Source"))
                step[f"{side}_pH"] = _as_float(row.get(f"{prefix}{i}_{cap}_pH"))
                step[f"{side}_salt_concentration_M"] = _as_float(row.get(f"{prefix}{i}_{cap}_Salt_M"))
                step[f"{side}_conductivity_mS_cm"] = _as_float(row.get(f"{prefix}{i}_{cap}_Conductivity_mS_cm"))
                step[f"{side}_salt_input_mode"] = salt_input_mode({
                    "salt_input_mode": row.get(f"{prefix}{i}_{cap}_Salt_Input_Mode"),
                    "salt_concentration_M": step[f"{side}_salt_concentration_M"],
                    "conductivity_mS_cm": step[f"{side}_conductivity_mS_cm"],
                })
    return cfg


def load_batch_run_config(run_number: int, base_config: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = load_batch_rows()
    if not 1 <= int(run_number) <= len(rows):
        raise ValueError(f"Run number must be 1-{MAX_RUNS}.")
    base = base_config if base_config is not None else load_config()
    run_cfg = batch_row_to_run_config(rows[int(run_number)-1])
    effective, _overrides = apply_batch_shared_parameters(base, run_cfg)
    return effective


def batch_max_pH_span(simulation_count: int, base_config: dict[str, Any] | None = None) -> float:
    base = base_config or load_config()
    if not uses_cpa(str(base.get("model") or "")):
        return 0.0
    rows = load_batch_rows()
    n = min(max(int(simulation_count), 1), MAX_RUNS)
    spans = []
    for row in rows[:n]:
        cfg, _ = apply_batch_shared_parameters(base, batch_row_to_run_config(row))
        spans.append(process_pH_span(cfg))
    return max(spans) if spans else 0.0


ensure_batch_runs_csv()
