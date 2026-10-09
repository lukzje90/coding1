from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import jmp  # type: ignore
    JMP_AVAILABLE = True
    JMP_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    jmp = None
    JMP_AVAILABLE = False
    JMP_IMPORT_ERROR = exc

from tdm_minimal_io import (
    BATCH_COUNT_FILE,
    BATCH_GALLERY_HTML,
    BATCH_RUNS_CSV,
    LSQ_RUNS_CSV,
    BATCH_SUMMARY_CSV,
    CONFIG_PATH,
    LATEST_RESULTS_CSV,
    LATEST_RESULTS_HTML,
    MAX_ELUTION_STEPS,
    MAX_BUFFER_IONS,
    MAX_IMPURITIES,
    MAX_LSQ_PROFILES,
    MAX_PLW_STEPS,
    MAX_RUNS,
    MODEL_EDM_CPA,
    MODEL_EDM_LANGMUIR,
    MODEL_LABELS,
    MODEL_TDM_CPA,
    MODEL_TDM_LANGMUIR,
    RESULT_GUARD_BUILD,
    normalize_config,
    is_hic,
    STATUS_PATH,
    ensure_batch_runs_csv,
    ensure_lsq_runs_csv,
    load_config,
    requirement_text,
)

PROJECT_DIR = Path(__file__).resolve().parent
BRIDGE_PATH = PROJECT_DIR / "tdm_bridge.py"
LAST_JSL = PROJECT_DIR / "last_generated_ui.jsl"
LSQ_ATTACHED_CSV = PROJECT_DIR / "least_squares_reference_chromatogram.csv"
LSQ_APPLY_JSL = PROJECT_DIR / "least_squares_apply_to_ui.jsl"
LSQ_REPORT_HTML = PROJECT_DIR / "least_squares_fit_report.html"
LSQ_HISTORY_CSV = PROJECT_DIR / "least_squares_objective_history.csv"
MAX_LSQ_REFERENCE_POINTS = 10


def _q(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False).replace("\\n", "\\!n")


def _path(path: Path) -> str:
    return _q(str(path.resolve()).replace("\\", "/"))


def _num(value: Any) -> str:
    try:
        if value is None:
            return "."
        return format(float(value), ".17g")
    except Exception:
        return "."


def _nrow(label: str, var: str, value: Any, label_width: int = 290,
          on_change: str | None = None, *, fit_path: str | None = None,
          fit_lock_vars: list[tuple[str, str]] | None = None,
          default_unlocked_paths: set[str] | None = None) -> str:
    callback = f', << Set Function(Function({{this}}, {on_change}))' if on_change else ""
    lock_control = ""
    if fit_path is not None:
        lock_var = f"{var}FitLock"
        selected = 2 if fit_path in (default_unlocked_paths or set()) else 1
        lock_control = (
            f', Text Box("LSQ", << Set Width(32)), '
            f'{lock_var} = Combo Box({{"Locked", "Unlocked"}}, << Set({selected}), << Set Width(94))'
        )
        if fit_lock_vars is not None:
            fit_lock_vars.append((lock_var, fit_path))
    return f'H List Box(Text Box({_q(label)}, << Set Width({label_width})), {var} = Number Edit Box({_num(value)}, 9, << Set Width(92){callback}){lock_control})'


def _trow(label: str, var: str, value: Any, label_width: int = 290) -> str:
    return f'H List Box(Text Box({_q(label)}, << Set Width({label_width})), {var} = Text Edit Box({_q(value)}, << Set Width(150)))'


def _component_panel(i: int, row: dict[str, Any], fit_lock_vars: list[tuple[str, str]],
                     default_unlocked_paths: set[str]) -> tuple[str, list[str], str]:
    p = f"c{i}"
    title = "Target protein" if i == 1 else f"Impurity {i-1}"
    qmax = row.get("qmax_g_L")
    b_value = row.get("b_L_g")
    h_value = row.get("H")
    if qmax is None and b_value not in (None, 0):
        try:
            qmax = float(h_value) / float(b_value)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    if qmax is not None and b_value is not None:
        try:
            h_value = float(qmax) * float(b_value)
        except (TypeError, ValueError):
            pass
    uv_response_control = (
        _nrow("UV → protein response factor [mAU·L/g]", f"{p}UVResponse", row.get("uv_response_factor_mAU_L_g"), 360,
              fit_path=f"components[{i-1}].uv_response_factor_mAU_L_g", fit_lock_vars=fit_lock_vars,
              default_unlocked_paths=default_unlocked_paths)
        if i > 1 else
        'Text Box("Target protein uses the shared UV response factor in Model Parameters.", << Set Wrap(620), << Set Font Style("Italic"))'
    )
    body = f'''{p}Panel = Outline Box({_q(title)},
        V List Box(
            {_trow("Species name", f"{p}Name", row.get("name", title))},
            {_nrow("Feed composition mass% of total protein [%]", f"{p}Mass", row.get("mass_percent"), 360, "If(!isLoading, UpdateUI())")},
            {uv_response_control},
            {p}TdmTransport = V List Box(
                {_nrow("Axial dispersion D_ax,i [mm²/s]", f"{p}Dax", row.get("D_ax_mm2_s"), fit_path=f"components[{i-1}].D_ax_mm2_s", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)},
                {_nrow("Effective mass transfer k_eff,i [µm/s]", f"{p}Keff", row.get("k_eff_um_s"), fit_path=f"components[{i-1}].k_eff_um_s", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)},
                {_nrow("Accessible particle porosity ε_p,i [-]", f"{p}EpsPi", row.get("accessible_particle_porosity"), fit_path=f"components[{i-1}].accessible_particle_porosity", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)}
            ),
            {p}EdmTransport = V List Box({_nrow("Apparent dispersion D_app,i [mm²/s]", f"{p}Dapp", row.get("D_app_mm2_s"), fit_path=f"components[{i-1}].D_app_mm2_s", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)}),
            {p}SaltRow = {_nrow("HIC salt sensitivity k_s,i [M⁻¹]", f"{p}SaltSensitivity", row.get("salt_sensitivity_per_M", 1.0), 360, "If(!isLoading, UpdateUI())", fit_path=f"components[{i-1}].salt_sensitivity_per_M", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)},
            {p}Langmuir = V List Box(
                {_nrow("Saturation capacity qmax,i [g/L stationary phase]", f"{p}Qmax", qmax, 360, "If(!isLoading, SyncLangmuirDerived(); UpdateUI())", fit_path=f"components[{i-1}].qmax_g_L", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)},
                {_nrow("Competitive Langmuir b_i [L/g]", f"{p}B", b_value, 290, "If(!isLoading, SyncLangmuirDerived(); UpdateUI())", fit_path=f"components[{i-1}].b_L_g", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)},
                Text Box("Salt-modified Langmuir affinity: b_eff,i = b_i × exp(k_s,i × salt concentration). Positive k_s means stronger binding at higher salt; estimate it for this protein, resin and salt system.", << Set Wrap(680), << Set Font Style("Italic")),
                Text Box("After Run selected, the status and result show the b and k_s received plus effective b at each stage. A STEP to very low salt can release protein at nearly the same CV despite different affinities; a LINEAR salt ramp makes retention changes easier to distinguish.", << Set Wrap(680), << Set Font Style("Italic")),
                H List Box(Text Box("Derived Henry constant H_i = qmax,i × b_i [-]", << Set Width(360)), {p}H = Number Edit Box({_num(h_value)}, 9, << Set Width(92), << Enable(0))),
                Text Box("The displayed H_i is the zero-salt reference. At local salt, H_eff,i = qmax,i × b_eff,i; both numerator and shared competitive denominator use their local salt-dependent affinities.", << Set Wrap(680), << Set Font Style("Italic")),
                Text Box("qmax uses stationary-phase volume. 1 mg/mL = 1 g/L. Packed-bed capacity also includes the stationary-phase volume fraction.", << Set Wrap(520), << Set Font Style("Italic"))
            ),
            {p}CPA = V List Box(
                {_nrow("Protein diameter [nm] (a_i = diameter/2)", f"{p}Diameter", row.get("diameter_nm"), fit_path=f"components[{i-1}].diameter_nm", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)},
                {_nrow("Accessible adsorber surface A_s,i [m⁻¹]", f"{p}As", row.get("As_m_inv"), fit_path=f"components[{i-1}].As_m_inv", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)},
                {p}ZrefRow = {_nrow("Protein charge Z_i [-]", f"{p}Zref", row.get("Z_ref"), fit_path=f"components[{i-1}].Z_ref", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)},
                {_nrow("CPA interaction parameter Δ_i [-]", f"{p}Delta", row.get("delta_ref"), fit_path=f"components[{i-1}].delta_ref", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)},
                {p}KkinRow = {_nrow("CPA kinetic parameter k*kin,i [s⁻¹]", f"{p}Kkin", row.get("kkin_star_s"), fit_path=f"components[{i-1}].kkin_star_s", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)},
                {p}PHPanel = Outline Box("pH-dependent CPA parameters",
                    V List Box(
                        {_nrow("Reference pH", f"{p}PHref", row.get("pH_ref"), fit_path=f"components[{i-1}].pH_ref", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)},
                        {_nrow("Z1,i [-]", f"{p}Z1", row.get("Z1_per_pH"), fit_path=f"components[{i-1}].Z1_per_pH", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)},
                        {_nrow("Z2,i [-]", f"{p}Z2", row.get("Z2_per_pH2"), fit_path=f"components[{i-1}].Z2_per_pH2", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)},
                        {p}Z3Row = {_nrow("Z3,i [-]", f"{p}Z3", row.get("Z3_per_pH3"), fit_path=f"components[{i-1}].Z3_per_pH3", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)},
                        {_nrow("Δ1,i [m²/C]", f"{p}DeltaSlope", row.get("delta_pH_slope_m2_C"), fit_path=f"components[{i-1}].delta_pH_slope_m2_C", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)}
                    )
                )
            )
        )
    )'''
    pairs = [
        (f"{p}Name << Get Text", f"tdm_ui_c{i}_name"),
        (f"{p}Mass << Get", f"tdm_ui_c{i}_mass_percent"),
        (f"{p}Dax << Get", f"tdm_ui_c{i}_D_ax_mm2_s"),
        (f"{p}Keff << Get", f"tdm_ui_c{i}_k_eff_um_s"),
        (f"{p}EpsPi << Get", f"tdm_ui_c{i}_accessible_particle_porosity"),
        (f"{p}Dapp << Get", f"tdm_ui_c{i}_D_app_mm2_s"),
        (f"{p}Qmax << Get", f"tdm_ui_c{i}_qmax_g_L"),
        (f"{p}B << Get", f"tdm_ui_c{i}_b_L_g"),
        (f"{p}SaltSensitivity << Get", f"tdm_ui_c{i}_salt_sensitivity_per_M"),
        (f"{p}Diameter << Get", f"tdm_ui_c{i}_diameter_nm"),
        (f"{p}As << Get", f"tdm_ui_c{i}_As_m_inv"),
        (f"{p}Zref << Get", f"tdm_ui_c{i}_Z_ref"),
        (f"{p}Delta << Get", f"tdm_ui_c{i}_delta_ref"),
        (f"{p}Kkin << Get", f"tdm_ui_c{i}_kkin_star_s"),
        (f"{p}PHref << Get", f"tdm_ui_c{i}_pH_ref"),
        (f"{p}Z1 << Get", f"tdm_ui_c{i}_Z1_per_pH"),
        (f"{p}Z2 << Get", f"tdm_ui_c{i}_Z2_per_pH2"),
        (f"{p}Z3 << Get", f"tdm_ui_c{i}_Z3_per_pH3"),
        (f"{p}DeltaSlope << Get", f"tdm_ui_c{i}_delta_pH_slope_m2_C"),
    ]
    if i > 1:
        pairs.append((f"{p}UVResponse << Get", f"tdm_ui_c{i}_uv_response_factor_mAU_L_g"))
    sends = [f'Python Send({expr}, Python Name({_q(name)}));' for expr, name in pairs]
    visible = f'''{p}Panel << Visibility(If({1 if i == 1 else f'impurityCount >= {i-1}'}, "Visible", "Collapse"));
    {p}TdmTransport << Visibility(If(isTDM, "Visible", "Collapse"));
    {p}EdmTransport << Visibility(If(isEDM, "Visible", "Collapse"));
    {p}Langmuir << Visibility(If(isLangmuir, "Visible", "Collapse"));
    {p}CPA << Visibility(If(isCPA, "Visible", "Collapse"));
    {p}KkinRow << Visibility(If(modelLabel == {_q(MODEL_LABELS[MODEL_TDM_CPA])}, "Visible", "Collapse"));
    {p}SaltRow << Visibility(If(isLangmuir | (isCPA & isHIC), "Visible", "Collapse"));
    {p}ZrefRow << Visibility(If(isCPA & !isHIC, "Visible", "Collapse"));
    {p}PHPanel << Visibility(If(isCPA & !isHIC & batchPHSpan > 1e-9, "Visible", "Collapse"));
    {p}Z3Row << Visibility(If(isCPA & !isHIC & batchPHSpan > 1, "Visible", "Collapse"));'''
    return body, sends, visible


def _salt_input_controls(prefix: str) -> str:
    # Editing any load-material salt field is an explicit user choice. Promote
    # the load selector automatically so a populated pH/salt/conductivity
    # value cannot be silently ignored while the load %B recipe remains stored.
    # A per-load pH/salt/conductivity edit is a direct material measurement.
    # Promote both selectors so an inherited BUFFER_A/B source cannot mask the
    # value the user just changed.
    load_hook = 'loadMaterialMode << Set(2); loadSource << Set(1); ' if prefix == "load" else ''
    return f'''H List Box(Text Box("Salt input", << Set Width(220)), {prefix}SaltMode = Combo Box({{"CONDUCTIVITY", "SALT_M"}}, << Set Width(170), << Set Function(Function({{this}}, If(!isLoading, {load_hook}SaveCurrentRun(); UpdateUI()))))),
        {prefix}SaltRow = H List Box(Text Box("Salt concentration [M salt]", << Set Width(220)), {prefix}Salt = Number Edit Box(., 8, << Set Width(92), << Set Function(Function({{this}}, If(!isLoading, {load_hook}SaveCurrentRun(); UpdateUI())))))'''


def _buffer_panel(prefix: str, title: str) -> str:
    default_name = "Buffer A" if prefix == "bufferA" else "Buffer B"
    ion_rows = []
    for i in range(1, MAX_BUFFER_IONS + 1):
        ion_rows.append(f'''{prefix}Ion{i}Panel = Outline Box("Ion {i}", V List Box(
            H List Box(Text Box("Species", << Set Width(180)), {prefix}Ion{i}Species = Text Edit Box("", << Set Width(130))),
            H List Box(Text Box("Concentration [M]", << Set Width(180)), {prefix}Ion{i}Conc = Number Edit Box(., 8, << Set Width(92), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); UpdateUI()))))),
            H List Box(Text Box("Valency z [-]", << Set Width(180)), {prefix}Ion{i}Z = Number Edit Box(., 8, << Set Width(92), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); UpdateUI())))))
        ))''')
    return f'''{prefix}Panel = Outline Box({_q(title)}, V List Box(
        H List Box(Text Box("Buffer name", << Set Width(220)), {prefix}Name = Text Edit Box({_q(default_name)}, << Set Width(150))),
        H List Box(Text Box("pH", << Set Width(220)), {prefix}PH = Number Edit Box(6, 8, << Set Width(92), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); UpdateUI()))))),
        {_salt_input_controls(prefix)},
        H List Box(Text Box("Conductivity [mS/cm]", << Set Width(220)), {prefix}Cond = Number Edit Box(5, 8, << Set Width(92), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); UpdateUI()))))),
        Text Box("CONDUCTIVITY uses conductivity × the calibrated conversion factor. SALT_M uses the entered salt molarity directly for binding; conductivity is then optional and used for the overlay. If blank, conductivity is estimated using the factor. The factor and HIC sensitivity depend on the salt and solution conditions.", << Set Wrap(690), << Set Font Style("Italic")),
        {prefix}IonicSourceRow = H List Box(Text Box("Ionic strength source", << Set Width(220)), {prefix}IonicSource = Combo Box({{"SALT_PROXY", "ION_COMPOSITION"}}, << Set Width(170), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); UpdateUI()))))),
        {prefix}IonSection = Outline Box("Buffer ion composition", V List Box(
        Text Box("ION_COMPOSITION uses I = 0.5 Σ cᵢzᵢ² for either CPA ion-exchange model. Enter individual ion molarities, including counterions. For 1 M ammonium sulfate: NH4+ = 2 M, z = +1; SO4-- = 1 M, z = -2; I = 3 M. HIC uses 1 M salt, not 3 M ionic strength. SALT_PROXY assumes I = salt, suitable only as a monovalent approximation.", << Set Wrap(680)),
            {', '.join(ion_rows)}
        )),
        {prefix}Derived = Text Box("", << Set Wrap(680))
    ))'''


def _step_panel(prefix: str, i: int, title: str) -> str:
    v = f"{prefix}{i}"
    return f'''{v}Panel = Outline Box({_q(title + ' ' + str(i))}, V List Box(
        H List Box(Text Box("Mode", << Set Width(220)), {v}Mode = Combo Box({{"SET_POINT", "LINEAR", "STEP"}}, << Set Width(125), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); UpdateUI()))))),
        H List Box(Text Box("Volume [CV]", << Set Width(220)), {v}CV = Number Edit Box(1, 8, << Set Width(92), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); UpdateUI()))))),
        H List Box(Text Box("Flow [mL/min]", << Set Width(220)), {v}Flow = Number Edit Box(1, 8, << Set Width(92), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); UpdateUI()))))),
        {v}StartBRow = H List Box(Text Box("Start / set-point Buffer B [%]", << Set Width(220)), {v}StartB = Number Edit Box(0, 8, << Set Width(92), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); UpdateUI()))))),
        {v}EndBRow = H List Box(Text Box("End Buffer B [%]", << Set Width(220)), {v}EndB = Number Edit Box(100, 8, << Set Width(92), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); UpdateUI()))))),
        {v}BlendDerived = Text Box("", << Set Wrap(680)),
        Text Box("Pump salt is the blend of BOTH buffers: (1 − %B/100) × Buffer A salt + (%B/100) × Buffer B salt. SET_POINT uses Start, STEP uses End, and LINEAR ramps between them. The preview below shows the salt and target affinity actually commanded for this step; the column responds after mixer and transport delay.", << Set Wrap(680), << Set Font Style("Italic")),
        {v}ChemPanel = V List Box(
            H List Box(Text Box("Chemistry control", << Set Width(220)), {v}ChemControl = Combo Box({{"BUFFER_B_PERCENT", "ENDPOINT_CHEMISTRY"}}, << Set Width(170), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); UpdateUI()))))),
            {v}PercentPanel = V List Box(
                Text Box("Buffer A = 0% B and Buffer B = 100% B. The %B values above resolve pH, salt concentration and ionic strength from the current Buffer A/B definitions. pH interpolation is a process approximation, not a buffer-speciation calculation.", << Set Wrap(680))
            ),
            {v}EndpointPanel = Outline Box("Advanced explicit endpoint chemistry", V List Box(
                Text Box("Use this only when the process cannot be represented by Buffer A/B %. When selected, the %B rows above remain stored but the explicit endpoint chemistry below is used for this step.", << Set Wrap(680), << Set Font Style("Italic")),
                {v}StartChem = Outline Box("Start chemistry", V List Box(
                    H List Box(Text Box("Source", << Set Width(220)), {v}StartSource = Combo Box({{"DIRECT", "BUFFER_A", "BUFFER_B"}}, << Set Width(125), << Set Function(Function({{this}}, If(!isLoading, ApplyChemSource({v}StartSource, {v}StartPH, {v}StartCond); SaveCurrentRun(); UpdateUI()))))),
                    H List Box({v}StartPHLabel = Text Box("Start pH", << Set Width(220)), {v}StartPH = Number Edit Box(6, 8, << Set Width(92))),
                    {_salt_input_controls(v + "Start")},
                    H List Box(Text Box("Start conductivity [mS/cm]", << Set Width(220)), {v}StartCond = Number Edit Box(5, 8, << Set Width(92)))
                )),
                {v}EndChem = Outline Box("End chemistry", V List Box(
                    H List Box(Text Box("Source", << Set Width(220)), {v}EndSource = Combo Box({{"DIRECT", "BUFFER_A", "BUFFER_B"}}, << Set Width(125), << Set Function(Function({{this}}, If(!isLoading, ApplyChemSource({v}EndSource, {v}EndPH, {v}EndCond); SaveCurrentRun(); UpdateUI()))))),
                    H List Box({v}EndPHLabel = Text Box("End pH", << Set Width(220)), {v}EndPH = Number Edit Box(6, 8, << Set Width(92))),
                    {_salt_input_controls(v + "End")},
                    H List Box(Text Box("End conductivity [mS/cm]", << Set Width(220)), {v}EndCond = Number Edit Box(5, 8, << Set Width(92)))
                ))
            ))
        )
    ))'''


def _lsq_reference_panel(k: int) -> tuple[str, list[str], str]:
    rows, sends, vis = [], [f'Python Send(lsqR{k}CV << Get, Python Name({_q(f"tdm_ui_lsq_r{k}_CV")}));'], []
    for i in range(1, 2 + MAX_IMPURITIES):
        row = f"lsqR{k}C{i}Row"
        box = f"lsqR{k}C{i}Mass"
        label = "Target protein mass%" if i == 1 else f"Impurity {i-1} mass%"
        rows.append(f'{row} = H List Box(Text Box({_q(label)}, << Set Width(215)), {box} = Number Edit Box(., 8, << Set Width(92)))')
        sends.append(f'Python Send({box} << Get, Python Name({_q(f"tdm_ui_lsq_r{k}_c{i}_mass_percent")}));')
        vis.append(f'{row} << Visibility(If({1 if i == 1 else f"impurityCount >= {i-1}"}, "Visible", "Collapse"));')
    body = f'''lsqRef{k}Panel = Outline Box("Composition reference {k}", V List Box(
        H List Box(Text Box("Reference CV", << Set Width(215)), lsqR{k}CV = Number Edit Box(., 8, << Set Width(92))),
        {', '.join(rows)}
    ))'''
    visibility = f'lsqRef{k}Panel << Visibility(If(lsqUsesComposition & lsqRefCount >= {k}, "Visible", "Collapse"));' + ''.join(vis)
    return body, sends, visibility


def _req_expr(model: str, span: float, chromatography_mode: str | None = None) -> str:
    vals = {n: _q(requirement_text(model, span, n, chromatography_mode)) for n in range(MAX_IMPURITIES + 1)}
    expr = vals[MAX_IMPURITIES]
    for n in reversed(range(MAX_IMPURITIES)):
        expr = f'If(impurityCount == {n}, {vals[n]}, {expr})'
    return expr


def _default_lsq_unlocked_paths(config: dict[str, Any]) -> set[str]:
    """Mirror the previous LSQ defaults while exposing every other fit lock."""
    persisted = config.get("least_squares", {}).get("unlocked_paths")
    if persisted is not None:
        return set(persisted)
    model = str(config.get("model") or "")
    n = min(max(int(config.get("impurity_count", 0) or 0), 0), MAX_IMPURITIES) + 1
    paths: set[str] = set()
    for i in range(n):
        prefix = f"components[{i}]"
        if model in {MODEL_EDM_LANGMUIR, MODEL_TDM_LANGMUIR}:
            paths.update({f"{prefix}.qmax_g_L", f"{prefix}.b_L_g", f"{prefix}.salt_sensitivity_per_M"})
        else:
            paths.update({f"{prefix}.delta_ref", f"{prefix}.As_m_inv"})
            paths.add(f"{prefix}.salt_sensitivity_per_M" if is_hic(config) else f"{prefix}.Z_ref")
            if model == MODEL_TDM_CPA:
                paths.add(f"{prefix}.kkin_star_s")
        if i > 0:
            paths.add(f"{prefix}.uv_response_factor_mAU_L_g")
    return paths


def build_jsl(config: dict[str, Any] | None = None) -> str:
    ensure_batch_runs_csv()
    ensure_lsq_runs_csv()
    cfg = normalize_config(config or load_config(CONFIG_PATH))
    selected = MODEL_LABELS.get(cfg.get("model"), MODEL_LABELS[MODEL_TDM_CPA])
    labels = [MODEL_LABELS[MODEL_EDM_LANGMUIR], MODEL_LABELS[MODEL_TDM_LANGMUIR], MODEL_LABELS[MODEL_EDM_CPA], MODEL_LABELS[MODEL_TDM_CPA]]
    model_items = "{" + ", ".join(_q(x) for x in ([selected] + [x for x in labels if x != selected])) + "}"
    imp_count = min(max(int(cfg.get("impurity_count", 0) or 0), 0), MAX_IMPURITIES)
    c, t, e, cp, conv = cfg["column"], cfg["tdm"], cfg["edm"], cfg["cpa"], cfg.get("conversion", {})
    try:
        batch_count = min(max(int(BATCH_COUNT_FILE.read_text().strip()), 1), MAX_RUNS)
    except Exception:
        batch_count = 1

    component_bodies, component_sends, component_vis = [], [], []
    fit_lock_vars: list[tuple[str, str]] = []
    default_unlocked_paths = _default_lsq_unlocked_paths(cfg)
    for i, row in enumerate(cfg["components"], start=1):
        b, s, v = _component_panel(i, row, fit_lock_vars, default_unlocked_paths)
        component_bodies.append(b); component_sends.extend(s); component_vis.append(v)

    buffer_a_panel = _buffer_panel("bufferA", "Buffer A (0% B)")
    buffer_b_panel = _buffer_panel("bufferB", "Buffer B (100% B)")
    plw_panels = [_step_panel("plw", i, "Post-load wash") for i in range(1, MAX_PLW_STEPS + 1)]
    el_panels = [_step_panel("el", i, "Elution") for i in range(1, MAX_ELUTION_STEPS + 1)]

    ref_bodies, ref_sends, ref_vis = [], [], []
    for k in range(1, MAX_LSQ_REFERENCE_POINTS + 1):
        b, s, v = _lsq_reference_panel(k)
        ref_bodies.append(b); ref_sends.extend(s); ref_vis.append(v)

    sends = [
        f'Python Send(modelBox << Get Selected, Python Name("tdm_ui_model"));',
        f'Python Send(impurityCountBox << Get Selected, Python Name("tdm_ui_impurity_count"));',
        f'Python Send(columnVolume << Get, Python Name("tdm_ui_column_volume_mL"));',
        f'Python Send(columnLength << Get, Python Name("tdm_ui_column_length_mm"));',
        f'Python Send(beadRadius << Get, Python Name("tdm_ui_bead_radius"));',
        f'Python Send(voidFraction << Get, Python Name("tdm_ui_void_fraction"));',
        f'Python Send(particlePorosity << Get, Python Name("tdm_ui_particle_porosity"));',
        f'Python Send(saltDax << Get, Python Name("tdm_ui_salt_Dax"));',
        f'Python Send(edmPorosity << Get, Python Name("tdm_ui_edm_porosity"));',
        f'Python Send(uvToProtein << Get, Python Name("tdm_ui_uv_to_protein_factor"));',
        f'Python Send(condToSalt << Get, Python Name("tdm_ui_cond_to_salt_factor"));',
        f'Python Send(ligandSurface << Get, Python Name("tdm_ui_ligand_surface_density"));',
        f'Python Send(systemAdsorption << Get, Python Name("tdm_ui_system_adsorption_parameter"));',
    ] + component_sends
    sys = cfg["system"]
    system_rows = []
    for field, var, label in (
        ("buffer_dispersion_mL", "bufferDispersion", "Buffer mixer dispersion volume [mL]"),
        ("salt_dispersion_mm2_s", "systemSaltDax", "Buffer / salt axial dispersion [mm²/s]"),
        ("uv_dead_volume_mL", "uvDeadVolume", "System dead volume to UV [mL]"),
        ("conductivity_dead_volume_mL", "condDeadVolume", "System dead volume to conductivity [mL]"),
    ):
        system_rows.append(_nrow(label, var, sys[field], 360, fit_path="system."+field,
                                fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths))
        sends.append(f'Python Send({var} << Get, Python Name("tdm_ui_system_{field}"));')
    for field, var in (("buffer_dispersion_enabled", "bufferDispersionToggle"),
                       ("salt_dispersion_enabled", "saltDispersionToggle"),
                       ("show_percent_B", "percentBToggle"), ("show_conductivity", "conductivityToggle"),
                       ("show_flow_rate", "flowRateToggle")):
        sends.append(f'Python Send({var} << Get Selected, Python Name("tdm_ui_system_{field}"));')
    sends += [
        'Python Send(chromatographyMode << Get Selected, Python Name("tdm_ui_chromatography_mode"));',
        'Python Send(lsqObjective << Get Selected, Python Name("tdm_ui_lsq_objective"));',
        'Python Send(lsqBaselineMode << Get Selected, Python Name("tdm_ui_lsq_baseline_mode"));',
        'Python Send(lsqBaseline << Get, Python Name("tdm_ui_lsq_detector_baseline"));',
        'Python Send(lsqMaxEvaluations << Get, Python Name("tdm_ui_lsq_max_nfev"));',
    ]
    lsq_run_sections = []
    lsq_selection_visibility = []
    lsq_selection_sends = []
    run_choices = "{" + ", ".join(_q(f"Run {r}") for r in range(1, MAX_LSQ_PROFILES + 1)) + "}"
    saved_selection = cfg["least_squares"].get("selected_run_numbers")
    saved_selection = saved_selection if isinstance(saved_selection, list) and 1 <= len(saved_selection) <= 5 else [1]
    initial_runs = []
    for value in saved_selection:
        try:
            run = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= run <= MAX_LSQ_PROFILES and run not in initial_runs:
            initial_runs.append(run)
    initial_runs.extend(run for run in range(1, MAX_LSQ_PROFILES + 1) if run not in initial_runs)
    for slot in range(1, 6):
        selected_run = initial_runs[slot - 1]
        lsq_run_sections.append(f'''lsqRunSlot{slot}Row = Outline Box("Run section {slot}", V List Box(
            H List Box(Text Box("Process recipe", << Set Width(180)), lsqRunSlot{slot} = Combo Box({run_choices}, << Set({selected_run}), << Set Width(140), << Set Function(Function({{this}}, If(!isLoading, ChangeLSQSlot({slot}, RunNumberFromLabel(this << Get Selected))))))),
            lsqRunSummary{slot} = Text Box("", << Set Wrap(720)),
            H List Box(Text Box("Excel / CSV file", << Set Width(180)), lsqCsvPath{slot} = Text Edit Box("", << Set Width(440))),
            H List Box(Text Box("X units", << Set Width(180)), lsqXUnit{slot} = Combo Box({{"AUTO", "CV", "ML", "MIN", "S", "H", "INDEX"}}, << Set(1), << Set Width(150))),
            H List Box(Text Box("X = 0 at", << Set Width(180)), lsqXOrigin{slot} = Combo Box({{"RUN_START", "ELUTION_START"}}, << Set(1), << Set Width(170))),
            H List Box(Button Box("Attach Excel / CSV", AttachLSQCSV({slot})), Button Box("Edit process recipe", SwitchProcessScope(1, RunNumberFromLabel(lsqRunSlot{slot} << Get Selected)); mainTabs << Set(1))),
            Outline Box("Optional columns and weight", << Close(1), V List Box(
                H List Box(Text Box("Excel worksheet", << Set Width(180)), lsqSheet{slot} = Text Edit Box("", << Set Width(380))),
                H List Box(Text Box("X column", << Set Width(180)), lsqXColumn{slot} = Text Edit Box("", << Set Width(380))),
                H List Box(Text Box("UV mAU column", << Set Width(180)), lsqSignalColumn{slot} = Text Edit Box("", << Set Width(380))),
                H List Box(Text Box("Run weight (default 1)", << Set Width(180)), lsqWeight{slot} = Number Edit Box(1, 9, << Set Width(100)))
            ))
        ))''')
        lsq_selection_visibility.append(f'lsqRunSlot{slot}Row << Visibility(If(Num(lsqRunCount << Get Selected) >= {slot}, "Visible", "Collapse"));')
        lsq_selection_sends.append(f'Python Send(RunNumberFromLabel(lsqRunSlot{slot} << Get Selected), Python Name("tdm_ui_lsq_selected_run_{slot}"));')
    # JMP does not preserve a display box as a scriptable object when it is
    # placed in a JSL list. Dispatch to the named controls directly instead.
    lsq_getters = []
    for function_name, prefix, message in (
        ("LSQSelectedRun", "lsqRunSlot", "Get Selected"),
        ("LSQPathText", "lsqCsvPath", "Get Text"),
        ("LSQWeightNumber", "lsqWeight", "Get"),
        ("LSQSheetText", "lsqSheet", "Get Text"),
        ("LSQXColumnText", "lsqXColumn", "Get Text"),
        ("LSQSignalColumnText", "lsqSignalColumn", "Get Text"),
        ("LSQUnitSelected", "lsqXUnit", "Get Selected"),
        ("LSQOriginSelected", "lsqXOrigin", "Get Selected"),
    ):
        branch_lines = []
        for i in range(1, MAX_LSQ_PROFILES + 1):
            expression = f'{prefix}{i} << {message}'
            if function_name == "LSQSelectedRun":
                expression = f'RunNumberFromLabel({expression})'
            branch_lines.append(f'    If(slot == {i}, Return({expression}));')
        branch = "\n".join(branch_lines)
        lsq_getters.append(f'{function_name} = Function({{slot}}, {{}},\n{branch}\n    Return("");\n);')
    lsq_setters = []
    for function_name, prefix, message in (
        ("LSQSetPathText", "lsqCsvPath", "Set Text"),
        ("LSQSetRunChoice", "lsqRunSlot", "Set"),
        ("LSQSetRunSummary", "lsqRunSummary", "Set Text"),
    ):
        branch = "\n".join(
            f'    If(slot == {i}, {prefix}{i} << {message}(value));'
            for i in range(1, MAX_LSQ_PROFILES + 1)
        )
        lsq_setters.append(f'{function_name} = Function({{slot, value}}, {{}},\n{branch}\n);')
    lsq_load_controls = "\n".join(
        f'''    If(slot == {i},
        lsqCsvPath{i} << Set Text(path);
        lsqWeight{i} << Set(w);
        lsqSheet{i} << Set Text(sheet);
        lsqXColumn{i} << Set Text(xcolumn);
        lsqSignalColumn{i} << Set Text(signal);
        lsqXUnit{i} << Set(unitIndex);
        lsqXOrigin{i} << Set(originIndex);
    );'''
        for i in range(1, MAX_LSQ_PROFILES + 1)
    )
    lsq_control_dispatch = "\n".join(lsq_getters + lsq_setters)
    lock_state_parts = []
    for lock_var, parameter_path in fit_lock_vars:
        lock_state_parts.extend([_q(parameter_path + "="), f'{lock_var} << Get Selected', _q(";")])
    if lock_state_parts:
        sends.append('Python Send(Concat(' + ", ".join(lock_state_parts) + '), Python Name("tdm_ui_lsq_parameter_lock_states"));')
    set_all_lock_lines = " ".join(
        f'{lock_var} << Set(If(Uppercase(Char(state)) == "UNLOCKED", 2, 1));'
        for lock_var, _parameter_path in fit_lock_vars
    )

    save_run_lines = [
        'Column(dtRuns, "Run_Name")[currentRun] = runNameBox << Get Text;',
        'Column(dtRuns, "Load_Flow_mL_min")[currentRun] = loadFlow << Get;',
        'Column(dtRuns, "Feed_Concentration_mg_mL")[currentRun] = feedConcentration << Get;',
        'Column(dtRuns, "Load_Amount_Basis")[currentRun] = LoadBasisToken(loadAmountBasis);',
        'Column(dtRuns, "Load_CV")[currentRun] = loadCV << Get;',
        'Column(dtRuns, "Load_Density_mg_mL_resin")[currentRun] = loadDensity << Get;',
        'Column(dtRuns, "Load_Mode")[currentRun] = loadMode << Get Selected;',
        'Column(dtRuns, "Load_Start_Percent_B")[currentRun] = loadStartB << Get;',
        'Column(dtRuns, "Load_End_Percent_B")[currentRun] = loadEndB << Get;',
        'Column(dtRuns, "Load_Chemistry_Control")[currentRun] = loadChemControl << Get Selected;',
        'Column(dtRuns, "Load_Material_Chemistry_Mode")[currentRun] = loadMaterialMode << Get Selected;',
        'Column(dtRuns, "Time_Steps")[currentRun] = timeStepsBox << Get;',
        'Column(dtRuns, "Axial_Positions")[currentRun] = axialPositionsBox << Get;',
        'Column(dtRuns, "Load_Source")[currentRun] = loadSource << Get Selected;',
        'Column(dtRuns, "Load_pH")[currentRun] = loadPH << Get;',
        'Column(dtRuns, "Load_Salt_M")[currentRun] = loadSalt << Get;',
        'Column(dtRuns, "Load_Salt_Input_Mode")[currentRun] = loadSaltMode << Get Selected;',
        'Column(dtRuns, "Load_Conductivity_mS_cm")[currentRun] = loadCond << Get;',
        'Column(dtRuns, "PLW_Count")[currentRun] = Num(plwCountBox << Get Selected);',
        'Column(dtRuns, "Elution_Count")[currentRun] = Num(elutionCountBox << Get Selected);',
    ]
    load_run_lines = [
        'runNameBox << Set Text(Char(Column(dtRuns, "Run_Name")[currentRun]));',
        'loadFlow << Set(Column(dtRuns, "Load_Flow_mL_min")[currentRun]);',
        'feedConcentration << Set(Column(dtRuns, "Feed_Concentration_mg_mL")[currentRun]);',
        'SetLoadBasisCombo(loadAmountBasis, Char(Column(dtRuns, "Load_Amount_Basis")[currentRun]));',
        'loadCV << Set(Column(dtRuns, "Load_CV")[currentRun]);',
        'loadDensity << Set(Column(dtRuns, "Load_Density_mg_mL_resin")[currentRun]);',
        'SetModeCombo(loadMode, Char(Column(dtRuns, "Load_Mode")[currentRun]));',
        'loadStartB << Set(Column(dtRuns, "Load_Start_Percent_B")[currentRun]);',
        'loadEndB << Set(Column(dtRuns, "Load_End_Percent_B")[currentRun]);',
        'If(Uppercase(Char(Column(dtRuns, "Load_Chemistry_Control")[currentRun])) == "ENDPOINT_CHEMISTRY", loadChemControl << Set(2), loadChemControl << Set(1));',
        'If(Uppercase(Char(Column(dtRuns, "Load_Material_Chemistry_Mode")[currentRun])) == "EXPLICIT", loadMaterialMode << Set(2), loadMaterialMode << Set(1));',
        'timeStepsBox << Set(Column(dtRuns, "Time_Steps")[currentRun]);',
        'axialPositionsBox << Set(Column(dtRuns, "Axial_Positions")[currentRun]);',
        'SetSourceCombo(loadSource, Char(Column(dtRuns, "Load_Source")[currentRun]));',
        'loadPH << Set(Column(dtRuns, "Load_pH")[currentRun]);',
        'loadCond << Set(Column(dtRuns, "Load_Conductivity_mS_cm")[currentRun]);',
        'loadSalt << Set(Column(dtRuns, "Load_Salt_M")[currentRun]);',
        'loadSaltMode << Set(If(Uppercase(Char(Column(dtRuns, "Load_Salt_Input_Mode")[currentRun])) == "SALT_M", 2, 1));',
        'plwCountBox << Set(Num(Column(dtRuns, "PLW_Count")[currentRun]) + 1);',
        'elutionCountBox << Set(Num(Column(dtRuns, "Elution_Count")[currentRun]) + 1);',
    ]
    copy_cols = ["Run_Name", "Load_Flow_mL_min", "Feed_Concentration_mg_mL", "Load_Amount_Basis", "Load_CV", "Load_Density_mg_mL_resin", "Load_Mode", "Load_Start_Percent_B", "Load_End_Percent_B", "Load_Chemistry_Control", "Load_Material_Chemistry_Mode", "Time_Steps", "Axial_Positions", "Load_Source", "Load_pH", "Load_Salt_M", "Load_Conductivity_mS_cm", "PLW_Count", "Elution_Count"]

    copy_cols.append("Load_Salt_Input_Mode")
    for bp, vp in (("BufferA", "bufferA"), ("BufferB", "bufferB")):
        fields = {
            "Name": f"{vp}Name << Get Text", "pH": f"{vp}PH << Get",
            "Conductivity_mS_cm": f"{vp}Cond << Get",
            "Ionic_Strength_Source": f"{vp}IonicSource << Get Selected",
            "Salt_M": f"{vp}Salt << Get",
            "Salt_Input_Mode": f"{vp}SaltMode << Get Selected",
        }
        for suffix, expr in fields.items():
            col = f"{bp}_{suffix}"; copy_cols.append(col)
            save_run_lines.append(f'Column(dtRuns, {_q(col)})[currentRun] = {expr};')
        load_run_lines += [
            f'{vp}Name << Set Text(Char(Column(dtRuns, {_q(bp+"_Name")})[currentRun]));',
            f'{vp}PH << Set(Column(dtRuns, {_q(bp+"_pH")})[currentRun]);',
            f'{vp}Cond << Set(Column(dtRuns, {_q(bp+"_Conductivity_mS_cm")})[currentRun]);',
            f'{vp}Salt << Set(Column(dtRuns, {_q(bp+"_Salt_M")})[currentRun]);',
            f'{vp}SaltMode << Set(If(Uppercase(Char(Column(dtRuns, {_q(bp+"_Salt_Input_Mode")})[currentRun])) == "SALT_M", 2, 1));',
            f'If(Uppercase(Char(Column(dtRuns, {_q(bp+"_Ionic_Strength_Source")})[currentRun])) == "ION_COMPOSITION", {vp}IonicSource << Set(2), {vp}IonicSource << Set(1));',
        ]
        for i in range(1, MAX_BUFFER_IONS + 1):
            for suffix, expr in {
                f"Ion{i}_Species": f"{vp}Ion{i}Species << Get Text",
                f"Ion{i}_Concentration_M": f"{vp}Ion{i}Conc << Get",
                f"Ion{i}_Valency": f"{vp}Ion{i}Z << Get",
            }.items():
                col=f"{bp}_{suffix}"; copy_cols.append(col)
                save_run_lines.append(f'Column(dtRuns, {_q(col)})[currentRun] = {expr};')
            load_run_lines += [
                f'{vp}Ion{i}Species << Set Text(Char(Column(dtRuns, {_q(bp+f"_Ion{i}_Species")})[currentRun]));',
                f'{vp}Ion{i}Conc << Set(Column(dtRuns, {_q(bp+f"_Ion{i}_Concentration_M")})[currentRun]);',
                f'{vp}Ion{i}Z << Set(Column(dtRuns, {_q(bp+f"_Ion{i}_Valency")})[currentRun]);',
            ]

    for prefix, max_steps, varprefix in (("PLW", MAX_PLW_STEPS, "plw"), ("Elution", MAX_ELUTION_STEPS, "el")):
        for i in range(1, max_steps + 1):
            v = f"{varprefix}{i}"
            cols = {
                "Mode": f"{v}Mode << Get Selected", "CV": f"{v}CV << Get", "Flow_mL_min": f"{v}Flow << Get",
                "Chemistry_Control": f"{v}ChemControl << Get Selected", "Start_Percent_B": f"{v}StartB << Get", "End_Percent_B": f"{v}EndB << Get",
                "Start_Source": f"{v}StartSource << Get Selected", "Start_pH": f"{v}StartPH << Get",
                "Start_Conductivity_mS_cm": f"{v}StartCond << Get",
                "End_Source": f"{v}EndSource << Get Selected", "End_pH": f"{v}EndPH << Get",
                "End_Conductivity_mS_cm": f"{v}EndCond << Get",
                "Start_Salt_M": f"{v}StartSalt << Get", "End_Salt_M": f"{v}EndSalt << Get",
                "Start_Salt_Input_Mode": f"{v}StartSaltMode << Get Selected",
                "End_Salt_Input_Mode": f"{v}EndSaltMode << Get Selected",
            }
            for suffix, expr in cols.items():
                col = f"{prefix}{i}_{suffix}"
                copy_cols.append(col)
                save_run_lines.append(f'Column(dtRuns, {_q(col)})[currentRun] = {expr};')
            for side in ("Start", "End"):
                load_run_lines += [
                    f'{v}{side}Salt << Set(Column(dtRuns, {_q(prefix+str(i)+"_"+side+"_Salt_M")})[currentRun]);',
                    f'{v}{side}SaltMode << Set(If(Uppercase(Char(Column(dtRuns, {_q(prefix+str(i)+"_"+side+"_Salt_Input_Mode")})[currentRun])) == "SALT_M", 2, 1));',
                ]
            load_run_lines += [
                f'SetModeCombo({v}Mode, Char(Column(dtRuns, {_q(prefix+str(i)+"_Mode")})[currentRun]));',
                f'{v}CV << Set(Column(dtRuns, {_q(prefix+str(i)+"_CV")})[currentRun]);',
                f'{v}Flow << Set(Column(dtRuns, {_q(prefix+str(i)+"_Flow_mL_min")})[currentRun]);',
                f'If(Uppercase(Char(Column(dtRuns, {_q(prefix+str(i)+"_Chemistry_Control")})[currentRun])) == "ENDPOINT_CHEMISTRY", {v}ChemControl << Set(2), {v}ChemControl << Set(1));',
                f'{v}StartB << Set(Column(dtRuns, {_q(prefix+str(i)+"_Start_Percent_B")})[currentRun]);',
                f'{v}EndB << Set(Column(dtRuns, {_q(prefix+str(i)+"_End_Percent_B")})[currentRun]);',
                f'SetSourceCombo({v}StartSource, Char(Column(dtRuns, {_q(prefix+str(i)+"_Start_Source")})[currentRun]));',
                f'{v}StartPH << Set(Column(dtRuns, {_q(prefix+str(i)+"_Start_pH")})[currentRun]);',
                f'{v}StartCond << Set(Column(dtRuns, {_q(prefix+str(i)+"_Start_Conductivity_mS_cm")})[currentRun]);',
                f'SetSourceCombo({v}EndSource, Char(Column(dtRuns, {_q(prefix+str(i)+"_End_Source")})[currentRun]));',
                f'{v}EndPH << Set(Column(dtRuns, {_q(prefix+str(i)+"_End_pH")})[currentRun]);',
                f'{v}EndCond << Set(Column(dtRuns, {_q(prefix+str(i)+"_End_Conductivity_mS_cm")})[currentRun]);',
            ]

    process_vis = []
    for prefix, max_steps, varprefix, countvar in (("PLW", MAX_PLW_STEPS, "plw", "plwCount"), ("Elution", MAX_ELUTION_STEPS, "el", "elutionCount")):
        for i in range(1, max_steps + 1):
            v = f"{varprefix}{i}"
            process_vis.append(f'''{v}Panel << Visibility(If({countvar} >= {i}, "Visible", "Collapse"));
            {v}ChemPanel << Visibility("Visible");
            stepMode = {v}Mode << Get Selected;
            stepControl = {v}ChemControl << Get Selected;
            {v}PercentPanel << Visibility(If(stepControl == "BUFFER_B_PERCENT", "Visible", "Collapse"));
            {v}EndpointPanel << Visibility(If(stepControl == "ENDPOINT_CHEMISTRY", "Visible", "Collapse"));
            {v}StartBRow << Visibility("Visible");
            {v}EndBRow << Visibility("Visible");
            {v}StartChem << Visibility("Visible");
            {v}EndChem << Visibility("Visible");
            {v}StartSaltRow << Visibility(If(Char({v}StartSaltMode << Get Selected) == "SALT_M", "Visible", "Collapse"));
            {v}EndSaltRow << Visibility(If(Char({v}EndSaltMode << Get Selected) == "SALT_M", "Visible", "Collapse"));
            startPct = {v}StartB << Get; endPct = {v}EndB << Get;
            startSalt = If(stepControl == "BUFFER_B_PERCENT", BlendBufferSalt(startPct),
                ResolvedStepSalt({v}StartSource, {v}StartCond, {v}StartSalt, {v}StartSaltMode));
            endSalt = If(stepControl == "BUFFER_B_PERCENT", BlendBufferSalt(endPct),
                ResolvedStepSalt({v}EndSource, {v}EndCond, {v}EndSalt, {v}EndSaltMode));
            activeStart = If(stepMode == "STEP", endSalt, startSalt);
            activeEnd = If(stepMode == "LINEAR", endSalt, activeStart);
            {v}BlendDerived << Set Text("Pump salt used (" || Char(stepMode) || "): " || Char(activeStart) ||
                If(stepMode == "LINEAR", " → " || Char(activeEnd), "") || " M" ||
                If(isLangmuir, "   |   target b_eff: " || Char(TargetLangmuirBEff(activeStart)) ||
                    If(stepMode == "LINEAR", " → " || Char(TargetLangmuirBEff(activeEnd)), "") || " L/g", "") ||
                If(stepControl == "BUFFER_B_PERCENT", "   |   Buffer A/B blend at " || Char(If(stepMode == "STEP", endPct, startPct)) || "% B", "   |   explicit endpoint chemistry"));''')

    req_edm_l = _req_expr(MODEL_EDM_LANGMUIR, 0)
    req_tdm_l = _req_expr(MODEL_TDM_LANGMUIR, 0)
    req_tdm_c0 = _req_expr(MODEL_TDM_CPA, 0)
    req_tdm_c05 = _req_expr(MODEL_TDM_CPA, 0.5)
    req_tdm_c2 = _req_expr(MODEL_TDM_CPA, 2.0)
    req_edm_c0 = _req_expr(MODEL_EDM_CPA, 0)
    req_edm_c05 = _req_expr(MODEL_EDM_CPA, 0.5)
    req_edm_c2 = _req_expr(MODEL_EDM_CPA, 2.0)
    req_edm_hic = _req_expr(MODEL_EDM_CPA, 0, "HIC")
    req_tdm_hic = _req_expr(MODEL_TDM_CPA, 0, "HIC")

    run_labels = "{" + ", ".join(_q(f"Run {i}") for i in range(1, MAX_RUNS + 1)) + "}"
    count_items = "{" + ", ".join(_q(str(i)) for i in range(0, 6)) + "}"
    imp_items = "{" + ", ".join(_q(str(i)) for i in range(0, MAX_IMPURITIES + 1)) + "}"
    ref_items = "{" + ", ".join(_q(str(i)) for i in range(1, MAX_LSQ_REFERENCE_POINTS + 1)) + "}"
    buffer_a_I_expr = "0.5 * (" + " + ".join(
        f"If(Is Missing(bufferAIon{i}Conc << Get) | Is Missing(bufferAIon{i}Z << Get), 0, (bufferAIon{i}Conc << Get) * (bufferAIon{i}Z << Get) * (bufferAIon{i}Z << Get))"
        for i in range(1, MAX_BUFFER_IONS + 1)
    ) + ")"
    buffer_b_I_expr = "0.5 * (" + " + ".join(
        f"If(Is Missing(bufferBIon{i}Conc << Get) | Is Missing(bufferBIon{i}Z << Get), 0, (bufferBIon{i}Conc << Get) * (bufferBIon{i}Z << Get) * (bufferBIon{i}Z << Get))"
        for i in range(1, MAX_BUFFER_IONS + 1)
    ) + ")"
    langmuir_sync_lines = []
    for i in range(1, MAX_IMPURITIES + 2):
        langmuir_sync_lines.append(
            f'''q = c{i}Qmax << Get; b = c{i}B << Get;
            If(Is Missing(q) | Is Missing(b), c{i}H << Set(.),
                If(q <= 0 | b <= 0, c{i}H << Set(.), c{i}H << Set(q*b))
            );'''
        )

    jsl = f'''Names Default To Here(1);
projectDir = {_path(PROJECT_DIR)};
bridgeFile = {_path(BRIDGE_PATH)};
batchRunsFile = {_path(BATCH_RUNS_CSV)};
lsqRunsFile = {_path(LSQ_RUNS_CSV)};
statusFile = {_path(STATUS_PATH)};
resultFile = {_path(LATEST_RESULTS_HTML)};
resultCSV = {_path(LATEST_RESULTS_CSV)};
batchGallery = {_path(BATCH_GALLERY_HTML)};
batchSummary = {_path(BATCH_SUMMARY_CSV)};
lsqApplyFile = {_path(LSQ_APPLY_JSL)};
lsqReport = {_path(LSQ_REPORT_HTML)};
currentRun = 1; copiedRun = 0; isLoading = 0; isLSQSetup = 0;
dtBatchRuns = Open(batchRunsFile, Invisible);
dtLSQRuns = Open(lsqRunsFile, Invisible);
dtRuns = dtBatchRuns; activeRunsFile = batchRunsFile;
// Preserve user-entered paths and worksheet/column names even when a new CSV has empty cells.
Column(dtLSQRuns, "LSQ_Chromatogram_CSV") << Data Type(Character);
Column(dtLSQRuns, "LSQ_Sheet") << Data Type(Character);
Column(dtLSQRuns, "LSQ_X_Column") << Data Type(Character);
Column(dtLSQRuns, "LSQ_Signal_Column") << Data Type(Character);
Column(dtLSQRuns, "LSQ_X_Unit") << Data Type(Character);
Column(dtLSQRuns, "LSQ_X_Origin") << Data Type(Character);

SetModeCombo = Function({{box, txt}}, {{}},
    If(Uppercase(txt) == "LINEAR", box << Set(2), Uppercase(txt) == "STEP", box << Set(3), box << Set(1));
);
SetSourceCombo = Function({{box, txt}}, {{}},
    If(Uppercase(txt) == "BUFFER_A", box << Set(2), Uppercase(txt) == "BUFFER_B", box << Set(3), box << Set(1));
);
SetAllFitLocks = Function({{state}}, {{}},
    {set_all_lock_lines}
);
SetLoadBasisCombo = Function({{box, txt}}, {{}},
    If(Uppercase(txt) == "LOAD_VOLUME" | Uppercase(txt) == "LOAD VOLUME [CV]", box << Set(2), box << Set(1));
);
LoadBasisToken = Function({{box}}, {{label}},
    label = Uppercase(Char(box << Get Selected));
    If(label == "CAPACITY [G/L RESIN]", "CAPACITY", "LOAD_VOLUME");
);
SyncLoadAmount = Function({{}}, {{basis, concentration, capacity, cv, requiredVolume}},
    If(isLoading, Return());
    basis = Uppercase(Char(loadAmountBasis << Get Selected));
    concentration = feedConcentration << Get;
    If(basis == "CAPACITY [G/L RESIN]",
        capacity = loadDensity << Get;
        If(!Is Missing(capacity) & !Is Missing(concentration) & concentration > 0,
            isLoading = 1; loadCV << Set(capacity / concentration); isLoading = 0
        ),
        cv = loadCV << Get;
        If(!Is Missing(cv) & !Is Missing(concentration) & concentration > 0,
            isLoading = 1; loadDensity << Set(cv * concentration); isLoading = 0
        )
    );
    requiredVolume = (loadCV << Get) * (columnVolume << Get);
    loadVolumeSummary << Set Text("Capacity: " || Char(loadDensity << Get) || " g/L resin; required feed volume: " || Char(requiredVolume) || " mL (" || Char(loadCV << Get) || " CV)");
    SaveCurrentRun(); UpdateUI();
);
RunNumberFromLabel = Function({{label}}, {{n}},
    n = Num(Substr(Char(label), 5));
    If(Is Missing(n), Return(1));
    If(n < 1 | n > {MAX_RUNS}, Return(1));
    Return(n);
);
ResolvedBufferSalt = Function({{condBox, saltBox, modeBox}}, {{c, f}},
    If(Char(modeBox << Get Selected) == "SALT_M", Return(saltBox << Get));
    c = condBox << Get; f = condToSalt << Get;
    If(Is Missing(c) | Is Missing(f), ., c * f)
);
BlendBufferPH = Function({{pct}}, {{f, a, b}},
    If(Is Missing(pct), Return(.));
    f = Max(0, Min(100, pct)) / 100; a = bufferAPH << Get; b = bufferBPH << Get;
    If(Is Missing(a) | Is Missing(b), ., (1-f)*a + f*b)
);
BlendBufferSalt = Function({{pct}}, {{f, a, b}},
    If(Is Missing(pct), Return(.));
    f = Max(0, Min(100, pct)) / 100; a = ResolvedBufferSalt(bufferACond, bufferASalt, bufferASaltMode); b = ResolvedBufferSalt(bufferBCond, bufferBSalt, bufferBSaltMode);
    If(Is Missing(a) | Is Missing(b), ., (1-f)*a + f*b)
);
ResolvedStepSalt = Function({{sourceBox, condBox, saltBox, modeBox}}, {{src}},
    src = Uppercase(Char(sourceBox << Get Selected));
    If(src == "BUFFER_A", Return(ResolvedBufferSalt(bufferACond, bufferASalt, bufferASaltMode)));
    If(src == "BUFFER_B", Return(ResolvedBufferSalt(bufferBCond, bufferBSalt, bufferBSaltMode)));
    Return(ResolvedBufferSalt(condBox, saltBox, modeBox));
);
TargetLangmuirBEff = Function({{salt}}, {{b, k}},
    b = c1B << Get; k = c1SaltSensitivity << Get;
    If(Is Missing(salt) | Is Missing(b) | Is Missing(k), Return(.));
    Return(b * Exp(Max(-80, Min(80, k * salt))));
);
ApplyChemSource = Function({{sourceBox, pHBox, condBox}}, {{src}},
    src = Uppercase(sourceBox << Get Selected);
    If(src == "BUFFER_A",
        pHBox << Set(bufferAPH << Get); condBox << Set(bufferACond << Get),
       src == "BUFFER_B",
        pHBox << Set(bufferBPH << Get); condBox << Set(bufferBCond << Get)
    );
);
ResolvedPHFromRow = Function({{r, sourceCol, pHCol}}, {{src}},
    src = Uppercase(Char(Column(dtRuns, sourceCol)[r]));
    If(src == "BUFFER_A", Column(dtRuns, "BufferA_pH")[r],
       src == "BUFFER_B", Column(dtRuns, "BufferB_pH")[r],
       Column(dtRuns, pHCol)[r])
);
ResolvedPHFromPercentBRow = Function({{r, pctCol}}, {{pct, a, b, f}},
    pct = Num(Column(dtRuns, pctCol)[r]); a = Num(Column(dtRuns, "BufferA_pH")[r]); b = Num(Column(dtRuns, "BufferB_pH")[r]);
    If(Is Missing(pct) | Is Missing(a) | Is Missing(b), Return(.));
    f = Max(0, Min(100, pct)) / 100; (1-f)*a + f*b
);

SyncLangmuirDerived = Function({{}}, {{q, b}},
    {''.join(langmuir_sync_lines)}
);

SaveCurrentRun = Function({{}}, {{}},
    If(isLoading, Return());
    {''.join(save_run_lines)}
    Column(dtRuns, "Run_Number")[currentRun] = currentRun;
    dtRuns << Save(activeRunsFile);
    UpdateLSQRunSummary();
);

LoadRun = Function({{n}}, {{}},
    If(isLSQSetup & n > {MAX_LSQ_PROFILES}, n = 1);
    isLoading = 1; currentRun = n;
    {''.join(load_run_lines)}
    runSelector << Set(currentRun);
    If(isLSQSetup, lsqTargetRunSelector << Set(currentRun));
    isLoading = 0; SyncLoadAmount(); UpdateUI(); UpdateLSQRunSummary();
);

{lsq_control_dispatch}
LSQLoadControls = Function({{slot, path, w, sheet, xcolumn, signal, unitIndex, originIndex}}, {{}},
{lsq_load_controls}
);

SaveLSQReference = Function({{slot}}, {{r, w}},
    If(isLoading, Return());
    r = lsqLoadedRuns[slot];
    If(r < 1 | r > {MAX_LSQ_PROFILES}, Return());
    w = LSQWeightNumber(slot);
    Column(dtLSQRuns, "LSQ_Chromatogram_CSV")[r] = Trim(Char(LSQPathText(slot)));
    Column(dtLSQRuns, "LSQ_Weight")[r] = If(Is Missing(w), 1, w);
    Column(dtLSQRuns, "LSQ_Sheet")[r] = Trim(Char(LSQSheetText(slot)));
    Column(dtLSQRuns, "LSQ_X_Column")[r] = Trim(Char(LSQXColumnText(slot)));
    Column(dtLSQRuns, "LSQ_Signal_Column")[r] = Trim(Char(LSQSignalColumnText(slot)));
    Column(dtLSQRuns, "LSQ_X_Unit")[r] = LSQUnitSelected(slot);
    Column(dtLSQRuns, "LSQ_X_Origin")[r] = LSQOriginSelected(slot);
    dtLSQRuns << Save(lsqRunsFile);
);
LoadLSQReference = Function({{slot, r}}, {{w, path, unit, origin, sheet, xcolumn, signal}},
    If(Is Missing(Num(r)), r = slot);
    r = Max(1, Min({MAX_LSQ_PROFILES}, Num(r)));
    path = Column(dtLSQRuns, "LSQ_Chromatogram_CSV")[r];
    w = Num(Column(dtLSQRuns, "LSQ_Weight")[r]);
    lsqLoadedRuns[slot] = r;
    sheet = Column(dtLSQRuns, "LSQ_Sheet")[r];
    xcolumn = Column(dtLSQRuns, "LSQ_X_Column")[r];
    signal = Column(dtLSQRuns, "LSQ_Signal_Column")[r];
    unit = Uppercase(Char(Column(dtLSQRuns, "LSQ_X_Unit")[r]));
    origin = Uppercase(Char(Column(dtLSQRuns, "LSQ_X_Origin")[r]));
    LSQLoadControls(slot,
        If(Is Missing(path), "", Char(path)), If(Is Missing(w), 1, w),
        If(Is Missing(sheet), "", Char(sheet)), If(Is Missing(xcolumn), "", Char(xcolumn)),
        If(Is Missing(signal), "", Char(signal)),
        If(unit == "CV", 2, unit == "ML", 3, unit == "MIN", 4, unit == "S", 5, unit == "H", 6, unit == "INDEX", 7, 1),
        If(origin == "ELUTION_START", 2, 1));
);
SaveAllLSQReferences = Function({{}}, {{i, j, n, ri}},
    n = Num(lsqRunCount << Get Selected);
    For(i = 1, i <= n, i++,
        ri = LSQSelectedRun(i);
        For(j = i + 1, j <= n, j++,
            If(ri == LSQSelectedRun(j),
                lsqStatusText << Set Text("Choose a different LSQ process run in each visible run section.");
                Return(0)
            )
        )
    );
    For(i = 1, i <= n, i++, SaveLSQReference(i));
    UpdateLSQRunSummary();
    Return(1);
);
ChangeLSQSlot = Function({{slot, r}}, {{i, n}},
    n = Num(lsqRunCount << Get Selected);
    For(i = 1, i <= n, i++,
        If(i != slot & r == lsqLoadedRuns[i],
            isLoading = 1; LSQSetRunChoice(slot, lsqLoadedRuns[slot]); isLoading = 0;
            lsqStatusText << Set Text("Choose a different LSQ process run in each visible run section.");
            Return()
        )
    );
    SaveLSQReference(slot);
    LoadLSQReference(slot, r);
    UpdateLSQRunSummary();
);
SwitchProcessScope = Function({{useLSQ, runNum}}, {{}},
    If(!SaveAllLSQReferences(), Return());
    SaveCurrentRun();
    isLSQSetup = If(useLSQ, 1, 0);
    dtRuns = If(isLSQSetup, dtLSQRuns, dtBatchRuns);
    activeRunsFile = If(isLSQSetup, lsqRunsFile, batchRunsFile);
    copiedRun = 0;
    scopeStatus << Set Text(If(isLSQSetup,
        "Editing INDEPENDENT LSQ process setup (1–5). Saving does not modify the normal batch.",
        "Editing NORMAL batch process setup (1–30). LSQ setups remain unchanged."));
    LoadRun(If(isLSQSetup, Max(1, Min({MAX_LSQ_PROFILES}, runNum)), Max(1, Min({MAX_RUNS}, runNum))));
);


LSQRunCellText = Function({{r, fieldName}}, {{v, text}},
    v = Column(dtLSQRuns, fieldName)[r];
    If(Is Missing(v), text = "—", text = Char(v));
    Return(text);
);

LSQBufferIonSummary = Function({{r, prefix}}, {{i, species, conc, valency, summary}},
    summary = "";
    For(i = 1, i <= {MAX_BUFFER_IONS}, i++,
        species = LSQRunCellText(r, prefix || "_Ion" || Char(i) || "_Species");
        conc = LSQRunCellText(r, prefix || "_Ion" || Char(i) || "_Concentration_M");
        valency = LSQRunCellText(r, prefix || "_Ion" || Char(i) || "_Valency");
        If(species != "" & conc != "—" & valency != "—",
            summary = summary || If(summary == "", "", "; ") || species || " " || conc || " M (z=" || valency || ")"
        )
    );
    If(summary == "", "no complete ion rows", summary);
);

LSQRunStageSummary = Function({{r, prefix, i}}, {{base, summary, control}},
    base = prefix || Char(i);
    control = LSQRunCellText(r, base || "_Chemistry_Control");
    summary = base || " " || LSQRunCellText(r, base || "_Mode") || " " ||
        LSQRunCellText(r, base || "_CV") || " CV @ " || LSQRunCellText(r, base || "_Flow_mL_min") ||
        " mL/min; B " || LSQRunCellText(r, base || "_Start_Percent_B") || "→" ||
        LSQRunCellText(r, base || "_End_Percent_B") || "% (" || control || ")";
    If(control != "BUFFER_B_PERCENT",
        summary = summary || "; start " || LSQRunCellText(r, base || "_Start_Source") ||
            " pH " || LSQRunCellText(r, base || "_Start_pH") || ", conductivity " ||
            LSQRunCellText(r, base || "_Start_Conductivity_mS_cm") || " mS/cm; end " ||
            LSQRunCellText(r, base || "_End_Source") || " pH " ||
            LSQRunCellText(r, base || "_End_pH") || ", conductivity " ||
            LSQRunCellText(r, base || "_End_Conductivity_mS_cm") || " mS/cm"
    );
    Return(summary);
);

UpdateLSQRunSummary = Function({{}}, {{slot, r, n, attached, nPLW, nElution, path, summary}},
    n = Num(lsqRunCount << Get Selected); attached = 0;
    For(slot = 1, slot <= {MAX_LSQ_PROFILES}, slot++,
        r = lsqLoadedRuns[slot];
        If(r >= 1 & r <= {MAX_LSQ_PROFILES},
            nPLW = Num(Column(dtLSQRuns, "PLW_Count")[r]); If(Is Missing(nPLW), nPLW = 0);
            nElution = Num(Column(dtLSQRuns, "Elution_Count")[r]); If(Is Missing(nElution), nElution = 0);
            path = Trim(Char(LSQPathText(slot)));
            If(slot <= n & !Is Empty(path), attached++);
            summary = "LSQ Run " || Char(r) || " — " || LSQRunCellText(r, "Run_Name") ||
                ": load " || LSQRunCellText(r, "Load_CV") || " CV, " || Char(nPLW) || " wash(es), " ||
                Char(nElution) || " elution step(s); column " || Char(columnVolume << Get) || " mL." ||
                If(Is Empty(path), " No chromatogram attached.", " Chromatogram entered.");
            LSQSetRunSummary(slot, summary)
        )
    );
    lsqProfilesStatus << Set Text(Char(n) || " run(s) selected; " || Char(attached) || " chromatogram(s) entered. Each selected run needs its own file.");
);

ComputeBatchPHSpan = Function({{}}, {{nRuns, r, i, mn, mx, v, c, stepMode, chemControl, loadMaterialMode, loadMode}},
    If(Contains(Char(modelBox << Get Selected), "CPA") == 0, Return(0));
    nRuns = Floor(batchCountBox << Get); If(Is Missing(nRuns), nRuns = 1); nRuns = Max(1, Min({MAX_RUNS}, nRuns));
    mn = 1e99; mx = -1e99;
    For(r = 1, r <= nRuns, r++,
        loadMaterialMode = Uppercase(Char(Column(dtRuns, "Load_Material_Chemistry_Mode")[r]));
        loadMode = Uppercase(Char(Column(dtRuns, "Load_Mode")[r]));
        chemControl = Uppercase(Char(Column(dtRuns, "Load_Chemistry_Control")[r]));
        If(loadMaterialMode == "EXPLICIT" | chemControl == "ENDPOINT_CHEMISTRY",
            v = ResolvedPHFromRow(r, "Load_Source", "Load_pH"); If(!Is Missing(v), mn = Min(mn, v); mx = Max(mx, v));
        ,
            If(loadMode != "STEP", v = ResolvedPHFromPercentBRow(r, "Load_Start_Percent_B"); If(!Is Missing(v), mn = Min(mn, v); mx = Max(mx, v)));
            If(loadMode != "SET_POINT", v = ResolvedPHFromPercentBRow(r, "Load_End_Percent_B"); If(!Is Missing(v), mn = Min(mn, v); mx = Max(mx, v)));
        );
        c = Num(Column(dtRuns, "PLW_Count")[r]);
        For(i = 1, i <= c, i++,
            stepMode = Uppercase(Char(Column(dtRuns, "PLW" || Char(i) || "_Mode")[r]));
            chemControl = Uppercase(Char(Column(dtRuns, "PLW" || Char(i) || "_Chemistry_Control")[r]));
            If(chemControl == "BUFFER_B_PERCENT",
                If(stepMode != "STEP", v = ResolvedPHFromPercentBRow(r, "PLW" || Char(i) || "_Start_Percent_B"); If(!Is Missing(v), mn = Min(mn, v); mx = Max(mx, v)));
                If(stepMode != "SET_POINT", v = ResolvedPHFromPercentBRow(r, "PLW" || Char(i) || "_End_Percent_B"); If(!Is Missing(v), mn = Min(mn, v); mx = Max(mx, v)));
            ,
                If(stepMode != "STEP", v = ResolvedPHFromRow(r, "PLW" || Char(i) || "_Start_Source", "PLW" || Char(i) || "_Start_pH"); If(!Is Missing(v), mn = Min(mn, v); mx = Max(mx, v)));
                If(stepMode != "SET_POINT", v = ResolvedPHFromRow(r, "PLW" || Char(i) || "_End_Source", "PLW" || Char(i) || "_End_pH"); If(!Is Missing(v), mn = Min(mn, v); mx = Max(mx, v)));
            );
        );
        c = Num(Column(dtRuns, "Elution_Count")[r]);
        For(i = 1, i <= c, i++,
            stepMode = Uppercase(Char(Column(dtRuns, "Elution" || Char(i) || "_Mode")[r]));
            chemControl = Uppercase(Char(Column(dtRuns, "Elution" || Char(i) || "_Chemistry_Control")[r]));
            If(chemControl == "BUFFER_B_PERCENT",
                If(stepMode != "STEP", v = ResolvedPHFromPercentBRow(r, "Elution" || Char(i) || "_Start_Percent_B"); If(!Is Missing(v), mn = Min(mn, v); mx = Max(mx, v)));
                If(stepMode != "SET_POINT", v = ResolvedPHFromPercentBRow(r, "Elution" || Char(i) || "_End_Percent_B"); If(!Is Missing(v), mn = Min(mn, v); mx = Max(mx, v)));
            ,
                If(stepMode != "STEP", v = ResolvedPHFromRow(r, "Elution" || Char(i) || "_Start_Source", "Elution" || Char(i) || "_Start_pH"); If(!Is Missing(v), mn = Min(mn, v); mx = Max(mx, v)));
                If(stepMode != "SET_POINT", v = ResolvedPHFromRow(r, "Elution" || Char(i) || "_End_Source", "Elution" || Char(i) || "_End_pH"); If(!Is Missing(v), mn = Min(mn, v); mx = Max(mx, v)));
            );
        );
    );
    If(mn > mx, 0, mx - mn)
);

UpdateUI = Function({{}}, {{modelLabel, impurityCount, isTDM, isEDM, isCPA, isLangmuir, isHIC, plwCount, elutionCount, stepMode, stepControl, startPct, endPct, startSalt, endSalt, activeStart, activeEnd, batchPHSpan, reqText, lsqUsesComposition, lsqRefCount, loadSaltValue, loadSourceLabel, stationaryFraction, nominalCapacity, targetDose}},
    modelLabel = Char(modelBox << Get Selected);
    loadVolumeSummary << Set Text("Capacity: " || Char(loadDensity << Get) || " g/L resin; required feed volume: " || Char((loadCV << Get) * (columnVolume << Get)) || " mL (" || Char(loadCV << Get) || " CV)");
    SyncLangmuirDerived();
    impurityCount = Num(impurityCountBox << Get Selected); If(Is Missing(impurityCount), impurityCount = 0);
    isTDM = Contains(modelLabel, "TDM") > 0; isEDM = Contains(modelLabel, "EDM") > 0;
    isCPA = Contains(modelLabel, "CPA") > 0; isLangmuir = Contains(modelLabel, "Competitive Langmuir") > 0;
    isHIC = Uppercase(Char(chromatographyMode << Get Selected)) == "HIC";
    If(isLangmuir & !Is Missing(c1Qmax << Get) & !Is Missing(c1Mass << Get) & !Is Missing(loadDensity << Get) &
       If(isEDM, !Is Missing(edmPorosity << Get), !Is Missing(voidFraction << Get) & !Is Missing(particlePorosity << Get)),
        stationaryFraction = If(isEDM, 1 - (edmPorosity << Get), (1 - (voidFraction << Get)) * (1 - (particlePorosity << Get)));
        nominalCapacity = (c1Qmax << Get) * stationaryFraction;
        targetDose = (loadDensity << Get) * (c1Mass << Get) / 100;
        loadSaturationStatus << Set Text("Target load " || Char(Round(targetDose, 0.0001)) ||
            " g/L packed bed; nominal stationary-phase saturation " || Char(Round(nominalCapacity, 0.0001)) ||
            " g/L packed bed (qmax × solid bed fraction)." ||
            If(targetDose > nominalCapacity, " Above saturation: protein can break through during load/wash even at high salt.", ""));
    ,
        loadSaturationStatus << Set Text("")
    );
    {''.join(lsq_selection_visibility)}
    plwCount = Num(plwCountBox << Get Selected); If(Is Missing(plwCount), plwCount = 0);
    elutionCount = Num(elutionCountBox << Get Selected); If(Is Missing(elutionCount), elutionCount = 0);
    batchPHSpan = ComputeBatchPHSpan();
    bufferSection << Visibility("Visible");
    bufferAIonicSourceRow << Visibility("Visible");
    bufferBIonicSourceRow << Visibility("Visible");
    bufferAIonSection << Visibility(If(Uppercase(Char(bufferAIonicSource << Get Selected)) == "ION_COMPOSITION", "Visible", "Collapse"));
    bufferBIonSection << Visibility(If(Uppercase(Char(bufferBIonicSource << Get Selected)) == "ION_COMPOSITION", "Visible", "Collapse"));
    bufferASaltRow << Visibility(If(Char(bufferASaltMode << Get Selected) == "SALT_M", "Visible", "Collapse"));
    bufferBSaltRow << Visibility(If(Char(bufferBSaltMode << Get Selected) == "SALT_M", "Visible", "Collapse"));
    loadSaltRow << Visibility(If(Char(loadSaltMode << Get Selected) == "SALT_M", "Visible", "Collapse"));
    bufferADerived << Set Text("Resolved Buffer A salt [M]: " || Char(ResolvedBufferSalt(bufferACond, bufferASalt, bufferASaltMode)) || "   |   ionic strength [M]: " || If(Uppercase(Char(bufferAIonicSource << Get Selected)) == "ION_COMPOSITION", Char({buffer_a_I_expr}), Char(ResolvedBufferSalt(bufferACond, bufferASalt, bufferASaltMode))));
    bufferBDerived << Set Text("Resolved Buffer B salt [M]: " || Char(ResolvedBufferSalt(bufferBCond, bufferBSalt, bufferBSaltMode)) || "   |   ionic strength [M]: " || If(Uppercase(Char(bufferBIonicSource << Get Selected)) == "ION_COMPOSITION", Char({buffer_b_I_expr}), Char(ResolvedBufferSalt(bufferBCond, bufferBSalt, bufferBSaltMode))));
    loadSourceLabel = Uppercase(Char(loadSource << Get Selected));
    loadSaltValue = If(Uppercase(Char(loadMaterialMode << Get Selected)) == "EXPLICIT" | Uppercase(Char(loadChemControl << Get Selected)) == "ENDPOINT_CHEMISTRY",
        If(loadSourceLabel == "BUFFER_A", ResolvedBufferSalt(bufferACond, bufferASalt, bufferASaltMode),
           loadSourceLabel == "BUFFER_B", ResolvedBufferSalt(bufferBCond, bufferBSalt, bufferBSaltMode),
           ResolvedBufferSalt(loadCond, loadSalt, loadSaltMode)),
        BlendBufferSalt(loadStartB << Get));
    saltFactorStatus << Set Text("Current run: Buffer A " || Char(ResolvedBufferSalt(bufferACond, bufferASalt, bufferASaltMode)) ||
        " M (" || Char(bufferASaltMode << Get Selected) || "); Buffer B " ||
        Char(ResolvedBufferSalt(bufferBCond, bufferBSalt, bufferBSaltMode)) || " M (" ||
        Char(bufferBSaltMode << Get Selected) || "); load start " || Char(loadSaltValue) || " M.");
    loadChemControlRow << Visibility("Visible");
    loadChemPanel << Visibility(If((isCPA | isLangmuir) & (Uppercase(Char(loadMaterialMode << Get Selected)) == "EXPLICIT" | Uppercase(Char(loadChemControl << Get Selected)) == "ENDPOINT_CHEMISTRY"), "Visible", "Collapse"));
    loadPH << Visibility(If(isCPA, "Visible", "Collapse")); loadPHLabel << Visibility(If(isCPA, "Visible", "Collapse"));
    loadCond << Visibility(If(isCPA | isLangmuir, "Visible", "Collapse")); loadCondLabel << Visibility(If(isCPA | isLangmuir, "Visible", "Collapse"));
    tdmPanel << Visibility(If(isTDM, "Visible", "Collapse")); edmPanel << Visibility(If(isEDM, "Visible", "Collapse"));
    cpaSystemPanel << Visibility(If(isCPA & !isHIC, "Visible", "Collapse"));
    condConversionRow << Visibility(If(isCPA | isLangmuir, "Visible", "Collapse"));
    tdmSaltRow << Visibility("Collapse");
    {''.join(component_vis)}
    {''.join(process_vis)}
    speciesStatus << Set Text("Showing target protein + " || Char(impurityCount) || If(impurityCount == 1, " impurity", " impurities") || ". Only these species are sent to the solver.");
    reqText = "";
    If(modelLabel == {_q(MODEL_LABELS[MODEL_EDM_LANGMUIR])}, reqText = {req_edm_l});
    If(modelLabel == {_q(MODEL_LABELS[MODEL_TDM_LANGMUIR])}, reqText = {req_tdm_l});
    If(modelLabel == {_q(MODEL_LABELS[MODEL_TDM_CPA])}, If(batchPHSpan > 1, reqText = {req_tdm_c2}, If(batchPHSpan > 1e-9, reqText = {req_tdm_c05}, reqText = {req_tdm_c0})));
    If(modelLabel == {_q(MODEL_LABELS[MODEL_EDM_CPA])}, If(batchPHSpan > 1, reqText = {req_edm_c2}, If(batchPHSpan > 1e-9, reqText = {req_edm_c05}, reqText = {req_edm_c0})));
    If(isHIC & modelLabel == {_q(MODEL_LABELS[MODEL_EDM_CPA])}, reqText = {req_edm_hic});
    If(isHIC & modelLabel == {_q(MODEL_LABELS[MODEL_TDM_CPA])}, reqText = {req_tdm_hic});
    requiredList << Set Text(reqText);
    batchPHStatus << Set Text(If(isCPA, "Selected-batch pH span: " || Char(Round(batchPHSpan, 6)), ""));
    lsqUsesComposition = Contains(Char(lsqModeBox << Get Selected), "mass%") > 0;
    lsqTargetRunRow << Visibility(If(lsqUsesComposition, "Visible", "Collapse"));
    lsqCompositionPanel << Visibility(If(lsqUsesComposition, "Visible", "Collapse"));
    lsqRefCount = Num(lsqReferenceCountBox << Get Selected); If(Is Missing(lsqRefCount), lsqRefCount = 1);
    {''.join(ref_vis)}
);

SendSharedModel = Function({{}}, {{}},
    {''.join(sends)}
);

SendAndRun = Function({{actionText}}, {{rc, statusValue}},
    If(isLSQSetup & (actionText == "RUN_SELECTED" | actionText == "RUN_BATCH" | actionText == "CHECK_RESULT_CURRENT"),
        statusText << Set Text("Switch to Normal batch setup before running a production chromatogram."); Return()
    );
    If(actionText == "FIT_LSQ" | actionText == "ATTACH_LSQ_CSV", If(!SaveAllLSQReferences(), Return()));
    SaveCurrentRun();
    UpdateLSQRunSummary(); SendSharedModel();
    Python Send(actionText, Python Name("tdm_ui_action"));
    Python Send(If(isLSQSetup, 1, currentRun), Python Name("tdm_ui_run_number"));
    Python Send(Floor(batchCountBox << Get), Python Name("tdm_ui_batch_count"));
    Python Send(lsqModeBox << Get Selected, Python Name("tdm_ui_lsq_mode"));
    Python Send(lsqRunCount << Get Selected, Python Name("tdm_ui_lsq_run_count"));
    {''.join(lsq_selection_sends)}
    Python Send(lsqReferenceCountBox << Get Selected, Python Name("tdm_ui_lsq_reference_count"));
    Python Send(LSQPathText(lsqAttachSlot), Python Name("tdm_ui_lsq_csv_path"));
    Python Send(If(actionText == "ATTACH_LSQ_CSV", lsqLoadedRuns[lsqAttachSlot], RunNumberFromLabel(lsqTargetRunSelector << Get Selected)), Python Name("tdm_ui_lsq_target_run"));
    Python Send(projectDir, Python Name("tdm_project_dir"));
    Python Send(projectDir, Python Name("tdm_project_dir_posix"));
    {''.join(ref_sends)}
    If(actionText == "CHECK_RESULT_CURRENT", Save Text File(statusFile, "CHECK_RESULT_CURRENT pending"));
    rc = Python Submit File(bridgeFile);
    If(File Exists(statusFile), statusValue = Load Text File(statusFile), statusValue = "No status file was produced. Check View > Log.");
    statusText << Set Text(statusValue); lsqStatusText << Set Text(statusValue);
    If(actionText == "FIT_LSQ" & Contains(statusValue, "Accepted/applied to shared batch parameters: True") > 0 & File Exists(lsqApplyFile), Eval(Parse(Load Text File(lsqApplyFile))));
    UpdateUI();
);

OpenFitReport = Function({{}}, {{}},
    If(File Exists(lsqReport), Web(lsqReport),
        lsqStatusText << Set Text("No least-squares report is available yet. Attach a chromatogram and run refinement first.")
    );
);

OpenBatchGallery = Function({{}}, {{}},
    If(File Exists(batchGallery), Web(batchGallery),
        statusText << Set Text("No batch gallery is available yet. Run the saved batch first.")
    );
);

OpenBatchSummary = Function({{}}, {{}},
    If(File Exists(batchSummary), Open(batchSummary),
        statusText << Set Text("No batch summary is available yet. Run the saved batch first.")
    );
);

OpenLSQReport = Function({{}}, {{}},
    If(File Exists(lsqReport), Web(lsqReport),
        statusText << Set Text("No least-squares report is available yet. Attach a chromatogram and run refinement first.")
    );
);

OpenLSQHistory = Function({{}}, {{}},
    If(File Exists({_path(LSQ_HISTORY_CSV)}), Open({_path(LSQ_HISTORY_CSV)}),
        statusText << Set Text("No least-squares objective history is available yet. Run a refinement first.")
    );
);

OpenCurrentResult = Function({{openData}}, {{statusValue}},
    SendAndRun("CHECK_RESULT_CURRENT");
    If(File Exists(statusFile), statusValue = Load Text File(statusFile), statusValue = "No freshness status was produced. Check View > Log.");
    If(Contains(statusValue, "RESULT_CURRENT:") > 0,
        If(openData == 1,
            If(File Exists(resultCSV), Open(resultCSV), statusText << Set Text("Current result data file is missing.")),
            If(File Exists(resultFile), Web(resultFile), statusText << Set Text("Current chromatogram file is missing."))
        ),
        statusText << Set Text(statusValue)
    );
);

AttachLSQCSV = Function({{slot}}, {{csvPath}},
    lsqAttachSlot = slot;
    csvPath = Trim(Char(LSQPathText(slot)));
    If(Is Empty(csvPath) | !File Exists(csvPath),
        csvPath = Pick File("Select raw chromatogram CSV/Excel", "", {{"Chromatogram files|csv;xlsx;xlsm", "All files|*"}}, 1, 0, "");
        If(Is Empty(csvPath), Return());
        LSQSetPathText(slot, csvPath);
    );
    SendAndRun("ATTACH_LSQ_CSV");
);

CopyRun = Function({{}}, {{}}, SaveCurrentRun(); copiedRun = currentRun; copyStatus << Set Text("Copied Run " || Char(currentRun)););
PasteRun = Function({{}}, {{i, nm}},
    If(copiedRun <= 0, copyStatus << Set Text("Copy a run first."),
        SaveCurrentRun();
        runCopyColumns = {{{', '.join(_q(x) for x in copy_cols)}}};
        For(i = 1, i <= N Items(runCopyColumns), i++,
            nm = runCopyColumns[i]; Column(dtRuns, nm)[currentRun] = Column(dtRuns, nm)[copiedRun];
        );
        Column(dtRuns, "Run_Number")[currentRun] = currentRun;
        Column(dtRuns, "Run_Name")[currentRun] = "Run " || Char(currentRun);
        dtRuns << Save(activeRunsFile); LoadRun(currentRun); copyStatus << Set Text("Pasted operating conditions into Run " || Char(currentRun));
    );
);

modelWindow = New Window("TDM 22 — Classic Process UI / Direct Mechanistic Inputs — {RESULT_GUARD_BUILD}",
    V List Box(
        V List Box(
            H List Box(
                Text Box("Runs in batch", << Set Width(110)), batchCountBox = Number Edit Box({batch_count}, 8, << Set Format(Format(Fixed, 10, 0)), << Set Width(80), << Set Function(Function({{this}}, If(this << Get < 1, this << Set(1)); If(this << Get > {MAX_RUNS}, this << Set({MAX_RUNS})); UpdateUI()))),
                Spacer Box(Size(20,1)), Text Box("Current run", << Set Width(100)), runSelector = Combo Box({run_labels}, << Set Width(130), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); LoadRun(RunNumberFromLabel(this << Get Selected)))))),
                Button Box("Copy", CopyRun()), Button Box("Paste", PasteRun())
            ),
            H List Box(
                Button Box("Save model + run", SendAndRun("SAVE_MODEL")),
                Button Box("Run selected", SendAndRun("RUN_SELECTED")),
                Button Box("Run saved batch", SendAndRun("RUN_BATCH"))
            )
        ),
        copyStatus = Text Box("", << Set Wrap(780)),
        H List Box(
            Button Box("Edit normal batch setup", SwitchProcessScope(0, 1); mainTabs << Set(1)),
            Button Box("Edit LSQ process setup", SwitchProcessScope(1, lsqLoadedRuns[1]); mainTabs << Set(1)),
            Button Box("Save current process recipe", SaveCurrentRun(); copyStatus << Set Text("Saved recipe in " || If(isLSQSetup, "independent LSQ runs", "normal batch runs")))
        ),
        scopeStatus = Text Box("Editing NORMAL batch process setup. The independent LSQ runs are kept separately.", << Set Wrap(780)),
        mainTabs = Tab Box(
            "Process Setup",
            V Scroll Box(Size(850, 690), V List Box(
                Text Box("This restores the original run-oriented workflow. Operating conditions are entered directly; there are no process templates. Up to 30 saved runs share the selected mechanistic parameter set.", << Set Wrap(760), << Set Font Style("Italic")),
                bufferSection = V List Box(
                    Text Box("Buffer A / Buffer B definitions (current run)", << Set Font Style("Bold")),
                    Text Box("Buffer A is the 0% B endpoint and Buffer B is the 100% B endpoint. Select conductivity or direct salt molarity for each buffer. HIC binding uses local salt molarity; CPA ion exchange uses ionic strength. Every model reads the selected run's load, PLW and elution chemistry. Result receipts show the resolved inputs reaching the solver.", << Set Wrap(720)),
                    {buffer_a_panel},
                    {buffer_b_panel}
                ),
                Spacer Box(Size(1,8)),
                Outline Box("Run & load",
                    V List Box(
                        {_trow("Run name", "runNameBox", "")},
        {_nrow("Load flow [mL/min]", "loadFlow", None, 290, "If(!isLoading, SaveCurrentRun(); UpdateUI())")},
                        {_nrow("Feed concentration [mg/mL]", "feedConcentration", None, 290, "If(!isLoading, SyncLoadAmount())")},
                        H List Box(Text Box("Load amount specified by", << Set Width(290)), loadAmountBasis = Combo Box({{"Capacity [g/L resin]", "Load volume [CV]"}}, << Set Width(180), << Set Function(Function({{this}}, If(!isLoading, SyncLoadAmount()))))),
                        {_nrow("Load capacity [g/L resin]", "loadDensity", 10.0, 290, "If(!isLoading, SyncLoadAmount())")},
                        {_nrow("Required load volume [CV]", "loadCV", 10.0, 290, "If(!isLoading, SyncLoadAmount())")},
                        loadVolumeSummary = Text Box("Capacity and total feed volume are calculated from the selected load basis.", << Set Wrap(680)),
                        loadSaturationStatus = Text Box("", << Set Wrap(680)),
                        H List Box(Text Box("Load mode", << Set Width(290)), loadMode = Combo Box({{"SET_POINT", "LINEAR", "STEP"}}, << Set Width(125), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); UpdateUI()))))),
                        H List Box(Text Box("Load start / set-point Buffer B [%]", << Set Width(290)), loadStartB = Number Edit Box(0, 9, << Set Width(92), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); UpdateUI()))))),
                        H List Box(Text Box("Load end Buffer B [%]", << Set Width(290)), loadEndB = Number Edit Box(0, 9, << Set Width(92), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); UpdateUI()))))),
                        loadChemControlRow = H List Box(Text Box("Load chemistry control", << Set Width(290)), loadChemControl = Combo Box({{"BUFFER_B_PERCENT", "ENDPOINT_CHEMISTRY"}}, << Set Width(190), << Set Function(Function({{this}}, If(!isLoading, If(Uppercase(Char(loadChemControl << Get Selected)) == "ENDPOINT_CHEMISTRY", loadMaterialMode << Set(2)); SaveCurrentRun(); UpdateUI()))))),
                        H List Box(Text Box("Load material chemistry", << Set Width(290)), loadMaterialMode = Combo Box({{"BUFFER_RECIPE", "EXPLICIT"}}, << Set Width(190), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); UpdateUI()))))),
                        Text Box("Select load capacity [g/L resin] or load volume [CV] as the input. Capacity is numerically equal to mg/mL resin; with feed concentration [mg/mL], CV = capacity ÷ concentration. The model applies the resulting total feed volume = CV × column volume. BUFFER_RECIPE resolves chemistry from the load Buffer B % program. EXPLICIT uses the load material source, pH, salt input and conductivity fields below, so feed chemistry reaches the solver even when the load %B display is retained.", << Set Wrap(700), << Set Font Style("Italic")),
                        {_nrow("Number of time steps", "timeStepsBox", 700)},
                        {_nrow("Number of axial positions", "axialPositionsBox", 20)},
                        Text Box("Numerical resolution is stored per run. Time steps set the reported time grid and cap the adaptive integration step; axial positions set the finite-volume cells along the column. For comparing subtle peak shifts from b or k_s changes, use at least 300 reported time steps; 50 points may hide a shift between adjacent samples.", << Set Wrap(700), << Set Font Style("Italic")),
                        loadChemPanel = V List Box(
                            H List Box(Text Box("Load chemistry source", << Set Width(220)), loadSource = Combo Box({{"DIRECT", "BUFFER_A", "BUFFER_B"}}, << Set Width(125), << Set Function(Function({{this}}, If(!isLoading, loadMaterialMode << Set(2); ApplyChemSource(loadSource, loadPH, loadCond); SaveCurrentRun(); UpdateUI()))))),
                            loadPHRow = H List Box(loadPHLabel = Text Box("Process / load pH", << Set Width(220)), loadPH = Number Edit Box(6, 9, << Set Width(92), << Set Function(Function({{this}}, If(!isLoading, loadMaterialMode << Set(2); loadSource << Set(1); SaveCurrentRun(); UpdateUI()))))),
                            {_salt_input_controls("load")},
                            loadCondRow = H List Box(loadCondLabel = Text Box("Load conductivity [mS/cm]", << Set Width(220)), loadCond = Number Edit Box(5, 9, << Set Width(92), << Set Function(Function({{this}}, If(!isLoading, loadMaterialMode << Set(2); loadSource << Set(1); SaveCurrentRun(); UpdateUI())))))
                        )
                    )
                ),
                Spacer Box(Size(1,8)),
                Outline Box("Standard process sequence",
                    V List Box(
                        H List Box(Text Box("Post-load washes [0-5]", << Set Width(220)), plwCountBox = Combo Box({count_items}, << Set Width(75), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); UpdateUI()))))),
                        H List Box(Text Box("Elution steps [0-5]", << Set Width(220)), elutionCountBox = Combo Box({count_items}, << Set Width(75), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); UpdateUI()))))),
                        Text Box("Load and every PLW/Elution step store Start and End %B. Buffer A = 0% B and Buffer B = 100% B. SET_POINT holds Start %B, LINEAR interpolates Start to End across the CV, and STEP switches to End %B. Both endpoints remain visible and saved in every mode. HIC Competitive Langmuir uses local salt concentration through b_eff = b × exp(k_s × salt); CPA uses pH/salt through its CPA equations.", << Set Wrap(740))
                    )
                ),
                Spacer Box(Size(1,8)),
                plwSection = V List Box(Text Box("Post-load washes", << Set Font Style("Bold")), {', '.join(plw_panels)}),
                Spacer Box(Size(1,8)),
                elutionSection = V List Box(Text Box("Elution", << Set Font Style("Bold")), {', '.join(el_panels)})
            )),
            "Model Parameters",
            V Scroll Box(Size(850, 690), V List Box(
                Outline Box("Model selection", V List Box(
                    H List Box(Text Box("Model combination", << Set Width(275)), modelBox = Combo Box({model_items}, << Set Width(235), << Set Function(Function({{this}}, UpdateUI())))),
                    H List Box(Text Box("Number of impurities [0-5]", << Set Width(275)), impurityCountBox = Combo Box({imp_items}, << Set Width(75), << Set Function(Function({{this}}, UpdateUI())))),
                    speciesStatus = Text Box("", << Set Wrap(760)), batchPHStatus = Text Box("", << Set Wrap(760))
                )),
                Outline Box("Required parameters for the selected combination", requiredList = Text Box("", << Set Wrap(760), << Set Height(250))),
                Outline Box("Column geometry — direct inputs", V List Box(
                    {_nrow("Column volume V_col [mL]", "columnVolume", c.get("volume_mL"), 360, "If(!isLoading, SyncLoadAmount())")},
                    {_nrow("Column length L_C [mm]", "columnLength", c.get("length_mm"), 360)}
                )),
                Outline Box("HIC / ion exchange and system response", V List Box(
                    H List Box(Text Box("Chromatography mode", << Set Width(360)), chromatographyMode = Combo Box({{"HIC", "ION_EXCHANGE"}}, << Set({1 if is_hic(cfg) else 2}), << Set Function(Function({{this}}, UpdateUI())))),
                    Text Box("For 100% B loading followed by a 0% B gradient, Buffer B must be the high-salt endpoint. Native CPA is an ion-exchange model. CPA in HIC mode uses empirical exp(k_s × salt) affinity with CPA surface competition; calibrate it for your protein and resin.", << Set Wrap(740)),
                    {', '.join(system_rows)},
                    H List Box(Text Box("Buffer mixer dispersion", << Set Width(360)), bufferDispersionToggle = Combo Box({{"On", "Off"}}, << Set({1 if sys['buffer_dispersion_enabled'] else 2}))),
                    H List Box(Text Box("Salt axial dispersion", << Set Width(360)), saltDispersionToggle = Combo Box({{"On", "Off"}}, << Set({1 if sys['salt_dispersion_enabled'] else 2}))),
                    H List Box(Text Box("Show %B overlay", << Set Width(360)), percentBToggle = Combo Box({{"On", "Off"}}, << Set({1 if sys['show_percent_B'] else 2}))),
                    H List Box(Text Box("Show conductivity overlay", << Set Width(360)), conductivityToggle = Combo Box({{"On", "Off"}}, << Set({1 if sys['show_conductivity'] else 2}))),
                    H List Box(Text Box("Show flow-rate overlay", << Set Width(360)), flowRateToggle = Combo Box({{"On", "Off"}}, << Set({1 if sys['show_flow_rate'] else 2})))
                )),
                signalConversionPanel = Outline Box("Signal conversion parameters — direct inputs", V List Box(
                    {_nrow("Target UV → protein response factor [mAU·L/g]", "uvToProtein", conv.get("uv_to_protein_mAU_L_g"), 360,
                           fit_path="conversion.uv_to_protein_mAU_L_g", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)},
                    condConversionRow = {_nrow("Conductivity → salt factor [M per (mS/cm)]", "condToSalt", conv.get("conductivity_to_salt_M_per_mS_cm"), 360,
                           "If(!isLoading, UpdateUI())", fit_path="conversion.conductivity_to_salt_M_per_mS_cm",
                           fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)},
                    saltFactorStatus = Text Box("", << Set Wrap(720)),
                    Text Box("Only CONDUCTIVITY salt inputs use this factor for binding. SALT_M uses its entered molarity. In CONDUCTIVITY mode, a 0 mS/cm endpoint stays at 0 M for any factor; an elution to that endpoint can release protein. HIC inputs above 10 M are rejected before solving.", << Set Wrap(720), << Set Font Style("Italic"))
                )),
                tdmPanel = Outline Box("TDM parameters", V List Box(
                    {_nrow("Bead radius r_p [µm]", "beadRadius", t.get("bead_radius_um"), 360, fit_path="tdm.bead_radius_um", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)},
                    {_nrow("Void fraction ε_v [-]", "voidFraction", t.get("void_fraction"), 360, "If(!isLoading, UpdateUI())", fit_path="tdm.void_fraction", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)},
                    {_nrow("Particle porosity ε_p [-]", "particlePorosity", t.get("particle_porosity"), 360, "If(!isLoading, UpdateUI())", fit_path="tdm.particle_porosity", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)},
                    tdmSaltRow = {_nrow("Salt axial dispersion D_ax,salt [mm²/s]", "saltDax", t.get("salt_axial_dispersion_mm2_s"), 360, fit_path="tdm.salt_axial_dispersion_mm2_s", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)}
                )),
                edmPanel = Outline Box("EDM parameters", V List Box({_nrow("Total bed porosity ε [-]", "edmPorosity", e.get("total_bed_porosity"), 360, "If(!isLoading, UpdateUI())", fit_path="edm.total_bed_porosity", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)})),
                cpaSystemPanel = Outline Box("CPA resin/system parameters", V List Box(
                    {_nrow("Ligand surface density Γ_L [µmol/m²]", "ligandSurface", cp.get("ligand_surface_density_umol_m2"), 360, fit_path="cpa.ligand_surface_density_umol_m2", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)},
                    {_nrow("System-specific adsorption parameter [-]", "systemAdsorption", cp.get("system_specific_adsorption_parameter"), 360, fit_path="cpa.system_specific_adsorption_parameter", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)}
                )),
                Text Box("Every visible numeric box is sent to the solver. For least-squares fitting, each mechanistic field has an adjacent Locked/Unlocked control. Column geometry, feed composition, and each profile's process recipe remain fixed experimental inputs.", << Set Wrap(760), << Set Font Style("Italic")),
                {', '.join(component_bodies)}
            )),
            "Least-Squares Refinement",
            V Scroll Box(Size(850, 690), V List Box(
                H List Box(Text Box("Number of runs to fit [1–5]", << Set Width(240)), lsqRunCount = Combo Box({{"1", "2", "3", "4", "5"}}, << Set({len(saved_selection)}), << Set Width(100), << Set Function(Function({{this}}, UpdateUI(); UpdateLSQRunSummary())))),
                lsqProfilesStatus = Text Box("", << Set Wrap(740)),
                Text Box("Attach one measured CV/mL versus UV mAU chromatogram in each run section. The run count immediately shows or hides sections; each section keeps its own file, X units, X=0 origin and process recipe.", << Set Wrap(760)),
                {', '.join(lsq_run_sections)},
                Outline Box("Fit settings", V List Box(
                    H List Box(Text Box("Refinement mode", << Set Width(200)), lsqModeBox = Combo Box({{"Normal least-squares: chromatogram only", "Least-squares: chromatogram + species mass% at CV"}}, << Set Width(340), << Set Function(Function({{this}}, UpdateUI())))),
                    lsqTargetRunRow = H List Box(Text Box("Species reference run", << Set Width(200)), lsqTargetRunSelector = Combo Box({run_choices}, << Set Width(140))),
                    H List Box(Button Box("Lock all parameters", SetAllFitLocks("Locked")), Button Box("Unlock all parameters", SetAllFitLocks("Unlocked"))),
                    Text Box("Leave objective and weights alone for ordinary pointwise least-squares. The optimizer keeps the best improvement and stops when the objective, gradient, or parameter step converges; the evaluation limit is a safety cap. Set individual parameter locks on Model Parameters.", << Set Wrap(740)),
                    Outline Box("Optional objective and stopping settings", << Close(1), V List Box(
                        H List Box(Text Box("Objective", << Set Width(240)), lsqObjective = Combo Box({{"RAW_SSE", "NORMALIZED_MSE", "WEIGHTED_RMSE"}}, << Set({1 if cfg['least_squares']['objective']=='RAW_SSE' else 2 if cfg['least_squares']['objective']=='NORMALIZED_MSE' else 3}), << Set Width(220))),
                        H List Box(Text Box("Baseline policy", << Set Width(240)), lsqBaselineMode = Combo Box({{"NONE", "INITIAL_MEDIAN"}}, << Set({1 if cfg['least_squares']['baseline_mode']=='NONE' else 2}), << Set Width(220))),
                        {_nrow("Fixed detector baseline [mAU]", "lsqBaseline", cfg['least_squares']['detector_baseline'], 240)},
                        {_nrow("Safety cap: solver evaluations [2–10000]", "lsqMaxEvaluations", cfg['least_squares'].get('max_nfev', 120), 240)}
                    ))
                )),
                Button Box("Run least-squares refinement", SendAndRun("FIT_LSQ")),
                H List Box(Button Box("Open fit report", OpenFitReport())),
                lsqStatusText = Text Box("", << Set Wrap(760), << Set Height(130)),
                lsqCompositionPanel = Outline Box("Species mass% reference points", V List Box(
                    Text Box("Enter the expected species mass% at one or more CV positions. Only the target + selected impurities are shown. Each reference CV must sum to 100%.", << Set Wrap(740)),
                    H List Box(Text Box("Reference CV count [1-10]", << Set Width(240)), lsqReferenceCountBox = Combo Box({ref_items}, << Set Width(75), << Set Function(Function({{this}}, UpdateUI())))),
                    {', '.join(ref_bodies)}
                ))
            )),
            "Run / Results",
            V Scroll Box(Size(850, 690), V List Box(
                Text Box("The result buttons verify the selected run and its settings before opening a file. If status says RESULT_STALE, run the selected model again. All models export UV, conductivity, programmed salt, column salt, and local binding salt / ionic strength. For TDM, binding chemistry is the pore-phase chemistry.", << Set Wrap(760)),
                H List Box(
                    Button Box("Open latest chromatogram", OpenCurrentResult(0)),
                    Button Box("Open latest result data", OpenCurrentResult(1))
                ),
                H List Box(
                    Button Box("Open batch gallery", OpenBatchGallery()),
                    Button Box("Open batch summary", OpenBatchSummary()),
                    Button Box("Open LSQ report", OpenLSQReport()),
                    Button Box("Open LSQ objective log", OpenLSQHistory())
                ),
                Spacer Box(Size(1,8)), statusText = Text Box("Ready.", << Set Wrap(760), << Set Height(520))
            ))
        )
    )
);
modelWindow << Set Window Size(920, 850);
impurityCountBox << Set({imp_count + 1});
lsqReferenceCountBox << Set(1);
lsqLoadedRuns = {{0, 0, 0, 0, 0}};
lsqAttachSlot = 1;
For(lsqInitSlot = 1, lsqInitSlot <= {MAX_LSQ_PROFILES}, lsqInitSlot++,
    LoadLSQReference(lsqInitSlot, LSQSelectedRun(lsqInitSlot))
);
LoadRun(1);
lsqTargetRunSelector << Set(1);
UpdateUI();
UpdateLSQRunSummary();
'''
    final_lock_parts = []
    for lock_var, path in fit_lock_vars:
        final_lock_parts.extend([_q(path + "="), f'{lock_var} << Get Selected', _q(";")])
    final_send = 'Python Send(Concat(' + ", ".join(final_lock_parts) + '), Python Name("tdm_ui_lsq_parameter_lock_states"));'
    jsl = jsl.replace(sends[-1], final_send) if 'tdm_ui_lsq_parameter_lock_states' in sends[-1] else jsl
    final_lock_lines = " ".join(f'{var} << Set(If(Uppercase(Char(state)) == "UNLOCKED", 2, 1));' for var, _ in fit_lock_vars)
    jsl = jsl.replace(set_all_lock_lines, final_lock_lines)
    return jsl


def _validate_jsl(jsl: str) -> None:
    import re
    structural = re.sub(r'"(?:\\.|[^"\\])*"', '""', jsl)
    for l, r in [("(", ")"), ("{", "}")]:
        if structural.count(l) != structural.count(r):
            raise ValueError(f"Unbalanced JSL delimiters: {l}{r}")
    if re.search(r'\blsq\w*Boxes\s*\[', jsl):
        raise ValueError("JMP display boxes cannot be sent messages through an indexed JSL list.")
    if re.search(r'\bLoadLSQReference\s*\(\s*\d+\s*\)', jsl):
        raise ValueError("LoadLSQReference requires both a slot and a run number.")
    required = [
        "Post-load washes [0-5]", "Elution steps [0-5]", "Runs in batch",
        "Run saved batch", "Model combination", "Number of impurities [0-5]",
        "Normal least-squares: chromatogram only", "species mass% at CV",
        "Feed composition mass% of total protein", "UV → protein response factor",
        "Conductivity → salt factor", "Number of time steps", "Number of axial positions", "Number Edit Box", "tdm_batch_runs.csv",
        "Chemistry control", "Start / set-point Buffer B [%]", "End Buffer B [%]",
        "Buffer A (0% B)", "Buffer B (100% B)",
        "Saturation capacity qmax,i [g/L stationary phase]", "SyncLangmuirDerived",
        "AttachLSQCSV = Function", "Pick File(\"Select raw chromatogram CSV/Excel\"",
        "LSQSetPathText(slot, csvPath)", "!File Exists(csvPath)",
        "LSQ_Chromatogram_CSV", "lsqTargetRunSelector << Set(currentRun)", "RunNumberFromLabel(this << Get Selected)",
        "RunNumberFromLabel = Function", "Column(dtRuns, \"PLW_Count\")[currentRun] = Num(plwCountBox << Get Selected)",
        "Column(dtRuns, \"Elution_Count\")[currentRun] = Num(elutionCountBox << Get Selected)",
        "dtRuns << Save(activeRunsFile)",
        "modelLabel = Char(modelBox << Get Selected)",
        'tdmPanel << Visibility(If(isTDM, "Visible", "Collapse"))',
        'edmPanel << Visibility(If(isEDM, "Visible", "Collapse"))',
        "CHECK_RESULT_CURRENT", "OpenCurrentResult(0)", "OpenCurrentResult(1)",
        "statusValue = Load Text File(statusFile)", "CHECK_RESULT_CURRENT pending", "RESULT_CURRENT:", RESULT_GUARD_BUILD,
    ]
    for item in required:
        if item not in jsl:
            raise ValueError(f"Missing restored UI construct: {item}")
    # Validate actual UI controls rather than searching all generated text.
    # Absolute project paths can contain words such as "Model Template" or
    # "SMA", and are embedded in JSL string literals for the bridge files.
    model_combo = re.search(r'modelBox\s*=\s*Combo Box\(\s*\{([^{}]*)\}', jsl)
    if model_combo is None:
        raise ValueError("Missing model selector.")
    try:
        model_choices = json.loads("[" + model_combo.group(1) + "]")
    except (ValueError, TypeError) as exc:
        raise ValueError("Could not parse model selector choices.") from exc
    if len(model_choices) != len(MODEL_LABELS) or set(model_choices) != set(MODEL_LABELS.values()):
        raise ValueError(f"Model selector contains unsupported choices: {model_choices}")
    forbidden_labels = ("Model template", "Resin template", "Mechanistic model templates")
    for label in re.findall(r'\b(?:Text Box|Outline Box|Button Box)\(\s*"([^"]*)"', jsl):
        for forbidden in forbidden_labels:
            if forbidden.casefold() in label.casefold():
                raise ValueError(f"Removed template construct leaked into UI: {forbidden}")
    selected_text_controls = [
        "modelBox", "impurityCountBox", "plwCountBox", "elutionCountBox", "loadSource", "loadMaterialMode",
        "lsqModeBox", "lsqReferenceCountBox", "lsqTargetRunSelector",
    ]
    for control in selected_text_controls:
        if f"{control} << Get Selected" not in jsl:
            raise ValueError(f"Combo Box {control} must be read as selected text.")
        if f"{control} << Get()" in jsl:
            raise ValueError(f"Combo Box {control} is incorrectly read as an item index.")
    if jsl.count("RunNumberFromLabel(this << Get Selected)") != 1 + MAX_LSQ_PROFILES:
        raise ValueError("The batch and all LSQ run selectors must parse their selected Run label.")
    if "dtRuns << Save;" in jsl:
        raise ValueError("Run profiles must be persisted to the explicit batch CSV path.")
    if "Python Send(Floor(batchCountBox << Get), Python Name(\"tdm_ui_batch_count\"))" not in jsl:
        raise ValueError("Numeric batch count must continue to use Number Edit Box Get().")
    for prefix, variable, max_steps in (
        ("PLW", "plw", MAX_PLW_STEPS), ("Elution", "el", MAX_ELUTION_STEPS)
    ):
        for i in range(1, max_steps + 1):
            expected_controls = {
                "Mode": (f"{variable}{i}Mode", "Get Selected"),
                "CV": (f"{variable}{i}CV", "Get"),
                "Flow_mL_min": (f"{variable}{i}Flow", "Get"),
                "Chemistry_Control": (f"{variable}{i}ChemControl", "Get Selected"),
                "Start_Percent_B": (f"{variable}{i}StartB", "Get"),
                "End_Percent_B": (f"{variable}{i}EndB", "Get"),
            }
            for field, (control, getter) in expected_controls.items():
                expected = f'Column(dtRuns, "{prefix}{i}_{field}")[currentRun] = {control} << {getter};'
                if expected not in jsl:
                    raise ValueError(f"{prefix} {i} {field} is not saved from its current UI control.")
            for control in (f"{variable}{i}CV", f"{variable}{i}Flow"):
                callback_fragment = (
                    f"{control} = Number Edit Box(1, 8, << Set Width(92), "
                    "<< Set Function(Function({this}, If(!isLoading, SaveCurrentRun(); UpdateUI()))))"
                )
                if callback_fragment not in jsl:
                    raise ValueError(f"{prefix} {i} {control} must save on numeric edit commit.")
            for control in (f"{variable}{i}StartSource", f"{variable}{i}EndSource"):
                if f"{control} << Get Selected" not in jsl:
                    raise ValueError(f"{prefix} {i} source Combo Box must be read as selected text.")
                if f"{control} << Get()" in jsl:
                    raise ValueError(f"{prefix} {i} source Combo Box is incorrectly read as an item index.")
    for slot in range(1, MAX_LSQ_PROFILES + 1):
        if f'Button Box("Attach Excel / CSV", AttachLSQCSV({slot}))' not in jsl:
            raise ValueError(f"LSQ run section {slot} must call its own attachment handler")
        if f'Return(RunNumberFromLabel(lsqRunSlot{slot} << Get Selected))' not in jsl:
            raise ValueError(f"LSQ run section {slot} must read its selector directly")
        if f'lsqCsvPath{slot} << Set Text(path)' not in jsl:
            raise ValueError(f"LSQ run section {slot} must load its file path directly")


def open_in_jmp() -> bool:
    if not JMP_AVAILABLE:
        print("JMP Python module unavailable:", JMP_IMPORT_ERROR)
        return False
    text = build_jsl(load_config(CONFIG_PATH))
    _validate_jsl(text)
    LAST_JSL.write_text(text, encoding="utf-8")
    try:
        jmp.run_jsl(text)
    except Exception as exc:
        print("JSL submission failed:", type(exc).__name__, exc)
        return False
    return True


if __name__ == "__main__":
    text = build_jsl(load_config(CONFIG_PATH))
    _validate_jsl(text)
    LAST_JSL.write_text(text, encoding="utf-8")
    print(LAST_JSL)
