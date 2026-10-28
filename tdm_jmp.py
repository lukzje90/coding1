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
    LSQ_MASS_CSV,
    LSQ_MASS_FIELDS,
    LSQ_RUNS_CSV,
    BATCH_SUMMARY_CSV,
    CONFIG_PATH,
    LATEST_RESULTS_CSV,
    LATEST_RESULTS_HTML,
    MAX_ELUTION_STEPS,
    MAX_BUFFER_IONS,
    MAX_IMPURITIES,
    MAX_LSQ_PROFILES,
    MAX_LSQ_FRACTIONS,
    MAX_PLW_STEPS,
    MAX_RUNS,
    batch_run_fields,
    batch_run_numeric_fields,
    MODEL_EDM_CPA,
    MODEL_EDM_LANGMUIR,
    MODEL_LABELS,
    MODEL_TDM_CPA,
    MODEL_TDM_LANGMUIR,
    MODEL_TDM_WANG,
    RESULT_GUARD_BUILD,
    normalize_config,
    is_hic,
    STATUS_PATH,
    ensure_batch_runs_csv,
    ensure_lsq_runs_csv,
    ensure_lsq_mass_csv,
    load_config,
    requirement_text,
)

PROJECT_DIR = Path(__file__).resolve().parent
BRIDGE_PATH = PROJECT_DIR / "tdm_bridge.py"
LAST_JSL = PROJECT_DIR / "last_generated_ui.jsl"
LSQ_ATTACHED_CSV = PROJECT_DIR / "least_squares_reference_chromatogram.csv"
LSQ_APPLY_JSL = PROJECT_DIR / "least_squares_apply_to_ui.jsl"
LSQ_REPORT_HTML = PROJECT_DIR / "least_squares_fit_report.html"
LSQ_ALIGNMENT_HTML = PROJECT_DIR / "least_squares_alignment_preview.html"
LSQ_HISTORY_CSV = PROJECT_DIR / "least_squares_objective_history.csv"
MAX_LSQ_REFERENCE_POINTS = MAX_LSQ_FRACTIONS


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


class _FitControlRegistry(list[tuple[str, str]]):
    """Keep fit-lock paths and their saved optional range limits together."""

    def __init__(self, fit_range_percent: dict[str, Any]):
        super().__init__()
        self.fit_range_percent = fit_range_percent


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
            saved_range = getattr(fit_lock_vars, "fit_range_percent", {}).get(fit_path)
            range_enabled = saved_range is not None
            range_value = saved_range if range_enabled else 20.0
            lock_control += (
                f', Text Box("Fit ±%", << Set Width(48)), '
                f'{var}FitRangeMode = Combo Box({{"Default", "Limited"}}, '
                f'<< Set({2 if range_enabled else 1}), << Set Width(82)), '
                f'{var}FitRangePercent = Number Edit Box({_num(range_value)}, 9, << Set Width(70))'
            )
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
        'Text Box("Uses the shared target UV response factor.", << Set Wrap(620))'
    )
    body = f'''{p}Panel = Outline Box({_q(title)},
        V List Box(
            H List Box(Text Box("Species name", << Set Width(290)), {p}Name = Text Edit Box({_q(row.get('name', title))}, << Set Width(150), << Set Function(Function({{this}}, If(!isLoading, UpdateUI()))))),
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
                H List Box(Text Box("Derived Henry constant H_i = qmax,i × b_i [-]", << Set Width(360)), {p}H = Number Edit Box({_num(h_value)}, 9, << Set Width(92), << Enable(0)))
            ),
            {p}Wang = V List Box(
                {_nrow("Wang K′kin,i [s]", f"{p}WangKkin", row.get("wang_K_kin_s"), fit_path=f"components[{i-1}].wang_K_kin_s", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)},
                {_nrow("Wang k_eq,i", f"{p}WangKeq", row.get("wang_k_eq"), fit_path=f"components[{i-1}].wang_k_eq", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)},
                {_nrow("Wang qmax,i [g/L stationary phase]", f"{p}WangQmax", row.get("wang_qmax_g_L"), fit_path=f"components[{i-1}].wang_qmax_g_L", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)},
                {_nrow("Wang n_i [-]", f"{p}WangN", row.get("wang_n"), fit_path=f"components[{i-1}].wang_n", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)},
                {_nrow("Wang β0,i [-]", f"{p}WangBeta0", row.get("wang_beta0"), fit_path=f"components[{i-1}].wang_beta0", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)},
                {_nrow("Wang β1,i [M⁻¹]", f"{p}WangBeta1", row.get("wang_beta1_per_M"), fit_path=f"components[{i-1}].wang_beta1_per_M", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)},
                {_nrow("Wang β2,i [L/g]", f"{p}WangBeta2", row.get("wang_beta2_L_g"), fit_path=f"components[{i-1}].wang_beta2_L_g", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)},
                {_nrow("Wang β3,i [pH⁻¹]", f"{p}WangBeta3", row.get("wang_beta3_per_pH"), fit_path=f"components[{i-1}].wang_beta3_per_pH", fit_lock_vars=fit_lock_vars, default_unlocked_paths=default_unlocked_paths)}
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
        (f"{p}WangKkin << Get", f"tdm_ui_c{i}_wang_K_kin_s"),
        (f"{p}WangKeq << Get", f"tdm_ui_c{i}_wang_k_eq"),
        (f"{p}WangQmax << Get", f"tdm_ui_c{i}_wang_qmax_g_L"),
        (f"{p}WangN << Get", f"tdm_ui_c{i}_wang_n"),
        (f"{p}WangBeta0 << Get", f"tdm_ui_c{i}_wang_beta0"),
        (f"{p}WangBeta1 << Get", f"tdm_ui_c{i}_wang_beta1_per_M"),
        (f"{p}WangBeta2 << Get", f"tdm_ui_c{i}_wang_beta2_L_g"),
        (f"{p}WangBeta3 << Get", f"tdm_ui_c{i}_wang_beta3_per_pH"),
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
    {p}Wang << Visibility(If(isWang, "Visible", "Collapse"));
    {p}CPA << Visibility(If(isCPA, "Visible", "Collapse"));
    {p}KkinRow << Visibility(If(modelLabel == {_q(MODEL_LABELS[MODEL_TDM_CPA])}, "Visible", "Collapse"));
    {p}SaltRow << Visibility(If(isLangmuir | (isCPA & isHIC), "Visible", "Collapse"));
    {p}ZrefRow << Visibility(If(isCPA & !isHIC, "Visible", "Collapse"));
    {p}PHPanel << Visibility(If(isCPA & !isHIC & fitPHSpan > 1e-9, "Visible", "Collapse"));
    {p}Z3Row << Visibility(If(isCPA & !isHIC & fitPHSpan > 1, "Visible", "Collapse"));'''
    return body, sends, visible


def _salt_input_controls(prefix: str) -> str:
    # Editing any load-material salt field is an explicit user choice. Promote
    # the load selector automatically so a populated pH/salt/conductivity
    # value cannot be silently ignored while the load %B recipe remains stored.
    # A per-load pH/salt/conductivity edit is a direct material measurement.
    # Promote both selectors so an inherited BUFFER_A/B source cannot mask the
    # value the user just changed.
    load_hook = 'loadMaterialMode << Set(2); loadChemControl << Set(2); loadSource << Set(1); ' if prefix == "load" else ''
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
        {prefix}IonicSourceRow = H List Box(Text Box("Ionic strength source", << Set Width(220)), {prefix}IonicSource = Combo Box({{"SALT_PROXY", "ION_COMPOSITION"}}, << Set Width(170), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); UpdateUI()))))),
        {prefix}IonSection = Outline Box("Buffer ion composition", V List Box(
        Text Box("For CPA ion exchange, enter ion concentrations and charges to calculate ionic strength. SALT_PROXY uses salt molarity as an approximation.", << Set Wrap(680)),
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
        {v}ChemPanel = V List Box(
            H List Box(Text Box("Chemistry control", << Set Width(220)), {v}ChemControl = Combo Box({{"BUFFER_B_PERCENT", "ENDPOINT_CHEMISTRY"}}, << Set Width(170), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); UpdateUI()))))),
            {v}PercentPanel = V List Box(Spacer Box(Size(1, 1))),
            {v}EndpointPanel = Outline Box("Advanced explicit endpoint chemistry", V List Box(
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


def _lsq_reference_table() -> tuple[str, list[str]]:
    """Keep each fraction heading and its 30 inputs in one physical column."""
    columns: list[list[str]] = [
        ['Text Box("#", << Set Width(28), << Set Height(64))'],
        ['Text Box("Start", << Set Width(76), << Set Height(64))'],
        ['Text Box("End", << Set Width(76), << Set Height(64))'],
    ]
    for i in range(1, 2 + MAX_IMPURITIES):
        columns.append([
            f'lsqMassHeaderC{i} = Text Box("", << Set Width(112), << Set Wrap(112), << Set Height(64))'
        ])
    visibility: list[str] = []
    for k in range(1, MAX_LSQ_REFERENCE_POINTS + 1):
        columns[0].append(f'lsqR{k}Index = Number Edit Box({k}, 8, << Set Width(3), << Enable(0))')
        visibility.append(f'lsqR{k}Index << Visibility(If(lsqUsesComposition & lsqRefCount >= {k}, "Visible", "Collapse"));')
        for column, field in ((1, "Start"), (2, "End")):
            box = f"lsqR{k}{field}"
            columns[column].append(f'{box} = Number Edit Box(., 8, << Set Width(7))')
            visibility.append(f'{box} << Visibility(If(lsqUsesComposition & lsqRefCount >= {k}, "Visible", "Collapse"));')
        for i in range(1, 2 + MAX_IMPURITIES):
            box = f"lsqR{k}C{i}Mass"
            columns[i + 2].append(f'{box} = Number Edit Box(., 8, << Set Width(8))')
            visibility.append(f'{box} << Visibility(If(lsqUsesComposition & lsqRefCount >= {k}, "Visible", "Collapse"));')
    table = 'H List Box(\n' + ',\n'.join(
        f'lsqMassColumnC{i-2} = V List Box(' + ', '.join(column) + ')'
        if i >= 3 else 'V List Box(' + ', '.join(column) + ')'
        for i, column in enumerate(columns)
    ) + '\n)'
    return table, visibility


def _req_expr(model: str, span: float, chromatography_mode: str | None = None) -> str:
    vals = {n: _q(requirement_text(model, span, n, chromatography_mode)) for n in range(MAX_IMPURITIES + 1)}
    expr = vals[MAX_IMPURITIES]
    for n in reversed(range(MAX_IMPURITIES)):
        expr = f'If(impurityCount == {n}, {vals[n]}, {expr})'
    return expr


def _default_lsq_unlocked_paths(config: dict[str, Any]) -> set[str]:
    """Use a small initial TDM–CPA fit; preserve explicit saved lock choices."""
    persisted = config.get("least_squares", {}).get("unlocked_paths")
    if persisted is not None:
        return set(persisted)
    model = str(config.get("model") or "")
    if model == MODEL_TDM_CPA:
        return {"components[0].delta_ref", "components[0].Z_ref"}
    n = min(max(int(config.get("impurity_count", 0) or 0), 0), MAX_IMPURITIES) + 1
    paths: set[str] = set()
    for i in range(n):
        prefix = f"components[{i}]"
        if model == MODEL_TDM_WANG:
            paths.update({f"{prefix}.wang_k_eq", f"{prefix}.wang_beta1_per_M"})
        elif model in {MODEL_EDM_LANGMUIR, MODEL_TDM_LANGMUIR}:
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
    ensure_lsq_mass_csv()
    cfg = normalize_config(config or load_config(CONFIG_PATH))
    selected = MODEL_LABELS.get(cfg.get("model"), MODEL_LABELS[MODEL_TDM_CPA])
    labels = [MODEL_LABELS[model] for model in (
        MODEL_EDM_LANGMUIR, MODEL_TDM_LANGMUIR, MODEL_EDM_CPA,
        MODEL_TDM_CPA, MODEL_TDM_WANG,
    )]
    model_items = "{" + ", ".join(_q(x) for x in ([selected] + [x for x in labels if x != selected])) + "}"
    imp_count = min(max(int(cfg.get("impurity_count", 0) or 0), 0), MAX_IMPURITIES)
    c, t, e, cp, conv = cfg["column"], cfg["tdm"], cfg["edm"], cfg["cpa"], cfg.get("conversion", {})
    try:
        batch_count = min(max(int(BATCH_COUNT_FILE.read_text().strip()), 1), MAX_RUNS)
    except Exception:
        batch_count = 1

    component_bodies, component_sends, component_vis = [], [], []
    fit_lock_vars = _FitControlRegistry(cfg["least_squares"].get("fit_range_percent") or {})
    default_unlocked_paths = _default_lsq_unlocked_paths(cfg)
    if cfg["least_squares"].get("unlocked_paths") is None and cfg["model"] != MODEL_TDM_CPA:
        # Seed each model's usual fit controls only when no explicit locks were
        # saved. Switching model in JMP then exposes sensible active defaults;
        # a saved Lock all choice remains locked even for hidden combinations.
        for i in range(imp_count + 1):
            prefix = f"components[{i}]"
            default_unlocked_paths.update({
                f"{prefix}.qmax_g_L", f"{prefix}.b_L_g",
                f"{prefix}.salt_sensitivity_per_M", f"{prefix}.delta_ref",
                f"{prefix}.As_m_inv", f"{prefix}.Z_ref",
                f"{prefix}.kkin_star_s", f"{prefix}.wang_k_eq",
                f"{prefix}.wang_beta1_per_M",
            })
    for i, row in enumerate(cfg["components"], start=1):
        b, s, v = _component_panel(i, row, fit_lock_vars, default_unlocked_paths)
        component_bodies.append(b); component_sends.extend(s); component_vis.append(v)
    wang_q0_row = _nrow("Wang shared q0 [g/L stationary phase]", "wangQ0", cfg["wang"]["q0_g_L"], 360,
                        fit_path="wang.q0_g_L", fit_lock_vars=fit_lock_vars,
                        default_unlocked_paths=default_unlocked_paths)
    wang_eta_row = _nrow("Wang shared η [-]", "wangEta", cfg["wang"]["eta"], 360,
                         fit_path="wang.eta", fit_lock_vars=fit_lock_vars,
                         default_unlocked_paths=default_unlocked_paths)

    buffer_a_panel = _buffer_panel("bufferA", "Buffer A (0% B)")
    buffer_b_panel = _buffer_panel("bufferB", "Buffer B (100% B)")
    plw_panels = [_step_panel("plw", i, "Post-load wash") for i in range(1, MAX_PLW_STEPS + 1)]
    el_panels = [_step_panel("el", i, "Elution") for i in range(1, MAX_ELUTION_STEPS + 1)]

    ref_table, ref_vis = _lsq_reference_table()
    mass_header_updates = []
    for i in range(1, 2 + MAX_IMPURITIES):
        role = "Target" if i == 1 else f"Impurity {i-1}"
        mass_header_updates.extend([
            f'speciesName = Trim(Char(c{i}Name << Get Text));',
            f'If(speciesName == "", speciesName = {_q(role)});',
            f'lsqMassHeaderC{i} << Set Text(speciesName || " mass%");',
            f'lsqMassColumnC{i} << Visibility(If({1 if i == 1 else f"impurityCount >= {i-1}"}, "Visible", "Collapse"));',
        ])
    mass_save_rows, mass_load_rows = [], []
    mass_numeric_columns = "\n".join(
        f'Column(dtLSQMass, {_q(field)}) << Data Type(Numeric);'
        for field in LSQ_MASS_FIELDS if field not in {"Enabled", "Unit"}
    )
    numeric_process_fields = set(batch_run_numeric_fields())
    process_column_types = "\n".join(
        f'Column({table}, {_q(field)}) << Data Type({"Numeric" if field in numeric_process_fields else "Character"});'
        for field in batch_run_fields() for table in ("dtBatchRuns", "dtLSQRuns")
    )
    for k in range(1, MAX_LSQ_FRACTIONS + 1):
        mass_save_rows.extend([
            f'Column(dtLSQMass, "Start")[massRow + {k - 1}] = lsqR{k}Start << Get;',
            f'Column(dtLSQMass, "End")[massRow + {k - 1}] = lsqR{k}End << Get;',
        ])
        mass_load_rows.extend([
            f'lsqR{k}Start << Set(Num(Column(dtLSQMass, "Start")[massRow + {k - 1}]));',
            f'lsqR{k}End << Set(Num(Column(dtLSQMass, "End")[massRow + {k - 1}]));',
        ])
        for species in range(1, MAX_IMPURITIES + 2):
            field = "Target_mass_percent" if species == 1 else f"Impurity{species - 1}_mass_percent"
            mass_save_rows.append(
                f'Column(dtLSQMass, "{field}")[massRow + {k - 1}] = lsqR{k}C{species}Mass << Get;')
            mass_load_rows.append(
                f'lsqR{k}C{species}Mass << Set(Num(Column(dtLSQMass, "{field}")[massRow + {k - 1}]));')

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
        f'Python Send(wangQ0 << Get, Python Name("tdm_ui_wang_q0_g_L"));',
        f'Python Send(wangEta << Get, Python Name("tdm_ui_wang_eta"));',
    ] + component_sends
    sys = cfg["system"]
    system_rows = []
    for field, var, label in (
        ("buffer_dispersion_mL", "bufferDispersion", "Buffer mixer dispersion volume [mL]"),
        ("salt_dispersion_mm2_s", "systemSaltDax", "Buffer / salt axial dispersion [mm²/s]"),
        ("uv_dead_volume_mL", "uvDeadVolume", "System dead volume to UV [mL]"),
        ("conductivity_dead_volume_mL", "condDeadVolume", "System dead volume to conductivity [mL]"),
    ):
        # Conductivity detector delay is eligible only when a selected LSQ
        # profile has measured conductivity enabled.
        fit_path = "system." + field
        system_rows.append(_nrow(label, var, sys[field], 360, fit_path=fit_path,
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
        'Python Send(lsqMaxTrialMinutes << Get, Python Name("tdm_ui_lsq_max_trial_minutes"));',
        'Python Send(lsqPhysicalAcceptance << Get Selected, Python Name("tdm_ui_lsq_physical_acceptance_mode"));',
        'Python Send(lsqMinBound << Get, Python Name("tdm_ui_lsq_min_bound_percent"));',
        'Python Send(lsqMinRecovery << Get, Python Name("tdm_ui_lsq_min_recovery_percent"));',
        'Python Send(lsqMinImprovement << Get, Python Name("tdm_ui_lsq_min_improvement_percent"));',
    ]
    lsq_run_sections = []
    lsq_selection_visibility = []
    lsq_selection_sends = []
    run_choices = "{" + ", ".join(_q(f"Run {r}") for r in range(1, MAX_LSQ_PROFILES + 1)) + "}"
    model_run_choices = "{" + ", ".join([_q("Independent LSQ recipe")] + [_q(f"Run {r}") for r in range(1, MAX_RUNS + 1)]) + "}"
    saved_selection = cfg["least_squares"].get("selected_run_numbers")
    saved_selection = saved_selection if isinstance(saved_selection, list) and 1 <= len(saved_selection) <= MAX_LSQ_PROFILES else [1]
    initial_runs = []
    for value in saved_selection:
        try:
            run = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= run <= MAX_LSQ_PROFILES and run not in initial_runs:
            initial_runs.append(run)
    initial_runs.extend(run for run in range(1, MAX_LSQ_PROFILES + 1) if run not in initial_runs)
    for slot in range(1, MAX_LSQ_PROFILES + 1):
        selected_run = initial_runs[slot - 1]
        lsq_run_sections.append(f'''lsqRunSlot{slot}Row = Outline Box("LSQ run {slot}", V List Box(
            H List Box(Text Box("Process recipe", << Set Width(180)), lsqRunSlot{slot} = Combo Box({run_choices}, << Set({selected_run}), << Set Width(140), << Set Function(Function({{this}}, If(!isLoading, ChangeLSQSlot({slot}, RunNumberFromLabel(this << Get Selected))))))),
            lsqRunSummary{slot} = Text Box("", << Set Wrap(720)),
            H List Box(Text Box("Model run", << Set Width(180)), lsqModelRun{slot} = Combo Box({model_run_choices}, << Set(1), << Set Width(210), << Set Function(Function({{this}}, If(!isLoading, UpdateUI(); UpdateLSQRunSummary()))))),
            H List Box(Text Box("Excel / CSV file", << Set Width(180)), lsqCsvPath{slot} = Text Edit Box("", << Set Width(440))),
            H List Box(Text Box("X units", << Set Width(180)), lsqXUnit{slot} = Combo Box({{"AUTO", "CV", "ML", "MIN", "S", "H", "INDEX"}}, << Set(1), << Set Width(150))),
            H List Box(Text Box("X = 0 at", << Set Width(180)), lsqXOrigin{slot} = Combo Box({{"RUN_START", "ELUTION_START"}}, << Set(1), << Set Width(170))),
            H List Box(Text Box("Use this run's fraction mass%", << Set Width(180)), lsqMassEnabled{slot} = Combo Box({{"Off", "On"}}, << Set(1), << Set Width(100), << Set Function(Function({{this}}, If(!isLoading, UpdateLSQRunSummary()))))),
            H List Box(Text Box("Fit measured conductivity", << Set Width(180)), lsqConductivityFit{slot} = Combo Box({{"Off", "On"}}, << Set(1), << Set Width(100))),
            H List Box(Button Box("Attach Excel / CSV", AttachLSQCSV({slot})), Button Box("Edit process recipe", SwitchProcessScope(1, RunNumberFromLabel(lsqRunSlot{slot} << Get Selected)); mainTabs << Set(1)), Button Box("Preview this run", PreviewLSQSlot({slot}))),
            Outline Box("Optional columns and weight", << Close(1), V List Box(
                H List Box(Text Box("Excel worksheet", << Set Width(180)), lsqSheet{slot} = Text Edit Box("", << Set Width(380))),
                H List Box(Text Box("X column", << Set Width(180)), lsqXColumn{slot} = Text Edit Box("", << Set Width(380))),
                H List Box(Text Box("UV mAU column", << Set Width(180)), lsqSignalColumn{slot} = Text Edit Box("", << Set Width(380))),
                H List Box(Text Box("Conductivity column [mS/cm]", << Set Width(180)), lsqConductivityColumn{slot} = Text Edit Box("", << Set Width(380))),
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
        ("LSQModelRunSelected", "lsqModelRun", "Get Selected"),
        ("LSQPathText", "lsqCsvPath", "Get Text"),
        ("LSQWeightNumber", "lsqWeight", "Get"),
        ("LSQSheetText", "lsqSheet", "Get Text"),
        ("LSQXColumnText", "lsqXColumn", "Get Text"),
        ("LSQSignalColumnText", "lsqSignalColumn", "Get Text"),
        ("LSQConductivityFitSelected", "lsqConductivityFit", "Get Selected"),
        ("LSQConductivityColumnText", "lsqConductivityColumn", "Get Text"),
        ("LSQUnitSelected", "lsqXUnit", "Get Selected"),
        ("LSQOriginSelected", "lsqXOrigin", "Get Selected"),
        ("LSQMassEnabledSelected", "lsqMassEnabled", "Get Selected"),
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
        lsqModelRun{i} << Set(modelIndex);
        lsqWeight{i} << Set(w);
        lsqSheet{i} << Set Text(sheet);
        lsqXColumn{i} << Set Text(xcolumn);
        lsqSignalColumn{i} << Set Text(signal);
        lsqConductivityFit{i} << Set(condFitIndex);
        lsqConductivityColumn{i} << Set Text(condColumn);
        lsqXUnit{i} << Set(unitIndex);
        lsqXOrigin{i} << Set(originIndex);
        lsqMassEnabled{i} << Set(massEnabledIndex);
    );'''
        for i in range(1, MAX_LSQ_PROFILES + 1)
    )
    lsq_control_dispatch = "\n".join(lsq_getters + lsq_setters)
    # Some controls are created while interpolating the UI template below.
    # Fill these placeholders after that template has registered every control.
    sends.extend(("__LSQ_LOCK_SENDS__", "__LSQ_FIT_RANGE_SENDS__"))
    set_all_lock_lines = " ".join(
        f'{lock_var} << Set(If(Uppercase(Char(state)) == "UNLOCKED", 2, 1));'
        for lock_var, _parameter_path in fit_lock_vars
    )

    save_run_lines = [
        'Column(dtRuns, "Run_Name")[currentRun] = runNameBox << Get Text;',
        'Column(dtRuns, "Load_Flow_mL_min")[currentRun] = loadFlow << Get;',
        'Column(dtRuns, "Feed_Concentration_mg_mL")[currentRun] = feedConcentration << Get;',
        'Column(dtRuns, "Load_Amount_Basis")[currentRun] = "CAPACITY";',
        'Column(dtRuns, "Load_CV")[currentRun] = loadCV << Get;',
        'Column(dtRuns, "Load_Volume_mL")[currentRun] = loadVolumeML << Get;',
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
        'Column(dtRuns, "Equilibration_Source")[currentRun] = equilibrationSource << Get Selected;',
        'Column(dtRuns, "Equilibration_pH")[currentRun] = equilibrationPH << Get;',
        'Column(dtRuns, "Equilibration_Salt_M")[currentRun] = equilibrationSalt << Get;',
        'Column(dtRuns, "Equilibration_Salt_Input_Mode")[currentRun] = equilibrationSaltMode << Get Selected;',
        'Column(dtRuns, "Equilibration_Conductivity_mS_cm")[currentRun] = equilibrationCond << Get;',
        'Column(dtRuns, "PLW_Count")[currentRun] = Num(plwCountBox << Get Selected);',
        'Column(dtRuns, "Elution_Count")[currentRun] = Num(elutionCountBox << Get Selected);',
    ]
    load_run_lines = [
        'runNameBox << Set Text(Char(Column(dtRuns, "Run_Name")[currentRun]));',
        'loadFlow << Set(Column(dtRuns, "Load_Flow_mL_min")[currentRun]);',
        'feedConcentration << Set(Column(dtRuns, "Feed_Concentration_mg_mL")[currentRun]);',
        'loadCV << Set(Column(dtRuns, "Load_CV")[currentRun]);',
        'loadVolumeML << Set(Column(dtRuns, "Load_Volume_mL")[currentRun]);',
        'loadDensity << Set(Column(dtRuns, "Load_Density_mg_mL_resin")[currentRun]);',
        'loadBasis = Uppercase(Char(Column(dtRuns, "Load_Amount_Basis")[currentRun]));',
        'If(loadBasis == "LOAD_VOLUME" | loadBasis == "LOAD VOLUME [CV]", loadDensity << Set(If(Is Missing(loadCV << Get) | Is Missing(feedConcentration << Get), ., (loadCV << Get) * (feedConcentration << Get))));',
        'If(loadBasis == "VOLUME_ML" | loadBasis == "LOAD VOLUME [ML]", loadDensity << Set(If(Is Missing(loadVolumeML << Get) | Is Missing(columnVolume << Get) | (columnVolume << Get) <= 0 | Is Missing(feedConcentration << Get), ., (loadVolumeML << Get) / (columnVolume << Get) * (feedConcentration << Get))));',
        'SetModeCombo(loadMode, Char(Column(dtRuns, "Load_Mode")[currentRun]));',
        'loadStartB << Set(Column(dtRuns, "Load_Start_Percent_B")[currentRun]);',
        'loadEndB << Set(Column(dtRuns, "Load_End_Percent_B")[currentRun]);',
        'If(Uppercase(Char(Column(dtRuns, "Load_Chemistry_Control")[currentRun])) == "ENDPOINT_CHEMISTRY", loadChemControl << Set(2), loadChemControl << Set(1));',
        'If(Uppercase(Char(Column(dtRuns, "Load_Material_Chemistry_Mode")[currentRun])) == "EXPLICIT", loadMaterialMode << Set(2), loadMaterialMode << Set(1));',
        'loadChemControl << Set(If(Uppercase(Char(loadMaterialMode << Get Selected)) == "EXPLICIT", 2, 1));',
        'timeStepsBox << Set(Column(dtRuns, "Time_Steps")[currentRun]);',
        'axialPositionsBox << Set(Column(dtRuns, "Axial_Positions")[currentRun]);',
        'SetSourceCombo(loadSource, Char(Column(dtRuns, "Load_Source")[currentRun]));',
        'loadPH << Set(Column(dtRuns, "Load_pH")[currentRun]);',
        'loadCond << Set(Column(dtRuns, "Load_Conductivity_mS_cm")[currentRun]);',
        'loadSalt << Set(Column(dtRuns, "Load_Salt_M")[currentRun]);',
        'loadSaltMode << Set(If(Uppercase(Char(Column(dtRuns, "Load_Salt_Input_Mode")[currentRun])) == "SALT_M", 2, 1));',
        'equilibrationSource << Set(If(Uppercase(Char(Column(dtRuns, "Equilibration_Source")[currentRun])) == "BUFFER_A", 2, Uppercase(Char(Column(dtRuns, "Equilibration_Source")[currentRun])) == "BUFFER_B", 3, Uppercase(Char(Column(dtRuns, "Equilibration_Source")[currentRun])) == "DIRECT", 4, 1));',
        'equilibrationPH << Set(Column(dtRuns, "Equilibration_pH")[currentRun]);',
        'equilibrationSalt << Set(Column(dtRuns, "Equilibration_Salt_M")[currentRun]);',
        'equilibrationSaltMode << Set(If(Uppercase(Char(Column(dtRuns, "Equilibration_Salt_Input_Mode")[currentRun])) == "SALT_M", 2, 1));',
        'equilibrationCond << Set(Column(dtRuns, "Equilibration_Conductivity_mS_cm")[currentRun]);',
        'plwCountBox << Set(Num(Column(dtRuns, "PLW_Count")[currentRun]) + 1);',
        'elutionCountBox << Set(Num(Column(dtRuns, "Elution_Count")[currentRun]) + 1);',
    ]
    copy_cols = ["Run_Name", "Load_Flow_mL_min", "Feed_Concentration_mg_mL", "Load_Amount_Basis", "Load_CV", "Load_Volume_mL", "Load_Density_mg_mL_resin", "Load_Mode", "Load_Start_Percent_B", "Load_End_Percent_B", "Load_Chemistry_Control", "Load_Material_Chemistry_Mode", "Time_Steps", "Axial_Positions", "Load_Source", "Load_pH", "Load_Salt_M", "Load_Conductivity_mS_cm", "Equilibration_Source", "Equilibration_pH", "Equilibration_Salt_M", "Equilibration_Salt_Input_Mode", "Equilibration_Conductivity_mS_cm", "PLW_Count", "Elution_Count"]

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

    req_wang = _req_expr(MODEL_TDM_WANG, 0, "HIC")
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
    lsq_run_count_items = "{" + ", ".join(_q(str(i)) for i in range(1, MAX_LSQ_PROFILES + 1)) + "}"
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

    shared_sends = "\n    ".join(sends)
    jsl = f'''Names Default To Here(1);
projectDir = {_path(PROJECT_DIR)};
bridgeFile = {_path(BRIDGE_PATH)};
batchRunsFile = {_path(BATCH_RUNS_CSV)};
lsqRunsFile = {_path(LSQ_RUNS_CSV)};
lsqMassFile = {_path(LSQ_MASS_CSV)};
statusFile = {_path(STATUS_PATH)};
resultFile = {_path(LATEST_RESULTS_HTML)};
resultCSV = {_path(LATEST_RESULTS_CSV)};
batchGallery = {_path(BATCH_GALLERY_HTML)};
batchSummary = {_path(BATCH_SUMMARY_CSV)};
lsqApplyFile = {_path(LSQ_APPLY_JSL)};
lsqReport = {_path(LSQ_REPORT_HTML)};
lsqPreviewReport = {_path(LSQ_ALIGNMENT_HTML)};
currentRun = 1; copiedRun = 0; isLoading = 1; isLSQSetup = 0; lsqMassEditorRun = 1; lsqPreviewRun = 0;
dtBatchRuns = Open(batchRunsFile, Invisible);
dtLSQRuns = Open(lsqRunsFile, Invisible);
dtLSQMass = Open(lsqMassFile, Invisible);
dtRuns = dtBatchRuns; activeRunsFile = batchRunsFile;
// Both run tables use the same explicit CSV schema. Empty salt and ion cells
// must be numeric, while blank chromatogram paths and names stay character.
{process_column_types}
// Empty fraction columns can otherwise import as character on a new setup.
Column(dtLSQMass, "Enabled") << Data Type(Character);
Column(dtLSQMass, "Unit") << Data Type(Character);
{mass_numeric_columns}
SetModeCombo = Function({{box, txt}}, {{}},
    If(Uppercase(txt) == "LINEAR", box << Set(2), Uppercase(txt) == "STEP", box << Set(3), box << Set(1));
);
SetSourceCombo = Function({{box, txt}}, {{}},
    If(Uppercase(txt) == "BUFFER_A", box << Set(2), Uppercase(txt) == "BUFFER_B", box << Set(3), box << Set(1));
);
SetAllFitLocks = Function({{state}}, {{}},
    {set_all_lock_lines}
);
SyncLoadAmount = Function({{}}, {{concentration, capacity, cv, volume, columnVol}},
    If(isLoading, Return());
    concentration = feedConcentration << Get;
    capacity = loadDensity << Get;
    columnVol = columnVolume << Get;
    cv = If(Is Missing(capacity) | Is Missing(concentration) | concentration <= 0, ., capacity / concentration);
    volume = If(Is Missing(cv) | Is Missing(columnVol) | columnVol <= 0, ., cv * columnVol);
    isLoading = 1;
    loadCV << Set(cv);
    loadVolumeML << Set(volume);
    isLoading = 0;
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
ResolvedBufferCond = Function({{condBox, saltBox, modeBox}}, {{c, factor}},
    c = condBox << Get;
    If(Char(modeBox << Get Selected) == "SALT_M",
        If(!Is Missing(c), Return(c));
        factor = condToSalt << Get;
        If(Is Missing(factor) | factor <= 0, Return(.));
        Return((saltBox << Get) / factor)
    );
    Return(c)
);
BlendBufferCond = Function({{pct}}, {{f, a, b}},
    If(Is Missing(pct), Return(.));
    f = Max(0, Min(100, pct)) / 100;
    a = ResolvedBufferCond(bufferACond, bufferASalt, bufferASaltMode);
    b = ResolvedBufferCond(bufferBCond, bufferBSalt, bufferBSaltMode);
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
ResolvedPHFromRow = Function({{runsTable, r, sourceCol, pHCol}}, {{src}},
    src = Uppercase(Char(Column(runsTable, sourceCol)[r]));
    If(src == "BUFFER_A", Column(runsTable, "BufferA_pH")[r],
       src == "BUFFER_B", Column(runsTable, "BufferB_pH")[r],
       Column(runsTable, pHCol)[r])
);
ResolvedPHFromPercentBRow = Function({{runsTable, r, pctCol}}, {{pct, a, b, f}},
    pct = Num(Column(runsTable, pctCol)[r]); a = Num(Column(runsTable, "BufferA_pH")[r]); b = Num(Column(runsTable, "BufferB_pH")[r]);
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

LoadRun = Function({{n}}, {{loadBasis}},
    If(isLSQSetup & n > {MAX_LSQ_PROFILES}, n = 1);
    isLoading = 1; currentRun = n;
    {''.join(load_run_lines)}
    runSelector << Set(currentRun);
    isLoading = 0; SyncLoadAmount(); UpdateUI(); UpdateLSQRunSummary();
);

{lsq_control_dispatch}
LSQLoadControls = Function({{slot, path, w, sheet, xcolumn, signal, unitIndex, originIndex, modelIndex, massEnabledIndex, condFitIndex, condColumn}}, {{}},
{lsq_load_controls}
);

SaveLSQMassReference = Function({{r}}, {{massRow}},
    If(r < 1 | r > {MAX_LSQ_PROFILES}, Return());
    massRow = (r - 1) * {MAX_LSQ_FRACTIONS} + 1;
    Column(dtLSQMass, "Fraction_Count")[massRow] = Num(lsqReferenceCountBox << Get Selected);
    Column(dtLSQMass, "Unit")[massRow] = lsqFractionUnit << Get Selected;
    {''.join(mass_save_rows)}
    dtLSQMass << Save(lsqMassFile);
);
LoadLSQMassReference = Function({{r}}, {{massRow, count, unit, wasLoading}},
    If(r < 1 | r > {MAX_LSQ_PROFILES}, Return());
    wasLoading = isLoading; isLoading = 1;
    massRow = (r - 1) * {MAX_LSQ_FRACTIONS} + 1;
    count = Num(Column(dtLSQMass, "Fraction_Count")[massRow]);
    lsqReferenceCountBox << Set(If(Is Missing(count), 1, Max(1, Min({MAX_LSQ_FRACTIONS}, count))));
    unit = Uppercase(Char(Column(dtLSQMass, "Unit")[massRow]));
    lsqFractionUnit << Set(If(unit == "ML", 2, 1));
    {''.join(mass_load_rows)}
    lsqMassEditorRun = r;
    lsqMassEditorStatus << Set Text("Editing fraction mass% for LSQ Run " || Char(r) || " — " || Char(Column(dtLSQRuns, "Run_Name")[r]));
    isLoading = wasLoading;
    UpdateUI();
);
ChangeLSQMassEditorRun = Function({{r}}, {{}},
    If(isLoading | r == lsqMassEditorRun, Return());
    SaveLSQMassReference(lsqMassEditorRun);
    LoadLSQMassReference(r);
);

SaveLSQReference = Function({{slot}}, {{r, w, modelChoice, massRow}},
    If(isLoading, Return());
    r = lsqLoadedRuns[slot];
    If(r < 1 | r > {MAX_LSQ_PROFILES}, Return());
    w = LSQWeightNumber(slot);
    Column(dtLSQRuns, "LSQ_Chromatogram_CSV")[r] = Trim(Char(LSQPathText(slot)));
    Column(dtLSQRuns, "LSQ_Weight")[r] = If(Is Missing(w), 1, w);
    Column(dtLSQRuns, "LSQ_Sheet")[r] = Trim(Char(LSQSheetText(slot)));
    Column(dtLSQRuns, "LSQ_X_Column")[r] = Trim(Char(LSQXColumnText(slot)));
    Column(dtLSQRuns, "LSQ_Signal_Column")[r] = Trim(Char(LSQSignalColumnText(slot)));
    Column(dtLSQRuns, "LSQ_Conductivity_Fit")[r] = LSQConductivityFitSelected(slot);
    Column(dtLSQRuns, "LSQ_Conductivity_Column")[r] = Trim(Char(LSQConductivityColumnText(slot)));
    Column(dtLSQRuns, "LSQ_X_Unit")[r] = LSQUnitSelected(slot);
    Column(dtLSQRuns, "LSQ_X_Origin")[r] = LSQOriginSelected(slot);
    modelChoice = LSQModelRunSelected(slot);
    Column(dtLSQRuns, "LSQ_Model_Source")[r] = If(modelChoice == "Independent LSQ recipe", "INDEPENDENT_LSQ", "NORMAL_BATCH");
    Column(dtLSQRuns, "LSQ_Model_Run_Number")[r] = If(modelChoice == "Independent LSQ recipe", r, RunNumberFromLabel(modelChoice));
    massRow = (r - 1) * {MAX_LSQ_FRACTIONS} + 1;
    Column(dtLSQMass, "Enabled")[massRow] = LSQMassEnabledSelected(slot);
    dtLSQRuns << Save(lsqRunsFile);
    dtLSQMass << Save(lsqMassFile);
);
LoadLSQReference = Function({{slot, r}}, {{w, path, unit, origin, sheet, xcolumn, signal, modelSource, modelNumber, modelIndex, massEnabled, condFit, condColumn}},
    If(Is Missing(Num(r)), r = slot);
    r = Max(1, Min({MAX_LSQ_PROFILES}, Num(r)));
    path = Column(dtLSQRuns, "LSQ_Chromatogram_CSV")[r];
    w = Num(Column(dtLSQRuns, "LSQ_Weight")[r]);
    lsqLoadedRuns[slot] = r;
    sheet = Column(dtLSQRuns, "LSQ_Sheet")[r];
    xcolumn = Column(dtLSQRuns, "LSQ_X_Column")[r];
    signal = Column(dtLSQRuns, "LSQ_Signal_Column")[r];
    condFit = Uppercase(Char(Column(dtLSQRuns, "LSQ_Conductivity_Fit")[r]));
    condColumn = Column(dtLSQRuns, "LSQ_Conductivity_Column")[r];
    unit = Uppercase(Char(Column(dtLSQRuns, "LSQ_X_Unit")[r]));
    origin = Uppercase(Char(Column(dtLSQRuns, "LSQ_X_Origin")[r]));
    modelSource = Uppercase(Char(Column(dtLSQRuns, "LSQ_Model_Source")[r]));
    modelNumber = Num(Column(dtLSQRuns, "LSQ_Model_Run_Number")[r]);
    modelIndex = If(modelSource == "NORMAL_BATCH" & !Is Missing(modelNumber) & modelNumber >= 1 & modelNumber <= {MAX_RUNS}, modelNumber + 1, 1);
    massEnabled = Uppercase(Char(Column(dtLSQMass, "Enabled")[(r - 1) * {MAX_LSQ_FRACTIONS} + 1]));
    LSQLoadControls(slot,
        If(Is Missing(path), "", Char(path)), If(Is Missing(w), 1, w),
        If(Is Missing(sheet), "", Char(sheet)), If(Is Missing(xcolumn), "", Char(xcolumn)),
        If(Is Missing(signal), "", Char(signal)),
        If(unit == "CV", 2, unit == "ML", 3, unit == "MIN", 4, unit == "S", 5, unit == "H", 6, unit == "INDEX", 7, 1),
        If(origin == "ELUTION_START", 2, 1), modelIndex, If(massEnabled == "ON", 2, 1),
        If(condFit == "ON", 2, 1), If(Is Missing(condColumn), "", Char(condColumn)));
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
    SaveLSQMassReference(lsqMassEditorRun);
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
    UpdateUI();
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
        "Editing LSQ run recipes (1–{MAX_LSQ_PROFILES}).",
        "Editing batch run recipes (1–30)."));
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

LSQRunLoadCVText = Function({{r}}, {{basis, volume, columnVol}},
    basis = Uppercase(Char(Column(dtLSQRuns, "Load_Amount_Basis")[r]));
    If(basis == "VOLUME_ML" | basis == "LOAD VOLUME [ML]",
        volume = Num(Column(dtLSQRuns, "Load_Volume_mL")[r]);
        columnVol = columnVolume << Get;
        If(Is Missing(volume) | Is Missing(columnVol) | columnVol <= 0,
            Return("—"),
            Return(Char(Round(volume / columnVol, 0.0001)))
        )
    );
    Return(LSQRunCellText(r, "Load_CV"));
);

UpdateLSQRunSummary = Function({{}}, {{slot, r, n, attached, massEnabled, nPLW, nElution, path, summary, modelChoice, modelNumber}},
    n = Num(lsqRunCount << Get Selected); attached = 0; massEnabled = 0;
    For(slot = 1, slot <= {MAX_LSQ_PROFILES}, slot++,
        r = lsqLoadedRuns[slot];
        If(r >= 1 & r <= {MAX_LSQ_PROFILES},
            nPLW = Num(Column(dtLSQRuns, "PLW_Count")[r]); If(Is Missing(nPLW), nPLW = 0);
            nElution = Num(Column(dtLSQRuns, "Elution_Count")[r]); If(Is Missing(nElution), nElution = 0);
            path = Trim(Char(LSQPathText(slot)));
            modelChoice = LSQModelRunSelected(slot);
            If(slot <= n & !Is Empty(path), attached++);
            If(slot <= n & LSQMassEnabledSelected(slot) == "On", massEnabled++);
            If(modelChoice == "Independent LSQ recipe",
                summary = "LSQ Run " || Char(r) || " — " || LSQRunCellText(r, "Run_Name") ||
                    ": load " || LSQRunLoadCVText(r) || " CV, " || Char(nPLW) || " washes, " ||
                    Char(nElution) || " elution steps; model: LSQ recipe.",
                modelNumber = RunNumberFromLabel(modelChoice);
                summary = "LSQ Run " || Char(r) || " — " || LSQRunCellText(r, "Run_Name") ||
                    "; model: batch " || modelChoice || " — " || Char(Column(dtBatchRuns, "Run_Name")[modelNumber])
            );
            summary = summary || If(Is Empty(path), "; no UV file.", "; UV file attached.");
            LSQSetRunSummary(slot, summary)
        )
    );
    lsqProfilesStatus << Set Text("Selected: " || Char(n) || " runs, " || Char(attached) || " UV files, " || Char(massEnabled) || " mass% fits.");
);

ComputePHSpanForRuns = Function({{runsTable, useSelectedLSQ}}, {{nRuns, slot, r, i, mn, mx, v, c, stepMode, chemControl, loadMaterialMode, loadMode, modelChoice, initialSource}},
    If(Contains(Char(modelBox << Get Selected), "CPA") == 0, Return(0));
    nRuns = If(useSelectedLSQ, Num(lsqRunCount << Get Selected), Floor(batchCountBox << Get));
    If(Is Missing(nRuns), nRuns = 1);
    nRuns = Max(1, Min(If(useSelectedLSQ, {MAX_LSQ_PROFILES}, {MAX_RUNS}), nRuns));
    mn = 1e99; mx = -1e99;
    For(slot = 1, slot <= nRuns, slot++,
        If(useSelectedLSQ,
            modelChoice = LSQModelRunSelected(slot);
            runsTable = If(modelChoice == "Independent LSQ recipe", dtLSQRuns, dtBatchRuns);
            r = If(modelChoice == "Independent LSQ recipe", LSQSelectedRun(slot), RunNumberFromLabel(modelChoice));
        ,
            runsTable = dtBatchRuns;
            r = slot;
        );
        initialSource = Uppercase(Char(Column(runsTable, "Equilibration_Source")[r]));
        If(initialSource == "BUFFER_A" | initialSource == "BUFFER_B" | initialSource == "DIRECT",
            v = ResolvedPHFromRow(runsTable, r, "Equilibration_Source", "Equilibration_pH");
            If(!Is Missing(v), mn = Min(mn, v); mx = Max(mx, v))
        );
        loadMaterialMode = Uppercase(Char(Column(runsTable, "Load_Material_Chemistry_Mode")[r]));
        loadMode = Uppercase(Char(Column(runsTable, "Load_Mode")[r]));
        chemControl = Uppercase(Char(Column(runsTable, "Load_Chemistry_Control")[r]));
        If(loadMaterialMode == "EXPLICIT",
            v = ResolvedPHFromRow(runsTable, r, "Load_Source", "Load_pH"); If(!Is Missing(v), mn = Min(mn, v); mx = Max(mx, v));
        ,
            If(loadMode != "STEP", v = ResolvedPHFromPercentBRow(runsTable, r, "Load_Start_Percent_B"); If(!Is Missing(v), mn = Min(mn, v); mx = Max(mx, v)));
            If(loadMode != "SET_POINT", v = ResolvedPHFromPercentBRow(runsTable, r, "Load_End_Percent_B"); If(!Is Missing(v), mn = Min(mn, v); mx = Max(mx, v)));
        );
        c = Num(Column(runsTable, "PLW_Count")[r]);
        For(i = 1, i <= c, i++,
            stepMode = Uppercase(Char(Column(runsTable, "PLW" || Char(i) || "_Mode")[r]));
            chemControl = Uppercase(Char(Column(runsTable, "PLW" || Char(i) || "_Chemistry_Control")[r]));
            If(chemControl == "BUFFER_B_PERCENT",
                If(stepMode != "STEP", v = ResolvedPHFromPercentBRow(runsTable, r, "PLW" || Char(i) || "_Start_Percent_B"); If(!Is Missing(v), mn = Min(mn, v); mx = Max(mx, v)));
                If(stepMode != "SET_POINT", v = ResolvedPHFromPercentBRow(runsTable, r, "PLW" || Char(i) || "_End_Percent_B"); If(!Is Missing(v), mn = Min(mn, v); mx = Max(mx, v)));
            ,
                If(stepMode != "STEP", v = ResolvedPHFromRow(runsTable, r, "PLW" || Char(i) || "_Start_Source", "PLW" || Char(i) || "_Start_pH"); If(!Is Missing(v), mn = Min(mn, v); mx = Max(mx, v)));
                If(stepMode != "SET_POINT", v = ResolvedPHFromRow(runsTable, r, "PLW" || Char(i) || "_End_Source", "PLW" || Char(i) || "_End_pH"); If(!Is Missing(v), mn = Min(mn, v); mx = Max(mx, v)));
            );
        );
        c = Num(Column(runsTable, "Elution_Count")[r]);
        For(i = 1, i <= c, i++,
            stepMode = Uppercase(Char(Column(runsTable, "Elution" || Char(i) || "_Mode")[r]));
            chemControl = Uppercase(Char(Column(runsTable, "Elution" || Char(i) || "_Chemistry_Control")[r]));
            If(chemControl == "BUFFER_B_PERCENT",
                If(stepMode != "STEP", v = ResolvedPHFromPercentBRow(runsTable, r, "Elution" || Char(i) || "_Start_Percent_B"); If(!Is Missing(v), mn = Min(mn, v); mx = Max(mx, v)));
                If(stepMode != "SET_POINT", v = ResolvedPHFromPercentBRow(runsTable, r, "Elution" || Char(i) || "_End_Percent_B"); If(!Is Missing(v), mn = Min(mn, v); mx = Max(mx, v)));
            ,
                If(stepMode != "STEP", v = ResolvedPHFromRow(runsTable, r, "Elution" || Char(i) || "_Start_Source", "Elution" || Char(i) || "_Start_pH"); If(!Is Missing(v), mn = Min(mn, v); mx = Max(mx, v)));
                If(stepMode != "SET_POINT", v = ResolvedPHFromRow(runsTable, r, "Elution" || Char(i) || "_End_Source", "Elution" || Char(i) || "_End_pH"); If(!Is Missing(v), mn = Min(mn, v); mx = Max(mx, v)));
            );
        );
    );
    If(mn > mx, 0, mx - mn)
);

UpdateUI = Function({{}}, {{modelLabel, impurityCount, isTDM, isEDM, isCPA, isLangmuir, isWang, isHIC, plwCount, elutionCount, stepMode, stepControl, startPct, endPct, startSalt, endSalt, activeStart, activeEnd, batchPHSpan, fitPHSpan, reqText, lsqUsesComposition, lsqRefCount, loadSaltValue, loadSourceLabel, loadPHValue, loadCondValue, stationaryFraction, nominalCapacity, targetDose, speciesName}},
    modelLabel = Char(modelBox << Get Selected);
    SyncLangmuirDerived();
    impurityCount = Num(impurityCountBox << Get Selected); If(Is Missing(impurityCount), impurityCount = 0);
    isTDM = Contains(modelLabel, "TDM") > 0; isEDM = Contains(modelLabel, "EDM") > 0;
    isCPA = Contains(modelLabel, "CPA") > 0; isLangmuir = Contains(modelLabel, "Competitive Langmuir") > 0; isWang = Contains(modelLabel, "Modified Wang") > 0;
    If(isCPA,
        If(Uppercase(Char(chromatographyMode << Get Selected)) != "ION_EXCHANGE", chromatographyMode << Set(2)),
        If(Uppercase(Char(chromatographyMode << Get Selected)) != "HIC", chromatographyMode << Set(1))
    );
    chromatographyMode << Enable(0);
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
    fitPHSpan = ComputePHSpanForRuns(dtLSQRuns, 1);
    batchPHSpan = If(isLSQSetup, fitPHSpan, ComputePHSpanForRuns(dtBatchRuns, 0));
    fitPHSpan = Max(fitPHSpan, batchPHSpan);
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
    loadSaltValue = If(Uppercase(Char(loadMaterialMode << Get Selected)) == "EXPLICIT",
        If(loadSourceLabel == "BUFFER_A", ResolvedBufferSalt(bufferACond, bufferASalt, bufferASaltMode),
           loadSourceLabel == "BUFFER_B", ResolvedBufferSalt(bufferBCond, bufferBSalt, bufferBSaltMode),
           ResolvedBufferSalt(loadCond, loadSalt, loadSaltMode)),
        BlendBufferSalt(loadStartB << Get));
    saltFactorStatus << Set Text("Current run: Buffer A " || Char(ResolvedBufferSalt(bufferACond, bufferASalt, bufferASaltMode)) ||
        " M (" || Char(bufferASaltMode << Get Selected) || "); Buffer B " ||
        Char(ResolvedBufferSalt(bufferBCond, bufferBSalt, bufferBSaltMode)) || " M (" ||
        Char(bufferBSaltMode << Get Selected) || "); load start " || Char(loadSaltValue) || " M.");
    loadChemControlRow << Visibility("Collapse");
    loadMaterialModeRow << Visibility("Collapse");
    equilibrationDirectPanel << Visibility(If(Uppercase(Char(equilibrationSource << Get Selected)) == "DIRECT", "Visible", "Collapse"));
    loadChemPanel << Visibility("Collapse");
    If(Uppercase(Char(loadMaterialMode << Get Selected)) == "EXPLICIT",
        loadPHValue = If(loadSourceLabel == "BUFFER_A", bufferAPH << Get,
            loadSourceLabel == "BUFFER_B", bufferBPH << Get, loadPH << Get);
        loadCondValue = If(loadSourceLabel == "BUFFER_A", ResolvedBufferCond(bufferACond, bufferASalt, bufferASaltMode),
            loadSourceLabel == "BUFFER_B", ResolvedBufferCond(bufferBCond, bufferBSalt, bufferBSaltMode),
            ResolvedBufferCond(loadCond, loadSalt, loadSaltMode));
        loadMaterialStatus << Set Text("Saved explicit load chemistry: pH " || Char(loadPHValue) ||
            ", conductivity " || Char(loadCondValue) || " mS/cm. Edit load %B to use the Buffer A/B blend.");
    ,
        loadMaterialStatus << Set Text("Load chemistry from Buffer A/B: start pH " || Char(BlendBufferPH(loadStartB << Get)) ||
            ", conductivity " || Char(BlendBufferCond(loadStartB << Get)) || " mS/cm; end pH " ||
            Char(BlendBufferPH(loadEndB << Get)) || ", conductivity " || Char(BlendBufferCond(loadEndB << Get)) || " mS/cm.");
    );
    tdmPanel << Visibility(If(isTDM, "Visible", "Collapse")); edmPanel << Visibility(If(isEDM, "Visible", "Collapse"));
    cpaSystemPanel << Visibility(If(isCPA & !isHIC, "Visible", "Collapse"));
    wangSharedPanel << Visibility(If(isWang, "Visible", "Collapse"));
    wangModeHelp << Visibility(If(isWang, "Visible", "Collapse"));
    legacyModeHelp << Visibility(If(isWang, "Collapse", "Visible"));
    condConversionRow << Visibility(If(isCPA | isLangmuir | isWang, "Visible", "Collapse"));
    tdmSaltRow << Visibility("Collapse");
    {''.join(component_vis)}
    {''.join(process_vis)}
    speciesStatus << Set Text("Target protein + " || Char(impurityCount) || If(impurityCount == 1, " impurity", " impurities"));
    reqText = "";
    If(modelLabel == {_q(MODEL_LABELS[MODEL_EDM_LANGMUIR])}, reqText = {req_edm_l});
    If(isWang, reqText = {req_wang});
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
    {''.join(mass_header_updates)}
    {''.join(ref_vis)}
);

SendSharedModel = Function({{}}, {{}},
    {shared_sends}
);

SendAndRun = Function({{actionText}}, {{rc, statusValue}},
    If(isLSQSetup & (actionText == "RUN_SELECTED" | actionText == "RUN_BATCH" | actionText == "CHECK_RESULT_CURRENT"),
        statusText << Set Text("Switch to Normal batch setup before running a production chromatogram."); Return()
    );
    If(actionText == "FIT_LSQ" | actionText == "PREVIEW_LSQ_ALIGNMENT" | actionText == "ATTACH_LSQ_CSV", If(!SaveAllLSQReferences(), Return()));
    SaveCurrentRun();
    UpdateLSQRunSummary(); SendSharedModel();
    Python Send(actionText, Python Name("tdm_ui_action"));
    Python Send(If(isLSQSetup, 1, currentRun), Python Name("tdm_ui_run_number"));
    Python Send(Floor(batchCountBox << Get), Python Name("tdm_ui_batch_count"));
    Python Send(lsqModeBox << Get Selected, Python Name("tdm_ui_lsq_mode"));
    Python Send(lsqRunCount << Get Selected, Python Name("tdm_ui_lsq_run_count"));
    {''.join(lsq_selection_sends)}
    Python Send(If(actionText == "PREVIEW_LSQ_ALIGNMENT", lsqPreviewRun, 0), Python Name("tdm_ui_lsq_preview_run"));
    Python Send(LSQPathText(lsqAttachSlot), Python Name("tdm_ui_lsq_csv_path"));
    Python Send(If(actionText == "ATTACH_LSQ_CSV", lsqLoadedRuns[lsqAttachSlot], RunNumberFromLabel(lsqTargetRunSelector << Get Selected)), Python Name("tdm_ui_lsq_target_run"));
    Python Send(projectDir, Python Name("tdm_project_dir"));
    Python Send(projectDir, Python Name("tdm_project_dir_posix"));
    If(actionText == "FIT_LSQ",
        lsqStatusText << Set Text("Refinement running. Progress: least_squares_objective_history.csv. JMP will respond when the fit finishes.");
        Wait(0.01)
    );
    If(actionText == "PREVIEW_LSQ_ALIGNMENT",
        lsqStatusText << Set Text("Building alignment preview for the selected runs.");
        Wait(0.01)
    );
    If(actionText == "RUN_SELECTED",
        statusText << Set Text("Running prediction for the selected run. JMP will respond when it finishes.");
        Wait(0.01)
    );
    If(actionText == "CHECK_RESULT_CURRENT", Save Text File(statusFile, "CHECK_RESULT_CURRENT pending"));
    rc = Python Submit File(bridgeFile);
    If(File Exists(statusFile), statusValue = Load Text File(statusFile), statusValue = "No status file was produced. Check View > Log.");
    statusText << Set Text(statusValue); lsqStatusText << Set Text(statusValue);
    If(((actionText == "FIT_LSQ" & Contains(statusValue, "Accepted/applied to shared batch parameters: True") > 0) |
        Contains(statusValue, "FITTED_PARAMETERS_RESTORED:") > 0) & File Exists(lsqApplyFile),
        Eval(Parse(Load Text File(lsqApplyFile)))
    );
    If(actionText == "PREVIEW_LSQ_ALIGNMENT" & Contains(statusValue, "LSQ_ALIGNMENT_READY:") > 0 & File Exists(lsqPreviewReport), Web(lsqPreviewReport));
    UpdateUI();
);

PreviewLSQSlot = Function({{slot}}, {{}},
    lsqPreviewRun = LSQSelectedRun(slot);
    SendAndRun("PREVIEW_LSQ_ALIGNMENT");
    lsqPreviewRun = 0;
);

OpenAlignmentPreview = Function({{}}, {{}},
    lsqPreviewRun = 0;
    SendAndRun("PREVIEW_LSQ_ALIGNMENT");
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

modelWindow = New Window("Chromatography Model — {RESULT_GUARD_BUILD}",
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
        scopeStatus = Text Box("Editing batch run recipes (1–30).", << Set Wrap(780)),
        mainTabs = Tab Box(
            "Process Setup",
            V Scroll Box(Size(850, 690), V List Box(
                bufferSection = V List Box(
                    Text Box("Buffers for this run", << Set Font Style("Bold")),
                    Text Box("Buffer A = 0% B; Buffer B = 100% B. Choose conductivity or salt molarity as the salt input.", << Set Wrap(720)),
                    {buffer_a_panel},
                    {buffer_b_panel}
                ),
                Spacer Box(Size(1,8)),
                Outline Box("Run & load",
                    V List Box(
                        {_trow("Run name", "runNameBox", "")},
        {_nrow("Load flow [mL/min]", "loadFlow", None, 290, "If(!isLoading, SaveCurrentRun(); UpdateUI())")},
                        {_nrow("Feed concentration [mg/mL]", "feedConcentration", None, 290, "If(!isLoading, SyncLoadAmount())")},
                        {_nrow("Load capacity [g/L resin]", "loadDensity", 10.0, 290, "If(!isLoading, SyncLoadAmount())")},
                        H List Box(Text Box("Calculated load volume [CV]", << Set Width(290)), loadCV = Number Edit Box(., 9, << Set Width(92), << Enable(0))),
                        H List Box(Text Box("Calculated load volume [mL]", << Set Width(290)), loadVolumeML = Number Edit Box(., 9, << Set Width(92), << Enable(0))),
                        loadSaturationStatus = Text Box("", << Set Wrap(680)),
                        H List Box(Text Box("Load mode", << Set Width(290)), loadMode = Combo Box({{"SET_POINT", "LINEAR", "STEP"}}, << Set Width(125), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); UpdateUI()))))),
                        H List Box(Text Box("Load start / set-point Buffer B [%]", << Set Width(290)), loadStartB = Number Edit Box(0, 9, << Set Width(92), << Set Function(Function({{this}}, If(!isLoading, loadMaterialMode << Set(1); loadChemControl << Set(1); SaveCurrentRun(); UpdateUI()))))),
                        H List Box(Text Box("Load end Buffer B [%]", << Set Width(290)), loadEndB = Number Edit Box(0, 9, << Set Width(92), << Set Function(Function({{this}}, If(!isLoading, loadMaterialMode << Set(1); loadChemControl << Set(1); SaveCurrentRun(); UpdateUI()))))),
                        loadMaterialStatus = Text Box("", << Set Wrap(700)),
                        loadChemControlRow = H List Box(Text Box("Load chemistry control", << Set Width(290)), loadChemControl = Combo Box({{"BUFFER_B_PERCENT", "ENDPOINT_CHEMISTRY"}}, << Set Width(190), << Set Function(Function({{this}}, If(!isLoading, If(Uppercase(Char(this << Get Selected)) == "ENDPOINT_CHEMISTRY", loadMaterialMode << Set(2), loadMaterialMode << Set(1)); SaveCurrentRun(); UpdateUI()))))),
                        loadMaterialModeRow = H List Box(Text Box("Load chemistry mode", << Set Width(290)), loadMaterialMode = Combo Box({{"BUFFER_RECIPE", "EXPLICIT"}}, << Set Width(190), << Set Function(Function({{this}}, If(!isLoading, If(Uppercase(Char(this << Get Selected)) == "BUFFER_RECIPE", loadChemControl << Set(1), loadChemControl << Set(2)); If(Uppercase(Char(this << Get Selected)) == "EXPLICIT", loadSource << Set(1)); SaveCurrentRun(); UpdateUI()))))),
                        Text Box("Load volume is calculated from capacity, feed concentration and column volume.", << Set Wrap(700)),
                        {_nrow("Number of time steps", "timeStepsBox", 700)},
                        {_nrow("Number of axial positions", "axialPositionsBox", 20)},
                        Text Box("Time steps and axial positions control this run's resolution.", << Set Wrap(700)),
                        loadChemPanel = Outline Box("Load material pH and salt", V List Box(
                            Text Box("Editing pH, conductivity or salt selects explicit load chemistry.", << Set Wrap(700)),
                            H List Box(Text Box("Load chemistry source", << Set Width(220)), loadSource = Combo Box({{"DIRECT", "BUFFER_A", "BUFFER_B"}}, << Set Width(125), << Set Function(Function({{this}}, If(!isLoading, loadMaterialMode << Set(2); loadChemControl << Set(2); ApplyChemSource(loadSource, loadPH, loadCond); SaveCurrentRun(); UpdateUI()))))),
                            loadPHRow = H List Box(loadPHLabel = Text Box("Load material pH", << Set Width(220)), loadPH = Number Edit Box(6, 9, << Set Width(92), << Set Function(Function({{this}}, If(!isLoading, loadMaterialMode << Set(2); loadChemControl << Set(2); loadSource << Set(1); SaveCurrentRun(); UpdateUI()))))),
                            {_salt_input_controls("load")},
                            loadCondRow = H List Box(loadCondLabel = Text Box("Load material conductivity [mS/cm]", << Set Width(220)), loadCond = Number Edit Box(5, 9, << Set Width(92), << Set Function(Function({{this}}, If(!isLoading, loadMaterialMode << Set(2); loadChemControl << Set(2); loadSource << Set(1); SaveCurrentRun(); UpdateUI())))))
                        )),
                        Outline Box("Initial column equilibration (before load)", V List Box(
                            Text Box("Initial column chemistry before load; this does not add CV.", << Set Wrap(700)),
                            H List Box(Text Box("Initial equilibration source", << Set Width(290)), equilibrationSource = Combo Box({{"MATCH_LOAD", "BUFFER_A", "BUFFER_B", "DIRECT"}}, << Set(1), << Set Width(175), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); UpdateUI()))))),
                            equilibrationDirectPanel = V List Box(
                                H List Box(Text Box("Initial column pH", << Set Width(290)), equilibrationPH = Number Edit Box(., 9, << Set Width(92), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); UpdateUI()))))),
                                H List Box(Text Box("Initial salt input", << Set Width(290)), equilibrationSaltMode = Combo Box({{"CONDUCTIVITY", "SALT_M"}}, << Set(1), << Set Width(175), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); UpdateUI()))))),
                                H List Box(Text Box("Initial salt concentration [M]", << Set Width(290)), equilibrationSalt = Number Edit Box(., 9, << Set Width(92), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); UpdateUI()))))),
                                H List Box(Text Box("Initial conductivity [mS/cm]", << Set Width(290)), equilibrationCond = Number Edit Box(., 9, << Set Width(92), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); UpdateUI())))))
                            )
                        ))
                    )
                ),
                Spacer Box(Size(1,8)),
                Outline Box("Standard process sequence",
                    V List Box(
                        H List Box(Text Box("Post-load washes [0-5]", << Set Width(220)), plwCountBox = Combo Box({count_items}, << Set Width(75), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); UpdateUI()))))),
                        H List Box(Text Box("Elution steps [0-5]", << Set Width(220)), elutionCountBox = Combo Box({count_items}, << Set Width(75), << Set Function(Function({{this}}, If(!isLoading, SaveCurrentRun(); UpdateUI()))))),
                        Text Box("SET_POINT holds Start %B; LINEAR ramps to End; STEP switches to End.", << Set Wrap(740))
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
                Outline Box("Required parameters", << Close(1), requiredList = Text Box("", << Set Wrap(760), << Set Height(250))),
                Outline Box("Column geometry — direct inputs", V List Box(
                    {_nrow("Column volume V_col [mL]", "columnVolume", c.get("volume_mL"), 360, "If(!isLoading, SyncLoadAmount())")},
                    {_nrow("Column length L_C [mm]", "columnLength", c.get("length_mm"), 360)}
                )),
                Outline Box("HIC / ion exchange and system response", V List Box(
                    H List Box(Text Box("Chromatography mode", << Set Width(360)), chromatographyMode = Combo Box({{"HIC", "ION_EXCHANGE"}}, << Set({1 if is_hic(cfg) else 2}), << Set Function(Function({{this}}, UpdateUI())))),
                    legacyModeHelp = Text Box("For a descending-salt elution, set Buffer B as the high-salt buffer.", << Set Wrap(740)),
                    wangModeHelp = Text Box("TDM–Wang uses HIC with Buffer B as the high-salt buffer.", << Set Wrap(740)),
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
                    Text Box("This factor converts conductivity inputs to salt molarity; direct salt inputs use their entered value.", << Set Wrap(720))
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
                wangSharedPanel = Outline Box("Modified Wang shared parameters", V List Box(
                    {wang_q0_row},
                    {wang_eta_row}
                )),
                Text Box("LSQ changes only unlocked model parameters. Run recipes and column geometry stay fixed.", << Set Wrap(760)),
                Text Box("Fit ±%: choose Limited to restrict an unlocked parameter around its starting value. Default uses the model bounds.", << Set Wrap(760)),
                {', '.join(component_bodies)}
            )),
            "Least-Squares Refinement",
            V Scroll Box(Size(850, 690), V List Box(
                H List Box(Text Box("Number of runs to fit [1–{MAX_LSQ_PROFILES}]", << Set Width(240)), lsqRunCount = Combo Box({lsq_run_count_items}, << Set({len(saved_selection)}), << Set Width(100), << Set Function(Function({{this}}, If(!isLoading, UpdateUI(); UpdateLSQRunSummary()))))),
                lsqProfilesStatus = Text Box("", << Set Wrap(740)),
                Text Box("For each run, attach a measured UV file and choose the model run to compare. Conductivity and fraction mass% are optional.", << Set Wrap(760)),
                {', '.join(lsq_run_sections)},
                Outline Box("Fit settings", V List Box(
                    H List Box(Text Box("Fit data", << Set Width(200)), lsqModeBox = Combo Box({{"UV chromatogram", "UV + fraction mass%"}}, << Set Width(220), << Set Function(Function({{this}}, UpdateUI())))),
                    H List Box(Button Box("Lock all parameters", SetAllFitLocks("Locked")), Button Box("Unlock all parameters", SetAllFitLocks("Unlocked"))),
                    Text Box("Set parameter locks on Model Parameters. The safety cap counts model evaluations; progress is saved to least_squares_objective_history.csv.", << Set Wrap(740)),
                    {_nrow("Safety cap: model evaluations [2–10000]", "lsqMaxEvaluations", cfg['least_squares'].get('max_nfev', 120), 320)},
                    {_nrow("TDM–CPA trial timeout [minutes; 0 = off]", "lsqMaxTrialMinutes", cfg['least_squares'].get('max_trial_minutes', 30), 320)},
                    Outline Box("TDM–CPA fit acceptance", << Close(1), V List Box(
                        H List Box(Text Box("Physical check", << Set Width(240)), lsqPhysicalAcceptance = Combo Box({{"Require bind + elute", "Allow flow-through fit"}}, << Set({1 if cfg['least_squares']['physical_acceptance_mode']=='REQUIRE_BIND_ELUTE' else 2}), << Set Width(250))),
                        {_nrow("Minimum target bound after load [%]", "lsqMinBound", cfg['least_squares']['minimum_bound_percent_after_load'], 240)},
                        {_nrow("Minimum elution recovery of retained target [%]", "lsqMinRecovery", cfg['least_squares']['minimum_elution_recovery_percent'], 240)},
                        {_nrow("Minimum objective improvement [%]", "lsqMinImprovement", cfg['least_squares']['minimum_objective_improvement_percent'], 240)}
                    )),
                    Outline Box("Optional objective and stopping settings", << Close(1), V List Box(
                        H List Box(Text Box("Objective", << Set Width(240)), lsqObjective = Combo Box({{"RAW_SSE", "NORMALIZED_MSE", "WEIGHTED_RMSE"}}, << Set({1 if cfg['least_squares']['objective']=='RAW_SSE' else 2 if cfg['least_squares']['objective']=='NORMALIZED_MSE' else 3}), << Set Width(220))),
                        H List Box(Text Box("Baseline policy", << Set Width(240)), lsqBaselineMode = Combo Box({{"NONE", "INITIAL_MEDIAN"}}, << Set({1 if cfg['least_squares']['baseline_mode']=='NONE' else 2}), << Set Width(220))),
                        {_nrow("Fixed detector baseline [mAU]", "lsqBaseline", cfg['least_squares']['detector_baseline'], 240)}
                    ))
                )),
                H List Box(
                    Button Box("Preview UV, conductivity and mass%", lsqPreviewRun = 0; SendAndRun("PREVIEW_LSQ_ALIGNMENT")),
                    Button Box("Run least-squares refinement", SendAndRun("FIT_LSQ"))
                ),
                H List Box(Button Box("Open alignment preview", OpenAlignmentPreview()), Button Box("Open fit report", OpenFitReport())),
                lsqStatusText = Text Box("", << Set Wrap(760), << Set Height(130)),
                lsqCompositionPanel = Outline Box("Species mass% reference points", V List Box(
                    lsqTargetRunRow = H List Box(Text Box("Edit mass% for LSQ run", << Set Width(200)), lsqTargetRunSelector = Combo Box({run_choices}, << Set Width(140), << Set Function(Function({{this}}, ChangeLSQMassEditorRun(RunNumberFromLabel(this << Get Selected))))), Text Box("Turn on mass% for this run above to fit it.", << Set Wrap(390))),
                    lsqMassEditorStatus = Text Box("", << Set Wrap(740)),
                    Text Box("Enter fraction start and end from run start. Fit uses each midpoint; blank species cells are skipped, while 0 is measured.", << Set Wrap(740)),
                    H List Box(Text Box("Fraction rows [1-30]", << Set Width(240)), lsqReferenceCountBox = Combo Box({ref_items}, << Set Width(75), << Set Function(Function({{this}}, UpdateUI()))), Text Box("Fraction axis"), lsqFractionUnit = Combo Box({{"CV", "mL"}}, << Set(1), << Set Width(90))),
                    {ref_table}
                ))
            )),
            "Run / Results",
            V Scroll Box(Size(850, 690), V List Box(
                Text Box("If a result is stale, rerun the selected model.", << Set Wrap(760)),
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
lsqLoadedRuns = {{{', '.join('0' for _ in range(MAX_LSQ_PROFILES))}}};
lsqAttachSlot = 1;
isLoading = 1;
For(lsqInitSlot = 1, lsqInitSlot <= {MAX_LSQ_PROFILES}, lsqInitSlot++,
    LoadLSQReference(lsqInitSlot, LSQSelectedRun(lsqInitSlot))
);
isLoading = 0;
LoadRun(1);
lsqTargetRunSelector << Set(1);
LoadLSQMassReference(1);
UpdateUI();
UpdateLSQRunSummary();
'''
    final_lock_sends = []
    final_range_sends = []
    for offset in range(0, len(fit_lock_vars), 20):
        lock_parts = []
        range_parts = []
        for lock_var, path in fit_lock_vars[offset:offset + 20]:
            var = lock_var.removesuffix("FitLock")
            lock_parts.extend((_q(path + "="), f'{lock_var} << Get Selected', _q(";")))
            range_parts.extend((_q(path + "|"), f'{var}FitRangeMode << Get Selected',
                                _q("|"), f'Char({var}FitRangePercent << Get)', _q(";")))
        final_lock_sends.append('Concat(' + ", ".join(lock_parts) + ')')
        final_range_sends.append('Concat(' + ", ".join(range_parts) + ')')
    lock_send_lines = [f'Python Send({len(final_lock_sends)}, Python Name("tdm_ui_lsq_parameter_lock_chunk_count"));']
    lock_send_lines.extend(
        f'Python Send({chunk}, Python Name("tdm_ui_lsq_parameter_lock_states_{i}"));'
        for i, chunk in enumerate(final_lock_sends, 1)
    )
    range_send_lines = [f'Python Send({len(final_range_sends)}, Python Name("tdm_ui_lsq_parameter_fit_range_chunk_count"));']
    range_send_lines.extend(
        f'Python Send({chunk}, Python Name("tdm_ui_lsq_parameter_fit_ranges_{i}"));'
        for i, chunk in enumerate(final_range_sends, 1)
    )
    jsl = jsl.replace("__LSQ_LOCK_SENDS__", "\n    ".join(lock_send_lines))
    jsl = jsl.replace("__LSQ_FIT_RANGE_SENDS__", "\n    ".join(range_send_lines))
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
        "UV chromatogram", "UV + fraction mass%",
        "Feed composition mass% of total protein", "UV → protein response factor",
        "Conductivity → salt factor", "Number of time steps", "Number of axial positions", "Number Edit Box", "tdm_batch_runs.csv",
        "Chemistry control", "Start / set-point Buffer B [%]", "End Buffer B [%]",
        "Buffer A (0% B)", "Buffer B (100% B)",
        "Saturation capacity qmax,i [g/L stationary phase]", "SyncLangmuirDerived",
        "AttachLSQCSV = Function", "Pick File(\"Select raw chromatogram CSV/Excel\"",
        "LSQSetPathText(slot, csvPath)", "!File Exists(csvPath)",
        "LSQ_Chromatogram_CSV", "ChangeLSQMassEditorRun", "RunNumberFromLabel(this << Get Selected)",
        "SaveLSQMassReference", "LoadLSQMassReference", "tdm_lsq_fraction_mass.csv",
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
    expected_models = [MODEL_LABELS[model] for model in (
        MODEL_EDM_LANGMUIR, MODEL_TDM_LANGMUIR, MODEL_EDM_CPA,
        MODEL_TDM_CPA, MODEL_TDM_WANG,
    )]
    if sorted(model_choices) != sorted(expected_models):
        raise ValueError(f"Model selector contains unsupported choices: {model_choices}")
    mode_combo = re.search(r'chromatographyMode\s*=\s*Combo Box\(\s*\{([^{}]*)\}', jsl)
    if mode_combo is None or json.loads("[" + mode_combo.group(1) + "]") != ["HIC", "ION_EXCHANGE"]:
        raise ValueError("Chromatography mode selector must retain HIC and ion exchange.")
    forbidden_labels = ("Model template", "Resin template", "Mechanistic model templates")
    for label in re.findall(r'\b(?:Text Box|Outline Box|Button Box)\(\s*"([^"]*)"', jsl):
        for forbidden in forbidden_labels:
            if forbidden.casefold() in label.casefold():
                raise ValueError(f"Removed template construct leaked into UI: {forbidden}")
    selected_text_controls = [
        "modelBox", "impurityCountBox", "plwCountBox", "elutionCountBox", "loadSource", "loadMaterialMode",
        "equilibrationSource", "equilibrationSaltMode",
        "lsqModeBox", "lsqReferenceCountBox", "lsqFractionUnit", "lsqTargetRunSelector",
    ]
    for control in selected_text_controls:
        if f"{control} << Get Selected" not in jsl:
            raise ValueError(f"Combo Box {control} must be read as selected text.")
        if f"{control} << Get()" in jsl:
            raise ValueError(f"Combo Box {control} is incorrectly read as an item index.")
    if jsl.count("RunNumberFromLabel(this << Get Selected)") != 2 + MAX_LSQ_PROFILES:
        raise ValueError("The batch, mass editor, and all LSQ run selectors must parse their selected Run label.")
    if "dtRuns << Save;" in jsl:
        raise ValueError("Run profiles must be persisted to the explicit batch CSV path.")
    if "Python Send(Floor(batchCountBox << Get), Python Name(\"tdm_ui_batch_count\"))" not in jsl:
        raise ValueError("Numeric batch count must continue to use Number Edit Box Get().")
    for field, control, getter in (
        ("Equilibration_Source", "equilibrationSource", "Get Selected"),
        ("Equilibration_pH", "equilibrationPH", "Get"),
        ("Equilibration_Salt_M", "equilibrationSalt", "Get"),
        ("Equilibration_Salt_Input_Mode", "equilibrationSaltMode", "Get Selected"),
        ("Equilibration_Conductivity_mS_cm", "equilibrationCond", "Get"),
    ):
        if f'Column(dtRuns, "{field}")[currentRun] = {control} << {getter};' not in jsl:
            raise ValueError(f"Initial equilibration {field} must be saved from the current run control.")
        if f'{control} << Set(' not in jsl:
            raise ValueError(f"Initial equilibration {field} must load with the current run.")
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
        if f'Button Box("Preview this run", PreviewLSQSlot({slot}))' not in jsl:
            raise ValueError(f"LSQ run section {slot} must preview its own selected run")
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
