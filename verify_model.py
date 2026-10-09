from __future__ import annotations

import copy
import csv
import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tdm_minimal_io import (
    MAX_RUNS, MAX_PLW_STEPS, MAX_ELUTION_STEPS, MAX_IMPURITIES, MAX_BUFFER_IONS,
    MODEL_EDM_LANGMUIR, MODEL_TDM_LANGMUIR, MODEL_EDM_CPA, MODEL_TDM_CPA,
    MODEL_LABELS, blank_config, uses_tdm, uses_cpa, uses_langmuir,
    normalize_config,
    validate_config, validate_mechanistic_config, validate_operating_conditions,
    required_global_fields, required_component_fields, required_parameter_paths,
    sanitize_config_for_selected_model, batch_parameter_fingerprint,
    apply_batch_shared_parameters, process_pH_span, load_batch_rows, buffer_ionic_strength_M, buffer_salt_concentration_M, buffer_blend_chemistry, resolve_process_chemistry,
    batch_row_to_run_config, default_batch_row, save_batch_rows, save_config, BATCH_RUNS_CSV,
)
from tdm_minimal_model import simulate, write_outputs, run_batch_and_write, _build_program, _trapezoidal_integral, _langmuir_q_jac, _langmuir_c_q_from_total, check_result_freshness
from tdm_jmp import build_jsl, _validate_jsl
import tdm_bridge
import tdm_least_squares as lsq
from tdm_least_squares import (
    load_chromatogram_reference, validate_composition_references,
    _objective_parts, MODE_CHROMATOGRAM, MODE_CHROMATOGRAM_AND_COMPOSITION,
    fit_parameter_specs,
)


def make_config(model: str, n_imp: int = 1, *, multistep: bool = True) -> dict:
    c = blank_config()
    c['model'] = model
    c['impurity_count'] = n_imp
    c['column'].update(volume_mL=1.0, length_mm=50.0, flow_mL_min=1.0)
    c['feed'].update(total_concentration_mg_mL=2.0, load_density_mg_mL_resin=0.4)
    c['numerics'].update(time_steps=180, axial_positions=20)
    c['conversion'].update(uv_to_protein_mAU_L_g=200.0, conductivity_to_salt_M_per_mS_cm=0.01)
    p = c['process']
    p.update(load_pH=6.0, load_conductivity_mS_cm=5.0, plw_count=(2 if multistep else 1), elution_count=(3 if multistep else 1))
    p['buffer_A'].update(name='Buffer A', pH=6.0, salt_concentration_M=0.05, conductivity_mS_cm=5.0)
    p['buffer_B'].update(name='Buffer B', pH=6.0, salt_concentration_M=0.50, conductivity_mS_cm=50.0)
    for buf in (p['buffer_A'], p['buffer_B']):
        for ion in buf['ions']:
            ion.update(species='', concentration_M=None, valency=None)
    p['plw_steps'][0].update(mode='SET_POINT', CV=0.06, flow_mL_min=1.1, chemistry_control='BUFFER_B_PERCENT', start_percent_B=0.0, end_percent_B=0.0, start_pH=6.0, end_pH=11.0, start_conductivity_mS_cm=5.0, end_conductivity_mS_cm=90.0)
    p['plw_steps'][1].update(mode='STEP', CV=0.05, flow_mL_min=0.9, chemistry_control='BUFFER_B_PERCENT', start_percent_B=0.0, end_percent_B=20.0, start_pH=2.0, end_pH=6.0, start_conductivity_mS_cm=90.0, end_conductivity_mS_cm=8.0)
    p['elution_steps'][0].update(mode='LINEAR', CV=0.18, flow_mL_min=0.8, chemistry_control='BUFFER_B_PERCENT', start_percent_B=0.0, end_percent_B=70.0, start_pH=6.0, end_pH=6.0, start_conductivity_mS_cm=8.0, end_conductivity_mS_cm=35.0)
    p['elution_steps'][1].update(mode='STEP', CV=0.10, flow_mL_min=1.2, chemistry_control='BUFFER_B_PERCENT', start_percent_B=70.0, end_percent_B=100.0, start_pH=1.0, end_pH=6.0, start_conductivity_mS_cm=1.0, end_conductivity_mS_cm=50.0)
    p['elution_steps'][2].update(mode='SET_POINT', CV=0.08, flow_mL_min=1.0, chemistry_control='BUFFER_B_PERCENT', start_percent_B=100.0, end_percent_B=100.0, start_pH=6.0, end_pH=12.0, start_conductivity_mS_cm=55.0, end_conductivity_mS_cm=100.0)

    n = n_imp + 1
    fracs = [100.0/n] * n
    fracs[-1] = 100.0 - sum(fracs[:-1])
    for i, row in enumerate(c['components'][:n]):
        row['name'] = 'Target protein' if i == 0 else f'Impurity {i}'
        row['mass_percent'] = fracs[i]

    if uses_tdm(model):
        c['tdm'].update(bead_radius_um=25.0, void_fraction=0.42, particle_porosity=0.56)
        for i, row in enumerate(c['components'][:n]):
            row.update(D_ax_mm2_s=1.15 + 0.05*i, k_eff_um_s=8.0 + 0.1*i,
                       accessible_particle_porosity=0.41 - 0.01*i)
    else:
        c['edm']['total_bed_porosity'] = 0.75
        for i, row in enumerate(c['components'][:n]):
            row['D_app_mm2_s'] = 0.067 + 0.002*i

    if uses_langmuir(model):
        for i, row in enumerate(c['components'][:n]):
            h = 1.5 + 0.2*i
            b = 0.005 + 0.0005*i
            row.update(qmax_g_L=h/b, H=h, b_L_g=b)
    else:
        c['cpa'].update(ligand_surface_density_umol_m2=2.89,
                        system_specific_adsorption_parameter=0.03)
        if model == MODEL_TDM_CPA:
            c['tdm']['salt_axial_dispersion_mm2_s'] = 0.17
        for i, row in enumerate(c['components'][:n]):
            row.update(diameter_nm=11.0 + 0.4*i, As_m_inv=2.2e8 - 7e6*i,
                       Z_ref=40.0 + 3*i, delta_ref=0.012 + 0.001*i)
            if model == MODEL_TDM_CPA:
                row['kkin_star_s'] = 1e5
    return c


def assert_close(a, b, tol=1e-9):
    assert abs(float(a)-float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b))), (a,b)


def verify_button_paths() -> None:
    print('[23/23] execute JMP bridge routes for save, run, freshness, batch, CSV attach and fit')
    with tempfile.TemporaryDirectory() as td:
        project = Path(td) / 'model'
        project.mkdir()
        for filename in ('tdm_bridge.py', 'tdm_minimal_io.py', 'tdm_minimal_model.py',
                         'tdm_least_squares.py', 'tdm_jmp.py', 'tdm_mobile_phase.py', 'tdm_inputs.json'):
            shutil.copy2(ROOT / filename, project / filename)

        bridge_cfg = make_config(MODEL_EDM_LANGMUIR, 1, multistep=False)
        bridge_cfg['numerics'].update(time_steps=50, axial_positions=6)
        bridge_cfg['conversion']['uv_to_protein_mAU_L_g'] = 200.0
        bridge_cfg['components'][0]['salt_sensitivity_per_M'] = 3.0
        save_config(bridge_cfg, project / 'tdm_inputs.json')
        bridge_rows = [default_batch_row(i) for i in (1, 2)]
        for i, row in enumerate(bridge_rows, start=1):
            row.update(
                Run_Name=f'Button audit run {i}', Load_Flow_mL_min=1.0,
                Feed_Concentration_mg_mL=2.0, Load_CV=0.2, Load_Density_mg_mL_resin=0.4,
                Time_Steps=50, Axial_Positions=6, PLW_Count=0, Elution_Count=1,
                Elution1_Mode='LINEAR', Elution1_CV=0.12,
                Elution1_Start_Percent_B=0.0, Elution1_End_Percent_B=100.0,
            )
        # Use distinctive per-stage values to exercise the saved-row handoff,
        # including multiple elution stages, non-default flows and %B ramps.
        bridge_rows[0].update(
            Load_CV=10.0, Load_Density_mg_mL_resin=20.0,
            Load_Amount_Basis='Capacity [g/L resin]',
            PLW_Count=3,
            PLW1_Mode='LINEAR', PLW1_CV=0.47, PLW1_Flow_mL_min=1.35,
            PLW1_Start_Percent_B=12.0, PLW1_End_Percent_B=34.0,
            PLW2_Mode='LINEAR', PLW2_CV=0.31, PLW2_Flow_mL_min=0.78,
            PLW2_Start_Percent_B=34.0, PLW2_End_Percent_B=55.0,
            PLW3_Mode='LINEAR', PLW3_CV=0.22, PLW3_Flow_mL_min=1.70,
            PLW3_Start_Percent_B=55.0, PLW3_End_Percent_B=80.0,
            Elution_Count=2,
            Elution1_Mode='LINEAR', Elution1_CV=1.26, Elution1_Flow_mL_min=0.63,
            Elution1_Start_Percent_B=21.0, Elution1_End_Percent_B=66.0,
            Elution2_Mode='LINEAR', Elution2_CV=2.34, Elution2_Flow_mL_min=1.44,
            Elution2_Start_Percent_B=66.0, Elution2_End_Percent_B=93.0,
        )
        save_batch_rows(bridge_rows, project / 'tdm_batch_runs.csv')

        persisted_recipe = load_batch_rows(project / 'tdm_batch_runs.csv')[0]
        mapped_recipe = batch_row_to_run_config(persisted_recipe)
        recipe_config, _ = apply_batch_shared_parameters(bridge_cfg, mapped_recipe)
        recipe_program = _build_program(recipe_config, np.zeros(1))
        assert [stage.name for stage in recipe_program.stages] == [
            'Load', 'PLW 1', 'PLW 2', 'PLW 3', 'Elution 1', 'Elution 2'
        ]
        assert [stage.duration_CV for stage in recipe_program.stages] == [10.0, 0.47, 0.31, 0.22, 1.26, 2.34]
        assert [stage.flow_mL_min for stage in recipe_program.stages] == [1.0, 1.35, 0.78, 1.70, 0.63, 1.44]
        # Exercise the saved-run mapper used by JMP. Legacy direct-salt CSV
        # values are ignored; conductivity and the required factor are used.
        chemistry_row = copy.deepcopy(persisted_recipe)
        chemistry_row.update(BufferA_Salt_M=0.22, BufferA_Conductivity_mS_cm=5.0,
                             BufferB_Salt_M=0.11, BufferB_Conductivity_mS_cm=75.0)
        chemistry_config, _ = apply_batch_shared_parameters(bridge_cfg, batch_row_to_run_config(chemistry_row))
        chemistry_program = _build_program(chemistry_config, np.zeros(1))
        assert_close(chemistry_program.stages[0].start_salt_M, 0.05)
        assert_close(chemistry_program.stages[-1].start_salt_M, 0.05 * 0.34 + 0.75 * 0.66)
        assert_close(chemistry_program.stages[-1].end_salt_M, 0.05 * 0.07 + 0.75 * 0.93)
        assert [
            (recipe_config['process']['plw_steps'][i]['start_percent_B'], recipe_config['process']['plw_steps'][i]['end_percent_B'])
            for i in range(3)
        ] == [(12.0, 34.0), (34.0, 55.0), (55.0, 80.0)]
        assert [
            (recipe_config['process']['elution_steps'][i]['start_percent_B'], recipe_config['process']['elution_steps'][i]['end_percent_B'])
            for i in range(2)
        ] == [(21.0, 66.0), (66.0, 93.0)]
        assert_close(recipe_program.total_CV, 14.6)

        sent = {
            'tdm_ui_model': MODEL_LABELS[bridge_cfg['model']],
            'tdm_ui_impurity_count': bridge_cfg['impurity_count'],
            'tdm_ui_column_volume_mL': bridge_cfg['column']['volume_mL'],
            'tdm_ui_column_length_mm': bridge_cfg['column']['length_mm'],
            'tdm_ui_bead_radius': bridge_cfg['tdm']['bead_radius_um'],
            'tdm_ui_void_fraction': bridge_cfg['tdm']['void_fraction'],
            'tdm_ui_particle_porosity': bridge_cfg['tdm']['particle_porosity'],
            'tdm_ui_salt_Dax': bridge_cfg['tdm']['salt_axial_dispersion_mm2_s'],
            'tdm_ui_edm_porosity': bridge_cfg['edm']['total_bed_porosity'],
            'tdm_ui_uv_to_protein_factor': bridge_cfg['conversion']['uv_to_protein_mAU_L_g'],
            'tdm_ui_cond_to_salt_factor': 0.01,
            'tdm_ui_ligand_surface_density': bridge_cfg['cpa']['ligand_surface_density_umol_m2'],
            'tdm_ui_system_adsorption_parameter': bridge_cfg['cpa']['system_specific_adsorption_parameter'],
            'tdm_ui_run_number': 1,
            'tdm_ui_batch_count': 2,
            'tdm_ui_lsq_target_run': 2,
            'tdm_ui_lsq_run_count': 2,
            'tdm_ui_lsq_selected_run_1': 1,
            'tdm_ui_lsq_selected_run_2': 2,
            'tdm_ui_lsq_baseline_mode': 'INITIAL_MEDIAN',
            'tdm_ui_lsq_reference_count': 1,
            'tdm_ui_lsq_mode': 'Normal least-squares: chromatogram only',
            'tdm_ui_lsq_csv_path': '',
            'tdm_project_dir': str(project),
            'tdm_project_dir_posix': str(project),
        }
        for i, component in enumerate(bridge_cfg['components'], start=1):
            for key in ('name', 'mass_percent', 'D_ax_mm2_s', 'k_eff_um_s',
                        'uv_response_factor_mAU_L_g',
                        'accessible_particle_porosity', 'D_app_mm2_s', 'qmax_g_L',
                        'b_L_g', 'salt_sensitivity_per_M', 'diameter_nm', 'As_m_inv', 'Z_ref', 'delta_ref',
                        'kkin_star_s', 'pH_ref', 'Z1_per_pH', 'Z2_per_pH2',
                        'Z3_per_pH3', 'delta_pH_slope_m2_C'):
                sent[f'tdm_ui_c{i}_{key}'] = component.get(key)

        bridge_source = (project / 'tdm_bridge.py').read_text(encoding='utf-8')

        def run_ui_bridge(action, overrides=None):
            namespace = dict(sent)
            namespace['__name__'] = '__main__'
            namespace['tdm_ui_action'] = action
            if overrides:
                namespace.update(overrides)
            exec(compile(bridge_source, str(project / 'tdm_bridge.py'), 'exec'), namespace)
            status = (project / 'tdm_status.txt').read_text(encoding='utf-8')
            return namespace['RETURN_CODE'], status

        code, status = run_ui_bridge('SAVE_MODEL')
        assert code == 0 and 'Model parameters saved' in status, status

        code, status = run_ui_bridge('RUN_SELECTED', {'tdm_ui_c1_qmax_g_L': None})
        assert code == 1 and status.startswith('ERROR:')
        # JMP-sent actions return their status instead of raising through Submit File.

        code, status = run_ui_bridge('RUN_SELECTED')
        assert code == 0 and 'Button audit run 1' in status and 'completed successfully' in status, status
        assert 'Recipe used: Run 1 — Button audit run 1:' in status
        assert 'PLW 1 0.47 CV @ 1.35 mL/min' in status and '%B 12→34' in status
        assert 'Elution 1 1.26 CV @ 0.63 mL/min' in status and '%B 21→66' in status
        assert 'Elution 2 2.34 CV @ 1.44 mL/min' in status and '%B 66→93' in status
        for filename in ('latest_results.html', 'latest_results.csv', 'latest_results.svg', 'latest_results_manifest.json'):
            assert (project / filename).is_file(), filename

        with open(project / 'latest_results.csv', encoding='utf-8', newline='') as fh:
            bridge_base_table = list(csv.DictReader(fh))
        assert {'CV', 'time_min', 'flow_mL_min', 'percent_B', 'program_stage'} <= set(bridge_base_table[0])
        assert any(row['percent_B'] for row in bridge_base_table)
        baseline_svg = (project / 'latest_results.svg').read_text(encoding='utf-8')
        baseline_manifest = json.loads((project / 'latest_results_manifest.json').read_text(encoding='utf-8'))
        assert 'Target affinity used: b=' in status
        base_affinity = baseline_manifest['langmuir_affinity_used'][0]
        base_b = base_affinity['b_L_g']
        base_k = base_affinity['salt_sensitivity_per_M']
        assert math.isclose(base_b, bridge_cfg['components'][0]['b_L_g'])
        assert math.isclose(base_k, bridge_cfg['components'][0]['salt_sensitivity_per_M'])

        # Exercise separate JMP edits of b and k_s through Run selected. The
        # result receipt and outlet-cell affinity must change even when a
        # steep step or breakthrough keeps the UV peak near the same CV.
        for name, value in (('b_L_g', base_b * 2), ('salt_sensitivity_per_M', base_k + 1)):
            code, changed_status = run_ui_bridge('RUN_SELECTED', {f'tdm_ui_c1_{name}': value})
            assert code == 0 and 'Target affinity used:' in changed_status, changed_status
            changed_manifest = json.loads((project / 'latest_results_manifest.json').read_text(encoding='utf-8'))
            changed_affinity = changed_manifest['langmuir_affinity_used'][0]
            key = 'b_L_g' if name == 'b_L_g' else 'salt_sensitivity_per_M'
            assert math.isclose(changed_affinity[key], value)
            assert changed_manifest['configuration_fingerprint'] != baseline_manifest['configuration_fingerprint']
            with open(project / 'latest_results.csv', encoding='utf-8', newline='') as fh:
                changed_table = list(csv.DictReader(fh))
            assert 'Target protein_effective_b_L_g' in changed_table[0]
            salt = float(changed_table[-1]['binding_salt_concentration_M'])
            expected_b_eff = changed_affinity['b_L_g'] * math.exp(changed_affinity['salt_sensitivity_per_M'] * salt)
            assert_close(float(changed_table[-1]['Target protein_effective_b_L_g']), expected_b_eff)
            code, stale_status = run_ui_bridge('CHECK_RESULT_CURRENT')
            assert code == 0 and stale_status.startswith('RESULT_STALE:'), stale_status
        code, status = run_ui_bridge('RUN_SELECTED')
        assert code == 0 and 'completed successfully' in status, status
        baseline_manifest = json.loads((project / 'latest_results_manifest.json').read_text(encoding='utf-8'))

        # The actual JMP button route must reject a factor that resolves to
        # impossible HIC molarity and must not replace the previous result.
        code, status = run_ui_bridge('RUN_SELECTED', {'tdm_ui_cond_to_salt_factor': 1000.0})
        assert code == 1 and 'no new chromatogram was generated' in status, status
        assert 'resolves to' in status and 'HIC model limit is 10 M' in status, status
        assert json.loads((project / 'latest_results_manifest.json').read_text(encoding='utf-8')) == baseline_manifest
        code, status = run_ui_bridge('CHECK_RESULT_CURRENT', {'tdm_ui_cond_to_salt_factor': 1000.0})
        assert code == 0 and status.startswith('RESULT_STALE:'), status

        # Change one saved JMP table input at a time, invoke the same RUN_SELECTED
        # bridge action, and inspect the CSV/SVG it writes from that table row.
        changed_rows = copy.deepcopy(bridge_rows)
        changed_rows[0]['PLW1_CV'] = 0.55
        save_batch_rows(changed_rows, project / 'tdm_batch_runs.csv')
        assert float(load_batch_rows(project / 'tdm_batch_runs.csv')[0]['PLW1_CV']) == 0.55
        code, status = run_ui_bridge('RUN_SELECTED')
        assert code == 0 and 'completed successfully' in status, status
        cv_manifest = json.loads((project / 'latest_results_manifest.json').read_text(encoding='utf-8'))
        with open(project / 'latest_results.csv', encoding='utf-8', newline='') as fh:
            cv_table = list(csv.DictReader(fh))
        assert abs(cv_manifest['total_CV'] - (baseline_manifest['total_CV'] + 0.08)) < 1e-9
        assert float(cv_table[-1]['CV']) != float(bridge_base_table[-1]['CV'])
        assert (project / 'latest_results.svg').read_text(encoding='utf-8') != baseline_svg

        changed_rows = copy.deepcopy(bridge_rows)
        changed_rows[0]['PLW1_Flow_mL_min'] = 2.15
        save_batch_rows(changed_rows, project / 'tdm_batch_runs.csv')
        assert float(load_batch_rows(project / 'tdm_batch_runs.csv')[0]['PLW1_Flow_mL_min']) == 2.15
        code, status = run_ui_bridge('RUN_SELECTED')
        assert code == 0 and 'completed successfully' in status, status
        with open(project / 'latest_results.csv', encoding='utf-8', newline='') as fh:
            flow_table = list(csv.DictReader(fh))
        assert float(flow_table[-1]['CV']) == float(bridge_base_table[-1]['CV'])
        assert float(flow_table[-1]['time_min']) != float(bridge_base_table[-1]['time_min'])
        assert any(float(row['flow_mL_min']) == 2.15 for row in flow_table)
        assert (project / 'latest_results.svg').read_text(encoding='utf-8') != baseline_svg

        changed_rows = copy.deepcopy(bridge_rows)
        changed_rows[0].update(PLW1_Start_Percent_B=20.0, PLW1_End_Percent_B=60.0)
        save_batch_rows(changed_rows, project / 'tdm_batch_runs.csv')
        assert float(load_batch_rows(project / 'tdm_batch_runs.csv')[0]['PLW1_Start_Percent_B']) == 20.0
        code, status = run_ui_bridge('RUN_SELECTED')
        assert code == 0 and 'completed successfully' in status, status
        with open(project / 'latest_results.csv', encoding='utf-8', newline='') as fh:
            b_table = list(csv.DictReader(fh))
        base_b = [row['percent_B'] for row in bridge_base_table if row['percent_B']]
        changed_b = [row['percent_B'] for row in b_table if row['percent_B']]
        assert base_b != changed_b
        assert (project / 'latest_results.svg').read_text(encoding='utf-8') != baseline_svg
        # Restore the input row before the subsequent saved-batch test.
        save_batch_rows(bridge_rows, project / 'tdm_batch_runs.csv')
        code, status = run_ui_bridge('RUN_SELECTED')
        assert code == 0 and 'completed successfully' in status, status

        # Run the real JMP bridge against the saved recipe twice. Since direct
        # salt is blank, changing conductivity must change the consumed salt
        # profile and the HIC protein chromatogram.
        salt_rows = copy.deepcopy(bridge_rows)
        salt_rows[0].update(
            Run_Name='Saved conductivity salt test', Load_CV=0.4,
            Load_Density_mg_mL_resin=0.8, Feed_Concentration_mg_mL=2.0,
            Time_Steps=50, Axial_Positions=6, PLW_Count=0, Elution_Count=1,
            Elution1_Mode='LINEAR', Elution1_CV=5.0,
            Elution1_Flow_mL_min=1.0, Elution1_Start_Percent_B=0.0,
            Elution1_End_Percent_B=100.0,
            BufferA_Salt_M='', BufferA_Conductivity_mS_cm=5.0,
            BufferB_Salt_M='', BufferB_Conductivity_mS_cm=5.0,
        )
        save_batch_rows(salt_rows, project / 'tdm_batch_runs.csv')
        code, status = run_ui_bridge('RUN_SELECTED')
        assert code == 0 and 'completed successfully' in status, status
        low_manifest = json.loads((project / 'latest_results_manifest.json').read_text(encoding='utf-8'))
        with open(project / 'latest_results.csv', encoding='utf-8', newline='') as fh:
            low_salt_table = list(csv.DictReader(fh))
        assert 'conductivity 5→5 mS/cm' in status and 'salt 0.05→0.05 M' in status, status
        assert max(abs(float(row['salt_concentration_M']) - 0.05) for row in low_salt_table) < 1e-10

        salt_rows[0]['BufferA_Conductivity_mS_cm'] = 75.0
        salt_rows[0]['BufferB_Conductivity_mS_cm'] = 75.0
        save_batch_rows(salt_rows, project / 'tdm_batch_runs.csv')
        code, status = run_ui_bridge('RUN_SELECTED')
        assert code == 0 and 'completed successfully' in status, status
        high_manifest = json.loads((project / 'latest_results_manifest.json').read_text(encoding='utf-8'))
        with open(project / 'latest_results.csv', encoding='utf-8', newline='') as fh:
            high_salt_table = list(csv.DictReader(fh))
        assert 'conductivity 75→75 mS/cm' in status and 'salt 0.75→0.75 M' in status, status
        salt_delta = max(abs(float(a['salt_concentration_M']) - float(b['salt_concentration_M']))
                         for a, b in zip(low_salt_table, high_salt_table))
        chromatogram_delta = max(abs(float(a['total_protein_g_L']) - float(b['total_protein_g_L']))
                                 for a, b in zip(low_salt_table, high_salt_table))
        assert salt_delta > 0.69, salt_delta
        assert chromatogram_delta > 1e-4, chromatogram_delta
        assert low_manifest['stages'][-1]['end_salt_M'] == 0.05
        assert high_manifest['stages'][-1]['end_salt_M'] == 0.75

        # Hold conductivity fixed and change only the conversion factor. This
        # checks that the JMP parameter sent to Python reaches the solver too.
        code, status = run_ui_bridge('RUN_SELECTED', {'tdm_ui_cond_to_salt_factor': 0.02})
        assert code == 0 and 'completed successfully' in status, status
        assert 'conductivity 75→75 mS/cm' in status and 'salt 1.5→1.5 M' in status, status
        with open(project / 'latest_results.csv', encoding='utf-8', newline='') as fh:
            changed_factor_table = list(csv.DictReader(fh))
        factor_salt_delta = max(abs(float(a['salt_concentration_M']) - float(b['salt_concentration_M']))
                                for a, b in zip(high_salt_table, changed_factor_table))
        factor_chromatogram_delta = max(abs(float(a['total_protein_g_L']) - float(b['total_protein_g_L']))
                                        for a, b in zip(high_salt_table, changed_factor_table))
        assert factor_salt_delta > 0.74, factor_salt_delta
        assert factor_chromatogram_delta > 1e-4, factor_chromatogram_delta
        assert all(abs(float(row['salt_concentration_M']) - 1.5) < 1e-10 for row in changed_factor_table)
        print(f'    conductivity 5→75 mS/cm changed salt by {salt_delta:.3f} M and chromatogram by {chromatogram_delta:.6g} g/L; with conductivity fixed at 75, factor 0.01→0.02 changed chromatogram by {factor_chromatogram_delta:.6g} g/L')

        save_batch_rows(bridge_rows, project / 'tdm_batch_runs.csv')
        code, status = run_ui_bridge('RUN_SELECTED')
        assert code == 0 and 'completed successfully' in status, status

        code, status = run_ui_bridge('CHECK_RESULT_CURRENT')
        assert code == 0 and status.startswith('RESULT_CURRENT:') and 'Button audit run 1' in status, status
        code, status = run_ui_bridge('CHECK_RESULT_CURRENT', {'tdm_ui_run_number': 2})
        assert code == 0 and status.startswith('RESULT_STALE:') and (
            'different selected run' in status or 'Current settings differ' in status
        ), status

        single_result_receipt = (project / 'latest_results_manifest.json').read_bytes()
        code, status = run_ui_bridge('RUN_BATCH')
        assert code == 0 and 'Batch completed: 2 run(s).' in status, status
        assert 'Run 1 — Button audit run 1: load capacity 20 g/L resin; required feed volume 10 mL; 14.6 CV total' in status
        assert 'Elution 2 2.34 CV @ 1.44 mL/min' in status
        first_batch_dir = Path(next(
            line.split(': ', 1)[1] for line in status.splitlines()
            if line.startswith('Batch result folder: ')
        ))
        assert first_batch_dir.is_dir() and first_batch_dir.parent == project / 'batch_results'
        assert (project / 'latest_results_manifest.json').read_bytes() == single_result_receipt
        assert (project / 'tdm_batch_summary.csv').is_file()
        root_gallery = (project / 'tdm_batch_gallery.html').read_text(encoding='utf-8')
        assert f'<base href="{first_batch_dir.relative_to(project).as_posix()}/">' in root_gallery
        for run_number in (1, 2):
            run_dir = first_batch_dir / f'run_{run_number:03d}'
            assert (run_dir / 'latest_results.html').is_file()
            assert (run_dir / 'latest_results.svg').is_file()
            assert f'run_{run_number:03d}/latest_results.svg' in root_gallery

        recipe_manifest = json.loads(
            (first_batch_dir / 'run_001' / 'latest_results_manifest.json').read_text(encoding='utf-8')
        )
        assert_close(recipe_manifest['total_CV'], 14.6)
        assert [stage['name'] for stage in recipe_manifest['stages']] == [
            'Load', 'PLW 1', 'PLW 2', 'PLW 3', 'Elution 1', 'Elution 2'
        ]
        assert [stage['duration_CV'] for stage in recipe_manifest['stages']] == [10.0, 0.47, 0.31, 0.22, 1.26, 2.34]
        assert [stage['flow_mL_min'] for stage in recipe_manifest['stages']] == [1.0, 1.35, 0.78, 1.70, 0.63, 1.44]
        assert [(recipe_manifest['stages'][i]['start_percent_B'], recipe_manifest['stages'][i]['end_percent_B']) for i in range(1, 6)] == [
            (12.0,34.0),(34.0,55.0),(55.0,80.0),(21.0,66.0),(66.0,93.0)
        ]
        recipe_html = (first_batch_dir / 'run_001' / 'latest_results.html').read_text(encoding='utf-8')
        assert all(name in recipe_html for name in ('PLW 1','PLW 2','PLW 3','Elution 1','Elution 2'))
        assert 'Total run: 14.6 CV' in recipe_html

        # Running again must create a separate output tree; existing/open results
        # are preserved instead of being recursively removed on Windows.
        code, status = run_ui_bridge('RUN_BATCH')
        assert code == 0 and 'Batch completed: 2 run(s).' in status, status
        second_batch_dir = Path(next(
            line.split(': ', 1)[1] for line in status.splitlines()
            if line.startswith('Batch result folder: ')
        ))
        assert second_batch_dir != first_batch_dir and second_batch_dir.is_dir()
        assert (first_batch_dir / 'run_001' / 'latest_results.html').is_file()
        root_gallery = (project / 'tdm_batch_gallery.html').read_text(encoding='utf-8')
        assert f'<base href="{second_batch_dir.relative_to(project).as_posix()}/">' in root_gallery

        target_config, _ = apply_batch_shared_parameters(bridge_cfg, batch_row_to_run_config(bridge_rows[1]))
        target_trace = simulate(target_config)['trace']
        raw = project / 'run_2_target.csv'
        with raw.open('w', encoding='utf-8', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['CV', 'UV_mAU'])
            for cv, concentration in zip(target_trace['CV'], target_trace['total_g_L']):
                writer.writerow([float(cv), 5.0 + 200.0 * float(concentration)])
        code, status = run_ui_bridge('ATTACH_LSQ_CSV', {'tdm_ui_lsq_csv_path': str(raw)})
        assert code == 0 and 'attached to LSQ profile Run 2' in status, status
        saved_profile = load_batch_rows(project / 'tdm_batch_runs.csv')[1]['LSQ_Chromatogram_CSV']
        assert Path(saved_profile) == raw.resolve()

        # Attach a second run with its own process recipe and ensure the JMP
        # bridge includes both saved profiles in one shared global fit.
        first_config, _ = apply_batch_shared_parameters(bridge_cfg, batch_row_to_run_config(bridge_rows[0]))
        first_trace = simulate(first_config)['trace']
        first_raw = project / 'run_1_target.csv'
        with first_raw.open('w', encoding='utf-8', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['CV', 'UV_mAU'])
            for cv, signal in zip(first_trace['CV'], first_trace['uv_mAU']):
                writer.writerow([float(cv), 5.0 + float(signal)])
        code, status = run_ui_bridge('ATTACH_LSQ_CSV', {
            'tdm_ui_lsq_target_run': 1,
            'tdm_ui_lsq_csv_path': str(first_raw),
        })
        assert code == 0 and 'attached to LSQ profile Run 1' in status, status
        assert Path(load_batch_rows(project / 'tdm_batch_runs.csv')[0]['LSQ_Chromatogram_CSV']) == first_raw.resolve()

        code, status = run_ui_bridge('FIT_LSQ', {'tdm_ui_lsq_csv_path': str(raw)})
        assert code == 0 and 'Least-squares refinement completed.' in status, status
        assert 'Global profiles fitted: 2' in status
        assert 'Run 1 — Button audit run 1:' in status and 'Run 2 — Button audit run 2:' in status
        assert (project / 'least_squares_fit_report.html').is_file()
        assert (project / 'least_squares_apply_to_ui.jsl').is_file()
        assert 'Accepted/applied to shared batch parameters: True' in status


def main() -> None:
    print('[1/22] stacked classic UI structure, direct editable controls and buffer chemistry')
    jsl = build_jsl()
    _validate_jsl(jsl)
    assert 'Runs in batch' in jsl and 'Run saved batch' in jsl
    assert all(f'Run {i}' in jsl for i in (1, 15, 30))
    assert 'Post-load washes [0-5]' in jsl and 'Elution steps [0-5]' in jsl
    assert 'Start pH' in jsl and 'End pH' in jsl
    assert 'Start conductivity [mS/cm]' in jsl and 'End conductivity [mS/cm]' in jsl
    assert 'Buffer A / Buffer B definitions (current run)' in jsl and 'TDM ion composition' in jsl
    assert 'Conductivity is the entered buffer property' in jsl
    assert 'Column(dtRuns, "BufferA_Salt_M")[currentRun] = .;' in jsl
    assert 'Column(dtRuns, "BufferA_Conductivity_mS_cm")[currentRun] = bufferACond << Get;' in jsl
    assert 'bufferASalt =' not in jsl and 'loadSalt =' not in jsl
    assert 'BUFFER_A' in jsl and 'BUFFER_B' in jsl
    assert 'Start / set-point Buffer B [%]' in jsl and 'End Buffer B [%]' in jsl
    assert 'BUFFER_B_PERCENT' in jsl and 'ENDPOINT_CHEMISTRY' in jsl
    assert 'Saturation capacity qmax,i [g/L stationary phase]' in jsl
    assert 'Derived Henry constant H_i = qmax,i × b_i [-]' in jsl
    assert 'HIC salt sensitivity k_s,i [M⁻¹]' in jsl
    assert 'b_eff,i = b_i × exp(k_s,i × salt concentration)' in jsl
    assert 'Python Send(condToSalt << Get, Python Name("tdm_ui_cond_to_salt_factor"));' in jsl
    assert 'Load volume [CV]' in jsl and 'Load start / set-point Buffer B [%]' in jsl
    assert 'Load end Buffer B [%]' in jsl and 'Load flow [mL/min]' in jsl
    assert 'StartBRow << Visibility("Visible")' in jsl and 'EndBRow << Visibility("Visible")' in jsl
    assert 'SyncLangmuirDerived = Function' in jsl
    assert 'AttachLSQCSV = Function' in jsl
    assert 'Pick File("Select raw chromatogram CSV"' in jsl
    assert 'lsqCsvPath << Set Text(csvPath)' in jsl
    assert 'lsqTargetRunSelector = Combo Box' in jsl
    assert 'UpdateLSQRunSummary = Function' in jsl
    assert 'SetAllFitLocks = Function' in jsl
    assert 'Button Box("Lock all parameters", SetAllFitLocks("Locked"))' in jsl
    assert 'Button Box("Unlock all parameters", SetAllFitLocks("Unlocked"))' in jsl
    assert 'Attached chromatograms: ' in jsl and 'Global fit profiles (up to 5)' in jsl
    assert 'LSQBufferIonSummary = Function' in jsl
    assert 'This chromatogram profile' in jsl
    assert 'LSQ_Chromatogram_CSV' in jsl
    assert 'Column(dtRuns, "LSQ_Chromatogram_CSV") << Data Type(Character)' in jsl
    assert 'lsqTargetRunSelector << Set(currentRun)' in jsl
    assert jsl.count('LoadRun(RunNumberFromLabel(this << Get Selected))') == 2
    assert 'Time_Steps' in jsl and 'Axial_Positions' in jsl
    assert 'Python Send(RunNumberFromLabel(lsqTargetRunSelector << Get Selected), Python Name("tdm_ui_lsq_target_run"))' in jsl
    assert 'Python Send(projectDir, Python Name("tdm_project_dir"))' in jsl
    assert 'projectDir =' in jsl
    assert '!File Exists(csvPath)' in jsl
    assert 'Button Box("Attach / validate chromatogram CSV", AttachLSQCSV())' in jsl
    assert jsl.index('lsqCsvPath << Set Text(csvPath)') < jsl.index('SendAndRun("ATTACH_LSQ_CSV")')
    assert jsl.index('Python Send(RunNumberFromLabel(lsqTargetRunSelector << Get Selected), Python Name("tdm_ui_lsq_target_run"))') < jsl.index('SendAndRun("FIT_LSQ")')
    assert 'modelLabel = Char(modelBox << Get Selected)' in jsl
    assert 'isTDM = Contains(modelLabel, "TDM") > 0' in jsl
    assert 'isEDM = Contains(modelLabel, "EDM") > 0' in jsl
    assert 'tdmPanel << Visibility(If(isTDM, "Visible", "Collapse"))' in jsl
    assert 'edmPanel << Visibility(If(isEDM, "Visible", "Collapse"))' in jsl
    assert 'c1TdmTransport << Visibility(If(isTDM, "Visible", "Collapse"))' in jsl
    assert 'c1EdmTransport << Visibility(If(isEDM, "Visible", "Collapse"))' in jsl
    assert 'tdm_ui_c1_qmax_g_L' in jsl and 'tdm_ui_c1_H' not in jsl
    # In every PLW/Elution panel, %B must be direct input immediately after CV/flow,
    # not buried in a secondary chemistry panel.
    plw1 = jsl[jsl.index('Post-load wash 1'):jsl.index('Post-load wash 2')]
    assert plw1.index('Volume [CV]') < plw1.index('Flow [mL/min]') < plw1.index('Start / set-point Buffer B [%]') < plw1.index('End Buffer B [%]') < plw1.index('Chemistry control')
    el1 = jsl[jsl.index('Elution 1'):jsl.index('Elution 2')]
    assert el1.index('Volume [CV]') < el1.index('Flow [mL/min]') < el1.index('Start / set-point Buffer B [%]') < el1.index('End Buffer B [%]') < el1.index('Chemistry control')
    assert 'Buffer A (0% B)' in jsl and 'Buffer B (100% B)' in jsl
    assert 'Number of time steps' in jsl and 'Number of axial positions' in jsl
    assert 'Set Width(92)' in jsl, 'numeric input boxes were not compacted'
    assert all(f'plw{i}Panel' in jsl for i in range(1, 6))
    assert all(f'el{i}Panel' in jsl for i in range(1, 6))
    assert jsl.count('Number Edit Box') > 90, 'direct numeric inputs unexpectedly missing'
    assert 'H Splitter Box' not in jsl, 'horizontal splitter reintroduced; UI should be vertically stacked'
    assert 'Size(850, 690)' in jsl and 'Set Window Size(920, 850)' in jsl
    for label in MODEL_LABELS.values(): assert label in jsl
    assert 'Mechanistic model templates' not in jsl and 'Resin template' not in jsl
    # Every << reference must resolve to a declared object/local variable.
    s2 = re.sub(r'"(?:\\.|[^"\\])*"', '""', jsl)
    s2 = re.sub(r'//.*', '', s2)
    refs = set(re.findall(r'\b([A-Za-z_]\w*)\s*<<', s2))
    declared = set(re.findall(r'\b([A-Za-z_]\w*)\s*=', s2))
    for a,b in re.findall(r'Function\s*\(\s*\{([^}]*)\}\s*,\s*\{([^}]*)\}', s2):
        for name in (a+','+b).split(','):
            name=name.strip()
            if re.match(r'^[A-Za-z_]\w*$', name): declared.add(name)
    assert not (refs - declared - {'this'}), sorted(refs-declared-{'this'})

    print('[2/22] exact per-model parameter contracts')
    expected_globals = {
        MODEL_EDM_LANGMUIR: {'column.volume_mL','column.length_mm','conversion.uv_to_protein_mAU_L_g','conversion.conductivity_to_salt_M_per_mS_cm','edm.total_bed_porosity'},
        MODEL_TDM_LANGMUIR: {'column.volume_mL','column.length_mm','conversion.uv_to_protein_mAU_L_g','conversion.conductivity_to_salt_M_per_mS_cm','tdm.bead_radius_um','tdm.void_fraction','tdm.particle_porosity'},
        MODEL_EDM_CPA: {'column.volume_mL','column.length_mm','conversion.uv_to_protein_mAU_L_g','conversion.conductivity_to_salt_M_per_mS_cm','edm.total_bed_porosity','cpa.ligand_surface_density_umol_m2','cpa.system_specific_adsorption_parameter'},
        MODEL_TDM_CPA: {'column.volume_mL','column.length_mm','conversion.uv_to_protein_mAU_L_g','conversion.conductivity_to_salt_M_per_mS_cm','tdm.bead_radius_um','tdm.void_fraction','tdm.particle_porosity','tdm.salt_axial_dispersion_mm2_s','cpa.ligand_surface_density_umol_m2','cpa.system_specific_adsorption_parameter'},
    }
    expected_component = {
        MODEL_EDM_LANGMUIR: {'mass_percent','D_app_mm2_s','qmax_g_L','b_L_g','salt_sensitivity_per_M'},
        MODEL_TDM_LANGMUIR: {'mass_percent','D_ax_mm2_s','k_eff_um_s','accessible_particle_porosity','qmax_g_L','b_L_g','salt_sensitivity_per_M'},
        MODEL_EDM_CPA: {'mass_percent','D_app_mm2_s','diameter_nm','As_m_inv','Z_ref','delta_ref'},
        MODEL_TDM_CPA: {'mass_percent','D_ax_mm2_s','k_eff_um_s','accessible_particle_porosity','diameter_nm','As_m_inv','Z_ref','delta_ref','kkin_star_s'},
    }
    for m in expected_globals:
        assert set(required_global_fields(m)) == expected_globals[m]
        assert set(required_component_fields(m,0)) == expected_component[m]
    legacy = make_config(MODEL_EDM_LANGMUIR, 1, multistep=False)
    for row in legacy['components'][:2]:
        row.pop('qmax_g_L')
    migrated = normalize_config(legacy)
    for row in migrated['components'][:2]:
        assert_close(row['qmax_g_L'], row['H']/row['b_L_g'])
        assert_close(row['H'], row['qmax_g_L']*row['b_L_g'])

    # Capacity is an editable basis, and its derived feed volume is the load
    # duration passed into the solver.  Accept the historical JMP display
    # label as well as the canonical token written by current UI versions.
    capacity_row = default_batch_row(1)
    capacity_row.update(
        Load_Amount_Basis='Capacity [g/L resin]',
        Load_Density_mg_mL_resin=12.0,
        Feed_Concentration_mg_mL=3.0,
        Load_CV=99.0,  # stale volume must be replaced in capacity mode
    )
    capacity_run = batch_row_to_run_config(capacity_row)
    assert capacity_run['process']['load_amount_basis'] == 'CAPACITY'
    assert_close(capacity_run['feed']['load_density_mg_mL_resin'], 12.0)
    assert_close(capacity_run['process']['load_CV'], 4.0)
    capacity_model = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
    capacity_model['column']['volume_mL'] = 2.5
    capacity_model, _ = apply_batch_shared_parameters(capacity_model, capacity_run)
    capacity_program = _build_program(capacity_model, np.zeros(1))
    assert_close(capacity_program.stages[0].duration_CV, 4.0)
    capacity_result = simulate(capacity_model)
    receipt = capacity_result['parameter_receipt']['load_input_used']
    assert receipt['specified_basis'] == 'CAPACITY'
    assert_close(receipt['capacity_g_L_resin'], 12.0)
    assert_close(receipt['required_feed_volume_CV'], 4.0)
    assert_close(receipt['required_feed_volume_mL'], 10.0)
    assert_close(receipt['total_protein_input_g'], 0.03)

    volume_row = default_batch_row(1)
    volume_row.update(
        Load_Amount_Basis='Load volume [CV]',
        Load_CV=2.5,
        Load_Density_mg_mL_resin=99.0,  # stale capacity must follow volume mode
        Feed_Concentration_mg_mL=3.0,
    )
    volume_run = batch_row_to_run_config(volume_row)
    assert volume_run['process']['load_amount_basis'] == 'LOAD_VOLUME'
    assert_close(volume_run['process']['load_CV'], 2.5)
    assert_close(volume_run['feed']['load_density_mg_mL_resin'], 7.5)
    assert 'LoadBasisToken(loadAmountBasis)' in build_jsl()
    assert 'loadAmountBasis = Combo Box' in build_jsl()
    assert 'required feed volume:' in build_jsl()
    missing_capacity = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
    missing_capacity['components'][0].update(qmax_g_L=None, H=None)
    capacity_errors = validate_mechanistic_config(missing_capacity)
    assert any('qmax' in error.lower() and 'required' in error.lower() for error in capacity_errors)
    ph = set(required_component_fields(MODEL_EDM_CPA,0.5))
    assert ph == expected_component[MODEL_EDM_CPA] | {'pH_ref','Z1_per_pH','Z2_per_pH2','delta_pH_slope_m2_C'}
    assert set(required_component_fields(MODEL_EDM_CPA,1.5)) == ph | {'Z3_per_pH3'}
    ph_tdm = set(required_component_fields(MODEL_TDM_CPA,0.5))
    assert ph_tdm == expected_component[MODEL_TDM_CPA] | {'pH_ref','Z1_per_pH','Z2_per_pH2','delta_pH_slope_m2_C'}
    assert set(required_component_fields(MODEL_TDM_CPA,1.5)) == ph_tdm | {'Z3_per_pH3'}

    print('[3/22] hidden process values do not become hidden model requirements')
    c = make_config(MODEL_EDM_CPA, 0)
    # SET_POINT end pH and STEP start pH are intentionally extreme and unused.
    c['process']['plw_steps'][0].update(mode='SET_POINT', start_pH=6.0, end_pH=13.0)
    c['process']['plw_steps'][1].update(mode='STEP', start_pH=1.0, end_pH=6.0)
    c['process']['elution_count']=1
    c['process']['elution_steps'][0].update(mode='SET_POINT', start_pH=6.0, end_pH=12.0)
    assert abs(process_pH_span(c)) < 1e-12, process_pH_span(c)
    assert not validate_mechanistic_config(c), validate_mechanistic_config(c)
    assert not validate_operating_conditions(c), validate_operating_conditions(c)

    print('[4/22] JMP-to-Python direct parameter mapping')
    # Sentinels prove editable boxes feed the bridge fields rather than a template/database.
    sent = {
        'tdm_ui_model': MODEL_LABELS[MODEL_TDM_LANGMUIR], 'tdm_ui_impurity_count': 1,
        'tdm_ui_column_volume_mL': 1.234, 'tdm_ui_column_length_mm': 56.7,
        'tdm_ui_bead_radius': 23.4, 'tdm_ui_void_fraction': .411, 'tdm_ui_particle_porosity': .522,
        'tdm_ui_salt_Dax': 9.99, 'tdm_ui_edm_porosity': .812,
        'tdm_ui_uv_to_protein_factor': 345.6, 'tdm_ui_cond_to_salt_factor': .0123,
        'tdm_ui_c1_salt_sensitivity_per_M': 2.75,
        'tdm_ui_ligand_surface_density': 7.77, 'tdm_ui_system_adsorption_parameter': .044,
    }
    for i in range(1,7):
        b_l_g = .004+i/10000
        qmax_g_L = (2+i/10)/b_l_g
        sent.update({f'tdm_ui_c{i}_name': f'S{i}', f'tdm_ui_c{i}_mass_percent': 70 if i==1 else (30 if i==2 else None),
                     f'tdm_ui_c{i}_D_ax_mm2_s': 1.0+i/10, f'tdm_ui_c{i}_k_eff_um_s': 8+i/10,
                     f'tdm_ui_c{i}_accessible_particle_porosity': .4-i/100,
                     f'tdm_ui_c{i}_D_app_mm2_s': .06+i/1000,
                     f'tdm_ui_c{i}_qmax_g_L': qmax_g_L,
                     f'tdm_ui_c{i}_b_L_g': b_l_g, f'tdm_ui_c{i}_diameter_nm': 10+i,
                     f'tdm_ui_c{i}_As_m_inv': 2e8+i, f'tdm_ui_c{i}_Z_ref': 20+i,
                     f'tdm_ui_c{i}_delta_ref': .01+i/1000, f'tdm_ui_c{i}_kkin_star_s': 1e5+i,
                     f'tdm_ui_c{i}_pH_ref': 6, f'tdm_ui_c{i}_Z1_per_pH': .1,
                     f'tdm_ui_c{i}_Z2_per_pH2': .01, f'tdm_ui_c{i}_Z3_per_pH3': .001,
                     f'tdm_ui_c{i}_delta_pH_slope_m2_C': .2})
    old = {k:getattr(tdm_bridge,k,None) for k in sent}
    try:
        for k,v in sent.items(): setattr(tdm_bridge,k,v)
        mapped=tdm_bridge.build_shared_config_from_jmp()
    finally:
        for k,v in old.items():
            if v is None and hasattr(tdm_bridge,k): delattr(tdm_bridge,k)
            else: setattr(tdm_bridge,k,v)
    assert mapped['model']==MODEL_TDM_LANGMUIR and mapped['impurity_count']==1
    assert_close(mapped['column']['volume_mL'],1.234); assert_close(mapped['tdm']['bead_radius_um'],23.4)
    assert_close(mapped['conversion']['uv_to_protein_mAU_L_g'],345.6); assert_close(mapped['conversion']['conductivity_to_salt_M_per_mS_cm'],.0123)
    assert_close(mapped['components'][0]['qmax_g_L'], 2.1/.0041)
    assert_close(mapped['components'][0]['H'],2.1); assert_close(mapped['components'][1]['b_L_g'],.0042)
    assert_close(mapped['components'][0]['salt_sensitivity_per_M'],2.75)
    clean=sanitize_config_for_selected_model(mapped)
    assert clean['components'][0]['D_app_mm2_s'] is None and clean['components'][0]['diameter_nm'] is None

    print('[5/22] 30-run storage and PLW/elution row mapping')
    # The legacy recipe fixture has fixed expected values. The delivered
    # startup run is independently verified and must remain user-editable.
    rows=load_batch_rows(ROOT / 'prior_inputs' / 'tdm_batch_runs_R12.csv')
    assert len(rows)==MAX_RUNS==30
    cfgrow=batch_row_to_run_config(rows[0])
    assert len(cfgrow['process']['plw_steps'])==MAX_PLW_STEPS==5
    assert len(cfgrow['process']['elution_steps'])==MAX_ELUTION_STEPS==5
    assert cfgrow['process']['plw_count'] == 3 and cfgrow['process']['elution_count'] == 1
    assert [cfgrow['process']['plw_steps'][i]['CV'] for i in range(3)] == [2.0, 1.0, 1.0]
    assert cfgrow['process']['elution_steps'][0]['CV'] == 4.0
    assert cfgrow['process']['load_conductivity_mS_cm'] is not None
    assert_close(cfgrow['process']['load_CV'], 10.0)
    assert cfgrow['process']['load_mode'] == 'LINEAR'
    assert cfgrow['process']['load_chemistry_control'] == 'BUFFER_B_PERCENT'
    assert cfgrow['numerics']['time_steps'] == 700
    assert cfgrow['numerics']['axial_positions'] == 40
    assert 'LSQ_Chromatogram_CSV' in rows[0]
    with tempfile.TemporaryDirectory() as td:
        profile_table = Path(td) / 'profile_runs.csv'
        profile_rows = [dict(row) for row in rows]
        profile_rows[0]['LSQ_Chromatogram_CSV'] = '/profiles/overload.csv'
        profile_rows[1]['LSQ_Chromatogram_CSV'] = '/profiles/standard.csv'
        profile_rows[2]['LSQ_Chromatogram_CSV'] = '/profiles/gradient.csv'
        save_batch_rows(profile_rows, profile_table)
        reloaded_profiles = load_batch_rows(profile_table)
        assert [reloaded_profiles[i]['LSQ_Chromatogram_CSV'] for i in range(3)] == [
            '/profiles/overload.csv', '/profiles/standard.csv', '/profiles/gradient.csv'
        ]
    direct_load = default_batch_row(1)
    direct_load.update(
        Load_CV=3.5, Load_Flow_mL_min=1.7, Feed_Concentration_mg_mL=2.0,
        Load_Mode='LINEAR', Load_Start_Percent_B=25.0, Load_End_Percent_B=75.0,
        Load_Chemistry_Control='BUFFER_B_PERCENT',
        BufferA_Salt_M=0.1, BufferB_Salt_M=0.5,
    )
    load_cfg = batch_row_to_run_config(direct_load)
    load_cfg['model'] = MODEL_EDM_LANGMUIR
    load_cfg['column'].update(volume_mL=1.0, length_mm=50.0)
    load_prog = _build_program(load_cfg, np.zeros(1))
    assert_close(load_prog.stages[0].duration_CV, 3.5)
    assert_close(load_prog.stages[0].flow_mL_min, 1.7)
    assert_close(load_cfg['feed']['load_density_mg_mL_resin'], 7.0)
    assert_close(load_prog.stages[0].start_percent_B, 25.0)
    assert_close(load_prog.stages[0].end_percent_B, 75.0)
    assert_close(load_prog.percent_B_at_cv(1.75), 50.0)
    for mode, effective_b in (('SET_POINT', 25.0), ('STEP', 75.0)):
        mode_cfg = copy.deepcopy(load_cfg)
        mode_cfg['process']['load_mode'] = mode
        mode_program = _build_program(mode_cfg, np.zeros(1))
        assert_close(mode_program.percent_B_at_cv(1.75), effective_b)

    print('[6/22] all four solvers with restored multistep operating program')
    results={}
    for m in (MODEL_EDM_LANGMUIR,MODEL_TDM_LANGMUIR,MODEL_EDM_CPA,MODEL_TDM_CPA):
        cfg=make_config(m,1)
        if m==MODEL_EDM_CPA:
            # This test program is constant pH despite extreme hidden values.
            assert abs(process_pH_span(cfg)) < 1e-12
        errors=validate_config(cfg)
        assert not errors, (m, errors)
        r=simulate(cfg); results[m]=(cfg,r)
        assert len(r['trace']['CV']) == cfg['numerics']['time_steps'] + 1
        assert r['simulation']['diagnostics']['axial_cells'] == cfg['numerics']['axial_positions']
        assert r['simulation']['diagnostics']['requested_time_steps'] == cfg['numerics']['time_steps']
        assert r['simulation']['diagnostics']['internal_time_steps'] >= 1
        assert_close(r['trace']['CV'][-1], _build_program(cfg,np.zeros(2)).total_CV, 1e-6)
        for vals in r['mass_balance'].values():
            assert abs(vals['closure_error_percent_of_input']) < .05, (m,vals)
        if uses_langmuir(m):
            rows = cfg['components'][:2]
            qmax = np.asarray([row['qmax_g_L'] for row in rows])
            b = np.asarray([row['b_L_g'] for row in rows])
            conc = np.asarray([0.7, 1.3])
            q, _jac = _langmuir_q_jac(conc, qmax*b, b)
            expected_q = qmax*b*conc/(1.0+float(np.dot(b,conc)))
            assert np.allclose(q, expected_q, rtol=1e-12, atol=1e-12)
            k_s = np.asarray([row['salt_sensitivity_per_M'] for row in rows])
            phase_ratio = np.asarray([0.6, 1.3])
            q_salt, _jac_salt = _langmuir_q_jac(conc, qmax*b, b, salt_M=0.31, salt_sensitivity_per_M=k_s)
            recovered_c, recovered_q = _langmuir_c_q_from_total(
                conc + phase_ratio*q_salt, qmax*b, b, salt_M=0.31,
                salt_sensitivity_per_M=k_s, phase_ratio=phase_ratio,
            )
            assert np.allclose(recovered_c, conc, rtol=2e-11, atol=1e-12)
            assert np.allclose(recovered_q, q_salt, rtol=2e-11, atol=1e-12)
            q_sat, _ = _langmuir_q_jac(np.asarray([1e12]), np.asarray([qmax[0]*b[0]]), np.asarray([b[0]]))
            assert_close(q_sat[0], qmax[0], 1e-9)
            assert r['simulation']['diagnostics']['competitive_langmuir_equation'] == 'q_i*=qmax_i*b_i*exp(k_s_i*m_s)*c_i/(1+sum_j b_j*exp(k_s_j*m_s)*c_j)'
            assert 'hic_salt_model' in r['simulation']['diagnostics']
            assert r['simulation']['diagnostics']['hic_salt_sensitivity_per_M'] == [row['salt_sensitivity_per_M'] for row in rows]

    print('[6a] HIC salt-dependent competitive Langmuir responds in both column solvers')
    for model in (MODEL_EDM_LANGMUIR, MODEL_TDM_LANGMUIR):
        salt_cfg = make_config(model, 0, multistep=False)
        salt_cfg['numerics'].update(time_steps=100, axial_positions=10)
        salt_cfg['components'][0]['salt_sensitivity_per_M'] = 3.0
        p = salt_cfg['process']
        p['load_CV'] = 0.4
        salt_cfg['feed']['load_density_mg_mL_resin'] = 0.8
        p['plw_count'] = 0
        p['elution_count'] = 1
        p['buffer_A'].update(salt_concentration_M=None, conductivity_mS_cm=5.0)
        p['buffer_B'].update(salt_concentration_M=None, conductivity_mS_cm=5.0)
        p['elution_steps'][0].update(mode='LINEAR', CV=5.0, flow_mL_min=1.0, start_percent_B=0.0, end_percent_B=100.0)
        low = simulate(salt_cfg)
        high_cfg = copy.deepcopy(salt_cfg)
        high_cfg['process']['buffer_A']['conductivity_mS_cm'] = 75.0
        high_cfg['process']['buffer_B']['conductivity_mS_cm'] = 75.0
        high = simulate(high_cfg)
        assert np.max(np.abs(low['trace']['salt_concentration_M'] - high['trace']['salt_concentration_M'])) > 0.5
        assert np.max(np.abs(low['trace']['total_g_L'] - high['trace']['total_g_L'])) > 1e-8, model
        assert high['simulation']['diagnostics']['hic_salt_sensitivity_per_M'] == [3.0]
        for result in (low, high):
            assert abs(result['mass_balance']['Target protein']['closure_error_percent_of_input']) < 0.05, (model,result['mass_balance'])

        legacy_salt_cfg = copy.deepcopy(salt_cfg)
        for key in ('buffer_A', 'buffer_B'):
            legacy_salt_cfg['process'][key].update(salt_concentration_M=0.75, conductivity_mS_cm=5.0)
        legacy_salt = simulate(legacy_salt_cfg)
        assert np.max(np.abs(low['trace']['salt_concentration_M'] - legacy_salt['trace']['salt_concentration_M'])) < 1e-12
        assert np.max(np.abs(low['trace']['total_g_L'] - legacy_salt['trace']['total_g_L'])) < 1e-10
        assert abs(legacy_salt['mass_balance']['Target protein']['closure_error_percent_of_input']) < 0.05

    print('[7/22] process stage mode, conductivity conversion and variable-flow semantics')
    cfg=make_config(MODEL_TDM_CPA,0)
    prog=_build_program(cfg,np.array([1.0]))
    # With an isocratic-pH program TDM+CPA stays at the load pH; conductivity is converted to salt and obeys each stage mode.
    for cv in np.linspace(0,prog.total_CV,15):
        _feed,pH,_I,_stage=prog.at_cv(float(cv)); assert_close(pH,6.0)
    # Different stage flow rates must alter time/CV conversion and be invertible.
    for cv in np.linspace(0,prog.total_CV,25):
        assert_close(prog.time_to_cv(prog.cv_to_time(float(cv))),cv,1e-7)
    assert len({round(x.flow_mL_min,6) for x in prog.stages}) > 1

    print('[8/22] shared mechanistic parameters are locked across batch runs')
    base=make_config(MODEL_EDM_LANGMUIR,1,multistep=False)
    run1=copy.deepcopy(base); run1['column']['flow_mL_min']=.7; run1['feed']['load_density_mg_mL_resin']=.25
    run2=copy.deepcopy(base); run2['column']['flow_mL_min']=1.3; run2['feed']['load_density_mg_mL_resin']=.3
    run2['components'][0]['qmax_g_L']=999.0 # attempted illegal per-run shared override
    eff,ignored=apply_batch_shared_parameters(base,run2)
    assert 'components[0].qmax_g_L' in ignored and eff['components'][0]['qmax_g_L']==base['components'][0]['qmax_g_L']
    assert batch_parameter_fingerprint(eff)==batch_parameter_fingerprint(base)
    with tempfile.TemporaryDirectory() as td:
        info=run_batch_and_write(base,[run1,run2],Path(td))
        assert len(info['runs'])==2 and Path(info['gallery_html']).is_file() and Path(info['summary_csv']).is_file()

    print('[9/22] run-specific numerical resolution feeds the solvers')
    coarse=make_config(MODEL_EDM_LANGMUIR,0,multistep=False)
    coarse['numerics'].update(time_steps=80, axial_positions=10)
    fine=copy.deepcopy(coarse); fine['numerics'].update(time_steps=160, axial_positions=24)
    rc=simulate(coarse); rfine=simulate(fine)
    assert len(rc['trace']['CV'])==81 and len(rfine['trace']['CV'])==161
    assert rc['simulation']['diagnostics']['axial_cells']==10 and rfine['simulation']['diagnostics']['axial_cells']==24
    # Output grids and solver diagnostics must reflect the requested resolution;
    # physical convergence is covered by the TDM-CPA grid-sensitivity check below.
    assert np.isfinite(rc['trace']['total_g_L']).all() and np.isfinite(rfine['trace']['total_g_L']).all()
    assert rc['simulation']['diagnostics']['axial_cells'] != rfine['simulation']['diagnostics']['axial_cells']
    bad=copy.deepcopy(coarse); bad['numerics']['time_steps']=80.5
    assert any('time steps must be an integer' in x.lower() for x in validate_operating_conditions(bad))

    print('[10/22] limited axial transport reduces grid smearing and preserves mass balance')
    c20=make_config(MODEL_TDM_CPA,1,multistep=True)
    c20['numerics'].update(time_steps=180,axial_positions=20)
    c40=copy.deepcopy(c20); c40['numerics'].update(time_steps=360,axial_positions=40)
    r20=simulate(c20); r40=simulate(c40)
    peak20=float(np.max(r20['trace']['total_g_L']))
    peak40=float(np.max(r40['trace']['total_g_L']))
    assert abs(peak20-peak40)/max(peak40,1e-12) < 0.12, (peak20,peak40)
    for result in (r20,r40):
        assert np.isfinite(result['trace']['total_g_L']).all()
        assert max(abs(v['closure_error_percent_of_input']) for v in result['mass_balance'].values()) < 0.05
        assert 'slope-limited' in result['simulation']['diagnostics']['axial_transport_discretization']

    print('[11/22] results include the exact run recipe, stage markers, and stale-result rejection')
    cfg,r=results[MODEL_EDM_LANGMUIR]
    with tempfile.TemporaryDirectory() as td:
        paths=write_outputs(cfg,r,td)
        assert all(Path(v).is_file() for v in paths.values())
        svg_text = Path(paths['svg']).read_text(encoding='utf-8')
        assert '<svg' in svg_text and 'width=\"1100\"' in svg_text and 'fill=\"white\"' in svg_text
        import xml.etree.ElementTree as ET
        ET.fromstring(svg_text)
        html=Path(paths['html']).read_text(encoding='utf-8')
        assert 'Chromatography model results' in html and 'Inputs actually used by this run' in html
        assert 'svg{display:block;width:100%;max-width:1000px;height:auto' in html
        rows_out=list(csv.DictReader(open(paths['csv'], encoding='utf-8')))
        assert 'uv_mAU' in rows_out[0]

    # Recreate the settings shown in the user's screenshot through the same
    # saved-run CSV mapping that the JMP bridge uses before calling the solver.
    row = default_batch_row(4)
    row.update(Run_Name='Screenshot validation', Load_Flow_mL_min=0.115,
               Feed_Concentration_mg_mL=2.0, Load_Density_mg_mL_resin=20.0,
               Time_Steps=80, Axial_Positions=8, PLW_Count=3, Elution_Count=1)
    for i, cv in enumerate((2.0, 1.0, 1.0), start=1):
        row.update({f'PLW{i}_CV':cv, f'PLW{i}_Flow_mL_min':0.115})
    row.update(Elution1_CV=4.0, Elution1_Flow_mL_min=0.115)
    base = make_config(MODEL_EDM_LANGMUIR, 0, multistep=False)
    exact_config, _ = apply_batch_shared_parameters(base, batch_row_to_run_config(row))
    assert not validate_config(exact_config), validate_config(exact_config)
    exact_result = simulate(exact_config)
    exact_program = exact_result['simulation']['program']
    assert_close(exact_program.total_CV, 18.0)
    assert [round(v, 6) for v in exact_program.stage_end_CV] == [10.0, 12.0, 13.0, 14.0, 18.0]
    context = {'run_number':4, 'run_name':'Screenshot validation'}
    with tempfile.TemporaryDirectory() as td:
        paths = write_outputs(exact_config, exact_result, td, run_context=context)
        html_text = Path(paths['html']).read_text(encoding='utf-8')
        svg_text = Path(paths['svg']).read_text(encoding='utf-8')
        receipt = json.loads(Path(paths['manifest']).read_text(encoding='utf-8'))
        csv_rows = list(csv.DictReader(open(paths['csv'], encoding='utf-8')))
        assert 'Screenshot validation' in html_text and '2026-10-01-R14' in html_text
        assert '18 CV' in html_text and 'Start %B' in html_text and 'Flow mL/min' in html_text
        assert all(label in html_text for label in ('PLW 1','PLW 2','PLW 3','Elution 1'))
        assert 'run ends at 18 CV' in svg_text
        assert receipt['total_CV'] == 18.0 and receipt['time_steps'] == 80 and receipt['axial_cells'] == 8
        assert [s['end_CV'] for s in receipt['stages']] == [10.0,12.0,13.0,14.0,18.0]
        assert csv_rows[0]['run_number'] == '4' and csv_rows[0]['run_name'] == 'Screenshot validation'
        assert csv_rows[0]['settings_fingerprint'] == receipt['configuration_fingerprint']
        assert 'percent_B' in csv_rows[0] and 'flow_mL_min' in csv_rows[0]
        assert all(f'{name}_uv_mAU' in csv_rows[0] for name in exact_result['simulation']['component_names'])
        assert 'Elapsed time (min)' in svg_text and 'programmed %B' in svg_text and 'column %B' in svg_text
        assert abs(float(csv_rows[-1]['CV']) - 18.0) < 1e-10
        assert check_result_freshness(exact_config, context, td)[0]
        altered = copy.deepcopy(exact_config)
        altered['process']['plw_steps'][1]['CV'] = 2.0
        assert not check_result_freshness(altered, context, td)[0]
        assert not check_result_freshness(exact_config, {'run_number':3,'run_name':'other run'}, td)[0]
        with open(paths['csv'], 'a', encoding='utf-8') as fh:
            fh.write('\n')
        assert not check_result_freshness(exact_config, context, td)[0]

    print('[12/22] both least-squares objective modes consume raw CSV + optional composition')
    cfg=make_config(MODEL_EDM_LANGMUIR,1,multistep=False)
    cfg['feed']['load_density_mg_mL_resin']=2.0
    r=simulate(cfg)
    tr=r['trace']
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'synthetic.csv'
        # Sparse raw chromatogram is enough to exercise CSV detection/alignment.
        idx=np.linspace(0,len(tr['CV'])-1,80).astype(int)
        with p.open('w',newline='',encoding='utf-8') as fh:
            w=csv.writer(fh); w.writerow(['CV','UV_mAU'])
            for k in idx: w.writerow([tr['CV'][k], 200.0*tr['total_g_L'][k]])
        ref=load_chromatogram_reference(p,cfg)
        r0,info0=_objective_parts(cfg,ref,MODE_CHROMATOGRAM,[])
        assert np.all(np.isfinite(r0)) and info0['chromatogram_sse_normalized'] < 1e-10
        # Composition references must be taken where the signal is measurable;
        # a mid-run tail can contain only numerical round-off concentrations.
        k=int(np.argmax(tr['total_g_L']))
        cv=float(tr['CV'][k])
        comps=tr['component_g_L'][k]
        total=float(np.sum(comps))
        assert total > 1e-12
        names=r['simulation']['component_names']
        refcomp={'CV':cv,'mass_percent':{name:100.0*float(comps[i])/total for i,name in enumerate(names)}}
        refs=validate_composition_references(cfg,[refcomp])
        r1,info1=_objective_parts(cfg,ref,MODE_CHROMATOGRAM_AND_COMPOSITION,refs)
        assert np.all(np.isfinite(r1)) and info1['composition_sse_normalized'] < 1e-10
        assert [s.path.split('.')[-1] for s in fit_parameter_specs(cfg)][:3] == ['qmax_g_L','b_L_g','salt_sensitivity_per_M']
        assert 'components[1].uv_response_factor_mAU_L_g' in [s.path for s in fit_parameter_specs(cfg)]
        concentration_specs=fit_parameter_specs(cfg,fit_uv_response=False)
        assert not any(s.path.endswith('uv_response_factor_mAU_L_g') for s in concentration_specs)

    print('[13/22] feed mass% and UV conversion are consumed, not display-only')
    cfg=make_config(MODEL_EDM_LANGMUIR,1,multistep=False)
    cfg['numerics'].update(time_steps=100,axial_positions=10)
    cfg['process']['load_CV']=0.4
    cfg['feed']['load_density_mg_mL_resin']=0.8
    cfg['process']['elution_steps'][0].update(mode='LINEAR',CV=5.0,flow_mL_min=1.0,start_percent_B=0.0,end_percent_B=100.0)
    cfg['components'][0]['salt_sensitivity_per_M']=3.0
    for key in ('buffer_A','buffer_B'):
        cfg['process'][key].update(salt_concentration_M=None,conductivity_mS_cm=5.0)
    cfg['components'][0]['mass_percent']=80.0; cfg['components'][1]['mass_percent']=20.0
    r=simulate(cfg)
    assert_close(r['parameter_receipt']['derived_species_feed_concentration_mg_mL']['Target protein'], 1.6)
    assert_close(r['parameter_receipt']['derived_species_feed_concentration_mg_mL']['Impurity 1'], 0.4)
    assert np.max(np.abs(r['trace']['uv_mAU'] - 200.0*r['trace']['total_g_L'])) < 1e-10
    cfg2=copy.deepcopy(cfg); cfg2['conversion']['uv_to_protein_mAU_L_g']=100.0
    r2=simulate(cfg2)
    assert np.max(np.abs(r2['trace']['total_g_L'] - r['trace']['total_g_L'])) < 1e-10
    assert np.max(np.abs(r2['trace']['uv_mAU'] - 0.5*r['trace']['uv_mAU'])) < 1e-10
    cfg3=copy.deepcopy(cfg); cfg3['components'][1]['uv_response_factor_mAU_L_g']=100.0
    r3=simulate(cfg3)
    expected_uv=200.0*r3['trace']['component_g_L'][:,0]+100.0*r3['trace']['component_g_L'][:,1]
    assert np.max(np.abs(r3['trace']['uv_mAU']-expected_uv)) < 1e-10
    assert np.max(np.abs(r3['trace']['uv_mAU']-r['trace']['uv_mAU'])) > 1e-6
    assert_close(r3['parameter_receipt']['required_inputs_used']['components[1].uv_response_factor_mAU_L_g'],100.0)
    assert batch_parameter_fingerprint(cfg3) != batch_parameter_fingerprint(cfg)

    print('[14/22] conductivity × required conversion factor is the sole salt input')
    cfg=make_config(MODEL_TDM_CPA,0,multistep=False)
    cfg['process']['plw_steps'][0].update(chemistry_control='ENDPOINT_CHEMISTRY', start_source='DIRECT', end_source='DIRECT', start_conductivity_mS_cm=5.0, end_conductivity_mS_cm=5.0)
    cfg['process']['elution_steps'][0].update(chemistry_control='ENDPOINT_CHEMISTRY', start_source='DIRECT', end_source='DIRECT', start_conductivity_mS_cm=8.0, end_conductivity_mS_cm=35.0, start_salt_concentration_M=None, end_salt_concentration_M=None)
    prog=_build_program(cfg,np.array([1.0]))
    assert_close(prog.stages[0].start_salt_M, 5.0*0.01)
    assert_close(prog.stages[0].start_I_M, 5.0*0.01)
    assert_close(prog.stages[-1].start_salt_M, 8.0*0.01)
    assert_close(prog.stages[-1].end_salt_M, 35.0*0.01)
    cfg['process']['elution_steps'][0]['start_salt_concentration_M']=0.222
    prog2=_build_program(cfg,np.array([1.0]))
    assert_close(prog2.stages[-1].start_salt_M,0.08)
    assert_close(prog2.stages[-1].start_I_M,0.08)
    cfg2=copy.deepcopy(cfg); cfg2['conversion']['conductivity_to_salt_M_per_mS_cm']=0.02
    assert batch_parameter_fingerprint(cfg2) != batch_parameter_fingerprint(cfg)

    print('[15/22] Buffer A/B salt calculation and TDM ion-derived ionic strength feed solver')
    g=make_config(MODEL_TDM_CPA,0,multistep=False)
    p=g['process']
    p['buffer_A'].update(name='Low salt',pH=6.0,conductivity_mS_cm=10.0,salt_concentration_M=None)
    p['buffer_B'].update(name='High salt',pH=6.0,conductivity_mS_cm=40.0,salt_concentration_M=None)
    for buf in (p['buffer_A'],p['buffer_B']):
        for ion in buf['ions']: ion.update(species='',concentration_M=None,valency=None)
    p['buffer_A']['ions'][0].update(species='Na+',concentration_M=0.10,valency=1)
    p['buffer_A']['ions'][1].update(species='SO4--',concentration_M=0.05,valency=-2)
    p['buffer_B']['ions'][0].update(species='Na+',concentration_M=0.40,valency=1)
    p['buffer_B']['ions'][1].update(species='Cl-',concentration_M=0.40,valency=-1)
    p['buffer_A']['ionic_strength_source']='ION_COMPOSITION'
    p['buffer_B']['ionic_strength_source']='ION_COMPOSITION'
    p['load_source']='BUFFER_A'
    p['plw_steps'][0].update(chemistry_control='BUFFER_B_PERCENT', start_percent_B=0.0, end_percent_B=0.0)
    p['elution_steps'][0].update(chemistry_control='BUFFER_B_PERCENT', start_percent_B=0.0, end_percent_B=100.0, mode='LINEAR')
    assert_close(buffer_salt_concentration_M(g,p['buffer_A']),0.10)
    assert_close(buffer_ionic_strength_M(p['buffer_A']),0.15)
    assert_close(buffer_ionic_strength_M(p['buffer_B']),0.40)
    pg=_build_program(g,np.array([1.0]))
    assert_close(pg.stages[0].start_salt_M,0.10); assert_close(pg.stages[0].start_I_M,0.15)
    assert_close(pg.stages[-1].start_I_M,0.15); assert_close(pg.stages[-1].end_I_M,0.40)
    # Constant pH avoids adding pH-polynomial requirements in this ion-path test.
    assert not validate_config(g), validate_config(g)
    rg=simulate(g)
    assert 'salt_concentration_M' in rg['trace'] and np.all(np.isfinite(rg['trace']['salt_concentration_M']))
    altered=copy.deepcopy(g); altered['process']['buffer_B']['ions'][0]['valency']=2
    r_alt=simulate(altered)
    common=min(len(rg['trace']['total_g_L']),len(r_alt['trace']['total_g_L']))
    assert np.max(np.abs(rg['trace']['total_g_L'][:common]-r_alt['trace']['total_g_L'][:common])) > 1e-10
    salt_proxy=copy.deepcopy(g)
    salt_proxy['process']['buffer_A']['ionic_strength_source']='SALT_PROXY'
    salt_proxy['process']['buffer_B']['ionic_strength_source']='SALT_PROXY'
    salt_proxy_low=simulate(salt_proxy)
    salt_proxy_high=copy.deepcopy(salt_proxy)
    salt_proxy_high['process']['buffer_B']['conductivity_mS_cm']=80.0
    salt_proxy_high_result=simulate(salt_proxy_high)
    assert np.max(np.abs(salt_proxy_low['trace']['total_g_L']-salt_proxy_high_result['trace']['total_g_L'])) > 1e-10

    print('[16/22] PLW/elution pH + salt gradients feed both CPA column models')
    for model in (MODEL_TDM_CPA, MODEL_EDM_CPA):
        g = make_config(model,0,multistep=False)
        g['process']['plw_steps'][0].update(mode='LINEAR', CV=0.2, chemistry_control='ENDPOINT_CHEMISTRY', start_source='DIRECT', end_source='DIRECT', start_pH=6.0, end_pH=6.4, start_conductivity_mS_cm=5.0, end_conductivity_mS_cm=9.0, start_salt_concentration_M=0.75, end_salt_concentration_M=0.75)
        g['process']['elution_steps'][0].update(mode='LINEAR', CV=1.2, chemistry_control='ENDPOINT_CHEMISTRY', start_source='DIRECT', end_source='DIRECT', start_pH=6.4, end_pH=7.6, start_conductivity_mS_cm=9.0, end_conductivity_mS_cm=40.0, start_salt_concentration_M=0.75, end_salt_concentration_M=0.75)
        row=g['components'][0]
        row.update(pH_ref=6.0, Z1_per_pH=-6.0, Z2_per_pH2=0.0, Z3_per_pH3=0.0, delta_pH_slope_m2_C=0.0)
        assert process_pH_span(g) > 1.0
        assert not validate_config(g), (model, validate_config(g))
        pg=_build_program(g,np.array([1.0]))
        el=pg.stages[-1]
        mid_pH, mid_I=el.environment(0.5)
        assert_close(mid_pH,7.0); assert_close(mid_I,(0.09+0.40)/2.0)
        assert_close(el.chemistry(0.5)[1],(0.09+0.40)/2.0)
        rg=simulate(g)
        assert float(np.nanmax(rg['trace']['pH'])-np.nanmin(rg['trace']['pH'])) > 0.5
        assert np.all(np.isfinite(rg['trace']['salt_concentration_M']))
        fixed=copy.deepcopy(g)
        fixed['process']['plw_steps'][0].update(start_pH=6.0,end_pH=6.0)
        fixed['process']['elution_steps'][0].update(start_pH=6.0,end_pH=6.0)
        rf=simulate(fixed)
        common=min(len(rg['trace']['total_g_L']),len(rf['trace']['total_g_L']))
        assert np.max(np.abs(rg['trace']['total_g_L'][:common]-rf['trace']['total_g_L'][:common])) > 1e-8, model

    print('[17/22] six-species legend and browser-opening paths are not clipped/imported as HTML tables')
    six = make_config(MODEL_EDM_LANGMUIR, 5)
    six['numerics'].update(time_steps=60, axial_positions=10)
    six_result = simulate(six)
    with tempfile.TemporaryDirectory() as td:
        paths = write_outputs(six, six_result, td)
        svg_text = Path(paths['svg']).read_text(encoding='utf-8')
        import xml.etree.ElementTree as ET
        root = ET.fromstring(svg_text)
        labels = [e.text for e in root.iter() if e.tag.endswith('text') and e.text]
        for label in ['Target protein','Impurity 1','Impurity 2','Impurity 3','Impurity 4','Impurity 5']:
            assert label in labels, (label, labels)
        # All legend text anchors must remain inside the 1000-pixel viewBox.
        for e in root.iter():
            if e.tag.endswith('text') and e.text in labels[:6] and 'x' in e.attrib:
                assert 0 <= float(e.attrib['x']) <= 1000
    assert 'OpenCurrentResult(0)' in jsl and 'OpenCurrentResult(1)' in jsl
    assert 'CHECK_RESULT_CURRENT' in jsl and '2026-10-01-R14' in jsl
    assert 'statusValue = Load Text File(statusFile)' in jsl
    assert 'If(openData == 1' in jsl
    assert 'If(actionText == "CHECK_RESULT_CURRENT", Save Text File(statusFile, "CHECK_RESULT_CURRENT pending"))' in jsl
    callback = jsl[jsl.index('OpenCurrentResult = Function'):jsl.index('AttachLSQCSV = Function')]
    assert callback.index('SendAndRun("CHECK_RESULT_CURRENT")') < callback.index('statusValue = Load Text File(statusFile)')
    assert 'statusValue = SendAndRun(' not in callback
    assert 'If(File Exists(resultFile), Web(resultFile)' in callback
    bridge_text = (ROOT / 'tdm_bridge.py').read_text(encoding='utf-8')
    assert 'RETURN_CODE != 0 and "tdm_ui_action" not in globals()' in bridge_text
    assert 'Contains(statusValue, "Accepted/applied to shared batch parameters: True") > 0' in jsl
    assert 'OpenFitReport = Function' in jsl and 'OpenBatchGallery = Function' in jsl
    assert 'OpenBatchSummary = Function' in jsl and 'OpenLSQReport = Function' in jsl
    assert 'Button Box("Open batch gallery", OpenBatchGallery())' in jsl
    assert 'Button Box("Open batch summary", OpenBatchSummary())' in jsl
    assert 'Button Box("Open fit report", OpenFitReport())' in jsl
    assert 'Button Box("Open LSQ report", OpenLSQReport())' in jsl
    assert 'Button Box("Open LSQ objective log", OpenLSQHistory())' in jsl
    assert 'c2UVResponse =' in jsl and 'tdm_ui_c2_uv_response_factor_mAU_L_g' in jsl
    assert 'Web(batchGallery)' in jsl and 'Web(lsqReport)' in jsl
    assert 'Button Box("Open latest chromatogram", If(File Exists(resultFile)' not in jsl
    assert 'Button Box("Open latest result data", If(File Exists(resultCSV)' not in jsl

    print('[18/22] Buffer B % gradients resolve from Buffer A/B and feed CPA calculations')
    for model in (MODEL_EDM_CPA, MODEL_TDM_CPA):
        g=make_config(model,0,multistep=False)
        g['numerics'].update(time_steps=50, axial_positions=6)
        g['feed']['load_density_mg_mL_resin']=0.10
        p=g['process']
        p['buffer_A'].update(pH=5.5, salt_concentration_M=0.05, conductivity_mS_cm=5.0)
        p['buffer_B'].update(pH=7.5, salt_concentration_M=0.45, conductivity_mS_cm=45.0)
        for buf in (p['buffer_A'], p['buffer_B']):
            for ion in buf['ions']: ion.update(species='', concentration_M=None, valency=None)
        if model == MODEL_TDM_CPA:
            p['buffer_A']['ions'][0].update(species='Na+', concentration_M=0.10, valency=1)
            p['buffer_A']['ions'][1].update(species='Cl-', concentration_M=0.10, valency=-1)
            p['buffer_B']['ions'][0].update(species='Na+', concentration_M=0.50, valency=1)
            p['buffer_B']['ions'][1].update(species='Cl-', concentration_M=0.50, valency=-1)
            p['buffer_A']['ionic_strength_source'] = 'ION_COMPOSITION'
            p['buffer_B']['ionic_strength_source'] = 'ION_COMPOSITION'
        p['load_source']='BUFFER_A'
        p['plw_steps'][0].update(mode='LINEAR', CV=0.05, chemistry_control='BUFFER_B_PERCENT', start_percent_B=10.0, end_percent_B=30.0)
        p['elution_steps'][0].update(mode='LINEAR', CV=0.15, chemistry_control='BUFFER_B_PERCENT', start_percent_B=25.0, end_percent_B=75.0)
        row=g['components'][0]
        row.update(pH_ref=6.0, Z1_per_pH=-4.0, Z2_per_pH2=0.0, Z3_per_pH3=0.0, delta_pH_slope_m2_C=0.0)
        q=buffer_blend_chemistry(g,25.0)
        assert_close(q['pH'],6.0); assert_close(q['salt_concentration_M'],0.15)
        if model == MODEL_TDM_CPA: assert_close(q['ionic_strength_M'],0.20)
        else: assert_close(q['ionic_strength_M'],0.15)
        assert not validate_config(g), (model, validate_config(g))
        pg=_build_program(g,np.array([1.0]))
        el=pg.stages[-1]
        assert_close(el.start_pH,6.0); assert_close(el.end_pH,7.0)
        assert_close(el.start_salt_M,0.15); assert_close(el.end_salt_M,0.35)
        mid=el.chemistry(0.5)
        assert_close(mid[0],6.5); assert_close(mid[1],0.25)
        r_pct=simulate(g)
        assert np.all(np.isfinite(r_pct['trace']['total_g_L']))
        assert np.all(np.isfinite(r_pct['trace']['pH']))
        assert np.all(np.isfinite(r_pct['trace']['salt_concentration_M']))
        changed=copy.deepcopy(g); changed['process']['elution_steps'][0].update(start_percent_B=60.0,end_percent_B=100.0)
        pg_changed=_build_program(changed,np.array([1.0]))
        assert pg_changed.stages[-1].start_salt_M > el.start_salt_M and pg_changed.stages[-1].end_salt_M > el.end_salt_M
    row0=load_batch_rows()[0]
    assert 'PLW1_Chemistry_Control' in row0 and 'Elution1_Start_Percent_B' in row0 and 'Elution1_End_Percent_B' in row0
    mapped=batch_row_to_run_config(row0)
    assert mapped['process']['plw_steps'][0]['chemistry_control'] in {'BUFFER_B_PERCENT','ENDPOINT_CHEMISTRY'}

    print('[19/22] inline %B run inputs propagate through all four model programs and outputs')
    for model in (MODEL_EDM_LANGMUIR, MODEL_TDM_LANGMUIR, MODEL_EDM_CPA, MODEL_TDM_CPA):
        g=make_config(model,0,multistep=False)
        g['numerics'].update(time_steps=50, axial_positions=6)
        p=g['process']
        p['buffer_A'].update(pH=5.8, salt_concentration_M=0.04, conductivity_mS_cm=4.0)
        p['buffer_B'].update(pH=7.0, salt_concentration_M=0.44, conductivity_mS_cm=44.0)
        p['plw_steps'][0].update(mode='LINEAR', chemistry_control='BUFFER_B_PERCENT', start_percent_B=10.0, end_percent_B=30.0, CV=0.5)
        p['elution_steps'][0].update(mode='LINEAR', chemistry_control='BUFFER_B_PERCENT', start_percent_B=20.0, end_percent_B=80.0, CV=2.0)
        if uses_cpa(model):
            g['components'][0].update(pH_ref=6.0, Z1_per_pH=-2.0, Z2_per_pH2=0.0, Z3_per_pH3=0.0, delta_pH_slope_m2_C=0.0)
        pg=_build_program(g,np.array([1.0]))
        plw=pg.stages[-2]; el=pg.stages[-1]
        assert_close(plw.start_salt_M,0.08); assert_close(plw.end_salt_M,0.16)
        assert_close(el.start_salt_M,0.12); assert_close(el.end_salt_M,0.36)
        assert_close(el.start_pH,6.04); assert_close(el.end_pH,6.76)
        r=simulate(g)
        for key in ('pH','salt_concentration_M','conductivity_mS_cm','ionic_strength_M','stage_index'):
            assert key in r['trace'] and np.all(np.isfinite(r['trace'][key])), (model,key)
        assert 'percent_B' in r['trace'] and np.any(np.isfinite(r['trace']['percent_B'])), model
        assert_close(pg.percent_B_at_cv(pg.stage_start_CV[-1] + 0.5 * el.duration_CV), 50.0)
        changed=copy.deepcopy(g)
        changed['process']['elution_steps'][0].update(start_percent_B=60.0,end_percent_B=100.0)
        r2=simulate(changed)
        # Every model carries the changed process chemistry; CPA and the
        # exponentially modified HIC Langmuir both consume the salt profile.
        common=min(len(r['trace']['salt_concentration_M']),len(r2['trace']['salt_concentration_M']))
        assert np.max(np.abs(r['trace']['salt_concentration_M'][:common]-r2['trace']['salt_concentration_M'][:common])) > 1e-6
        common_y=min(len(r['trace']['total_g_L']),len(r2['trace']['total_g_L']))
        dy=np.max(np.abs(r['trace']['total_g_L'][:common_y]-r2['trace']['total_g_L'][:common_y]))
        assert dy > 1e-10, (model,dy)

    # End-to-end result files must respond independently when the saved JMP run
    # table changes CV, flow or %B. The CV axis remains in column volumes; the
    # elapsed-time axis and point table expose flow changes. The %B panel changes
    # for HIC Langmuir, and both CPA and Langmuir protein curves respond to salt.
    print('[19a] CV, flow and %B edits update result tables and chromatogram axes')
    table_base = make_config(MODEL_EDM_CPA, 0, multistep=False)
    table_base['numerics'].update(time_steps=50, axial_positions=6)
    table_base['process']['load_source'] = 'BUFFER_A'
    table_base['process']['plw_steps'][0].update(mode='LINEAR', CV=0.5, start_percent_B=10.0, end_percent_B=30.0)
    table_base['process']['elution_steps'][0].update(mode='LINEAR', CV=2.0, start_percent_B=20.0, end_percent_B=80.0)
    variants = {
        'base': copy.deepcopy(table_base),
        'CV': copy.deepcopy(table_base),
        'flow': copy.deepcopy(table_base),
        '%B': copy.deepcopy(table_base),
    }
    variants['CV']['process']['elution_steps'][0]['CV'] = 2.5
    variants['flow']['process']['elution_steps'][0]['flow_mL_min'] *= 2.0
    variants['%B']['process']['elution_steps'][0].update(start_percent_B=60.0, end_percent_B=100.0)
    with tempfile.TemporaryDirectory() as td:
        outputs = {}
        tables = {}
        sims = {}
        for key, cfg in variants.items():
            sims[key] = simulate(cfg)
            out_dir = Path(td) / key.replace('%', 'pct')
            outputs[key] = write_outputs(cfg, sims[key], out_dir)
            with open(outputs[key]['csv'], encoding='utf-8', newline='') as fh:
                tables[key] = list(csv.DictReader(fh))
        base_manifest = json.loads(Path(outputs['base']['manifest']).read_text(encoding='utf-8'))
        cv_manifest = json.loads(Path(outputs['CV']['manifest']).read_text(encoding='utf-8'))
        assert cv_manifest['total_CV'] > base_manifest['total_CV']
        assert float(tables['CV'][-1]['CV']) > float(tables['base'][-1]['CV'])
        assert Path(outputs['CV']['svg']).read_text(encoding='utf-8') != Path(outputs['base']['svg']).read_text(encoding='utf-8')
        assert float(tables['flow'][-1]['CV']) == float(tables['base'][-1]['CV'])
        assert float(tables['flow'][-1]['time_min']) != float(tables['base'][-1]['time_min'])
        assert {float(r['flow_mL_min']) for r in tables['flow']} != {float(r['flow_mL_min']) for r in tables['base']}
        assert Path(outputs['flow']['svg']).read_text(encoding='utf-8') != Path(outputs['base']['svg']).read_text(encoding='utf-8')
        base_b = [r['percent_B'] for r in tables['base'] if r['percent_B']]
        changed_b = [r['percent_B'] for r in tables['%B'] if r['percent_B']]
        assert base_b != changed_b
        assert Path(outputs['%B']['svg']).read_text(encoding='utf-8') != Path(outputs['base']['svg']).read_text(encoding='utf-8')
        b_dy = np.max(np.abs(sims['%B']['trace']['total_g_L'] - sims['base']['trace']['total_g_L']))
        assert b_dy > 1e-10

        langmuir = copy.deepcopy(table_base)
        langmuir['model'] = MODEL_EDM_LANGMUIR
        langmuir['edm']['total_bed_porosity'] = table_base['edm']['total_bed_porosity']
        # Supply the selected model's required competitive-Langmuir parameters.
        langmuir['components'][0].update(qmax_g_L=300.0, H=1.5, b_L_g=0.005, D_app_mm2_s=0.067)
        l_base = simulate(langmuir)
        l_b = copy.deepcopy(langmuir)
        l_b['process']['elution_steps'][0].update(start_percent_B=60.0, end_percent_B=100.0)
        l_changed = simulate(l_b)
        assert np.max(np.abs(l_base['trace']['total_g_L'] - l_changed['trace']['total_g_L'])) > 1e-10
        assert not np.allclose(l_base['trace']['percent_B'], l_changed['trace']['percent_B'], equal_nan=True)

    print('[20/22] mass-balance integration works when NumPy has no trapz attribute')
    if hasattr(np, 'trapezoid'):
        legacy_trapz = getattr(np, 'trapz', None)
        try:
            if legacy_trapz is not None:
                delattr(np, 'trapz')
            assert_close(_trapezoidal_integral(np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 2.0])), 1.0)
        finally:
            if legacy_trapz is not None:
                np.trapz = legacy_trapz

    print('[21/22] opening another extraction and each action refreshes JMP persistent Python imports')
    import importlib
    import open_tdm_setup
    module_names = ('tdm_bridge', 'tdm_jmp', 'tdm_least_squares', 'tdm_minimal_io', 'tdm_minimal_model')
    original_path = list(sys.path)
    original_modules = {name: sys.modules.get(name) for name in module_names}
    try:
        with tempfile.TemporaryDirectory() as td:
            first_root = Path(td) / 'first'
            second_root = Path(td) / 'second'
            first_root.mkdir(); second_root.mkdir()
            for name in module_names:
                (first_root / f'{name}.py').write_text("ORIGIN = 'first'\n", encoding='utf-8')
                (second_root / f'{name}.py').write_text("ORIGIN = 'second'\n", encoding='utf-8')
            sys.path.insert(0, str(first_root))
            for name in module_names:
                sys.modules.pop(name, None)
                assert importlib.import_module(name).ORIGIN == 'first'
            open_tdm_setup._prepare_project_imports(second_root)
            assert Path(sys.path[0]).resolve() == second_root.resolve()
            for name in module_names:
                fresh = importlib.import_module(name)
                assert fresh.ORIGIN == 'second' and Path(fresh.__file__).resolve().parent == second_root

            # Each JMP button submits the bridge again in one persistent Python
            # interpreter. Verify the bridge itself flushes old project modules
            # before importing from the currently active extraction.
            bridge_project_modules = module_names[1:]
            sys.path.insert(0, str(first_root))
            for name in bridge_project_modules:
                sys.modules.pop(name, None)
                assert importlib.import_module(name).ORIGIN == 'first'
            original_modules['tdm_bridge']._prepare_project_imports(second_root)
            assert Path(sys.path[0]).resolve() == second_root.resolve()
            for name in bridge_project_modules:
                fresh = importlib.import_module(name)
                assert fresh.ORIGIN == 'second' and Path(fresh.__file__).resolve().parent == second_root
    finally:
        sys.path[:] = original_path
        for name, module in original_modules.items():
            sys.modules.pop(name, None)
            if module is not None:
                sys.modules[name] = module

    print('[22/22] raw CSV attachment, composition fit improvement, and JMP parameter apply')
    assert lsq._normalize_chromatogram_path_text(r'\C:\Users\etm50491\Downloads\Corabotase DBC Run (1).csv') == r'C:\Users\etm50491\Downloads\Corabotase DBC Run (1).csv'
    assert lsq._normalize_chromatogram_path_text('/C:/Users/etm50491/Downloads/run.csv') == 'C:/Users/etm50491/Downloads/run.csv'
    assert lsq._normalize_chromatogram_path_text(r'\\server\share\run.csv') == r'\\server\share\run.csv'
    cfg = make_config(MODEL_EDM_LANGMUIR, 1, multistep=False)
    cfg['numerics'].update(time_steps=60, axial_positions=8)
    cfg['process']['load_CV'] = 0.4
    cfg['feed']['load_density_mg_mL_resin'] = 0.8
    cfg['process']['elution_steps'][0].update(mode='LINEAR',CV=5.0,flow_mL_min=1.0,start_percent_B=0.0,end_percent_B=100.0)
    cfg['components'][0]['salt_sensitivity_per_M'] = 3.0
    for key in ('buffer_A','buffer_B'):
        cfg['process'][key].update(salt_concentration_M=None,conductivity_mS_cm=5.0)
    truth = copy.deepcopy(cfg)
    for i, row in enumerate(truth['components'][:2]):
        row['qmax_g_L'] *= (1.7 if i == 0 else 0.65)
        row['b_L_g'] *= (0.7 if i == 0 else 1.8)
    truth['components'][1]['uv_response_factor_mAU_L_g'] = 80.0
    truth_result = simulate(truth)
    truth_trace = truth_result['trace']
    peak = int(np.argmax(truth_trace['total_g_L']))
    peak_total = float(truth_trace['total_g_L'][peak])
    composition = [{
        'CV': float(truth_trace['CV'][peak]),
        'mass_percent': {
            name: 100.0 * float(truth_trace['component_g_L'][peak, i]) / peak_total
            for i, name in enumerate(truth_result['simulation']['component_names'])
        },
    }]
    lsq_bridge_sent = {
        'tdm_ui_lsq_mode': 'Least-squares: chromatogram + species mass% at CV',
        'tdm_ui_lsq_target_run': 2,
        'tdm_ui_lsq_reference_count': 1,
        'tdm_ui_lsq_r1_CV': composition[0]['CV'],
    }
    for i, name in enumerate(truth_result['simulation']['component_names'], start=1):
        lsq_bridge_sent[f'tdm_ui_lsq_r1_c{i}_mass_percent'] = composition[0]['mass_percent'][name]
    bridge_old = {name: (hasattr(tdm_bridge, name), getattr(tdm_bridge, name, None)) for name in lsq_bridge_sent}
    try:
        for name, value in lsq_bridge_sent.items():
            setattr(tdm_bridge, name, value)
        assert tdm_bridge._lsq_mode() == MODE_CHROMATOGRAM_AND_COMPOSITION
        mapped_refs = tdm_bridge._composition_references_from_jmp(cfg)
        assert_close(mapped_refs[0]['CV'], composition[0]['CV'])
        for name, value in composition[0]['mass_percent'].items():
            assert_close(mapped_refs[0]['mass_percent'][name], value)
        assert tdm_bridge._lsq_target_run(1) == 2
        saved_rows = [dict(row) for row in load_batch_rows()]
        saved_rows[1].update({
            'Run_Name': 'LSQ target validation run',
            'Load_Flow_mL_min': '2.75',
            'Feed_Concentration_mg_mL': '3.25',
            'Load_CV': '1.6',
            'Load_Density_mg_mL_resin': '4.5',
            'Time_Steps': '123',
            'Axial_Positions': '17',
            'PLW_Count': '1',
            'PLW1_CV': '0.47',
            'PLW1_Flow_mL_min': '1.35',
            'PLW1_Start_Percent_B': '12',
            'PLW1_End_Percent_B': '34',
            'Elution_Count': '1',
            'Elution1_CV': '8.5',
            'Elution1_Flow_mL_min': '1.8',
            'Elution1_Start_Percent_B': '34',
            'Elution1_End_Percent_B': '95',
            'BufferA_Name': 'Target Buffer A',
            'BufferB_Name': 'Target Buffer B',
            'LSQ_Chromatogram_CSV': '/profiles/gradient.csv',
        })
        original_loader = tdm_bridge.load_batch_rows
        original_saver = tdm_bridge.save_batch_rows
        saved_profiles = []
        try:
            tdm_bridge.load_batch_rows = lambda: saved_rows
            tdm_bridge.save_batch_rows = lambda rows: saved_profiles.append(copy.deepcopy(rows))
            target_cfg = tdm_bridge._effective_run(cfg, tdm_bridge._lsq_target_run(1))
            assert_close(target_cfg['column']['flow_mL_min'], 2.75)
            assert_close(target_cfg['feed']['total_concentration_mg_mL'], 3.25)
            assert_close(target_cfg['process']['load_CV'], 1.6)
            assert_close(target_cfg['feed']['load_density_mg_mL_resin'], 5.2)
            assert target_cfg['numerics']['time_steps'] == 123 and target_cfg['numerics']['axial_positions'] == 17
            assert target_cfg['process']['buffer_A']['name'] == 'Target Buffer A'
            assert_close(target_cfg['process']['plw_steps'][0]['CV'], 0.47)
            assert_close(target_cfg['process']['elution_steps'][0]['end_percent_B'], 95)
            target_context = tdm_bridge._fit_run_context(target_cfg, 2)
            assert target_context['run_number'] == 2 and target_context['run_name'] == 'LSQ target validation run'
            assert target_context['chromatogram_csv'] == '/profiles/gradient.csv'
            assert target_context['feed'] == target_cfg['feed'] and target_context['process'] == target_cfg['process']
            assert tdm_bridge._lsq_profile_csv_path(2) == '/profiles/gradient.csv'
            assert tdm_bridge._lsq_profile_csv_path(2, '/profiles/override.csv') == '/profiles/override.csv'
            tdm_bridge._save_lsq_profile_csv_path(2, '/profiles/updated_gradient.csv')
            assert saved_profiles[0][1]['LSQ_Chromatogram_CSV'] == '/profiles/updated_gradient.csv'
        finally:
            tdm_bridge.load_batch_rows = original_loader
            tdm_bridge.save_batch_rows = original_saver
        lsq_jsl = build_jsl(cfg)
        for name in ('tdm_ui_lsq_r1_CV', 'tdm_ui_lsq_r1_c1_mass_percent', 'tdm_ui_lsq_r1_c2_mass_percent'):
            assert name in lsq_jsl
        assert 'tdm_ui_lsq_target_run' in lsq_jsl and 'Chromatogram target run and process recipe' in lsq_jsl
    finally:
        for name, (existed, value) in bridge_old.items():
            if existed:
                setattr(tdm_bridge, name, value)
            elif hasattr(tdm_bridge, name):
                delattr(tdm_bridge, name)
    lsq_paths = {
        'ATTACHED_CHROMATOGRAM_CSV': 'attached.csv',
        'FIT_RESULTS_CSV': 'fit_results.csv',
        'FIT_TRACE_CSV': 'fit_trace.csv',
        'FIT_COMPOSITION_CSV': 'fit_composition.csv',
        'FIT_REPORT_HTML': 'fit_report.html',
        'FIT_CONFIG_JSON': 'fitted.json',
        'FIT_APPLY_JSL': 'apply.jsl',
        'FIT_SETTINGS_JSON': 'settings.json',
        'FIT_HISTORY_CSV': 'history.csv',
        'CONFIG_PATH': 'active.json',
    }
    old_lsq_paths = {name: getattr(lsq, name) for name in lsq_paths}
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name, filename in lsq_paths.items():
                setattr(lsq, name, root / filename)
            common_export = root / 'common_export.csv'
            common_export.write_text(
                'Sample_ID,Column Volume (mL),UV1_280nm (mAU)\n1,0,0\n2,1,10\n3,2,0\n',
                encoding='utf-8',
            )
            detected = lsq.inspect_chromatogram_csv(common_export)
            assert detected['x_column'] == 'Column Volume (mL)' and detected['x_unit'] == 'ML'
            assert detected['signal_column'] == 'UV1_280nm (mAU)'
            instrument_export = root / 'instrument_export.csv'
            instrument_export.write_text(
                'Instrument export;Version 1\nSample;Run 7\n\nRetention Time (min);UV 280 (mAU)\n'
                '0,00;1,20\n0,25;15,50\n0,50;7,00\n0,75;2,10\n',
                encoding='utf-8',
            )
            instrument_info = lsq.inspect_chromatogram_csv(instrument_export)
            assert instrument_info['x_column'] == 'Retention Time (min)' and instrument_info['x_unit'] == 'MIN'
            assert instrument_info['signal_column'] == 'UV 280 (mAU)' and instrument_info['rows'] == 4
            instrument_ref = lsq.load_chromatogram_reference(instrument_export, cfg)
            assert np.allclose(instrument_ref.signal, [1.2, 15.5, 7.0, 2.1])
            timestamp_export = root / 'timestamp_export.csv'
            timestamp_export.write_text(
                'Time,Detector 1\n00:00:00,0\n00:00:30,10\n00:01:00,1\n', encoding='utf-8'
            )
            timestamp_info = lsq.inspect_chromatogram_csv(timestamp_export)
            assert timestamp_info['x_unit'] == 'S' and timestamp_info['signal_column'] == 'Detector 1'
            timestamp_ref = lsq.load_chromatogram_reference(timestamp_export, cfg)
            timestamp_program = _build_program(cfg, np.zeros(2))
            assert_close(timestamp_ref.cv[-1], timestamp_program.time_to_cv(60.0))
            signal_only_export = root / 'signal_only.csv'
            signal_only_export.write_text('UV 280 (mAU)\n1.2\n15.5\n7.0\n2.1\n', encoding='utf-8')
            signal_only_info = lsq.inspect_chromatogram_csv(signal_only_export)
            assert signal_only_info['x_column'] == 'sample order (assumed)' and signal_only_info['x_unit'] == 'INDEX'
            assert signal_only_info['x_axis_assumption']
            utf16_export = root / 'utf16_export.csv'
            utf16_export.write_bytes(
                'Sample_ID,CV,UV_mAU\n1,0,0\n2,1,10\n3,2,0\n'.encode('utf-16')
            )
            utf16_attached = lsq.attach_chromatogram_csv(utf16_export)
            assert utf16_attached['rows'] == 3 and utf16_attached['x_column'] == 'CV'
            ml_config = copy.deepcopy(cfg)
            ml_config['column']['volume_mL'] = 2.0
            ml_reference = lsq.load_chromatogram_reference(common_export, ml_config)
            assert np.allclose(ml_reference.cv, [0.0, 0.5, 1.0])
            index_reference = lsq.load_chromatogram_reference(signal_only_export, cfg)
            program_end_cv = _build_program(cfg, np.zeros(2)).total_CV
            assert_close(index_reference.cv[0], 0.0)
            assert_close(index_reference.cv[-1], program_end_cv)
            assert index_reference.x_axis_assumption
            raw = root / 'raw.csv'
            with raw.open('w', newline='', encoding='utf-8') as fh:
                writer = csv.writer(fh); writer.writerow(['CV', 'UV_mAU'])
                for cv, y in zip(truth_trace['CV'], truth_trace['uv_mAU']):
                    writer.writerow([float(cv), 5.0 + float(y)])
            attached = lsq.attach_chromatogram_csv(f'"{raw}"')
            assert attached['rows'] == len(truth_trace['CV']) and attached['x_unit'] == 'CV'
            attached_without_extension = lsq.attach_chromatogram_csv(raw.with_suffix(''))
            assert Path(attached_without_extension['source_path']) == raw.resolve()
            assert lsq.attach_chromatogram_csv(lsq.ATTACHED_CHROMATOGRAM_CSV)['rows'] == len(truth_trace['CV'])
            test_run_context = {
                'run_number': 3,
                'run_name': 'LSQ report target recipe',
                'chromatogram_csv': str(raw.resolve()),
                'model': cfg['model'],
                'column': copy.deepcopy(cfg['column']),
                'feed': copy.deepcopy(cfg['feed']),
                'numerics': copy.deepcopy(cfg['numerics']),
                'process': copy.deepcopy(cfg['process']),
            }
            fit = lsq.run_least_squares_refinement(
                cfg,
                mode=MODE_CHROMATOGRAM_AND_COMPOSITION,
                chromatogram_csv=raw,
                composition_references=composition,
                run_context=test_run_context,
                max_nfev=10,
            )
            assert fit['accepted'] and fit['final_objective'] < 0.9 * fit['initial_objective']
            for key in ('fit_results_csv', 'fit_trace_csv', 'fit_composition_csv', 'fit_report_html', 'fit_config_json', 'apply_jsl', 'fit_history_csv'):
                assert Path(fit[key]).is_file(), (key, fit[key])
            apply_text = Path(fit['apply_jsl']).read_text(encoding='utf-8')
            for variable in ('c1Qmax', 'c1B', 'c1SaltSensitivity', 'c2Qmax', 'c2B', 'c2SaltSensitivity', 'c2UVResponse'):
                assert f'{variable} << Set(' in apply_text and f'{variable} =' in build_jsl(cfg)
            fitted_uv=float(fit['fitted_config']['components'][1]['uv_response_factor_mAU_L_g'])
            assert not math.isclose(fitted_uv, float(cfg['conversion']['uv_to_protein_mAU_L_g']), rel_tol=1e-4)
            with open(fit['fit_history_csv'],encoding='utf-8',newline='') as history_fh:
                history_rows=list(csv.DictReader(history_fh))
            assert len(history_rows)>2 and all(float(row['best_objective']) <= float(row['objective']) + 1e-12 for row in history_rows)
            assert float(history_rows[-1]['best_objective']) <= float(history_rows[0]['objective'])
            assert 'UpdateUI();' in apply_text
            assert 'SyncLangmuirDerived();' in build_jsl(cfg)
            report = Path(fit['fit_report_html']).read_text(encoding='utf-8')
            assert 'sum of each species concentration multiplied by its own response factor' in report and 'Objective improvement over time' in report
            assert 'Fit target run and operating recipe' in report
            assert 'LSQ report target recipe' in report and 'PLW steps' in report
            fit_settings = json.loads(Path(lsq.FIT_SETTINGS_JSON).read_text(encoding='utf-8'))
            assert fit_settings['target_run_context']['run_number'] == 3
            assert fit_settings['target_run_context']['run_name'] == 'LSQ report target recipe'
            assert fit_settings['reference_csv'] == str(raw.resolve())
            assert fit_settings['reference_x_axis_assumption'] is None
            assert Path(fit['fit_report_html']).name.endswith('_run_03.html')
            assert Path(fit['fit_settings_json']).name.endswith('_run_03.json')
            assert Path(fit['profile_apply_jsl']).is_file()
            assert Path(lsq.FIT_REPORT_HTML).is_file() and Path(lsq.FIT_SETTINGS_JSON).is_file()
            assert fit_settings['target_run_context']['process'] == test_run_context['process']
            saved = json.loads(Path(lsq.CONFIG_PATH).read_text(encoding='utf-8'))
            assert_close(saved['components'][0]['qmax_g_L'], fit['fitted_config']['components'][0]['qmax_g_L'])
            assert_close(saved['components'][0]['H'], saved['components'][0]['qmax_g_L'] * saved['components'][0]['b_L_g'])

            # A locked global fit evaluates each chromatogram with its own
            # column and process recipe while preserving one shared model
            # parameter set and writing objective/recipe audit files.
            global_base = make_config(MODEL_EDM_LANGMUIR, 1, multistep=False)
            global_base['numerics'].update(time_steps=50, axial_positions=6)
            global_base['process']['elution_steps'][0].update(
                mode='LINEAR', CV=4.0, flow_mL_min=1.0,
                start_percent_B=0.0, end_percent_B=100.0,
            )
            global_second = copy.deepcopy(global_base)
            global_second['column'].update(volume_mL=1.7, length_mm=65.0)
            global_second['column']['flow_mL_min'] = 1.4
            global_second['process']['load_amount_basis'] = 'CAPACITY'
            global_second['process']['elution_steps'][0].update(
                CV=5.0, flow_mL_min=1.4, start_percent_B=10.0, end_percent_B=90.0,
            )
            global_profiles = []
            for profile_number, profile_config in enumerate((global_base, global_second), start=1):
                prediction = simulate(profile_config)['trace']
                profile_csv = root / f'global_profile_{profile_number}.csv'
                with profile_csv.open('w', encoding='utf-8', newline='') as profile_fh:
                    profile_writer = csv.writer(profile_fh)
                    profile_writer.writerow(['CV', 'UV_mAU'])
                    for cv, signal in zip(prediction['CV'], prediction['uv_mAU']):
                        profile_writer.writerow([float(cv), 5.0 + float(signal)])
                profile_process = copy.deepcopy(profile_config['process'])
                global_profiles.append({
                    'config': profile_config,
                    'chromatogram_csv': profile_csv,
                    'run_context': {
                        'run_number': profile_number,
                        'run_name': f'Global profile {profile_number}',
                        'column': copy.deepcopy(profile_config['column']),
                        'feed': copy.deepcopy(profile_config['feed']),
                        'numerics': copy.deepcopy(profile_config['numerics']),
                        'process': profile_process,
                    },
                })
            global_fit = lsq.run_global_least_squares_refinement(
                global_profiles, unlocked_paths=set(), max_nfev=2,
            )
            assert global_fit['accepted'] and global_fit['nfev'] == 0
            assert global_fit['parameters'] == []
            assert len(global_fit['profile_metrics']) == 2
            assert global_fit['final_objective'] <= global_fit['initial_objective'] + 1e-12
            global_settings = json.loads(Path(global_fit['fit_settings_json']).read_text(encoding='utf-8'))
            assert global_settings['profile_count'] == 2
            assert len(global_settings['locked_parameter_paths']) > 0
            assert global_settings['profiles'][0]['recipe']['process'] != global_settings['profiles'][1]['recipe']['process']
            assert Path(global_fit['fit_history_csv']).is_file()
            assert 'Global least-squares refinement' in Path(global_fit['fit_report_html']).read_text(encoding='utf-8')
            catalog_paths = {spec.path for spec in lsq.fit_parameter_lock_catalog(global_base)}
            assert 'components[0].qmax_g_L' in catalog_paths
            assert 'conversion.conductivity_to_salt_M_per_mS_cm' in catalog_paths
            assert 'column.volume_mL' not in catalog_paths and 'column.length_mm' not in catalog_paths
            assert len(lsq.fit_parameter_specs(
                global_base, unlocked_paths={'components[0].qmax_g_L'},
            )) == 1
            print(f"    fit objective: {fit['initial_objective']:.6g} -> {fit['final_objective']:.6g}")
    finally:
        for name, value in old_lsq_paths.items():
            setattr(lsq, name, value)

    verify_button_paths()
    print('ALL GRAPHICS / BUFFER / INLINE-%B / SALT / ION / P_H-GRADIENT / CLASSIC-UI / SOLVER CHECKS PASSED')


if __name__ == '__main__':
    if '--buttons-only' in sys.argv:
        verify_button_paths()
    else:
        main()
