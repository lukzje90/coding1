"""R14 qmax unit/schema migration and feed-concentration handoff checks."""
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import numpy as np

from verify_r13 import make_hic_config, MODELS_ORDER
from tdm_minimal_io import (
    MODEL_EDM_LANGMUIR, MODEL_TDM_LANGMUIR, MODEL_LABELS, RESULT_GUARD_BUILD,
    normalize_config, load_config, save_config, default_batch_row,
    save_batch_rows, load_batch_rows, batch_row_to_run_config,
    apply_batch_shared_parameters,
)
from tdm_minimal_model import simulate, _langmuir_arrays, _langmuir_q_jac
from tdm_jmp import build_jsl, _validate_jsl
from tdm_least_squares import fit_parameter_specs
import tdm_bridge as bridge


def legacy_config(config):
    result = copy.deepcopy(config)
    for row in result['components']:
        row['qmax_mg_ml'] = row.pop('qmax_g_L')
    result['least_squares']['unlocked_paths'] = ['components[0].qmax_mg_ml']
    return result


def main():
    checks = []
    def passed(message):
        checks.append(message)
        print('PASS', message, flush=True)

    with tempfile.TemporaryDirectory() as td:
        folder = Path(td)
        for value in [.003, 3.6, 3600.]:
            canonical = make_hic_config()
            canonical['components'][0]['qmax_g_L'] = value
            old = legacy_config(canonical)
            path = folder / 'legacy.json'
            path.write_text(json.dumps(old))
            migrated = load_config(path)
            assert migrated['components'][0]['qmax_g_L'] == value
            assert 'qmax_mg_ml' not in migrated['components'][0]
            assert migrated['components'][0]['H'] == value * .001
            assert migrated['least_squares']['unlocked_paths'] == ['components[0].qmax_g_L']
            save_config(migrated, path)
            saved = json.loads(path.read_text())
            assert 'qmax_g_L' in saved['components'][0]
            assert 'qmax_mg_ml' not in saved['components'][0]
        passed('Legacy mg/mL capacities migrate to g/L with factor 1; saved keys and locks use g/L')

        selection = make_hic_config()
        selection['least_squares'].update(unlocked_paths=[], selected_run_numbers=[7, 2, 25, 30, 11])
        path = folder / 'selection.json'
        save_config(selection, path)
        restored_selection = load_config(path)
        assert restored_selection['least_squares']['unlocked_paths'] == []
        assert restored_selection['least_squares']['selected_run_numbers'] == [7, 2, 25, 30, 11]
        assert fit_parameter_specs(restored_selection) == []
        _validate_jsl(build_jsl(restored_selection))
        passed('All-locked state and five selected saved runs survive save/reload without truncation')

        both = {'qmax_g_L': 18., 'qmax_mg_ml': 3.6}
        assert normalize_config(both) == {'qmax_g_L': 18.}
        assert normalize_config({'qmax_g_L': None, 'qmax_mg_ml': 3.6}) == {'qmax_g_L': None}
        passed('Explicit new-schema values take precedence over legacy aliases, including cleared input')

        config = make_hic_config()
        config['components'][0]['qmax_g_L'] = 3.6
        _, H, b = _langmuir_arrays(config)
        q, _ = _langmuir_q_jac(np.array([1e12]), H, b)
        assert np.isclose(q[0], 3.6, rtol=1e-8)
        assert np.isclose(H[0], 3.6 * .001)
        passed('g/L qmax reaches the isotherm saturation limit and derived H without a scaling error')

        for model in [MODEL_EDM_LANGMUIR, MODEL_TDM_LANGMUIR]:
            config = make_hic_config(model)
            specs = fit_parameter_specs(config,
                unlocked_paths=['components[0].qmax_mg_ml'],
                candidate_paths=['components[0].qmax_mg_ml'])
            assert len(specs) == 1 and specs[0].path == 'components[0].qmax_g_L'
            assert '[g/L stationary phase]' in specs[0].label
            specs = fit_parameter_specs(config, unlocked_paths=[])
            assert specs == []
        passed('Old and new fitting-lock paths address the same g/L parameter in both Langmuir models')

        sent = {
            'tdm_ui_model': MODEL_LABELS[MODEL_EDM_LANGMUIR],
            'tdm_ui_c1_qmax_g_L': 3.6,
            'tdm_ui_c1_b_L_g': .25,
            'tdm_ui_lsq_parameter_lock_states': 'components[0].qmax_mg_ml=Unlocked;',
        }
        old = {name: (hasattr(bridge, name), getattr(bridge, name, None)) for name in sent}
        try:
            for name, value in sent.items():
                setattr(bridge, name, value)
            mapped = bridge.build_shared_config_from_jmp()
            assert mapped['components'][0]['qmax_g_L'] == 3.6
            assert mapped['components'][0]['H'] == .9
            assert bridge._lsq_unlocked_paths_from_jmp() == {'components[0].qmax_g_L'}
            delattr(bridge, 'tdm_ui_c1_qmax_g_L')
            bridge.tdm_ui_c1_qmax_mg_ml = 7.2
            assert bridge.build_shared_config_from_jmp()['components'][0]['qmax_g_L'] == 7.2
            bridge.tdm_ui_c1_qmax_g_L = None
            assert bridge.build_shared_config_from_jmp()['components'][0]['qmax_g_L'] is None
        finally:
            if hasattr(bridge, 'tdm_ui_c1_qmax_mg_ml'):
                delattr(bridge, 'tdm_ui_c1_qmax_mg_ml')
            for name, (present, value) in old.items():
                if present:
                    setattr(bridge, name, value)
                elif hasattr(bridge, name):
                    delattr(bridge, name)
        passed('JMP sends g/L qmax, legacy sends remain readable, and clearing an input stays authoritative')

        jsl = build_jsl(make_hic_config(impurities=5))
        _validate_jsl(jsl)
        assert 'Feed concentration [mg/mL]' in jsl and 'Feed total protein' not in jsl
        assert 'Saturation capacity qmax,i [g/L stationary phase]' in jsl
        for i in range(1, 7):
            assert f'tdm_ui_c{i}_qmax_g_L' in jsl
            assert f'components[{i-1}].qmax_g_L' in jsl
        assert 'feedConcentration << Get' in jsl and 'Feed_Concentration_mg_mL' in jsl
        passed('New labels, all six species sends, CSV feed binding and JSL structure are valid')

        row = default_batch_row(1)
        row.update(Feed_Concentration_mg_mL=.5, Load_Amount_Basis='CAPACITY',
                   Load_Density_mg_mL_resin=1., Load_Start_Percent_B=100.,
                   Load_End_Percent_B=100., Load_Mode='SET_POINT',
                   PLW1_Start_Percent_B=100., PLW1_End_Percent_B=100.,
                   Elution1_Start_Percent_B=100., Elution1_End_Percent_B=0.,
                   Time_Steps=180, Axial_Positions=8)
        csv_path = folder / 'runs.csv'
        save_batch_rows([row], csv_path)
        restored = load_batch_rows(csv_path)[0]
        run = batch_row_to_run_config(restored)
        assert run['feed']['total_concentration_mg_mL'] == .5
        assert run['process']['load_CV'] == 2.
        restored['Feed_Concentration_mg_mL'] = 1.
        assert batch_row_to_run_config(restored)['process']['load_CV'] == 1.
        passed('Feed concentration persists through run CSV and correctly determines capacity-based load CV')

        model_checks = []
        for model in MODELS_ORDER:
            base = make_hic_config(model, impurities=1, steps=180, cells=8)
            config, _ = apply_batch_shared_parameters(base, run)
            reference = simulate(config)
            migrated = simulate(legacy_config(config))
            np.testing.assert_array_equal(reference['trace']['uv_mAU'], migrated['trace']['uv_mAU'])
            np.testing.assert_array_equal(reference['trace']['column_component_g_L'], migrated['trace']['column_component_g_L'])
            for species, mb in reference['mass_balance'].items():
                assert np.isclose(mb['input_g'], .0005)
            assert np.isclose(sum(v['input_g'] for v in reference['mass_balance'].values()), .001)
            assert reference['parameter_receipt']['load_input_used']['feed_concentration_mg_mL'] == .5
            assert reference['parameter_receipt']['load_input_used']['required_feed_volume_CV'] == 2.
            np.testing.assert_allclose(reference['trace']['uv_mAU'],
                                       reference['trace']['component_uv_mAU'].sum(axis=1))
            increased = copy.deepcopy(config)
            increased['process']['load_amount_basis'] = 'LOAD_VOLUME'
            increased['feed']['total_concentration_mg_mL'] = 1.
            changed = simulate(increased)
            assert np.isclose(sum(v['input_g'] for v in changed['mass_balance'].values()), .002)
            assert not np.allclose(reference['trace']['uv_mAU'], changed['trace']['uv_mAU'])
            model_checks.append({'model': model, 'legacy_and_g_L_outputs_identical': True,
                                 'feed_0_5_mg_mL_input_g': .001,
                                 'feed_1_0_mg_mL_fixed_volume_input_g': .002})
            passed(model + ' legacy/g/L predictions match exactly; feed concentration changes load mass and UV')

    receipt = {'build': RESULT_GUARD_BUILD, 'unit_identity': '1 mg/mL = 1 g/L',
               'qmax_volume_basis': 'non-mobile stationary phase',
               'synthetic_tests': True, 'native_JMP_executed': False,
               'checks_passed': checks, 'model_checks': model_checks}
    Path(__file__).resolve().with_name('verification').joinpath('r14_verification.json').write_text(
        json.dumps(receipt, indent=2))
    print('ALL R14 UNIT AND FEED CHECKS PASSED', flush=True)


if __name__ == '__main__':
    main()
