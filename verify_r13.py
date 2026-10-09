"""R13 physics, detector, equation-9, lock and 1–5-run regression tests.

All references generated here are synthetic numerical checks, not experimental
calibration data. Run from any directory: python verify_r13.py.
"""
from __future__ import annotations
import copy, csv, json, math, re, sys, tempfile, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tdm_minimal_io import *
from tdm_minimal_model import simulate, write_outputs, _trapezoidal_integral, _build_cpa_components, _cpa_model
import tdm_least_squares as lsq
from tdm_jmp import build_jsl, _validate_jsl

MODELS_ORDER = [MODEL_EDM_LANGMUIR, MODEL_TDM_LANGMUIR, MODEL_EDM_CPA, MODEL_TDM_CPA]


def make_hic_config(model=MODEL_EDM_LANGMUIR, impurities=0, *, steps=300, cells=12):
    c = blank_config()
    c.update(model=model, chromatography_mode='HIC', impurity_count=impurities)
    c['column'].update(volume_mL=1., length_mm=50., flow_mL_min=1.)
    c['feed'].update(total_concentration_mg_mL=1., load_density_mg_mL_resin=1.)
    c['numerics'].update(time_steps=steps, axial_positions=cells)
    c['conversion'].update(uv_to_protein_mAU_L_g=1000., conductivity_to_salt_M_per_mS_cm=.02)
    c['tdm'].update(bead_radius_um=25., void_fraction=.42, particle_porosity=.56, salt_axial_dispersion_mm2_s=.05)
    c['edm']['total_bed_porosity'] = .75
    p = c['process']
    p.update(load_amount_basis='LOAD_VOLUME', load_CV=1., load_mode='SET_POINT',
             load_start_percent_B=100., load_end_percent_B=100., plw_count=1, elution_count=2)
    p['buffer_A'].update(name='Low-salt Buffer A', pH=6., conductivity_mS_cm=0.)
    p['buffer_B'].update(name='High-salt Buffer B', pH=6., conductivity_mS_cm=100.)
    p['plw_steps'][0].update(mode='SET_POINT', CV=2., flow_mL_min=1., start_percent_B=100., end_percent_B=100.)
    p['elution_steps'][0].update(mode='LINEAR', CV=8., flow_mL_min=1., start_percent_B=100., end_percent_B=0.)
    p['elution_steps'][1].update(mode='SET_POINT', CV=4., flow_mL_min=1., start_percent_B=0., end_percent_B=0.)
    c['system'].update(buffer_dispersion_mL=.1, salt_dispersion_mm2_s=.05,
                       uv_dead_volume_mL=.15, conductivity_dead_volume_mL=.3)
    for i, r in enumerate(c['components'][:impurities+1]):
        r.update(mass_percent=100./(impurities+1), uv_response_factor_mAU_L_g=1000.+100.*i,
                 qmax_g_L=100., b_L_g=.001, salt_sensitivity_per_M=6.,
                 delta_ref=.1, As_m_inv=2e8, diameter_nm=11., kkin_star_s=1.,
                 D_app_mm2_s=.067, D_ax_mm2_s=.1, k_eff_um_s=20., accessible_particle_porosity=.41)
    return normalize_config(c)


def scalar_close(a, b, tolerance=1e-8):
    assert math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance), (a, b)


def before_elution(result):
    return result['simulation']['diagnostics']['protein_outlet_before_elution_percent_of_load']


def reference_csv(path, result, offset=0.0):
    tr = result['trace']
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['CV','UV_mAU'])
        w.writerows(zip(tr['CV'],tr['uv_mAU']+offset))


def redirected_fit_outputs(directory):
    names = ['CONFIG_PATH','FIT_RESULTS_CSV','FIT_TRACE_CSV','FIT_COMPOSITION_CSV','FIT_REPORT_HTML',
             'FIT_CONFIG_JSON','FIT_SETTINGS_JSON','FIT_HISTORY_CSV','FIT_APPLY_JSL']
    old={n:getattr(lsq,n) for n in names}
    for n, value in old.items(): setattr(lsq,n,directory/Path(value).name)
    return old


def main():
    started=time.perf_counter();checks=[]; retention=[]; fit_metrics=[]
    def passed(name):
        checks.append(name);print('PASS',name,flush=True)
    output=ROOT/'verification'; output.mkdir(exist_ok=True)
    print('Testing 100% B retention and 0% B elution in all four models',flush=True)
    for model in MODELS_ORDER:
        c=make_hic_config(model)
        result=simulate(c);mb=result['mass_balance']['Target protein'];tr=result['trace']
        pre=before_elution(result); recovered=100.*mb['outlet_g']/mb['input_g']
        peak=float(tr['CV'][np.argmax(tr['uv_mAU'])])
        assert pre<.01,(model,pre)
        assert recovered>99.5,(model,recovered)
        assert abs(mb['closure_error_percent_of_input'])<.05,(model,mb)
        assert 3.<peak<15.,(model,peak)
        assert np.max(tr['uv_mAU'])>0 and np.all(np.isfinite(tr['uv_mAU']))
        np.testing.assert_allclose(tr['uv_mAU'],np.sum(tr['component_uv_mAU'],axis=1))
        hold=copy.deepcopy(c)
        for stage in hold['process']['elution_steps'][:2]: stage.update(start_percent_B=100.,end_percent_B=100.)
        held=simulate(hold)['mass_balance']['Target protein']
        assert 100*held['outlet_g']/held['input_g']<.01,(model,held)
        low=copy.deepcopy(c)
        low['process'].update(load_start_percent_B=0.,load_end_percent_B=0.)
        low['process']['plw_steps'][0].update(start_percent_B=0.,end_percent_B=0.)
        low_result=simulate(low)
        assert before_elution(low_result)>90.,(model,before_elution(low_result))
        retention.append({'model':model,'before_elution_percent':pre,'recovery_percent':recovered,
                          'peak_CV':peak,'mass_closure_error_percent':mb['closure_error_percent_of_input']})
        folder=output/model;folder.mkdir(exist_ok=True)
        (folder/'hic_example.json').write_text(json.dumps(c,indent=2))
        write_outputs(c,result,folder,{'run_number':1,'run_name':'Synthetic HIC 100-to-0% B validation'})
        passed(model+' high-salt hold, low-salt breakthrough and gradient recovery')

    # Reference equation 9: exact residual vector, including loading/baseline points.
    c=make_hic_config();ref=lsq.ChromatogramReference(np.array([0.,1.,2.]),np.array([1.,3.,1.]),'CV','UV_mAU','CV','synthetic')
    trace={'CV':np.array([0.,1.,2.]),'total_g_L':np.array([0.,.002,.004]),'uv_mAU':np.array([0.,2.,4.])}
    residual,info=lsq._chromatogram_residuals(c,ref,trace)
    np.testing.assert_allclose(residual,[-1.,-1.,3.]);scalar_close(np.dot(residual,residual),11.)
    assert info['baseline']==0.
    c['least_squares']['objective']='NORMALIZED_MSE'
    normalized,info=lsq._chromatogram_residuals(c,ref,trace)
    scalar_close(np.dot(normalized,normalized),11./12.)
    passed('Equation 9 raw SSE and explicit optional normalized MSE')

    # HIC affinity is monotone with salt; native CPA still has the IEX salt trend.
    for model in (MODEL_EDM_CPA,MODEL_TDM_CPA):
        c=make_hic_config(model)
        comps=_build_cpa_components(c,need_tdm_kinetics=uses_tdm(model));native=_cpa_model(c,comps)
        assert native.equilibrium_factors(np.zeros(1),6.,2.)[0]>native.equilibrium_factors(np.zeros(1),6.,0.)[0]
        c['chromatography_mode']='ION_EXCHANGE';c['cpa'].update(ligand_surface_density_umol_m2=2.89,system_specific_adsorption_parameter=.03)
        c['components'][0]['Z_ref']=40.
        comps=_build_cpa_components(c,need_tdm_kinetics=uses_tdm(model));native=_cpa_model(c,comps)
        assert native.equilibrium_factors(np.zeros(1),6.,.05)[0]>native.equilibrium_factors(np.zeros(1),6.,2.)[0]
    bad=make_hic_config();bad['components'][0]['salt_sensitivity_per_M']=-1
    assert validate_config(bad)
    passed('HIC salt direction, IEX CPA preservation and negative HIC sensitivity rejection')

    # Changes in flow require volume-domain, independently delayed detectors.
    c=make_hic_config(MODEL_TDM_LANGMUIR,1,steps=200,cells=8)
    c['process']['plw_steps'][0]['flow_mL_min']=.4
    c['process']['elution_steps'][0]['flow_mL_min']=1.6
    base=simulate(c);tr=base['trace'];cv=tr['CV']
    expected=np.interp(cv-.3,cv,tr['column_conductivity_mS_cm'],left=tr['column_conductivity_mS_cm'][0])
    np.testing.assert_allclose(tr['conductivity_mS_cm'],expected)
    for i in range(2):
        np.testing.assert_allclose(tr['component_g_L'][:,i],np.interp(cv-.15,cv,tr['column_component_g_L'][:,i],left=0.))
    unlagged=copy.deepcopy(c);unlagged['system'].update(uv_dead_volume_mL=0.,conductivity_dead_volume_mL=0.)
    unlagged_result=simulate(unlagged)
    np.testing.assert_allclose(tr['column_component_g_L'],unlagged_result['trace']['column_component_g_L'])
    assert abs(base['mass_balance']['Target protein']['closure_error_percent_of_input'])<.05
    np.testing.assert_allclose(tr['programmed_percent_B'],[base['simulation']['program'].percent_B_at_cv(v) for v in cv])
    assert np.max(np.abs(tr['conductivity_mS_cm']-tr['programmed_conductivity_mS_cm']))>5.
    hidden=copy.deepcopy(c);hidden['system'].update(show_percent_B=False,show_conductivity=False)
    np.testing.assert_allclose(simulate(hidden)['trace']['uv_mAU'],tr['uv_mAU'])
    off=copy.deepcopy(c);off['system'].update(buffer_dispersion_enabled=False,salt_dispersion_enabled=False)
    zero=copy.deepcopy(c);zero['system'].update(buffer_dispersion_mL=0.,salt_dispersion_mm2_s=0.)
    off_r=simulate(off);zero_r=simulate(zero)
    np.testing.assert_allclose(off_r['trace']['uv_mAU'],zero_r['trace']['uv_mAU'])
    assert np.max(np.abs(off_r['trace']['uv_mAU']-tr['uv_mAU']))>.1
    passed('Independent detector volumes, UV units, variable flow, overlay and physical dispersion toggles')

    # Cross-check UI selectors and EVERY numeric fit lock, including global controls.
    jsl=build_jsl(make_hic_config());_validate_jsl(jsl)
    assert 'tdm_ui_lsq_run_count' in jsl
    for slot in range(1,6): assert f'tdm_ui_lsq_selected_run_{slot}' in jsl
    lock_send=next(line for line in jsl.splitlines() if 'tdm_ui_lsq_parameter_lock_states' in line)
    for var in ['uvToProtein','condToSalt','beadRadius','voidFraction','particlePorosity','edmPorosity',
                'ligandSurface','systemAdsorption','bufferDispersion','systemSaltDax','uvDeadVolume','condDeadVolume']:
        assert f'{var}FitLock << Get Selected' in lock_send,var
    # The bridge must exclude unselected attached CSVs and reject duplicate runs.
    import tdm_bridge as bridge
    rows=[default_batch_row(i) for i in range(1,31)]
    for i in range(30):rows[i]['LSQ_Chromatogram_CSV']=f'run{i+1}.csv'
    bridge.tdm_ui_lsq_run_count=3
    for slot,number in enumerate([7,2,25],1):setattr(bridge,f'tdm_ui_lsq_selected_run_{slot}',number)
    assert [n for n,_,_ in bridge._selected_lsq_profile_rows(rows)]==[7,2,25]
    bridge.tdm_ui_lsq_selected_run_3=7
    try:bridge._selected_lsq_profile_rows(rows);raise AssertionError('duplicate run accepted')
    except ValueError:pass
    bridge.tdm_ui_lsq_parameter_lock_states='components[0].b_L_g=Unlocked;components[0].qmax_g_L=Locked;system.uv_dead_volume_mL=Locked;'
    assert bridge._lsq_unlocked_paths_from_jmp()=={'components[0].b_L_g'}
    passed('Explicit arbitrary run selection, duplicate rejection and all global lock handoffs')

    with tempfile.TemporaryDirectory() as td:
        folder=Path(td);old=redirected_fit_outputs(folder)
        try:
            # An all-locked global evaluation proves every selected point enters
            # the raw sum, for every allowed number of runs.
            profiles=[]
            for n in range(1,6):
                cfg=make_hic_config(steps=90,cells=6)
                cfg['column']['volume_mL']=.8+.1*n
                cfg['process']['elution_steps'][0].update(CV=6.+n,flow_mL_min=.6+.2*n)
                r=simulate(cfg);path=folder/f'count_{n}.csv';reference_csv(path,r,offset=.123)
                profiles.append({'config':cfg,'chromatogram_csv':path,'run_context':{'run_number':n,'run_name':f'Count {n}'}})
            for count in range(1,6):
                fit=lsq.run_global_least_squares_refinement(profiles[:count],unlocked_paths=[])
                scalar_close(fit['initial_objective'],count*91*.123**2,1e-7)
                assert fit['parameters']==[] and fit['nfev']==0 and len(fit['profile_metrics'])==count
            for count in (0,6):
                try:lsq.run_global_least_squares_refinement((profiles+[profiles[0]])[:count],unlocked_paths=[]);raise AssertionError('invalid run count accepted')
                except ValueError:pass
            passed('Global RAW_SSE contains all points for 1, 2, 3, 4 and 5 profiles; invalid counts rejected')

            duplicate_path=folder/'duplicates.csv'
            duplicate_path.write_text('CV,UV_mAU\n0,1\n1,2\n1,3\n2,4\n')
            duplicate_ref=lsq.load_chromatogram_reference(duplicate_path,make_hic_config())
            assert len(duplicate_ref.cv)==4
            cfg=make_hic_config(impurities=1,steps=90,cells=6)
            r=simulate(cfg);path=folder/'composition_profile.csv';reference_csv(path,r)
            peak=int(np.argmax(r['trace']['uv_mAU']))
            fractions=100*r['trace']['component_g_L'][peak]/r['trace']['total_g_L'][peak]
            comp=[{'CV':float(r['trace']['CV'][peak]),'mass_percent':{row['name']:float(fractions[i]) for i,row in enumerate(active_components(cfg))}}]
            mixed_profiles=[{'config':cfg,'chromatogram_csv':path,'composition_references':comp}, {'config':cfg,'chromatogram_csv':path}]
            mixed=lsq.run_global_least_squares_refinement(mixed_profiles,mode='CHROMATOGRAM_AND_COMPOSITION',unlocked_paths=[])
            assert mixed['initial_objective']<1e-16
            assert 'Measured and predicted chromatograms' in Path(mixed['fit_report_html']).read_text()
            truth=make_hic_config(steps=120,cells=6);truth['system']['uv_dead_volume_mL']=.3
            r=simulate(truth);path=folder/'zero_volume.csv';reference_csv(path,r)
            guess=copy.deepcopy(truth);guess['system']['uv_dead_volume_mL']=0.
            zero_fit=lsq.run_least_squares_refinement(guess,mode='CHROMATOGRAM',chromatogram_csv=path,unlocked_paths=['system.uv_dead_volume_mL'],max_nfev=12)
            assert zero_fit['accepted'] and zero_fit['final_objective']<zero_fit['initial_objective']*.001,(zero_fit['initial_objective'],zero_fit['final_objective'])
            assert abs(zero_fit['fitted_config']['system']['uv_dead_volume_mL']-.3)<.01
            # Out-of-range points must trigger a clear validation failure.
            outside=lsq.ChromatogramReference(np.array([0.,1.,20.]),np.zeros(3),'CV','UV_mAU','CV','synthetic')
            try:lsq._chromatogram_residuals(truth,outside,r['trace']);raise AssertionError('out-of-range sample silently discarded')
            except ValueError:pass
            passed('Replicate samples, optional composition per profile, reference coverage, and a zero-start UV-volume fit')

            # Real nonlinear optimization in each model/isotherm, with separate
            # recipes and exactly one unlocked parameter.
            for model in MODELS_ORDER:
                profiles=[]
                parameter='components[0].b_L_g' if uses_langmuir(model) else 'components[0].delta_ref'
                field=parameter.rsplit('.',1)[1];truth_value=.001 if uses_langmuir(model) else .1
                for n in range(1,3):
                    truth=make_hic_config(model,steps=100,cells=6)
                    if n==2:
                        truth['process']['buffer_B']['conductivity_mS_cm']=85.
                        truth['process']['elution_steps'][0].update(CV=6.,flow_mL_min=1.3)
                    result=simulate(truth);path=folder/f'fit_{model}_{n}.csv';reference_csv(path,result)
                    guess=copy.deepcopy(truth);guess['components'][0][field]*=.65
                    profiles.append({'config':guess,'chromatogram_csv':path,'run_context':{'run_number':n,'run_name':f'{model} {n}'}})
                fit=lsq.run_global_least_squares_refinement(profiles,unlocked_paths=[parameter],max_nfev=12)
                assert fit['accepted'] and fit['final_objective']<fit['initial_objective']*.01,(model,fit['initial_objective'],fit['final_objective'])
                estimated=fit['fitted_config']['components'][0][field]
                assert abs(estimated/truth_value-1)<.03,(model,estimated)
                for spec in lsq.fit_parameter_lock_catalog(profiles[0]['config']):
                    if spec.path!=parameter:
                        scalar_close(lsq._get_path(fit['fitted_config'],spec.path),lsq._get_path(profiles[0]['config'],spec.path),1e-13)
                assert fit['parameters']==[parameter]
                fit_metrics.append({'model':model,'profiles':2,'parameter':parameter,'truth':truth_value,
                                    'fitted':estimated,'initial_SSE':fit['initial_objective'],'final_SSE':fit['final_objective'],
                                    'optimizer_success':fit['success']})
                passed(model+' two-profile nonlinear fit improves SSE >99%; locked values unchanged')
            # Five profiles refined together, rather than five independent fits.
            five=[]
            for n in range(1,6):
                truth=make_hic_config(steps=90,cells=6)
                truth['process']['buffer_B']['conductivity_mS_cm']=80.+4*n
                truth['process']['elution_steps'][0].update(CV=5.+n,flow_mL_min=.5+.2*n)
                result=simulate(truth);path=folder/f'five_{n}.csv';reference_csv(path,result)
                guess=copy.deepcopy(truth);guess['components'][0]['salt_sensitivity_per_M']=5.5
                five.append({'config':guess,'chromatogram_csv':path,'run_context':{'run_number':n,'run_name':f'Five {n}'}})
            fit=lsq.run_global_least_squares_refinement(five,unlocked_paths=['components[0].salt_sensitivity_per_M'],max_nfev=12)
            assert fit['accepted'] and len(fit['profile_metrics'])==5
            assert fit['final_objective']<fit['initial_objective']*.01
            assert abs(fit['fitted_config']['components'][0]['salt_sensitivity_per_M']-6.)<.02
            fit_metrics.append({'model':MODEL_EDM_LANGMUIR,'profiles':5,'parameter':'salt_sensitivity_per_M',
                                'truth':6.,'fitted':fit['fitted_config']['components'][0]['salt_sensitivity_per_M'],
                                'initial_SSE':fit['initial_objective'],'final_SSE':fit['final_objective'],'optimizer_success':fit['success']})
            passed('Five distinct recipes refined by one shared nonlinear parameter vector')
        finally:
            for n,v in old.items():setattr(lsq,n,v)

    # Both competitive isotherms must retain and recover an actual mixture.
    for model in (MODEL_TDM_LANGMUIR,MODEL_TDM_CPA):
        cfg=make_hic_config(model,1,steps=160,cells=8);result=simulate(cfg)
        assert before_elution(result)<.01
        for mb in result['mass_balance'].values():
            assert mb['outlet_g']/mb['input_g']>.995
            assert abs(mb['closure_error_percent_of_input'])<.05
    passed('Two-species competitive HIC retention, gradient elution and mass conservation')
    summary={'build':RESULT_GUARD_BUILD,'synthetic_references':True,'native_JMP_executed':False,
             'checks_passed':checks,'retention_elution':retention,'nonlinear_fits':fit_metrics,
             'elapsed_seconds':time.perf_counter()-started}
    (output/'r13_verification.json').write_text(json.dumps(summary,indent=2))
    print('ALL R13 CHECKS PASSED',len(checks),'groups',round(summary['elapsed_seconds'],1),'seconds',flush=True)

if __name__=='__main__':main()
