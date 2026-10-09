"""Matched reference must preserve original floats, without transform drift."""
import copy, json, tempfile
from pathlib import Path
from verify_r13 import make_hic_config, reference_csv, redirected_fit_outputs
from tdm_minimal_model import simulate
import tdm_least_squares as lsq

config=make_hic_config(steps=90,cells=6)
config['components'][0].update(qmax_g_L=111.123456789123,b_L_g=.005)
result=simulate(config)
with tempfile.TemporaryDirectory() as td:
    folder=Path(td);old=redirected_fit_outputs(folder)
    try:
        path=folder/'matched.csv';reference_csv(path,result)
        paths=['components[0].qmax_g_L','components[0].b_L_g']
        single=lsq.run_least_squares_refinement(config,mode='CHROMATOGRAM',chromatogram_csv=path,unlocked_paths=paths,max_nfev=8)
        global_fit=lsq.run_global_least_squares_refinement([{'config':config,'chromatogram_csv':path}],unlocked_paths=paths,max_nfev=8)
        for fit in [single,global_fit]:
            assert fit['accepted'] and fit['final_objective']<=fit['initial_objective']
            for field in ['qmax_g_L','b_L_g']:
                assert fit['fitted_config']['components'][0][field]==config['components'][0][field]
        receipt={'single_initial_SSE':single['initial_objective'],'single_final_SSE':single['final_objective'],
                 'global_initial_SSE':global_fit['initial_objective'],'global_final_SSE':global_fit['final_objective'],
                 'original_values_preserved_exactly':True}
        Path(__file__).resolve().with_name('verification').joinpath('unchanged_fit_verification.json').write_text(json.dumps(receipt,indent=2))
        print('PASS single and global already-matching fits retain original numeric values exactly',flush=True)
    finally:
        for name,value in old.items():setattr(lsq,name,value)
