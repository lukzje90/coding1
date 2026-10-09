"""Check that a STEP load starts equilibrated at its effective END value."""
import copy, json
from pathlib import Path
from verify_r13 import make_hic_config, MODELS_ORDER, before_elution
from tdm_minimal_model import simulate

results=[]
for model in MODELS_ORDER:
    c=make_hic_config(model,steps=150,cells=8)
    c['process'].update(load_mode='STEP',load_start_percent_B=0.,load_end_percent_B=100.)
    result=simulate(c)
    before=before_elution(result)
    assert before<.01,(model,before)
    assert result['trace']['conductivity_mS_cm'][0]==100.
    assert result['trace']['programmed_percent_B'][0]==100.
    recovered=100.*result['mass_balance']['Target protein']['outlet_g']/result['mass_balance']['Target protein']['input_g']
    assert recovered>99.5,(model,recovered)
    results.append({'model':model,'before_elution_percent':before,'recovery_percent':recovered})
    print('PASS',model,'STEP start=0/end=100 equilibrates at 100% B',flush=True)
Path(__file__).resolve().with_name('verification').joinpath('step_load_verification.json').write_text(json.dumps(results,indent=2))
print('ALL STEP LOAD CHECKS PASSED',flush=True)
