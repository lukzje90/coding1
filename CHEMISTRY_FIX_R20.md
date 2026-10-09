# R20 competitive Langmuir affinity handoff

Build `2026-10-08-R20` makes the `b` and `k_s` path observable from the JMP
boxes to the local column binding equation. The generated JMP script sends
`tdm_ui_c1_b_L_g` and `tdm_ui_c1_salt_sensitivity_per_M`; the bridge persists
both shared inputs and the EDM/TDM competitive Langmuir solvers calculate
`b_eff = b × exp(k_s × local binding salt)` at every cell and time point.

The R18 PDF's factor of 1000 M/(mS/cm) exceeded the HIC model domain, and its
1.3 g/L load exceeded the nominal 0.6 g/L stationary capacity. R19 already
rejects those salt inputs and identifies the early UV as breakthrough. With a
valid low-load recipe, an abrupt STEP from 100% B to 10% B drops affinity
enough that peak timing is dominated by displacement and detector delay.
At 300 reported points, doubling `b` or changing `k_s` still changes the
calculated elution trace, but a 50-point plot can make small timing shifts
look identical.

R20 reports each selected protein's received `b`, `k_s`, and calculated
`b_eff` at each programmed load/wash/elution endpoint. The `latest_results.csv`
column `<protein>_effective_b_L_g` records the local outlet-cell affinity at
every reported point, using the binding salt before the UV detector delay.
The result HTML explains that programmed endpoints and local salt differ
during mixer/column transients. JMP's **Run selected** status includes target
load and elution affinities. Changing `b` or `k_s` changes the settings
fingerprint; the previous result is stale until rerun.

`python verify_r20_langmuir_affinity.py` exercises generated JMP sends,
bridge globals, the shared run, solver local salt/affinity, CSV, HTML and
manifest. With the 1 mL EDM/Langmuir model, a 0.20 g/L target load, 100% B
load/wash, and a **LINEAR** 100→10% B elution over 6 CV, the checks predict:

| b [L/g] | k_s [M⁻¹] | UV peak [CV] |
| ---: | ---: | ---: |
| 0.0029 | 5 | 4.689 |
| 0.0058 | 5 | 5.223 |
| 0.0029 | 7 | 6.586 |

For this synthetic demonstration, Buffer A is 5.12 mS/cm, Buffer B is
86.38 mS/cm, and an **illustrative, uncalibrated** conductivity-to-salt
factor of 0.020 M/(mS/cm) resolves 0.1024 M and 1.7276 M. Use measured
salt molarities and calibrate the affinity parameters for experimental
predictions. A step at 10% B remains a valid run, but it is a poor profile
for fitting `b` and `k_s` from peak position alone.

Additional checks:

```text
python verify_jmp_interface_inputs.py
python verify_model.py --buttons-only
python verify_r18.py
python verify_r19_counterexample.py
```

The JMP source and Python bridge paths are executed or structurally audited;
a native JMP desktop session was not available in this verification environment.
