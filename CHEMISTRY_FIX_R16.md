# R16 buffer chemistry handoff

The defect was in the chemistry handoff, before the column solver. Buffer
conductivity was treated as the only source of salt, saved salt cells were
discarded, and the transport state carried ionic strength in the field that
the HIC models interpreted as salt. The result could show a buffer change in
the UI while the HIC affinity calculation saw the wrong salt value.

R16 adds a `CONDUCTIVITY` / `SALT_M` selector to each Buffer A/B definition
and to direct load, wash and elution endpoints. `CONDUCTIVITY` computes
`salt_M = conductivity_mS_cm * conductivity_to_salt_M_per_mS_cm`.
`SALT_M` sends the entered formula-unit salt molarity directly to the solver;
conductivity is retained as an independent overlay and is estimated only when
it is absent. Existing files with both values retain conductivity priority
until `SALT_M` is explicitly selected. Clearing a CSV value is preserved and
does not silently restore a default.

Salt molarity and ionic strength now travel as separate transported fields:

| Quantity | Consumed by |
| --- | --- |
| Salt molarity [M] | HIC competitive Langmuir and empirical CPA-HIC affinity |
| Ionic strength [M] | Native CPA ion-exchange electrostatic terms |
| Conductivity [mS/cm] | Detector/overlay response |

Both CPA models accept `ION_COMPOSITION` using
`I = 0.5 * sum(c_i * z_i^2)`. For 1 M ammonium sulfate, enter 2 M NH4+ with
`z = +1` and 1 M SO4-- with `z = -2`, giving `I = 3 M`; HIC still uses the
1 M salt molarity. `SALT_PROXY` remains available as a monovalent approximation.

The HIC process direction is preserved: Buffer B is the high-salt endpoint,
loading and wash are 100% B, and elution decreases from 100% B to 0% B. The
HIC affinity remains `K_i ∝ exp(k_s,i * salt_M)`, so a decreasing salt program
weakens binding and releases retained protein. Native CPA remains the supplied
ion-exchange formulation; CPA-HIC is the documented empirical extension.

The result receipt now includes resolved buffer chemistry, stage salt and
ionic-strength endpoints, programmed salt, column salt, and local binding
chemistry. TDM uses the pore-phase local chemistry for binding diagnostics.

## Validation

`python verify_r16.py` passes all four EDM/TDM × Langmuir/CPA HIC models with
conductivity held fixed while salt is changed. The checks show high-salt
retention, low-salt breakthrough, gradient recovery above 99.99% for the
synthetic cases, and protein mass-balance closure below 0.004%. The same suite
checks direct endpoint modes, saved-run migration, blank-field preservation,
the JMP handoff, sulfate ion stoichiometry, and native CPA salt screening.

`python verify_model.py --buttons-only` also passes the saved-run, batch,
chromatogram attachment and least-squares bridge routes. The full-resolution
startup preset remains at a peak near 11.8514 CV with negligible pre-elution
loss and a closure error below 0.001% of input.

These are synthetic numerical checks. HIC salt sensitivity and the
conductivity-to-salt factor still require calibration for the actual protein,
resin and salt system; pH blending remains a linear process approximation.
