# R18 buffer and load chemistry handoff

Build `2026-10-07-R18` closes a JMP load-chemistry handoff gap found while
following the salt inputs through the UI, bridge, saved run and solver.

When a user edited load pH, salt input mode, salt concentration or conductivity,
the UI selected `EXPLICIT` load chemistry but could leave `Load_Source` set to
`BUFFER_A` or `BUFFER_B`. Those sources correctly resolve against the selected
buffer recipe, so the separate direct load fields were ignored. The edit
callbacks now select both `EXPLICIT` and `DIRECT` before saving the run. Users
can still deliberately select Buffer A/B as the load source; editing a direct
measurement switches back to `DIRECT` so the new value is consumed.

The full chemistry path is verified from generated JMP control names through
the Python bridge, saved run row, batch parameter merge and solver receipt.
Changing the conductivity conversion factor from `0.02` to `0.01` changes the
100% Buffer B load salt from `2.0 M` to `1.0 M`. Changing `k_s` from `6` to `0`
changes the simulated UV trace. A direct load conductivity of `35 mS/cm` with
the `0.02 M/(mS/cm)` factor resolves to `0.70 M` even if the saved load source
started at Buffer A and load %B still shows 100%.

Chemistry input authority is explicit:

- `CONDUCTIVITY`: analytical salt is conductivity × the calibration factor.
- `SALT_M`: entered formula-unit salt molarity drives HIC binding; the factor
  does not change binding salt. Conductivity stays an independent display
  value, or is estimated only if it is blank.
- HIC Competitive Langmuir and the empirical CPA-HIC extension use local salt
  molarity through `exp(k_s × salt_M)`. Native CPA ion exchange uses ionic
  strength.

Run `python verify_r18.py` for the JMP handoff, salt sensitivity, affinity,
load-material and output checks. Run `python verify_r16.py` for all four HIC
models, salt transport, salt-mode persistence and native CPA ion-exchange
checks. These are synthetic regression cases; conductivity calibration and
`k_s` still need values measured for the actual salt, protein and resin.
