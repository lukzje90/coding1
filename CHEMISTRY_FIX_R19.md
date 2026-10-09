# R19 HIC factor and salt input preflight

Build `2026-10-08-R19` checks the chemistry that will actually reach the HIC
solver before generating a chromatogram. The supplied Run 1 uses Buffer A at
5 mS/cm and Buffer B at 100 mS/cm. At a factor of 1000 M/(mS/cm), those inputs
resolve to 5,000 M and 100,000 M salt. The run is now rejected with both
resolved values and their source fields in the JMP status message. A previous
chromatogram is marked stale for the changed settings.

The HIC application limit is 10 M analytical salt. This is a generous model
domain guard, not a salt-specific solubility calculation or a calibrated
prediction at 10 M. It prevents an impossible conversion from being mistaken
for a chromatographic result. A second preflight rejects `k_s × salt > 80`,
the affinity exponent at which the numerical implementation would otherwise
clip and make further input changes appear ineffective.

The Model Parameters tab displays the current Buffer A, Buffer B, and load
start salt values next to the conversion factor. The result receipt now shows
the selected salt source and input mode at each stage endpoint. Only
`CONDUCTIVITY` inputs use the conversion factor for binding. A conductivity
endpoint at 0 mS/cm stays at 0 M regardless of the factor, and a gradient to it can
release protein. `SALT_M` uses its entered molarity for binding; changing the
factor will not change that salt value.

To inspect a run, select the run in Process Setup, check the two resolved
buffer values, and then run it. The result's **Process recipe** table lists
the resolved salt at the load, wash, and elution stages. **Inputs actually
used by this run** records the conversion factor and salt modes. If validation
fails, there is no new chromatogram; the earlier result remains associated
with its earlier settings.

Verification:

```text
python verify_jmp_interface_inputs.py
python verify_model.py --buttons-only
python verify_r18.py
python verify_r16.py
python verify_r19_counterexample.py
```

These checks cover the generated JMP send names, saved run fields, Python
bridge, factor-1000 rejection, stale-result status, stage receipts, affinity
sensitivity, and high-salt retention for all four model combinations.
They use synthetic profiles and do not execute a native JMP desktop session.

## Attached R18 `why1.pdf` example

The PDF itself records a factor of 1000 M/(mS/cm), Buffer A at 5.12 mS/cm
→ 5120 M, Buffer B at 86.38 mS/cm → 86380 M, and load/wash/elution salt of
45750/53876/13246 M. Thus the factor and buffers reached the R18 solver,
but those values are outside the chemical and numerical domain. R19 rejects
this exact setup before generating a new result. The PDF's UV peak begins
during loading and is gone before elution; its diagnostics report 52.1% of
the protein leaving before the elution stage.

The load is also above the finite stationary capacity. With EDM bed porosity
0.7 and target `qmax = 2 g/L stationary phase`, nominal target capacity is
`2 × (1 − 0.7) = 0.6 g/L packed bed`, while the run loads 1.3 g/L packed
bed. The PDF reports 0.0006 g retained in the 1 mL bed and 0.00067776 g
at the outlet. This is saturation breakthrough, not high-salt desorption.
R19 now shows this nominal saturation comparison beside the load input, in
the result and in the run status. It is a ceiling rather than a dynamic
capacity prediction; competing species and transport can reduce retention.

The PDF has the flow-rate checkbox off while the supplied JMP screenshot
shows it on. The saved PDF is therefore not an exact snapshot of every
visible setting. Use **Open latest chromatogram** to check the result
fingerprint after changing any input, then run again before exporting.

`python verify_r19_counterexample.py` reconstructs the exact buffer and
loading values recorded in the PDF and verifies rejection. It also checks a
valid, fixed 2 M high-salt overload: early outlet protein is identified as
breakthrough rather than an elution-stage peak.
