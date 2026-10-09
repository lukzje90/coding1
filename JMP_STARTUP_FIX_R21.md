# R21 JMP startup path fix

This page records the R21 startup fix. For an R22 installation and saved-run
migration, follow `PAPER_LANGMUIR_SALT_CHECK_R22.md` instead. The one-file
replacement below applies only to the older R21 startup fix; R22 also updates
the solver and its build guard.

The R20 launcher failed when extracted under a parent folder named
`No Model Template 4.0`. The generated JSL includes absolute paths for its
Python bridge, saved batch, status and results. `_validate_jsl()` searched
all generated text for `Model template`, so it mistook that path for a removed
UI control and raised before calling `jmp.run_jsl()`.

R21 validates the **actual** model selector choices against the four
supported combinations and checks UI label constructors for obsolete
template controls. Absolute file paths and run data can contain the words
`Model Template`, `SMA`, or similar text without blocking startup. The
four-model selection is still enforced.

`python verify_r21_windows_path.py` supplies the reported OneDrive path to
the generator, runs `open_in_jmp()` with a desktop submission stub, checks
that JSL was submitted and saved, and proves that a real `SMA` model choice
or `Model template` UI label is rejected. Native JMP itself was not available
in the build environment.

To keep your current R20 settings, extract the R21 ZIP separately and copy
**only** `TDM22_HIC_LSQ_R21/tdm_jmp.py` over the same file in your existing
`TDM22_HIC_LSQ_R20` folder. Reopen that folder's
`00_RUN_CHROMATOGRAPHY_MODEL_IN_JMP.jsl`. Your existing `tdm_inputs.json` and
`tdm_batch_runs.csv` remain in place. This single-file R20 replacement was
tested with the R20 package under a `No Model Template 4.0` parent folder.

For a fresh installation, extract and run the R21 folder. It can remain inside
`No Model Template 4.0`. The R20 Langmuir affinity and R19 chemistry fixes
remain included.
