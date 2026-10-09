TDM22 — CLASSIC DIRECT-INPUT CHROMATOGRAPHY MODEL
==================================================

Purpose
-------
This build keeps the original run-oriented workflow while removing the template/database
layer. It supports up to 30 saved runs, PLW and Elution selectors, editable operating
conditions, Copy/Paste, Save, Run selected, Run batch, and global least-squares refinement
against up to five independently configured chromatograms.

The four supported combinations are deliberately limited to:
  • EDM – Competitive Langmuir
  • TDM – Competitive Langmuir
  • EDM – CPA
  • TDM – CPA

No SMA/SWA/MMC or model/resin/component/buffer template is required.

Opening the model
-----------------
1. When updating from an earlier build, close JMP completely first. JMP keeps one Python
   interpreter alive, so closing only the model window does not clear older imports.
2. Extract the ZIP to a normal local folder. It creates a folder ending in
   `_LSQ_GLOBAL_FIT_20261001_R12` so it is easy to identify.
3. In JMP run 00_RUN_CHROMATOGRAPHY_MODEL_IN_JMP.jsl from that newly extracted folder.
4. Use the tabs in the main window. The Attach status reports the active model folder and
   selected CSV path so you can confirm which extraction JMP is using.

UI layout
---------
The interface is vertically stacked. Major input sections are below one another rather
than in horizontal split panes, so the Model Parameters and Least-Squares tabs do not
require sideways scrolling to reveal the right-hand half of the controls.

Classic run workflow
--------------------
At the top of the window:
  • Runs in batch: 1-30
  • Current run: Run 1-30
  • Copy / Paste operating conditions
  • Save model + run
  • Run selected
  • Run saved batch

Process Setup
-------------
Run & load and Standard process sequence are stacked vertically.

Run 1 in the included saved-run CSV starts with the reported recipe: 10 CV load,
PLW 1 = 2 CV, PLW 2 = 1 CV, PLW 3 = 1 CV, then Elution 1 = 4 CV. The other saved
run rows remain available for separate process profiles.

For each run you can directly enter:
  • load flow [mL/min]
  • total protein feed concentration [mg/mL]
  • load basis: capacity [g/L resin] or volume [CV]
  • capacity mode derives load CV = capacity ÷ feed concentration; volume mode derives
    capacity = load CV × feed concentration
  • required feed volume is load CV × column volume and is echoed in the run receipt
  • load mode and start/end Buffer B [%]
  • number of time steps
  • number of axial positions
  • load chemistry control: Buffer B % or explicit endpoint chemistry
  • for CPA and HIC Langmuir, load chemistry source and conductivity when explicit endpoints are selected

Each saved run contains editable Buffer A and Buffer B definitions.
Buffer A is the 0% B endpoint and Buffer B is the 100% B endpoint. Each buffer has:
  • buffer name
  • pH
  • conductivity [mS/cm] (the entered salt-related value)
  • up to four ion rows: ion species, concentration [M], and valency z

The required conductivity-to-salt conversion factor in Model Parameters always converts
conductivity to salt concentration. Salt concentration is derived and is not directly editable
in the JMP workflow. Legacy direct-salt columns in older run CSV files are ignored.

For TDM–CPA, select SALT_PROXY (the default) to make resolved salt concentration feed
the CPA ionic-strength term. Select ION_COMPOSITION to instead use complete ion rows:

  I [M] = 0.5 × Σ c_i z_i²

When ION_COMPOSITION is selected, this ion-derived ionic strength feeds the TDM–CPA
electrostatic calculation. SALT_PROXY uses conductivity-derived salt concentration, so
conductivity edits affect the calculation even when saved ion rows are present. EDM–CPA uses
resolved salt concentration as its ionic-strength proxy.

The numerical-resolution controls are run-specific. "Number of time steps" sets the
reported time grid and caps the largest adaptive integration step at total run time /
number of time steps. The adaptive BDF solvers may take extra internal steps when
required by stiffness. "Number of axial positions" is the number of finite-volume
axial cells used to discretize the column. Both values are read from the selected run
and are used by all four model combinations.

PLWs and Elution remain count-controlled drop-down sections. Selecting 0-5 immediately
shows exactly that many editable step panels. Each active step has:
  • SET_POINT / LINEAR / STEP mode
  • CV
  • flow [mL/min]
  • chemistry control: BUFFER_B_PERCENT or ENDPOINT_CHEMISTRY
  • visible Start / set-point Buffer B [%] and End Buffer B [%] controls in every mode
  • direct Start/End pH and conductivity [mS/cm] in ENDPOINT_CHEMISTRY mode

BUFFER_B_PERCENT is the normal A/B pump program. Buffer A = 0% B and Buffer B = 100% B.
At any programmed percentage f = %B/100, salt concentration is calculated as:

  salt [M] = (1-f) × salt_A + f × salt_B

The entered Buffer A/B pH values are interpolated in the same way to provide the process pH
used by CPA. This is an explicit endpoint interpolation, not a full acid/base-equilibrium
buffer calculation. If non-linear mixed-buffer pH is known experimentally, use
ENDPOINT_CHEMISTRY to enter the desired pH program directly.

For TDM–CPA, each Buffer A/B endpoint uses I = 0.5 Σ c_i z_i² when complete ion rows
are available; otherwise that endpoint uses its resolved salt concentration as the
monovalent-electrolyte ionic-strength proxy. The two endpoint ionic strengths are then
blended with the same %B.

For ENDPOINT_CHEMISTRY, salt concentration is calculated from the entered conductivity:

  salt concentration [M] = conductivity [mS/cm] × required conductivity-to-salt factor

LINEAR interpolates Start to End across the entered CV. SET_POINT holds the Start/set-point
value for the whole phase; STEP switches immediately to the End value. Both endpoints stay
editable and are saved in all three modes, while the selected mode determines which endpoint
or interpolation is active. The same rule applies to the load profile.

Model Parameters
----------------
The entire tab is a single vertical scroll area. The live required-parameter list,
column parameters, conversion factors, transport/isotherm parameters, and each selected
protein species appear one below another. Numeric edit boxes, text boxes and drop-downs
are deliberately compact so the controls fit without horizontal scrolling on a typical
JMP window.

Two direct conversion-factor inputs are included:
  • UV → protein response factor [mAU·L/g] — required for all four combinations.
    This is the target protein's response factor. Each selected impurity has its own
    UV → protein response factor in that impurity's component panel, initialized from
    this shared value for older projects and then independently editable. The predicted
    UV trace sums each species concentration × its own response factor.
  • Conductivity → salt concentration factor [M per (mS/cm)] — required for CPA and
    Competitive Langmuir/HIC. It always converts the entered conductivity to salt concentration.
    In EDM–CPA the resulting salt concentration is used as the ionic-strength proxy. In
    TDM–CPA, ion rows determine effective ionic strength only when ION_COMPOSITION is selected.

Protein feed composition
------------------------
For the target protein and every selected impurity, Model Parameters contains an editable
"Feed composition mass% of total protein" box. The active species must sum to 100%.
For every run the solver calculates:

  species feed concentration [mg/mL]
      = run total protein concentration [mg/mL] × species mass% / 100

This value is then used in the inlet/feed vector, mass balance, and simulation. Unselected
impurity slots are not used.

Mass-balance outlet integration uses `numpy.trapezoid` when available and falls back to
`numpy.trapz` for older NumPy versions. This supports current JMP Python installations
where `numpy.trapz` has been removed.

Axial advection uses a conservative, monotonized-central slope-limited finite-volume
scheme to reduce artificial peak smearing. New runs default to 40 axial cells; the
number of cells remains editable per run for a grid-sensitivity check.

Per-species scientific parameter sets
-------------------------------------
EDM – Competitive Langmuir:
  feed mass%, D_app,i, saturation capacity qmax,i, b_i, salt sensitivity k_s,i
  H_i = qmax,i × b_i is derived and displayed, not an independent input.
  qmax is entered in mg/mL stationary phase; its numeric value equals g/L for the solver.

TDM – Competitive Langmuir:
  feed mass%, D_ax,i, k_eff,i, epsilon_p,i, saturation capacity qmax,i, b_i, salt sensitivity k_s,i
  H_i = qmax,i × b_i is derived and displayed, not an independent input.
  qmax is entered in mg/mL stationary phase; its numeric value equals g/L for the solver.
  F_acc,i = epsilon_p,i / epsilon_p is derived.

Both Langmuir solvers use the exponentially modified competitive Langmuir form for HIC
  b_eff,i = b_i × exp(k_s,i × m_s)
  q_i* = qmax,i × b_eff,i × c_i / (1 + sum_j(b_eff,j × c_j))
where m_s is the local resolved salt concentration. The entered molarity is used as a
molality surrogate. Salt is propagated through the bed using an unretained interstitial
convective delay. Positive k_s increases binding at higher salt; its value is resin-,
protein- and salt-system-specific and should be fitted or otherwise established from data.
The starter value 1 M⁻¹ is only an initial estimate. The least-squares refinement includes
k_s,i, but salt sensitivity needs data spanning salt conditions to be identifiable.
The displayed H_i = qmax,i × b_i remains the zero-salt reference coefficient.
The column solvers advance conserved total protein in each phase and solve the local
competitive equilibrium at the salt concentration reaching that cell. A salt front can
therefore change binding and the chromatogram without creating or losing protein in the
column balance.

EDM – CPA:
  feed mass%, D_app,i, protein diameter, A_s,i, Z_i, Delta_i
  plus pH-dependence coefficients only if the selected batch spans pH.
  a_i = diameter/2 is derived.

TDM – CPA:
  feed mass%, D_ax,i, k_eff,i, epsilon_p,i, protein diameter,
  A_s,i, Z_i, Delta_i, k*_kin,i
  plus pH-dependence coefficients only if the selected batch spans pH.
  F_acc,i and a_i are derived.

Signal/output handling
----------------------
Every run result CSV contains predicted UV_mAU calculated using the user-entered UV
response factor. CPA result CSVs additionally contain predicted pH, conductivity_mS_cm,
salt_concentration_M and ionic_strength_M. The separate salt and ionic-strength columns
make it possible to audit TDM buffer-ion calculations when multivalent ions are present.

Each single-run result also records the selected run number/name, build ID, generation
time, full-settings fingerprint, requested time steps and axial cells. The result page
contains the exact load/PLW/elution recipe used by the solver, including cumulative start
and end CV, duration, mode, flow and %B endpoints. Numbered phase bands and dashed CV
markers on the chromatogram match that table. The horizontal axis ends at the actual
program duration in CV, including the derived load CV; it is not extended to a fixed
default run length. The result CSV repeats the run identity and settings fingerprint and
includes the active program stage, flow and programmed %B at each reported time point.
The pointwise %B cell is blank when a stage uses direct endpoint chemistry instead of
Buffer A/B blending. The SVG shows elapsed time above the CV axis and an aligned %B
profile below the protein chromatogram. In HIC Langmuir and CPA, changing resolved salt
also changes the protein prediction through the salt-dependent affinity or CPA electrostatics.

The “Open latest chromatogram” and “Open latest result data” buttons first compare the
current selected run and complete normalized settings with `latest_results_manifest.json`
and verify the saved output hashes. If a run, recipe or model input has changed, the
button reports `RESULT_STALE` and asks for a new run instead of showing an old chart.
Successful runs report `RESULT_CURRENT`; build ID `2026-10-01-R12` appears in the window
title and result page so the active package can be identified.

Each saved batch writes to its own timestamped folder under `batch_results/`; rerunning
does not delete folders that JMP, a browser, or OneDrive may still have open. The batch
gallery points to that exact folder, and the run status lists each recipe actually written
to the results, including its total CV and every Load, PLW and Elution duration. The
single-run result receipt remains available after a batch. The report and batch-open
buttons show a status message when their files have not been generated yet. Bridge errors
are returned to the JMP window through `tdm_status.txt`, and a fit's parameter-apply
script is evaluated only after the bridge confirms that the fit was accepted.

JMP run controls
----------------
Text-valued Combo Boxes are read with JMP's `Get Selected` message; numeric edit boxes use
`Get()`. This keeps the selected model name as text for the EDM/TDM visibility checks and
the selected process and source labels in the saved run. The run selectors convert labels
such as `Run 2` to row number `2` before loading or sending a target profile. Selecting an
EDM model collapses the TDM parameter panel (and TDM transport fields); selecting a TDM
model reveals them. Counts, PLW and Elution settings are written to that row before every
model action; the bridge then reads the saved row as the operating recipe. The successful
run status echoes the recipe saved with its chromatogram, making the selected run's stages
visible before opening the detailed results.

Every PLW and elution CV/flow numeric box saves its committed value into the selected
run's CSV profile. The table is saved to the explicit package CSV path before Python is
started. The Run / Results status echoes the exact CV, flow and %B recipe consumed by the
solver, including every elution stage.

The R5 selector correction is documented in `R5_MODEL_SELECTOR_FIX_AUDIT.txt`; the R6
process-recipe handoff, R7 output/parameter response, and R8 HIC salt response are
documented in `R6_PROCESS_RECIPE_HANDOFF_AUDIT.txt`, `R7_TRACE_RESPONSE_AUDIT.txt`, and
`R8_HIC_SALT_DEPENDENCE_AUDIT.txt`. R8 supersedes the R7 statement that Competitive
Langmuir is salt-independent. The older R4 batch/recipe audit has a supersession note
because its Combo Box getter description caused the selector regression.

Least-Squares Refinement
------------------------
Two modes are retained:
  1. Normal least-squares against a raw chromatogram CSV.
  2. Chromatogram + protein-species mass% at 1-10 selected CV positions.

Each saved run is a separate chromatogram profile. Give runs names such as Overload,
Standard, and Gradient in Process Setup, attach a different CSV to each profile, and set
that run's flow, load/feed, buffers, PLWs, elution and numerical conditions in Process
Setup. Selecting a profile in the LSQ tab also loads that same run into Process Setup for
editing. Its CSV path is saved with that run in `tdm_batch_runs.csv`. Attach up to five
profiles; the next refinement fits one shared mechanistic parameter set against every
attached chromatogram at once. Every profile uses its own saved process recipe; the column
specifications are shared and remain fixed during fitting. Profile signal residuals are
normalized by each measured signal scale and point count, then weighted equally so a longer
or larger-signal CSV cannot dominate the global objective by scale alone.

The live summary shows the selected profile name, column size, load capacity and calculated
feed volume, load flow/feed, numerical resolution, load chemistry, Buffer A/B definitions
and ion rows, and each active PLW and elution step. The fit uses the saved recipe associated
with every attached CSV. Its report and settings JSON record each original CSV path and
complete recipe, and the report shows per-profile fit metrics plus objective improvement
over time. Fitted mechanistic parameters remain shared across all fitted profiles.

Every eligible mechanistic input has a Locked/Unlocked control in Model Parameters. Locked
values remain fixed; unlocked values may be optimized. The Lock all and Unlock all buttons
change those controls together. Column geometry and feed composition are fixed model inputs;
each profile's process recipe remains its own fixed experimental input.

Raw UV chromatograms use the shared target response factor and the independently entered
response factor for each impurity. Least squares includes every selected impurity factor
as a positive fitted parameter for UV-valued CSVs; concentration-valued CSV columns are
compared directly and do not fit detector response factors. The report contains an
objective-versus-evaluation plot, and `least_squares_objective_history.csv` records elapsed
time, current and best-so-far objective, percentage improvement, evaluation phase, and any
numerical-penalty status. Use “Open LSQ objective log” in JMP or the report link to view it.
With heavily overlapping peaks, response factors can be correlated with each other and the
adsorption parameters; species-resolved UV or composition constraints help identify them.
To attach a reference,
click “Attach / validate chromatogram CSV”: JMP opens a file picker if the path is blank
or invalid. If an entered path omits its extension, the bridge also tries the same path
with .csv appended. It also accepts JMP's `/C:/...` and `\C:\...` Windows path forms by
removing the extra separator before the drive letter. UTF-8, BOM-marked UTF-16/UTF-32,
and Windows-1252 encodings are supported, along with comma, semicolon, tab and pipe
delimiters and common instrument metadata rows. Numeric x columns can be CV, mL, elapsed
time or sample index. If the file has detector values but no numeric x axis, sample order
is mapped uniformly across the selected process recipe's duration and the fit report shows
that assumption; a measured time/CV/volume axis is preferred. Accepted fitted mechanistic
parameters become the shared parameter set for subsequent runs in the batch. The standard
results PDF shows the model prediction; the measured chromatogram comparison and fit
diagnostics are in the least-squares fit report.
For Competitive Langmuir, least squares estimates qmax,i and b_i; H_i is recalculated
as qmax,i × b_i for every trial and saved as a derived value.

Batch behaviour
---------------
Mechanistic/model parameters, conversion factors, impurity selection, and species feed
mass% are shared across the selected batch. Run-specific flow, total feed concentration,
load capacity or volume, numerical resolution (time steps and axial positions), Buffer
A/B definitions, load chemistry, PLWs, Elution steps, pH, salt, conductivity and buffer ion
compositions remain specific to each saved run.

Verification
------------
Run inside the extracted folder if desired:
  python verify_model.py

The verifier checks the stacked compact UI, direct editable controls, exact per-model
parameter contracts, JMP→Python mapping, per-impurity UV response propagation and fitting,
objective-history logging, 30-run storage, run-specific time-step/axial
resolution propagation, Buffer A/B definitions, 0-100% B mixing, direct endpoint chemistry,
PLW/Elution semantics, all four solvers, axial-grid sensitivity and mass balance, batch
parameter locking, result generation, both least-squares objective modes, meaningful
composition-constrained fit improvement, per-profile chromatogram path persistence and
recipe selection, feed mass%-to-species concentration propagation, UV conversion,
conductivity-to-salt conversion, and TDM ion-derived ionic strength. It checks that changing
conductivity through the saved-run JMP bridge changes the resolved salt trace and EDM HIC
Langmuir chromatogram, then holds conductivity fixed and changes the UI-sent conversion
factor to verify that it independently changes both salt and chromatogram. Stale direct-salt
columns are ignored. It checks raw CSV detection for metadata-prefixed,
semicolon/decimal-comma, timestamp,
and detector-only files; per-profile report/settings retention; accepted values and JSL
assignments; NumPy mass-balance integration without `numpy.trapz`; and reopening a second
extraction while refreshing cached modules before every JMP action. The supplied settings
recipe is rebuilt through the saved-run CSV mapper and checked for 10 CV loading, 2/1/1 CV
PLWs, 4 CV elution, and 18 CV total duration. The generated manifest and HTML are checked
for all four phases and matching CV markers. A second saved batch is run against the same
folder to verify that the first result tree remains intact and the gallery follows the new
batch. Stale-result rejection is checked after a recipe or selected-run change.

JMP itself is not installed in the automated build environment, so native JMP clicking
cannot be exercised there. The generated JSL is structurally checked and the bridge and
core solvers are executed end-to-end by the verifier.
The conductivity handoff and two independent sensitivity checks are recorded in
`R11_CONDUCTIVITY_SOLVER_REVERIFICATION_AUDIT.txt`. R12 adds verification that a saved
capacity value derives the solver load CV and total mL feed volume, and that two CSVs with
different run recipes can be attached through the bridge and included in a single global fit.
See `R12_CAPACITY_AND_GLOBAL_LSQ_AUDIT.txt`.

INLINE BUFFER-B % PROCESS INPUTS (2026-09-25)
----------------------------------------------
For the load and every selected post-load wash and elution step, direct process inputs include:
  Mode -> Volume [CV] -> Flow [mL/min] -> Start/set-point Buffer B [%] -> End Buffer B [%].
The %B boxes are therefore not hidden inside a secondary chemistry panel.

Buffer A is the 0% B endpoint and Buffer B is the 100% B endpoint. Enter conductivity for each buffer; the required `conductivity_to_salt_M_per_mS_cm` model parameter converts it to salt concentration. The run program blends the run-specific Buffer A/B pH and conductivity at each entered %B, then carries the resulting conductivity and derived salt through all four model programs and writes both to result traces. For TDM-CPA, ion-defined Buffer A/B ionic strengths are also blended.

Scientific use by model:
- EDM-CPA and TDM-CPA consume resolved pH/salt/ionic-strength conditions in the CPA calculation.
- EDM-Competitive Langmuir and TDM-Competitive Langmuir use resolved salt concentration in the exponentially modified competitive Langmuir affinity. pH is reported but does not enter that isotherm.

The advanced ENDPOINT_CHEMISTRY option remains available for load, PLW or elution conditions that cannot be represented by Buffer A/B mixing. When selected, entered conductivity and pH endpoints are used for that phase, and salt is derived with the same required conversion factor; the inline %B values remain saved but do not drive its chemistry. Legacy direct-salt CSV columns are retained for file compatibility and ignored by the solver.

JMP Python Submit File compatibility
------------------------------------
The JMP action bridge does not rely on Python's __file__ variable. JMP can execute Python Submit File content without defining __file__, so the bridge resolves the model folder from the path supplied by the launcher, the TDM_PROJECT_DIR environment variable, or the current extracted folder.
The launcher also refreshes this folder's TDM Python modules when opened from a persistent JMP session, so a previously opened extraction cannot keep supplying the UI, model code, or output paths.
