TDM – MODIFIED WANG, mL LOADING AND MULTICHANNEL REFINEMENT (R25w)

CPA BUFFER IONS AND ALIGNMENT DISPLAY (R25w)
For EDM/TDM–CPA ion exchange, each run now requires Buffer A and Buffer B ion
composition. In Process Setup, select ION_COMPOSITION for each buffer and enter
at least one cation and anion with species, concentration [M], and signed charge.
The model calculates ionic strength as 0.5 × sum(c × z²) and blends the two
buffer values according to %B. Salt molarity/conductivity remain separate
process inputs. Existing CPA recipes using SALT_PROXY must be completed before
prediction or LSQ. Direct chemistry endpoints still use their explicitly
entered salt value as an ionic-strength proxy; choose Buffer A/B or %B stages
when ion-derived ionic strength is needed through the full process.

Dense measured UV lines in the alignment preview and fit report now use a
short display-only median filter after pixel grouping, removing isolated dark
spikes. Every imported sample still contributes unchanged to LSQ. The attach
status and preview list the imported UV first/last and minimum/maximum values
in mAU so the worksheet and signal selection can be checked locally.

MAIN PROTEIN IN SPECIES PROFILE (R25v)
The separate predicted impurity chromatogram now also shows the main protein.
Impurity concentrations use the left g/L axis; the main protein uses the right
g/L axis so smaller impurity peaks remain visible. This applies to normal and
batch predictions, alignment previews, and LSQ fit reports.

FASTER EXCEL ATTACHMENT (R25u)
Attaching a chromatogram now reads its Excel workbook once. Column detection
parses each cell once, even when instrument metadata appears above the table.
The interface shows a reading message while JMP validates the file. A workbook
with many worksheets or a large number of populated cells can still take time
to read; selecting the intended worksheet in the LSQ run avoids scanning the
other sheets.

SAVED PREDICTIONS AND LSQ FITS (R25t)
The extracted package contains Predictions and LSQ folders. Each normal
prediction is copied to a new Predictions folder named for its run and UK
date/time. Batch predictions are grouped by batch, with each run in its own
named, dated folder. Each completed LSQ is copied to a dated global fit folder
and separate named, dated folders for its selected runs. The run folders
contain run-specific UV, conductivity, and fraction mass% CSVs, the starting
and fitted process recipes, a short report, and a copy of the attached
reference file when it is available. Latest result buttons still open the
latest files. Past results are retained. Folder names include GMT or BST and
a short uniqueness suffix. Long run names are shortened in folder names for
Windows; the full name remains in each run receipt.

TARGET UV RESPONSE (R25s)
The target protein has its own UV → protein response factor beside its other
Model Parameters. Each impurity keeps its separate response factor. The target
control reads and writes the same saved target conversion value as earlier
packages, so normal predictions, LSQ locks, and saved fit values still use
the selected target response factor without changing existing recipes.

CPA FRACTION FIT CHECK (R25s)
The TDM+CPA mass% objective now uses each complete fraction's measured UV
and protein/impurity mass percentages to constrain the individual predicted
species signals. This gives the optimizer a useful residual when a trial
predicts almost no protein in that fraction. The reported mass percentages
remain the actual integrated species fractions. Rows with blank species cells
still use only their entered mass% values. A four-species bind-and-elute
recovery test and CPA paper equation audit are in CPA_PAPER_AUDIT_R25S.md.


LSQ LOCK SELECTION HANDOFF (R25r)
The JMP model sends parameter lock/unlock states in small numbered messages,
as it already did for fit-range limits. Python checks that every message
arrived before starting refinement. This avoids a single oversized JMP
expression silently omitting unlocked parameters. The fit-range messages are
now also assembled after all model controls exist; nine late-created controls
(including the UV response factor and bead radius) were previously absent
from those messages. Solver equations and fitting objectives are unchanged.

PREDICTED IMPURITY CHROMATOGRAM (R25q)
Normal run results and the batch gallery now show a separate chromatogram
for every selected impurity, using its detector concentration in g/L. The
target is excluded from this chart so small impurity peaks stay visible.
Each run with impurities saves its own latest_impurity_profile.svg; the run
receipt checks that file along with the main chromatogram. The LSQ alignment preview shows
the current predicted impurity chromatogram, and each fit report shows the
predicted impurity chromatogram from that run's fitted parameters. These
species curves are predictions, not independently measured impurity data.

LOAD BLEND, FIT LIMITS AND LSQ DISPLAY (R25p)
The load %B entries now select the Buffer A/B blend for both pH and
conductivity. The calculated load values are shown in Process Setup, while
the old direct conductivity entry is hidden. Saved explicit-load recipes
retain their original chemistry until their load %B is edited; editing %B
switches that run to the buffer blend. The mechanistic load stage consumes
the resolved pH, conductivity and salt. Least-squares Fit ±% controls now
reach Python in checked chunks, avoiding an oversized JMP send expression.
The bounded optimizer uses the resulting per-parameter limits and lists the
actual allowed values in its fit report. The fraction mass% plot uses a
linear 0–100% scale, and LSQ plots name their selected model run.

TDM+CPA LSQ STALLED-TRIAL PROTECTION (R25o)
A model evaluation that stalls inside TDM+CPA integration cannot be stopped
by the overall evaluation cap. The LSQ tab now has an adjustable timeout for
each TDM+CPA model solve (30 minutes by default; 0 disables it). A timed-out
Jacobian probe is never treated as a valid derivative. The fitter tries the
opposite direction once and, if that also times out, reports the parameter
name and stops. least_squares_active_evaluation.txt shows the evaluation and
parameter currently running; the history CSV records completed evaluations.
The normal prediction has no timeout. Equations and solver tolerances are
unchanged. Restart an already-running fit to use this release.

TDM+CPA LSQ RECHECK (R25n)
Both the JMP multi-run fit and the single-run Python fit now avoid using an
invalid CPA trial's fixed penalty as a numerical Jacobian derivative. The
single-run route was found during the R25m review and has the same feasible
direction fallback. The adsorption equations and solver tolerances are unchanged.

TDM+CPA REFINEMENT AND SELECTED PREDICTION (R25m)
The TDM+CPA fit history can show impossible surface coverage during a
Jacobian probe. Those trial parameters are invalid: the solver now rejects
such a trajectory at its first accepted over-coverage state, rather than
finishing the full run before rejecting it. The LSQ numerical Jacobian tries
the opposite parameter direction and then smaller steps; a fixed penalty is
never treated as a measured derivative. Each retry counts toward the selected
model-evaluation safety cap. CPA scalar calculations reuse the electrostatic
energy and avoid array operations on single numbers. Regression checks found
identical chromatograms before and after that calculation cleanup. The
selected-run button now shows that a prediction is in progress while JMP is
waiting for the solver. A complex TDM+CPA prediction can still take minutes.

TDM+CPA LSQ ACCEPTANCE AND SOLVER AUDIT (R25l)
TDM+CPA fits now require a measurable objective improvement and, by default,
target binding after loading, a column-outlet peak during elution, and recovery
of retained target. Fit settings expose adjustable percentage limits and an
option to allow intentional flow-through. A numerically improved candidate
that fails the physical check remains in the reports and fitted inputs JSON,
but is not copied into the active model or JMP parameter boxes. The solver
reports actual mobile-phase transport inputs, phase-state minima, and stage
mass closure. Material negative states and severe (>5% of load) mass errors
stop the simulation. Parameter readback is checked before final simulation.
The System response salt-dispersion value is the active transport input; the
old TDM-only salt field is kept solely for legacy project loading.
For new TDM+CPA projects, the default LSQ selection unlocks only target Delta
and charge; saved locks remain in force. Isolated tests cover salt and charge
direction, saturation, kinetics, retention, release, mass balance, and grid/time
convergence on a representative bind-and-elute recipe.

TDM+CPA BINDING AUDIT (R25k)
Normal results and least-squares fit reports now show the target and impurity
mass adsorbed at the end of loading, the amount still in the column, and the
amount leaving during loading, washing and elution. Stage boundaries use the
adaptive solver's states rather than the plotted time grid, which can skip a
short load or wash. The report also shows zero-coverage CPA partition factors
at the load start and first elution end. These are diagnostics, not measured
or fitted capacity estimates. A target column-outlet peak before elution or
less than 1% adsorbed at the end of loading raises a visible warning. LSQ now
requires the configurable binding check by default; intentional flow-through
fits can turn it off. Core solving equations and LSQ residuals are unchanged.

The CPA check includes a one-component bind-and-elute test and a deliberately
weak-affinity breakthrough test. With several unlocked, correlated parameters,
fit adsorption first using a small set of CPA parameters, then transport and
impurities, and finish with a joint UV plus fraction mass% fit. Verify the
load-material salt or conductivity setting actually used in the run receipt;
when a buffer uses CONDUCTIVITY input, its entered salt-M value is not the
authority. The uploaded assessment contained no run-specific inputs, so it
cannot establish the physical validity of a particular experimental fit.

PER-PARAMETER LSQ FIT LIMITS (R25j)
Each model parameter keeps its LSQ Locked/Unlocked selector and now has an
optional Fit ±% limit. Choose Limited and enter a percentage greater than zero
and at most 1000. An unlocked parameter can move only that far from its value
at the start of refinement, within its existing model bounds. Default
retains the previous bounds. For a zero-valued linear parameter, the entered
percentage uses the full previous bound span so it can move away from zero.
The fit report and parameter CSV show the applied percentage and actual
allowed minimum and maximum. The former repeated residual-DOF column was
removed; it did not control fitting. Normal simulations do not use these LSQ
limits.

SAVED PROCESS INPUT CHECKS (R25i)
Active wash and elution counts must be whole numbers within the supported
range. The model rejects a recipe that declares more active stages than its
stored stage array contains, including saved run rows with invalid counts.
The result receipt now lists the bed length, volume, derived cross-section and
diameter, and load residence time actually used by the solver. Compare these
against measured column dimensions before fitting; the derived diameter is
not an independently measured input. Variable-flow fraction integration and
stage validation are covered by automated tests. Fraction mass% entries are
also rejected if their populated species sum exceeds 100.5% (a small allowance
for assay rounding); partial rows remain valid when other species are blank.
Run verify_model.py for the maintained regression suite; its older 22-step
checks remain available only with --legacy.

FRACTION MASS% AND CHROMATOGRAPHY MODE (R25h)
Each entered fraction's predicted species mass% now comes from integrating
the detector-delayed species concentrations across its full start–end CV or
mL interval. The CV coordinate represents collected volume, including when
flow varies between stages. Legacy point-only Python references remain point
observations. Overlapping fractions and fractions outside the simulated run
are rejected. The fit report has a separate chromatogram with measured and
predicted fraction mass% overlaid on UV, and lists the initial and fitted UV
RMSE so a joint fit can be assessed channel by channel. Each unlocked
parameter row identifies its protein or impurity. The fit CSV flags fractions below a
1e-12 g/L numerical concentration floor; this is not an assay detection limit.
CPA with TDM or EDM is ion
exchange; Competitive Langmuir with TDM or EDM and TDM Modified Wang are HIC.
The JMP mode selector follows the selected model automatically; incompatible
mode/model combinations are rejected by Python validation.

LOAD INPUTS IN JMP (R25g)
Enter feed concentration and load capacity for each run. The JMP window
calculates load volume in CV and mL as read-only values using the shared column
volume. Opening an older CV- or mL-based saved run converts its existing load
amount to capacity before saving, so the programmed load amount is preserved.
Older CSV and Python recipes still accept their saved load basis. Load material
pH, conductivity and salt are now visible in Process Setup; editing them
selects explicit load chemistry for that run. The buffer-recipe load mode is
still available for existing runs.

CLEARER JMP INTERFACE (R25f)
The JMP window now uses shorter labels and essential setup guidance. Repeated
explanatory text was removed, the required-parameters list starts collapsed,
and no UI text uses italic styling. Model calculations and saved inputs are
unchanged.

TDM + CPA MULTI-RUN pH REFINEMENT (R25e)
Least-squares fitting now builds its unlocked parameter list from the selected
run whose recipe requires the broadest set of model parameters. If run 1 has
constant pH but a later selected run changes pH, the pH-dependent CPA
parameters shown in JMP are included in optimization and applied to the
shared fitted model. Each run still uses its own process recipe and reference
chromatogram. A two-run synthetic test verifies that the later run's CPA pH
coefficient is fitted and written to the JMP parameter update file.

FITTED-MODEL HANDOFF AND LSQ EVALUATION CAP (R25d)
After an accepted least-squares fit, a normal Run now recovers the fitted
mechanistic values if JMP still sends the pre-fit parameter boxes. The fit
updates the active model file and records the changed values, so a stale
window cannot silently replace the accepted fit on the next prediction.
The boxes are refreshed when the stale handoff is detected. An intentional
new value outside the pre-fit/fitted pair remains under the user's control.
The LSQ safety cap now counts every residual evaluation, including finite-
difference Jacobian probes and final verification. The initial reference
calculation is displayed as evaluation 0. A fit with 42 unlocked parameters
needs about 43 evaluations per complete numerical Jacobian for each set of
runs, so a cap of 200 permits only a few optimizer steps. The fit report and
status warn when the best saved candidate still matches UV poorly. An
accepted candidate means the tested objective did not worsen; convergence
and chromatogram quality are reported separately. Model equations and
residual definitions are unchanged.

CPU WORKER CONTROLS REMOVED (R25d)
Normal batch runs and least-squares evaluations execute serially. The two
CPU-worker inputs and joblib process pools have been removed. Existing saved
CPU settings are ignored when a project is loaded. The mechanistic solver,
residuals, numerical grids and optimizer equations are unchanged.

LOAD INPUT AND CHEMISTRY HANDOFF (R25c)
R25c allowed each saved normal or independent LSQ run to select capacity
[g/L resin], load volume [CV], or load volume [mL] as the authority. R25g
keeps those saved recipes readable but uses capacity as the only editable
load amount in JMP. A selected load input left blank stops with a clear error
instead of silently reusing an old value from another load basis. The direct
load salt/conductivity setting and Buffer A/B recipe selection now agree in
the JMP controls and the solver, including after changing modes. SALT_M uses
the entered molarity for binding; CONDUCTIVITY uses the entered conductivity
times the shared conversion factor. The output receipt records the resolved
capacity, CV, mL, salt, conductivity, and source for each run.

OVERNIGHT FIT AUDIT (R25b)
The LSQ safety cap remains adjustable from 2 to 10000.
The fit still stops on normal convergence, a solver error, or the selected cap.
Check that value in the LSQ tab before an overnight run; the start message and
fit report state the setting used.
Saved run tables now publish whole CSV files atomically, so other processes
cannot read a half-written run table. Opening an unchanged table no
longer rewrites it, which also avoids needless OneDrive/Excel file-lock errors.
The model equations and fraction mass% residuals are unchanged.

LEAST-SQUARES FIT DURATION (R25a)
After refinement finishes, the JMP status shows elapsed fit time in hours,
minutes, seconds and decimal seconds. The fit report and settings JSON also
record it. The timer starts when selected reference profiles are loaded and
includes optimization, final verification and fitted output preparation.

AUTOMATIC JMP MODEL FOLDER LOCATION (R24z)
Extract the complete ZIP, open its saved
00_RUN_CHROMATOGRAPHY_MODEL_IN_JMP.jsl in JMP, and press Ctrl+R. The launcher
finds the model files beside itself. It also checks the editor window and the
current directory for this package's marker when JMP does not report the
script path. It no longer opens a folder or file picker. Keep
tdm_r25t_package.marker beside the launcher and Python files. Run the saved
launcher file; an unsaved pasted script has no folder for JMP to identify.

UK CHROMATOGRAM TIME AND ONE-KEY JMP START (R24s)
The output chromatogram shows its generation time in UK local time, using GMT
in winter and BST in summer (day/month/year and a 24-hour clock). The manifest
retains its UTC timestamp and also records the UK timestamp with its UTC
offset. The chromatogram time axis remains elapsed run minutes.
After extracting the ZIP, open 00_RUN_CHROMATOGRAPHY_MODEL_IN_JMP.jsl in JMP
and press Ctrl+R. The script launches the model window directly from its own
folder.

LAUNCHER FOLDER DETECTION (R24r)
Open 00_RUN_CHROMATOGRAPHY_MODEL_IN_JMP.jsl from the extracted ZIP folder.
The launcher uses its own folder even when JMP's default directory points
somewhere else. Keep all extracted files together.

The JMP model selector retains all four original combinations and adds
TDM – Modified Wang. Wang is paired only with TDM and uses HIC mode. The
solver uses the modified Wang kinetic equation (4) from Beryamysoltan et al.,
Journal of Chromatography A 1783 (2026) 467108, inside the existing TDM column
transport model. The shared q0 and eta values, all eight values per protein,
local pore salt and pH, and fitted parameter locks are passed to one solver.
R24a uses a sparse kinetic-solver Jacobian to speed up larger TDM meshes.
R24f also supplies the local Jacobian structure to TDM–CPA, TDM–Langmuir
and larger EDM–Langmuir meshes. EDM–CPA now reads the same column chemistry
once per time step instead of once per cell. Transport, adsorption, integration
tolerances and reported time grids are unchanged. The independent buffer
solver and the existing TDM–Wang sparse solver are unchanged. Comparisons
with the earlier build confirm matching chromatograms and mass balances to
numerical tolerance.
The provided starting values are illustrative, not calibrated to a measured
protein/resin pair. See WANG_IMPLEMENTATION_AND_USAGE.md for the equation,
parameter mapping, units, fitting workflow, and model limits. Open the complete
new folder with 00_RUN_CHROMATOGRAPHY_MODEL_IN_JMP.jsl, then select the model
you want and rerun the normal simulation or least-squares fit with the saved
process recipes. Earlier saved model choices remain selected when the JMP
window opens.

The JMP Process Setup takes load capacity [g/L resin] and feed concentration
[mg/mL] for each run. It calculates CV = capacity / feed concentration and
mL = CV × column volume. Older saved CV and mL recipes remain readable;
opening one in JMP converts its current load amount to capacity. The batch
and LSQ run tables still store independent values.
The LSQ run summary calculates mL-based CV from the current column volume,
including runs that have not been opened since the column volume changed.
Each normal and independent LSQ run keeps its own load, wash, elution and
buffer recipe. All runs use the column volume and mechanistic parameters in
Model Parameters. JMP imports both process tables' numeric fields explicitly,
including blank salt and ion cells, so editing a run cannot assign a number
to a character column.

INITIAL COLUMN EQUILIBRATION (R24m)
Each normal and independent LSQ process run can specify the chemistry already
in the column and upstream buffer mixer before the load begins. Choose Match
Load to retain the previous model behavior and identical results. Buffer A or
Buffer B uses that run's saved buffer definition, including its selected salt
input and optional ionic-strength ion composition. Direct accepts a measured
starting pH plus either salt molarity or conductivity as the authoritative
salt input. The entered load material remains a separate incoming stream.
During loading it displaces the initial liquid through the mixer and column;
TDM also tracks the displacement in the particle pores. This is an initial
condition, so it adds no programmed CV. The same transport is used by all five
model combinations. Each run's initial values are saved in its own process
table and displayed in the chromatogram result and LSQ alignment preview.
Changing the initial pH in CPA ion exchange may require the corresponding
pH-dependence coefficients for each protein. The model still uses its existing
isotherms: the Langmuir combinations do not gain a new pH-binding law.

Before least-squares refinement, click "Preview this run" in an LSQ run
section to simulate and display only that run. The main "Preview UV,
conductivity and mass%" button previews all selected runs; the report's
run picker displays one alignment at a time. In each run section, choose
"Model run for UV comparison":
the independent LSQ recipe or any saved normal batch Run 1–30. The measured
UV file stays attached to its LSQ profile. Both preview and refinement read
the current saved recipe for the chosen model run and reread the UV file each
time, so changes to a normal run or its measured chromatogram are picked up.
The plot and baseline policy use the same
calculation as the fit. Preview runs one simulation per selected run and does
not optimize or change fitted parameters. It reports the measured CV span,
programmed end CV and pre-fit UV RMSE so axis or recipe mismatch can be fixed
before a long refinement. The report names the model run, process stages,
column volume, attached file, and actual Excel worksheet for each profile.
"Open alignment preview" saves the current entries, reruns the selected
profiles, and opens a fresh report. If any run has enabled mass% observations,
the report initially displays the first such run and links to its mass% graph.
The run picker labels profiles with mass% observations. Existing saved LSQ
profiles retain their independent recipe choice. If no LSQ run table exists,
the newly generated first ten profiles default to the corresponding normal
batch runs.
For display, dense measured UV samples sharing a chart pixel column are shown
at their median value so repeated strokes do not look like a filled black
peak. Every imported UV sample remains in the least-squares calculation and
the fit trace CSV.
Refinement intersects Unlocked controls with the selected model combination's
required parameter list. Values or locks left from another model combination
cannot be fitted or written back as improvements.

CONDUCTIVITY REFINEMENT (OPTIONAL)
Each selected LSQ run has a separate "Fit measured conductivity" switch, Off
by default. Turn it On when that run's attached UV CSV or Excel worksheet also
contains conductivity. If there is exactly one numeric column headed
"Conductivity", it is found automatically; otherwise enter its column heading
under Optional columns and weight. Values are interpreted as mS/cm by default;
headings containing µS/cm or S/m are converted to mS/cm. The same X-axis,
units and run-start origin selected for UV are used for conductivity. Preview
shows measured versus predicted conductivity and its RMSE for enabled runs.
The fit adds one residual per numeric conductivity sample, scaled by that
run's measured conductivity range and sample count so detector units and
sampling density do not overwhelm UV. Fraction mass% remains an independent
explicit mode. The conductivity dead-volume parameter can be unlocked and
fitted when at least one run has conductivity enabled; it is omitted from
UV-only fits. The fit report and least_squares_fit_conductivity.csv include
the selected run number and conductivity comparison. Results CSV and HTML
label every species parameter with its target/impurity number and name.

FRACTION MASS% REFINEMENT (R24k)
Choose 1–10 LSQ runs, with each run's own measured UV and independent or
normal-batch model recipe. In mass% mode, set "Use this run's fraction mass%"
to On in every run section that has fraction data. This lets you choose any
number of mass% runs from 1 to the selected LSQ run count; other selected
runs contribute UV alone. Choose "Edit mass% for LSQ run" to enter up to 30
fractions for that particular run, then switch to another run to enter its
different fractions. The values, row count, and units are saved by LSQ run
number in tdm_lsq_fraction_mass.csv and restored when you switch the editor.
JMP explicitly imports the fraction fields as numeric columns so a new blank
mass% table remains editable before any values have been entered.
Choose CV or mL for each run, then enter
each fraction's start and end from run start. The midpoint is calculated;
mL midpoints are divided by column volume to align with the solver CV axis.
The fraction table puts each heading above its own compact numeric column.
The target and impurity headings follow the species names in Model Parameters
when those names are edited. Scroll down for later rows.
Each fraction column stays tied to its numbered model species even if an
earlier impurity has zero feed mass. Leave the mass% cell blank for a zero-feed
species; the fit rejects entered values for a species the solver has omitted.
Active species names must be distinct so observations cannot be assigned to
two model species under one name.
Enter target and up to five impurity mass% values. Leave a cell blank when
that species was not measured in a fraction; enter 0 for a measured zero.
Blank cells create visible gaps, and only populated species values enter
the least-squares objective. A shape-preserving smooth line connects adjacent
populated fractions for display, without creating extra fit points. The
preview and final report show entered and predicted species lines, with the
model line limited to measured segments. The target measured line is solid,
measured impurity lines are dotted. The model target is dashed and model
impurities use lighter dotted lines.
The final fit report also shows before/after mass% RMSE in percentage points
for each measured species in each LSQ run, so the fraction-profile improvement
can be checked separately from the combined UV and mass% objective.
Measured values are compared with 100 x integrated species mass over the
fraction / integrated total modeled-species mass over that fraction. Partial
rows are allowed; a full row that does not total 100% may have an irreducible
residual because the model normalizes its selected species to 100%.
RAW_SSE keeps its UV term in mAU squared and scales the mass% block using the
measured UV range (minimum 1 mAU) and sample count so mass% observations can
affect the fit. NORMALIZED_MSE gives UV and mass% dimensionless group
contributions, also using a 1 mAU floor for a flat UV trace.

CHROMATOGRAM CV RANGE UPDATE
The chart spans the cumulative CV of every enabled process step. For example,
1 CV load + 2 CV wash + 2 CV wash + 10 CV elution ends at 15 CV. Programmed
flow and %B overlays now turn at the exact stage boundaries, independently
of the reporting resolution. Standalone SVG charts resize to the window.
Least-squares reports show predicted UV over the complete programmed run,
even when imported UV measurements cover only part of it. The objective
still uses only the measured UV samples, plus mass% when explicitly selected.

Extract this entire ZIP to a new folder. To retain an existing setup, copy
tdm_inputs.json, tdm_batch_runs.csv, tdm_lsq_process_runs.csv,
tdm_lsq_fraction_mass.csv (if present) and
tdm_batch_count.txt from your old folder into the new one. Close the old JMP
setup window and launch 00_RUN_CHROMATOGRAPHY_MODEL_IN_JMP.jsl from the new
folder. Run the selected model again to regenerate the chromatogram for
your current recipe. The stage table below the chart records the CV used.

LEAST-SQUARES PERFORMANCE UPDATE
The LSQ tab now labels the safety cap as optimizer evaluations. Each numerical
Jacobian reruns the column once per unlocked parameter and selected chromatogram,
plus a base solve.
While JMP is busy, least_squares_objective_history.csv gains progress rows with
elapsed seconds and the best objective so far. A repeated base solve and repeated
full-file history rewrites have been removed. The conductivity-detector delay
is excluded from UV fitting because it cannot affect the fitted signal.
If Excel or OneDrive locks the objective-history CSV, the fit continues and
writes the complete history to a uniquely named recovered CSV. The completion
message, settings JSON and report identify the actual file. If progress files
cannot be written, the report still plots the in-memory history.
The safety cap counts optimizer candidates, numerical Jacobian probes, and
final verification. For example, 44 unlocked parameters require about 45
model evaluations per full Jacobian and candidate, so a cap of 100 permits
only a few optimizer steps.
The supplied 1400-step/100-cell LSQ recipe can still take considerable time with
all parameters unlocked. For a diagnostic fit, edit the separate LSQ recipe to
use fewer time steps/cells; check the final result at full resolution.

LSQ CHROMATOGRAM IMPORT UPDATE
JMP startup fix: the LSQ run controls are read by their own named
display boxes. The run-loader always receives both a slot and a run number.
Replace tdm_jmp.py in an existing coding1 folder and restart the JMP launcher
to apply this interface fix while keeping your saved run tables and inputs.
The Least-Squares Refinement tab starts with the number of runs and immediately
shows that many file/axis sections for measured CV/mL-versus-mAU traces.
Objective and run weights can be left at their defaults. For an elution-only conference chromatogram, set
"X = 0 at ELUTION_START"; for a full run set RUN_START. Measured mL is divided
by the column volume and the resulting CV is aligned with the saved LSQ
process recipe. See LSQ_R23_IMPLEMENTATION_AND_USAGE.md for the workflow.

R23 UPDATE — INDEPENDENT LEAST-SQUARES PROCESS SETUP + WEIGHTED RMSE
See LSQ_R23_IMPLEMENTATION_AND_USAGE.md for the 1–10 LSQ run setups, Excel chromatogram attachments, run/point weighting, objective equations, solver locks and improved parameter outputs.

TDM22 R22 - PAPER LANGMUIR EQUILIBRIUM AND %B/SALT RECEIPT
======================================

Read PAPER_LANGMUIR_SALT_CHECK_R22.md for the paper Eq. 4 audit and the
Buffer A/B mixing rule. Read JMP_STARTUP_FIX_R21.md for the OneDrive "No Model Template 4.0" startup
fix. Read CHEMISTRY_FIX_R20.md for the competitive Langmuir b/k_s receipt and
gradient sensitivity check. Read CHEMISTRY_FIX_R19.md for the factor-1000 preflight and live resolved-salt
display. CHEMISTRY_FIX_R18.md records the buffer/load chemistry handoff and
the R17 load-material chemistry and flow-output controls. CHEMISTRY_FIX_R16.md
records the preceding buffer handoff fix. The original R15 startup recipe and
fitted starting values are preserved. No experimental calibration was
performed.

To use R22 with an existing setup, extract this archive to a new folder, then
copy your saved tdm_inputs.json and tdm_batch_runs.csv into the extracted R22
folder before running its JMP launcher. R22 updates the solver, UI, and build
guard together; replacing only tdm_jmp.py in an older folder would omit the
high-affinity numerical fix. See PAPER_LANGMUIR_SALT_CHECK_R22.md.

What changed
------------
The two-component competitive Langmuir equation from Mutavdzin and
Seidel-Morgenstern (2024), Eq. 4 is applied at each cell with
  b_eff,i = b_i exp(k_s,i * local salt M)
  q_i = qmax,i * b_eff,i * c_i / (1 + sum_j b_eff,j * c_j).
The extra exponential is an empirical HIC extension that needs calibration;
the paper's own equation is salt independent. The common EDM/TDM equilibrium
inversion now solves the shared denominator accurately at very high affinity.
The JMP step preview displays the active pump salt and target b_eff, including
STEP's End set-point and explicit endpoint chemistry. The solver uses the
salt after the mixer and local column transport. A low %B blends with Buffer
A; high salt entered only in Buffer B does not keep the blend at that level.

The generated JSL validator now inspects the real model selector and UI
labels. A parent directory containing "No Model Template 4.0" no longer
prevents the JMP interface from opening; see JMP_STARTUP_FIX_R21.md.

Competitive Langmuir results now show the received b and k_s plus the
calculated b_eff = b*exp(k_s*salt) for each stage endpoint. CSV records the
local outlet-cell b_eff at every point; JMP run status shows load/elution
affinities. A LINEAR elution gradient and at least 300 time steps help resolve
peak-position changes that an abrupt low-salt STEP can hide. R20 verification
varies each JMP field independently and checks the shifted elution peak.

HIC runs now reject any selected chemistry resolving above 10 M analytical
salt, or an affinity exponent k_s × salt above the numerical limit of 80.
The JMP status names the affected buffer or direct endpoint, shows its
resolved salt, and states that no new chromatogram was generated. The Model
Parameters tab shows the current buffer and load start salt next to the
conversion factor. Results record the salt source and selected input mode
at each stage endpoint. A factor change has no binding effect for SALT_M;
0 mS/cm in CONDUCTIVITY mode resolves to 0 M and can be an elution endpoint. The 10 M
guard is a broad application limit, not a salt-specific solubility law.
The attached R18 why1.pdf is a saturation breakthrough example: the 1 mL
EDM bed can hold nominally 0.6 mg target protein at qmax 2 g/L stationary
phase and porosity 0.7, but 1.3 mg was loaded. The R19 setup and result show
target load against nominal stationary-phase capacity and label pre-elution
protein as breakthrough. See CHEMISTRY_FIX_R19.md for the exact receipts.

Each buffer and direct chemistry endpoint now has a Salt input selector:
  CONDUCTIVITY: salt [M] = entered conductivity [mS/cm] * calibrated factor.
  SALT_M: entered salt molarity is authoritative for binding. Conductivity is
          optional and independent; if absent it is estimated for display.

Select SALT_M when the actual prepared salt concentration is known. Enter
formula-unit molarity (for example, 1 M ammonium sulfate means 1 M salt).
ION_COMPOSITION separately gives I = 0.5*sum(c_i*z_i^2) in both CPA models.
For 1 M ammonium sulfate, enter 2 M NH4+ (z=+1) and 1 M sulfate (z=-2),
giving I=3 M. HIC consumes 1 M salt; native CPA consumes 3 M ionic strength.
SALT_PROXY is the monovalent approximation I=salt, not a general salt law.

Salt now travels through the buffer mixer and column independently of
conductivity. Results include resolved-buffer receipts and programmed / local
binding salt. Cleared CSV fields no longer revert to default concentrations;
invalid saved tables fail clearly without being overwritten. Older files with
both salt and conductivity retain conductivity priority until SALT_M is chosen.

Each saved run also has Load material chemistry. BUFFER_RECIPE (the default)
uses the load %B recipe. EXPLICIT uses the saved load material source, pH, salt
input, salt molarity and conductivity, even if the load %B values are retained
for display. This is the path for measured feed/load chemistry. The selected
mode and the resolved load chemistry are recorded in the result receipt.
Editing a load pH, salt mode/value or conductivity now also switches the load
source to DIRECT, so an inherited BUFFER_A/B source cannot mask the edited
measurement.

The Model Parameters tab has independent On/Off controls for %B, conductivity
and flow-rate overlays. Flow rate is plotted on its own axis and exported as
flow_mL_min at every reported time point.

Extract the complete ZIP to a new folder, close the previous setup window,
and run 00_RUN_CHROMATOGRAPHY_MODEL_IN_JMP.jsl from this extracted folder.
Use Process Setup > Buffer A/B > Salt input to choose the input basis. The
same choice is available under advanced DIRECT load/wash/elution chemistry.
For the supplied recipe, Buffer B is high salt, loading/washes are 100% B,
and elution ramps from 100% to 0% B. Check resolved salt, not %B alone.
Run the selected model again after changing chemistry.

Checking an early target peak
-----------------------------
The supplied Run 1 is an illustrative HIC setup, not a fit to measured data.
Its entered conductivity and conversion factor resolve Buffer A to 0.1 M
salt and Buffer B to 2.0 M. Loading and both washes stay at 2.0 M; the
8 CV elution ramp starts at total CV 5 and falls toward 0.1 M. At the saved
resolution, the target detector peak is at total CV 11.85 (6.85 CV after
elution starts), where local binding salt is about 0.575 M. Essentially no
target exits before elution and the simulated mass balance closes within
0.001%. The JMP run status and result receipt now state the peak CV, column
outlet CV and local binding salt explicitly.

If a measured peak is later, first compare the actual prepared Buffer A/B
salt molarities with the resolved values. Choose SALT_M for known molarity;
the default 0.02 M/(mS/cm) conductivity conversion is illustrative. Then
compare load capacity, qmax, b, k_s, and gradient duration with the run
receipt. b and k_s are empirical HIC values and need measured chromatograms
for calibration. The LSQ tool can fit selected unlocked parameters against
the measured trace using the matching process recipe. Do not shift a peak by
changing an unmeasured input without recording the new value.

Verification
------------
python -m unittest discover -s tests -v
python verify_r17.py
python verify_r18.py
python verify_r19_counterexample.py
python verify_r20_langmuir_affinity.py
python verify_r21_windows_path.py
python verify_r22_paper_salt_equilibrium.py
python -m py_compile tdm_minimal_io.py tdm_minimal_model.py tdm_bridge.py tdm_jmp.py tdm_least_squares.py

The older verify_model.py, verify_r16.py and verify_jmp_interface_inputs.py
still contain assertions for the pre-R23 combined batch/LSQ run table and
earlier JMP labels. Their failures on those assertions do not indicate that
the current independent LSQ table or solver failed; use the tests above for
the R23 layout.

All four synthetic HIC model tests passed, including high-salt retention,
low-salt breakthrough, gradient recovery, independent conductivity/salt,
sulfate ionic strength, persistence, validation and native CPA IEX response.
The full-resolution supplied preset retains its 11.8514 CV peak.
The R18 regression follows the load chemistry and affinity controls through the
generated JMP field names, bridge, saved run row and effective solver config.
The R19 regression rejects factor-1000 HIC runs through the same route and
checks that a change to an inactive SALT_M factor leaves binding salt unchanged.
The R20 regression changes b and k_s separately through generated JMP sends,
the bridge and local Langmuir affinity, then checks the shifted gradient peaks,
CSV b_eff trace and HTML/manifest receipt. Button tests also check both edits.
Native JMP interaction was unavailable; generated JSL and Python bridge routes
were checked. HIC salt sensitivity must still be fitted for the actual
protein, resin and salt. CPA HIC remains an empirical extension, not the
published CPA ion-exchange electrostatic law. pH is interpolated linearly.
The exact Run 1 buffer-to-solver receipt is saved in
verification/r16_saved_run_chemistry.json. The direct load-material and flow
overlay check is in verification/r17_load_material_chemistry.json.

Original preset and model documentation (updated for R17)
---------------------------------------------------------

R15 startup defaults
--------------------
The included settings open EDM - Competitive Langmuir in HIC mode with one
target protein (100% feed composition) and no impurities. These are the
previously tested synthetic starting values, not an experimental calibration.

Column: 1 mL volume, 50 mm length, 1 mL/min flow, bed porosity 0.75.
Feed concentration: 1 mg/mL; load amount: 1 CV (1 mg applied protein).
Target qmax: 100 g/L non-mobile stationary phase; b: 0.001 L/g;
salt sensitivity: 6 per M; apparent dispersion: 0.067 mm2/s.
UV response: 1000 mAU L/g. Conductivity-to-salt factor: 0.02 M/(mS/cm).
Buffer A: 5 mS/cm, pH 6. Buffer B: 100 mS/cm, pH 6.
Buffer mixer dispersion: 0.10 mL, enabled; salt dispersion: 0.05 mm2/s,
enabled. UV dead volume: 0.15 mL; conductivity dead volume: 0.30 mL.
Programmed %B, conductivity and flow-rate overlays are enabled.
Numerical resolution: 1400 time steps and 100 axial positions.

Run 1 recipe (all flows 1 mL/min):
  Load:      1 CV at 100% B.
  PLW1:      2 CV at 100% B.
  PLW2:      2 CV at 100% B.
  Elution1:  8 CV LINEAR gradient, 100% to 0% B.
  Elution2:  4 CV at 0% B.
Elution starts at 5 CV; the complete recipe is 17 CV. The startup defaults
are saved in both tdm_inputs.json and Run 1 of tdm_batch_runs.csv. Run 1
is selected for simulation and least-squares refinement. Existing saved
recipes in Runs 2-30 are preserved. Choose the desired parameters' lock
states before refining against your own reference chromatogram(s).

R15_VERIFICATION_REPORT.html and verification/r15_default_verification.json
record the saved-run handoff and numerical verification of this exact preset.
The single predicted UV peak is near 11.85 CV, during Elution1; protein
release during Load/PLW1/PLW2 is negligible at this synthetic parameter set.

R14 input changes
-----------------
Saturation capacity qmax,i is now displayed in g/L stationary phase, stored
as qmax_g_L, and consumed directly by both competitive Langmuir solvers.
Old qmax_mg_ml fields, old H/b projects and old fitting-lock paths are
readable. Legacy mg/mL values migrate with factor 1; no factor of 1000 is
applied. The volume basis is not changed by changing the mass/volume units.

The run input formerly labeled Feed total protein is now Feed concentration
[mg/mL]. It describes the concentration of the entire protein mixture in
the injected sample. Each species receives this concentration times its
feed mass fraction. At fixed load volume, increasing feed concentration
increases loaded mass. At fixed load capacity, load CV = capacity / feed
concentration. Total protein mass [g] = concentration [mg/mL] * volume [mL]
/ 1000. The existing Feed_Concentration_mg_mL CSV column is preserved.

Saved lock states (including all locked) and selected 1–10 run lists are now
preserved through normalization and file reload. R14 testing found that the
previous defaults merge could discard those optional saved fields.

Run python verify_r14.py for the g/L migration, input handoff, feed-mass/UV
and saved-state checks. R14_VERIFICATION_REPORT.html records these checks.
R13_VERIFICATION_REPORT.html and the R13 numerical receipts describe the
earlier physics validation; their original dates/build IDs are preserved.

Start
-----
Extract the whole ZIP into a new folder. Open
00_RUN_CHROMATOGRAPHY_MODEL_IN_JMP.jsl in JMP and run it. Keep the Python files
in the same folder. The launcher builds the UI using that folder's files.
Python 3.10 or later, NumPy and SciPy are required. requirements.txt lists the
Python dependencies. No network access is needed to run the model.

Use the included settings files to load the new R15 preset. To carry your
own edited settings forward instead, copy tdm_inputs.json, tdm_batch_runs.csv
and tdm_batch_count.txt from your previous folder before opening the launcher.
Older capacity keys and lock paths are migrated automatically. Saving writes
the new g/L keys. R12 inputs and the previous R14 startup settings are
preserved in prior_inputs/. Earlier reports retain their original recipes,
build IDs and verification dates; the current defaults are in the R15 report.

The supplied R12 recipes loaded at 0% B and ramped to 100% B. With high-salt
Buffer B, this is the opposite direction to the requested HIC process.
The R12 Langmuir affinities were also too weak to guarantee retention over
the stored loading duration. Native CPA in the attached paper is an
ion-exchange adsorption model, not a hydrophobic-interaction model.

Model choices and HIC
---------------------
The original four combinations remain supported, alongside TDM – Modified Wang:
  EDM + Competitive Langmuir
  TDM + Competitive Langmuir
  EDM + CPA
  TDM + CPA
  TDM + Modified Wang (HIC)

Choose HIC in Model Parameters for hydrophobic chromatography. HIC affinity
increases with local salt. The HIC salt sensitivities must be nonnegative.
There is no artificial rule that suppresses all protein output during loading:
weak binding, overload or insufficient TDM rates can physically cause
breakthrough. The result reports the fraction leaving before elution and
flags this behavior so its cause can be investigated.

Competitive Langmuir:
  b_eff,i = b_i exp(k_s,i * salt_M)
  q_i* = qmax_i b_eff,i c_i / (1 + sum_j b_eff,j c_j)
  H_i = qmax_i b_i  (derived; not an independent fitted parameter)

qmax is entered in g/L of non-mobile stationary phase. 1 mg/mL = 1 g/L,
so the numerical conversion factor is 1. Its stationary-phase volume basis
is retained. Packed-bed capacity also includes the stationary-phase fraction;
for EDM, maximum adsorbed mass [g] = qmax [g/L] * (1 - epsilon) * V_col [L]. Salt is resolved from the selected conductivity or direct-molarity input.
The conductivity conversion factor must match the actual buffer/salt. Molar salt is used in the empirical affinity law; it is not a
full activity or salt-composition model.

CPA in HIC mode is an explicit empirical extension:
  K_i(q,s) = Delta_i * available_surface_i(q) * exp(k_s,i * salt_M)
  q_i = K_i(q,s) c_i  (EDM)
  dq_i/dt = kkin_star_i [K_i(q,s) cp_i - q_i]  (TDM)

It retains CPA competition for accessible surface and excluded-area effects.
It uses hydrophobic salt affinity instead of the IEX electrostatic screening
law. This HIC extension is not claimed to be established or calibrated by
the supplied CPA paper. HIC uses Delta, accessible surface, protein diameter
and salt sensitivity; TDM also uses adsorption kinetics and mass transfer.
IEX charge/ligand fields are hidden and excluded from HIC fitting.

Choose ION_EXCHANGE to use native CPA's electrostatic salt/pH response.
The supplied native CPA and pH-dependent parameterization remain available.
Both transport models conserve protein when changing salt changes partitioning.

UV, conductivity, %B and dispersion
----------------------------------
The main chromatogram outputs UV absorbance in mAU:
  UV_total = sum_i response_i [mAU L/g] * c_i [g/L]
The target uses the global UV response; each impurity has its own response.
The CSV also records protein concentrations, species UV and total UV.

The same plot has:
  - dotted programmed %B (the pump command, with no column delay);
  - light dotted column %B after buffer/column transport;
  - dashed conductivity at its detector, on a separate mS/cm axis;
  - dashed stage flow rate, on a separate mL/min axis.
The %B, conductivity and flow-rate overlays have independent On/Off controls
in JMP and checkboxes in each HTML result. Hiding a trace does not change
physics.
Direct endpoint-chemistry stages have no defined programmed %B.

Buffer mixer dispersion volume [mL]: a stirred upstream volume that smooths
and delays the buffer program entering the column. It affects adsorption.
Buffer / salt axial dispersion [mm2/s]: conservative mobile-phase axial
transport, shared by all five model choices. TDM salt is fully pore-accessible;
its residence volume includes interstitial and particle pore fluid. Salt
and ionic strength are tracked separately when ion composition is selected.
The initial column is equilibrated to the EFFECTIVE load chemistry, including
the end value of a STEP load. Protein starts at zero in the equilibrated column.
The buffer mixer and physical salt dispersion can be disabled independently.
Disabling dispersion retains physical column residence delay. The pH transport
is an ideal scalar approximation, not a buffer acid/base equilibrium solver.

System dead volume to UV [mL]: an independent downstream plug-flow delay
of protein/UV at the UV detector.
System dead volume to conductivity [mL]: an independent downstream plug-flow
delay of conductivity. Both delays use accumulated volume, so they remain
correct when stage flows change. They do not alter column binding. Their
location is downstream; buffer mixer volume describes upstream buffer delay.

The column mass balance uses the undelayed outlet, not the delayed detector
trace. A finite run can end with protein still in the column or detector line;
include sufficient low-salt flush when comparing full experimental traces.

Least-squares refinement: 1–10 selected runs
------------------------------------------
1. Save each experimental recipe in its own run, with its actual load, flow,
   feed, Buffer A/B, PLW, elution, column geometry and numerical resolution.
2. Attach the appropriate chromatogram CSV to each saved run.
3. Choose Number of runs to refine [1–10], then choose exactly those LSQ runs
   in the Fit run selectors. Any of the 10 LSQ profile runs can be selected. A run
   cannot be selected twice. Other attached chromatograms are excluded.
4. Set Locked/Unlocked beside the eligible numeric fields. Lock all / Unlock
   all applies to both component AND global/instrument controls. Lock states
   are saved and restored. Disabled/inactive parameters cannot be refined.
5. Choose the objective and baseline policy, then run refinement.

RAW_SSE is the default and directly adapts Osberghaus et al., Eq. 9:
  minimize over theta_unlocked:
    sum_r sum_j [UV_predicted_r(x_rj; theta_fixed, theta_unlocked)
                 - UV_reference_r(x_rj)]^2

One shared mechanistic/instrument vector is simulated with each selected
run's own recipe. Predictions are interpolated to measured coordinates;
there is no automatic peak alignment, time warping or amplitude rescaling.
Fit references must contain a UV signal; AU-labelled values are converted
to mAU before comparison. Geometry, recipe, species names and
feed composition are fixed. Only eligible UNLOCKED values vary.

The solver uses scipy.optimize.least_squares(loss='linear') with bounds and
parameter transforms/scaling. It selects TRF for interior initial values and
dogbox when an unlocked value starts at a bound. An absolute finite-difference
step floor accommodates the numerical forward model. Unlocked zero-valued
nonnegative volumes/dispersion can move away from zero. This is the same
bounded residual-vector least-squares formulation used by MATLAB lsqnonlin,
not MATLAB's binary implementation, and it finds a local least-squares
solution. The reported objective is the FULL squared norm; SciPy internally
uses half that norm, which has the same minimizer. The best feasible tested
parameter vector is re-simulated and applied only if it does not worsen the
starting objective. Reports expose convergence separately from acceptance.

NORMALIZED_MSE is optional: residuals are divided by measured profile scale,
sqrt(point count) and sqrt(run count) to give equal normalized profile weight.
It changes the multi-run weighting and is not silently used as Eq. 9.

Baseline policy NONE is the default: no baseline is inferred from sample
application. Fixed detector baseline is an explicit input (default 0).
INITIAL_MEDIAN is opt-in and estimates the initial 5% median. Use it only
when those initial points represent a baseline rather than protein output.

Every paired finite measured sample is retained, including replicate x values.
References outside the simulated recipe are rejected with a clear message;
extend the recipe/flush or explicitly crop the reference rather than silently
losing measured points. Time axes are converted using the saved variable-flow
recipe. Common CSV encodings, delimiters and instrument metadata are supported.
A signal-only file uses an explicitly reported sample-order x-axis assumption.

Optional species mass% constraints can be assigned independently to any
selected LSQ run at fraction midpoints in CV or mL; other profiles may have
chromatogram data alone. These are
additional normalized composition residuals, not part of the raw UV Eq. 9.
With overlapping peaks, multiple parameters can remain correlated; one shared
fit against several recipes does not guarantee parameter identifiability.

Fit outputs include measured/predicted overlays for every selected run,
RMSE/NRMSE/R2, raw residuals, parameter/lock/bound records, objective history,
convergence/acceptance status, and complete per-profile recipe snapshots.
Accepted parameters update the global model and JMP boxes.

Verification and boundaries
---------------------------
Run python -m unittest discover -s tests -v for the current R23 solver,
JMP handoff and least-squares tests. The older verify_model.py includes
pre-R23 interface assertions and is retained as a historical audit.
Run python verify_r13.py for the R13 HIC, detector and fitting regressions.
Run python verify_r13_step_load.py to check effective STEP-load equilibration,
python verify_r13_unchanged_fit.py to check exact preservation of a matching
fit, and python verify_r13_exports.py to check residual signs and exported
objective contributions. The historical startup_verification.json records
the earlier R13 two-species demonstration. The current R15 startup verification
is recorded in r15_default_verification.json after the saved recipe and shared
parameters are combined.
The verification/ folder contains the numerical receipts and all four
synthetic HIC chromatograms. R13 physics tests include constant high-salt holds,
low-salt loading, high-to-low gradients, mass conservation, species-specific
UV, variable flow, independent detector volumes, physical/display toggles,
literal Eq. 9, reference coverage/replicates, all global locks, selected
1–10 profiles, actual nonlinear refinement in all four combinations, a shared
five-profile refinement, and a fit starting at zero UV dead volume.
An already-matching fit preserves the original floating-point parameter values
exactly instead of changing them through a transform-and-inverse round trip.

These are synthetic numerical tests. No experimental chromatogram was provided,
so the supplied protein/resin parameters have not been calibrated to your sample.
JMP itself is unavailable in this verification environment. Generated JSL
structure and the same Python bridge paths invoked by JMP have been checked;
native JMP drawing and clicking have not been executed.

Sources
-------
Attached SMA paper: Osberghaus et al., Determination of parameters for the steric
mass action model - A comparison between two approaches, J. Chromatogr. A 1233
(2012) 54-65, printed page 56, Eq. 9. This is the least-squares equation; Eq. 9
in the other attached papers is not this objective.
Attached CPA paper: Lorenz-Cristea et al., A systematic approach for estimating
colloidal particle adsorption model parameters, J. Chromatogr. A 1739 (2025)
465512. Its adsorption/calibration setting is IEX.
Attached competitive Langmuir paper: Mutavdzin and Seidel-Morgenstern,
Adsorption 30 (2024) 337-349, DOI 10.1007/s10450-024-00436-z.
https://www.mathworks.com/help/optim/ug/lsqnonlin.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
https://cdn.cytivalifesciences.com/api/public/content/digi-11660-pdf
