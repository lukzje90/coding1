LEAST-SQUARES PERFORMANCE UPDATE
The LSQ tab now labels the safety cap as optimizer evaluations. Each numerical
Jacobian reruns the column once per unlocked parameter and selected chromatogram,
plus a base solve.
While JMP is busy, least_squares_objective_history.csv gains progress rows with
elapsed seconds and the best objective so far. A repeated base solve and repeated
full-file history rewrites have been removed. The conductivity-detector delay
is excluded from UV fitting because it cannot affect the fitted signal.
The supplied 1400-step/100-cell LSQ recipe can still take considerable time with
all parameters unlocked. For a diagnostic fit, edit the separate LSQ recipe to
use fewer time steps/cells; check the final result at full resolution.

LSQ CHROMATOGRAM IMPORT UPDATE
JMP startup fix: the five LSQ run controls are now read by their own named
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
See LSQ_R23_IMPLEMENTATION_AND_USAGE.md for the new 1–5 LSQ run setups, Excel chromatogram attachments, run/point weighting, objective equations, solver locks and improved parameter outputs.

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

Saved lock states (including all locked) and selected 1-5 run lists are now
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
All four combinations remain supported:
  EDM + Competitive Langmuir
  TDM + Competitive Langmuir
  EDM + CPA
  TDM + CPA

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
transport, shared by all four models. TDM salt is fully pore-accessible;
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

Least-squares refinement: 1-5 selected runs
------------------------------------------
1. Save each experimental recipe in its own run, with its actual load, flow,
   feed, Buffer A/B, PLW, elution, column geometry and numerical resolution.
2. Attach the appropriate chromatogram CSV to each saved run.
3. Choose Number of runs to refine [1-5], then choose exactly those saved runs
   in the Fit run selectors. Any of the 30 saved runs can be selected. A run
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
For concentration-valued references, concentrations are compared directly
and UV response factors are excluded. Geometry, recipe, species names and
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

Optional species mass% constraints can be assigned to a selected profile at
specified CVs; other profiles may have chromatogram data alone. These are
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
1-5 profiles, actual nonlinear refinement in all four combinations, a shared
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
