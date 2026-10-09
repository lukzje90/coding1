TDM22 R13 - HIC, UV SYSTEM RESPONSE AND GLOBAL LEAST SQUARES
==========================================================

Start
-----
Extract the whole ZIP into a new folder. Open
00_RUN_CHROMATOGRAPHY_MODEL_IN_JMP.jsl in JMP and run it. Keep the Python files
in the same folder. The launcher builds the UI using that folder's files.
Python 3.10 or later, NumPy and SciPy are required. requirements.txt lists the
Python dependencies. No network access is needed to run the model.

R13 opens a working HIC demonstration in Run 1: 1 CV load at 100% B, 2 CV
wash at 100% B, an 8 CV LINEAR gradient from 100% to 0% B, and 4 CV flush
at 0% B. Buffer B is the high-salt endpoint (100 mS/cm); Buffer A is the
low-salt endpoint (0 mS/cm). The demonstration parameters are synthetic,
not fitted to your sample. Original R12 inputs and all original saved run
recipes are preserved in prior_inputs/. Other saved runs are retained.

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

qmax is per stationary-phase volume, as labeled in the UI, not automatically
per total bed volume. Salt is resolved from entered conductivity and the
conductivity-to-salt conversion factor. The factor must match the actual
buffer/salt. Molar salt is used in the empirical affinity law; it is not a
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
  - dashed conductivity at its detector, on a separate mS/cm axis.
The %B and conductivity overlays have independent On/Off controls in JMP
and checkboxes in each HTML result. Hiding a trace does not change physics.
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

The solver uses scipy.optimize.least_squares(method='trf', loss='linear'),
with bounds, parameter transforms/scaling and an absolute finite-difference
step floor suitable for a numerical forward model. Unlocked zero-valued
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
Run python verify_model.py for the existing UI/recipe/solver/CSV/bridge tests.
Run python verify_r13.py for the R13 HIC, detector and fitting regressions.
Run python verify_r13_step_load.py to check effective STEP-load equilibration,
python verify_r13_unchanged_fit.py to check exact preservation of a matching
fit, and python verify_r13_exports.py to check residual signs and exported
objective contributions. startup_verification.json verifies the delivered
two-species Run 1 after the saved recipe and shared parameters are combined.
The verification/ folder contains the numerical receipts and all four
synthetic HIC chromatograms. R13 tests include constant high-salt holds,
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
