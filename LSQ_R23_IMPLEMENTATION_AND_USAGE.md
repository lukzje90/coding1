# R23 — Independent least-squares process setup and weighted RMSE

## Model basis

Hahn et al., *Calibration-free inverse modeling of ion-exchange chromatography in industrial antibody purification*, **Engineering in Life Sciences** 16 (2016), pp. 107–113, equation (9), defines the nonlinear sum-of-squared chromatogram residuals:

\[\min_{\theta}\sum_j (y_{\mathrm{measured},j} - y_{\mathrm{model},j}(\theta))^2.\]

The existing GRM/EDM or TDM mechanistic HIC model is **not replaced** by a peak-fitting proxy. For each candidate parameter vector, the existing mechanistic `simulate` function runs every selected fixed process recipe. The predicted detector signal is the sum of simulated species UV signals using species response factors (or total g/L for a concentration-valued measured signal). Linear interpolation evaluates the simulated signal at every experimental CV. Measurements outside the simulated CV window **cause an explicit error**; they are never silently dropped.

The `WEIGHTED_RMSE` option implements:

\[\mathrm{WRMSE}= \sqrt{ \frac{\sum_{r=1}^R W_r\left(\frac{\sum_j w_{rj}(\widehat y_{rj}-y_{rj})^2}{\sum_j w_{rj}}\right)}{\sum_{r=1}^R W_r} }.\]

- `W_r` is **each LSQ run's positive weight**, default `1.0`.
- `w_rj` is **the measurement's optional nonnegative `Weight` column**, default `1.0` for all samples. Every run must have at least one positive point weight.
- Every run contributes a weighted **mean** squared error, so a denser sampled chromatogram does not automatically overwhelm a less densely sampled run. This is deliberate and differs from the raw SSE option.
- The optimizer receives residuals `sqrt(W_r / sum W) * sqrt(w_rj / sum_j w_rj) * (prediction - measurement)`; therefore its least-squares objective is exactly WRMSE squared. Minimizing WRMSE or WRMSE squared produces the same minimizer.
- All profiles in one WRMSE fit must use comparable signal units. Mixing UV and concentration profiles is rejected.
- `RAW_SSE` reproduces equation (9) without weighting or baseline inference. `NORMALIZED_MSE` retains the original equal-run normalized objective.

A bounded nonlinear SciPy least-squares solver fits **only the individually unlocked** mechanistic parameters (Model Parameters tab). It uses trust-region reflective steps for interior starting values and dogbox steps when an entered parameter starts at a bound (for example, zero axial dispersion). All per-run feed, flow, stage lengths, buffers, gradients, and time/space mesh settings remain fixed. Initial and final candidate objectives are re-evaluated. The best verified result is only applied when it is not worse than the starting parameter set. An optimizer stopping at the user-selected evaluation cap is reported distinctly from successful convergence.

## JMP workflow

1. Unzip the whole project and start it using `00_RUN_CHROMATOGRAPHY_MODEL_IN_JMP.jsl` as before. Python dependencies: `pip install -r requirements.txt`.
2. Click **Edit LSQ process setup** at the top of the window or **Edit this LSQ run's process setup** in **Least-Squares Refinement**. The existing **Process Setup** editor now edits a **separate LSQ process table** (`tdm_lsq_process_runs.csv`). The screen identifies which scope is active. Edit LSQ runs 1–5 separately: buffer chemistry, load flow and amount, load feed chemistry, post-load washes, elution gradient/steps and numerical resolution. Click **Save current process recipe**. **Edit normal batch setup** returns to the original independent batch table (`tdm_batch_runs.csv`). Changes to either process table never rewrite the other table.
3. Under **Least-Squares Refinement**, choose an LSQ run 1–5. Attach its measured chromatogram as **Excel `.xlsx`/`.xlsm` or `.csv`** (no need to manually convert Excel to CSV). The file path is stored with *that LSQ profile*; no shared attachment file is overwritten. Set **This LSQ run weight** if required. Each run can have a different attached file and weight. Attach all runs you will select.
   For each file you can optionally set **Excel worksheet**, **Reference X column**, **Measured signal column**, and **Reference X-axis units** independently (AUTO, CV, ML, MIN, S, H, INDEX). Leave the worksheet/columns blank for auto-detection. Set units explicitly if choosing a different X column.
   Weighted RMSE is a chromatogram-only objective; when fitting extra species composition constraints choose RAW_SSE or NORMALIZED_MSE to avoid mixing units.
4. Set **Number of runs to refine** (1–5) and select the desired, distinct LSQ runs. Choose `WEIGHTED_RMSE`, `RAW_SSE` or `NORMALIZED_MSE`. Adjust fixed baseline or opt in to initial-median baseline subtraction. Set **Maximum solver evaluations** to match the number of parameters and numerical expense.
5. In **Model Parameters**, set each candidate's **Locked / Unlocked** selector next to its value. Only **Unlocked** eligible parameters change; column geometry, each LSQ recipe and feed species composition stay fixed. Click **Run least-squares refinement**.
6. Check the success/accepted flags and the per-run measured-vs-predicted overlays. A smaller objective is not necessarily proof of a unique physical parameter vector, so use additional runs/validation and inspect bounds and residual shape.

## Accepted Excel/CSV chromatogram layout

A numeric **CV**, **Volume (mL)**, **Time (min/s/hour)** or sample index column and a numeric **UV/mAU/Signal/Concentration** column are supported. The program scans workbook sheets for a usable table; extra metadata sheets can be ignored. The optional `Weight` column must contain a finite number >= 0 at every signal sample. Example:

| CV | UV 280 [mAU] | Weight |
|---:|---:|---:|
| 0.00 | 0.1 | 0.5 |
| 0.25 | 2.5 | 1.0 |
| 0.50 | 8.1 | 2.0 |

For time axes, the profile's *own varying stage flows* are used to convert time to CV. An unlabeled, single-signal column is treated as uniformly spaced over the full process duration, and the output records this assumption. Rows with blank/missing signal are ignored; measured points beyond the simulated recipe are rejected. `.xls` files should be saved as `.xlsx` first. Formula cells in `.xlsx` need cached calculated values; open/recalculate/save in Excel if blank in Python.

## Output files (after an actual fit)

- `least_squares_improved_parameters.csv` and `.txt`: **all unlocked/optimized mechanistic parameters**, initial values, best improved values, percent changes, and fitted bound flags. With no unlocked parameters the list explicitly says so. These are the primary requested **list of improved parameters**.
- `least_squares_fit_results.csv`: full locked/unlocked candidate parameter audit.
- `least_squares_fit_trace.csv`: individual reference points, predicted values, errors, optional sample/run weights and exact contribution to the objective.
- `least_squares_fit_report.html`: measured-vs-simulated chromatograms, per-run RMSE and weighted RMSE, full selected recipes, improved parameters, optimization progress and convergence/acceptance results.
- `least_squares_objective_history.csv`: logged loss progress for every objective call.
- `least_squares_settings.json`: individual profile locations/weights/recipes, parameter lock list, baseline policy and exact weighting expression.
- `least_squares_fitted_inputs.json`: fitted mechanistic input values.

## Verification and limits

Run `python -m unittest discover -s tests -v` for model-backed synthetic chromatogram fitting, two-profile Excel+CSV ingestion and weight mathematics, separate process-table checks, invalid-weight rejection, and generated-JSL static checks. These tests use synthetic data with a known physical model and recover a perturbed UV parameter; they **do not constitute a fit to real laboratory chromatograms**.

This environment cannot launch the JMP desktop application, so UI interactions are statically generated and audited but should be checked in a local JMP installation. The actual optimum for a given protein/resin **cannot be produced until real chromatogram files and process recipes are supplied**. Large mechanistic multi-run fits may take substantial time; increasing the optimizer evaluation cap improves opportunity to converge but does not ensure global identifiability.
