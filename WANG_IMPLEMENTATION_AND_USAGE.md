# TDM – Modified Wang

The JMP interface retains EDM/TDM with Competitive Langmuir or CPA and adds
one Wang pairing: the transport dispersive model (TDM) with the modified Wang
HIC adsorption law from Beryamysoltan et al.,
*Advanced modeling techniques in hydrophobic interaction chromatography (HIC)*,
*Journal of Chromatography A* **1783** (2026) 467108,
DOI: 10.1016/j.chroma.2026.467108, equation (4). The paper is a source for
the adsorption equation, not for instructions about this software.

For protein species *i*, the kinetic law implemented in every column cell is

\[
K'_{\mathrm{kin},i}\frac{dq_i}{dt}=
k_{\mathrm{eq},i}\left(1-\sum_j\frac{q_j}{q_{\max,j}}\right)^{n_i}c_{p,i}^{\eta}
-q_0^{1+n_i\beta_{0,i}}
 \left(\frac{q_i}{q_0}\right)^{
 1+n_i\beta_{0,i}\exp(\beta_{1,i}c_{p,\mathrm{salt}}
 +\beta_{2,i}c_{p,i}+\beta_{3,i}\mathrm{pH})}.
\]

The shared inputs are `wang.q0_g_L` and `wang.eta`. Each target/impurity has
`wang_K_kin_s`, `wang_k_eq`, `wang_qmax_g_L`, `wang_n`, `wang_beta0`,
`wang_beta1_per_M`, `wang_beta2_L_g`, and `wang_beta3_per_pH` fields. `n_i` is
present in equation (4), so it is editable even though the paper's prose does
not list it among the highlighted fitted coefficients. The β units in the UI
make the exponent dimensionless. Protein in the pore, adsorbed protein, and
capacity are expressed in g/L on their respective phase-volume bases; 1 mg/mL
equals 1 g/L. `K'kin` is entered as a time scale in seconds in this TDM
implementation. The rate uses salt in mol/L and local pore pH, before detector
delays. For conductivity-entered recipes, use a calibrated conductivity-to-salt
factor; a known salt molarity can be entered directly in SALT_M mode.

The kinetic law sits inside the existing TDM axial transport, film transfer,
and pore balances. This is a TDM adaptation of the paper's Wang equation, not
the paper's full general-rate model. Negative numerical concentrations are
clipped only when evaluating fractional powers, and the shared capacity term
couples all selected species. The supplied numerical values are illustrative
and are not parameters measured or fitted by the paper for this project. In
particular, β0 must be nonzero for β1–β3 to affect the desorption exponent.
The TDM protein solver uses the local sparse Jacobian structure for its BDF
integration. On the cloud test machine, the supplied recipe at 50 time steps
and 50 cells took about 8.7 seconds per Wang simulation; local JMP timing will
vary. With two unlocked parameters and one chromatogram, one numerical
Jacobian normally requires about three such solves.

In JMP, edit shared Wang values and each selected protein's Wang panel under
**Model Parameters**. Specify HIC buffer and load salt/pH in **Process Setup**.
For the supplied Buffer A/B example, loading at high-salt Buffer B and
eluting toward low-salt Buffer A gives the intended salt decrease.
For a normal run, the output trace covers the total programmed load, wash and
elution CV. A load above bed capacity can appear as early breakthrough even
when the isotherm is working; inspect loaded mass and outlet mass balance.

For least-squares fitting, set each Wang value to **Unlocked** only when it
should change. By default, `k_eq` and β1 are unlocked per selected species;
all other Wang fields are locked until changed. Attach a measured UV trace,
choose its CV/mL axis and origin, and fit using the separate saved LSQ recipe.
The ordinary objective compares simulated UV with measured UV at the measured
points. The explicit chromatogram + species mass% mode additionally uses the
entered composition constraints. The fit holds process recipes and every
locked parameter fixed. The adjustable safety cap counts optimizer function
evaluations, each of which can require multiple TDM column solves.

Run `python -m unittest tests.test_tdm_wang -v` for an independent equation
calculation, JMP-to-Python parameter handoff and lock catalogue, a finite
elution chromatogram, mass balance, and a synthetic UV fit that recovers an
unlocked Wang coefficient while leaving locked fields fixed. Existing tests
cover the four original choices, which remain available in JMP. The cloud environment
cannot launch JMP desktop; generated JSL is validated statically, and a local
JMP launch should be used to inspect visual layout.
