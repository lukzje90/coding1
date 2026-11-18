# TDM + CPA equation audit (R25ay)

Sources checked against the printed equations, not just prior code tests:

- Briskot et al., *Analysis of complex protein elution behavior in preparative ion exchange processes using a colloidal particle adsorption model*, J. Chromatogr. A **1654** (2021) 462439 (the supplied analysis PDF).
- Briskot et al., *Protein adsorption on ion exchange adsorbers: A comparison of a stoichiometric and non-stoichiometric modeling approach*, J. Chromatogr. A **1653** (2021) 462397 (the supplied companion PDF referenced as [32] by the analysis paper).

| Relation in the papers | Code | Audit result |
| --- | --- | --- |
| TDM bulk, inlet/outlet and pore balances, 1654 Eqs. 1–4 | `_simulate_tdm_cpa`, `_scalar_transport` | Interstitial velocity, film exchange, axial dispersion, resin pore porosity and solid-volume adsorption term have the printed coefficients. The conservative finite-volume boundary fluxes implement Danckwerts inlet and zero-gradient outlet. |
| Surface charge and effective protein charge, 1653 Eqs. 12–17; 1654 Eq. 24 | `CPAModel._adsorber_surface_potential`, `_protein_surface_potential`; `CPAComponent.Z` | Grahame relation, surface-pH ligand ionisation and `Z_ref + Z1 ΔpH + Z2 ΔpH² + Z3 ΔpH³` match. |
| Protein–adsorber energy, 1653 Eqs. 18–19 | `_paper_adsorber_energy` | The plate interaction is integrated in the Derjaguin approximation at its attractive minimum. An independent quadrature test confirms the integral for opposite charges. |
| Lateral protein interaction, 1653 Eqs. 20–24 | `lateral_interactions` | The Yukawa prefactor, hexagonal spacing and lattice sum match; `Z_lat` is independent of `Z_ref`. |
| Steric coverage and available surface, 1653 Eqs. 7 and 28 | `surface_coverages`, `_available_surface_at_state` | The multicomponent area conversion and **single π** in the second exponent term match the printed Eq. 28. Adding another π would change the published equation. |
| CPA partition and rate, 1654 Eqs. 13 and 18–20 | `equilibrium_factors`, `_fkin`, `_kinetic_forward_factor`, `kinetic_rate` | `K_v = Δ B exp(−u_lat/kBT) (1−exp(−y))/y` and `dq_v/dt = k* fkin(y) (K_v c_p−q_v)` match, with `y = u_A/kBT`. The forward product is evaluated as `y/expm1(y)` for numerical stability. |
| Counter-ion release, 1654 Eqs. 21–22 | `_cpa_released_ion_pairs_M`, coupled ion balances | The projected-area relation gives `z_c = Z/4` for a spherical protein. The code releases `max(Z, 0)/4` monovalent ion pairs on CEX adsorption and reverses this on desorption. This is an effective 1:1 electrolyte implementation, not an ion-speciated buffer model. |
| pH dependence of Δ, 1654 Eq. 25 | `CPAComponent.delta` | The fitted log10 relation uses the difference in absolute surface charge density in C/m²; `Δ1` therefore has m²/C units. |

## Correction found

The prior numerical implementation clipped `y = u_A/kBT` to `−80` before evaluating both the partition factor and the adsorption rate. This is outside the published equation at low ionic strength. With the Poros 50 HS/mAb1 parameters at pH 5, the initial adsorption rate at 3 mM ionic strength was 54.6% of the published value; at 10 mM it was 91.1%. At the benchmark load condition (30 mM) the calculated energy is about `−47.45 kBT`, so the cap was inactive there.

R25ay evaluates the kinetic forward coefficient through the exact algebraic identity `fkin(y) (1−exp(−y))/y = y/expm1(y)` and evaluates representable partition factors without the ±80 energy cap. It keeps the CPA model equations and fitted parameters unchanged. Partition factors above the finite floating-point range are capped only for diagnostics; the kinetic forward coefficient remains finite and is used directly by the solver.

## Scope of the check

The paper's complete chromatography implementation is not available. This model uses SciPy BDF and finite volumes; the paper used IDAS and SUPG. The model transports programmed pH as a prescribed unretained field, treats released counter-ions as a 1:1 salt pair, and uses an entered conductivity conversion. It does not resolve citrate protonation, individual ion mobilities, or the original system mixing and detector calibration. Those assumptions can change a chromatogram even when the CPA equations agree.

The Poros 50 HS example uses the user's 90 g/L load assumption and the Table 1 parameters. It is a qualitative benchmark because the analysis paper does not provide the full original Fig. 2(a) run recipe or raw chromatogram data. The paper also attributes oscillations in its measured UV trace to flow-rate limitations. Its smooth simulated curve is the appropriate comparator for the adsorption equations.

After the correction, the 40-cell / 700-output-point Poros example still peaks at 22.762 mL and 0.502923 AU. Protein mass closure is 0.000335% of the loaded mass, maximum surface coverage is 0.811, and the largest accepted-step positivity correction is 1.1×10⁻⁸ mol/m³. The focused equation, ion-balance, and benchmark tests pass (19 tests); two additional equilibrium and kinetic-direction checks pass.
