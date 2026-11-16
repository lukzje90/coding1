# TDM + CPA charge and saturation audit (R25an)

The adsorption surface coverage is computed from each species' adsorbed amount
per resin solid volume, divided by its accessible surface area. The fractional
coverage is `theta = pi * N_A * sum(a_i^2 * qv_i / As_i)`. The multicomponent
available-surface probability follows the scaled-particle expression in
Briskot et al., *Journal of Chromatography A* 1653 (2021) 462397, Eq. 28.
The JMP area input is m²/mL of stationary solid and is converted to m⁻¹ by
multiplication by 10^6. These terms were already implemented consistently.

The adsorption-facing protein charge `Z_i(pH)` follows the polynomial in
Briskot et al., *Journal of Chromatography A* 1654 (2021) 462439, Eq. 24.
The pH dependence of the fitted boundary-layer factor `Delta_i` follows
Eq. 25. `Z_lat,i` independently controls lateral electrostatic repulsion.
The resin ligand pK controls the regulated adsorber charge.

**Fixed:** the solver previously replaced the fitted pH reference and
coefficients with a constant charge whenever a single run had no internal pH
change. Therefore a shared LSQ fit across several constant-pH runs ignored
their charge-versus-pH curve. The solver now uses a complete supplied curve
at each run's actual pH. Python batch and global-LSQ pH ranges now span all
selected runs, and the required fields and fitting catalog follow that range.
The JMP panel shows an existing complete curve even for a constant-pH batch.
Partial charge curves are rejected instead of silently ignored. The
ligand-ionisation helper now uses the configured ligand pK.

**Checked:** independent tests compare mixed-species surface blocking with
the paper equation, verify charge and ligand-pK sensitivity, and fit a charge
slope from two synthetic constant-pH chromatograms. The focused CPA and LSQ
tests pass. No CPA saturation or adsorption-energy equation was changed.

**Interpretation:** a very small area can make the model appear saturated
almost immediately. For example, `As = 1 m²/mL solid` and a 13.63 nm protein
have a geometric monolayer ceiling of about 0.0114 mol/m³ solid, or 1.7 g/L
solid for a 150 kDa protein. With void fraction 0.195 and particle porosity
0.752, that is only about 0.34 g/L packed bed before steric exclusion. This
is a rough upper bound, not a calibrated binding capacity. The cited paper
reports some accessible areas around 0.14–0.75 nm⁻¹ (140–750 m²/mL solid)
for the studied resin/protein systems; transfer to a different resin requires
measurement or fitting. Charge inputs near `Z_i = 1` are also much smaller
than the paper's fitted examples (roughly 30–140), though a particular
protein can differ.

**Scope:** the current TDM transports programmed buffer ions independently
of protein adsorption. It does not couple the paper's Eq. 22 counter-ion
release balance back into the bulk-ion transport and conductivity trace.
This limitation is separate from the fitted charge and saturation equations
audited here and matters most when protein loading substantially perturbs
the local mobile-phase ion balance.
