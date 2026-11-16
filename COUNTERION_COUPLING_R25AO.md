# TDM + CPA counter-ion exchange (R25ao)

The CPA solver now couples protein adsorption to free electrolyte transport.
Briskot et al., *Journal of Chromatography A* 1654 (2021) 462439, Eq. 21
approximates the charge neutralized by each adsorbed protein as its charge
density times its projected area divided by the elementary charge. For a
spherical protein this is `z_c,i = Z_i(pH)/4`. In the cation-exchange model,
only positive protein charge neutralizes the negatively charged adsorber,
so the implemented release is `max(Z_i(pH), 0)/4` monovalent ion pairs per
adsorbed protein. One cation and one anion form each released pair, consistent
with the monovalent electrolyte assumption in the paper's Section 2.3.

The counter-ion correction is solved with the protein TDM states. It has a
bulk excess ion-pair concentration and a pore conserved concentration. The
free pore excess is the conserved concentration plus the counter-ions
released by the current bound-protein inventory. Both phases share the same
conservative axial transport and bulk-to-pore exchange law as the existing
buffer salt. This form remains conservative when `Z_i(pH)` changes while a
protein is bound, without a numerical derivative of the pH program.

The free pore excess changes the ionic strength used by CPA adsorption and
desorption. The column outlet excess changes predicted salt, ionic strength
and conductivity. Conductivity uses the existing calibrated conversion
factor in M/(mS/cm); the detector delay is applied afterwards. The result
CSV adds `counterion_pair_excess_M`, and the result manifest and solver
diagnostics report the minimum free electrolyte levels, excess ion outflow,
remaining column ion inventory and closure error. The build ID was changed
so results from the old solver are marked stale.

Tests cover the projected-charge equation, charge regulation during a pH
change, zero-charge behavior, stronger conductivity response at higher
protein loading, ion conservation, the CSV and manifest, plus the existing
CPA adsorption and LSQ checks. A four-species simulation also closes its
protein and ion balances.

**Scope and assumptions:** the correction represents an effective 1:1
electrolyte. The model does not resolve which named buffer ions occupy the
diffuse layer, ion-specific mobilities, or acid-base buffer speciation. The
programmed pH remains externally specified. Conductivity changes are based
on the user's salt-to-conductivity calibration. These assumptions should be
checked against measured conductivity, particularly at high loading and low
salt. Existing TDM + CPA fitted parameters should be refitted or validated;
the TDM + Langmuir and TDM + Wang equations are unchanged.
