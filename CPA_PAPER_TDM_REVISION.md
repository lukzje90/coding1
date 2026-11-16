# TDM + CPA adsorption revision

The ion-exchange TDM + CPA model uses the Briskot et al. colloidal adsorption
equations for the adsorber surface: ligand charge is evaluated at the **surface
pH**, the surface potentials satisfy the Grahame charge balance, and the
protein–adsorber energy is the Derjaguin integral at its attractive minimum.
The CEX ligand pK is a resin input (default 2.3, the value used for the
papers' strong sulfonate exchangers), so another ligand can use its own value.
The arbitrary TDM + CPA system adsorption multiplier is no longer fitted or
used. The pore balance now uses the resin particle porosity in Eq. (4), rather
than an additional fitted species porosity. Species-specific accessible area
remains in the CPA surface state. The scaled-particle available-surface and
CPA kinetic expressions remain. The independently fitted lateral protein
charge Z_lat,i is available for the paper's electrostatic protein–protein
interaction; set it to zero only if that approximation is justified. If it is
to be fitted by LSQ, start it above zero: this interaction depends on the
product of lateral charges, so a zero vector has a zero first derivative.

Enter a measured or assigned **molecular weight [kDa] for each TDM + CPA
component**. The fitted adsorption diameter is a separate quantity. Old
TDM + CPA input files therefore need molecular weights and lateral charges
before simulation; use zero lateral charge only as an explicit approximation.
Old TDM + CPA fits must be refitted because the adsorption energy changed.

The implementation follows Briskot et al., *J. Chromatogr. A* **1611** (2020)
460608, **1653** (2021) 462397, and **1654** (2021) 462439. It uses the 2021
effective-charge description Z(pH); the full amino-acid-sequence charge
regulation described in the 2020 paper requires sequence and residue pK data
that this application does not collect. The other model combinations retain
their previous equations. The 2021 interaction formula itself uses a linear
Poisson–Boltzmann approximation, which the authors caution can lose accuracy
at high surface potentials.

The 2021 preparative examples used effective adsorption charges about 33–140,
lateral charges about 12–72, accessible areas about 140–750 m²/mL solid, and
kinetic parameters about 1e4–8e7 s⁻¹. These are examples from different
protein/resin systems, not universal bounds or automatic replacements for
measured inputs.
